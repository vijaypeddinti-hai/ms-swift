"""
Lazy Sharded Dataset for Megatron-SWIFT distributed training.

Each data parallel rank independently loads only its assigned chunk files:
    chunk_idx % data_parallel_size == data_parallel_rank

This uses the efficient non-streaming path (MegatronPretrainingRandomSampler)
instead of the bottlenecked streaming path (MegatronDataLoaderDispatcher).

Key insight: With data_sharding=True (default), each DP rank gets contiguous
blocks of indices, which aligns with modulo-based chunk assignment.

Supports two chunk formats:

1. RAW FORMAT (requires encode_fn):
   Each line is a raw sample that needs encoding:
   {"messages": [{"role": "user", "content": "..."}], ...}

2. PRE-PACKED FORMAT (pre_packed=True):
   Each line is already tokenized and packed by PackingProducer:
   {
       "input_ids": [...],
       "labels": [...],
       "position_ids": [...],
       "lengths": [L1, L2, ...],
       "pack_length": N
   }

Usage:
    # Raw format (default)
    megatron sft \
        --model /path/to/model \
        --dataset /workspace/data/chunks \
        --sharded_lazy true \
        --sharded_lazy_samples_per_chunk 1000 \
        --max_steps 20000

    # Pre-packed format
    megatron sft \
        --model /path/to/model \
        --dataset /workspace/data/packed_chunks \
        --sharded_lazy true \
        --sharded_lazy_pre_packed true \
        --sharded_lazy_samples_per_chunk 100 \
        --max_steps 20000

Producer must write chunks with sequential naming:
    chunk_00000.jsonl, chunk_00001.jsonl, chunk_00002.jsonl, ...
"""

import json
import time
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch.distributed as dist
from torch.utils.data import Dataset

from swift.utils import get_logger

logger = get_logger()


def get_data_parallel_info() -> tuple:
    """Get data parallel rank and world size.

    Tries Megatron's mpu first, falls back to torch.distributed.

    Returns:
        Tuple of (dp_rank, dp_world_size)
    """
    try:
        from megatron.core import mpu
        if mpu.is_initialized():
            return mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
    except ImportError:
        pass

    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    return 0, 1


class LazyShardedDataset(Dataset):
    """Map-style dataset where each DP rank reads only its assigned chunks.

    Chunks are assigned via modulo: chunk_idx % dp_world_size == dp_rank

    This dataset:
    - Provides __len__() so it uses MegatronPretrainingRandomSampler (efficient)
    - Does NOT use MegatronDataLoaderDispatcher (bottleneck)
    - Each rank loads its assigned chunks lazily (waits if not yet available)
    - Encodes samples with return_length=True for padding_free compatibility

    With data_sharding=True (the default), MegatronPretrainingRandomSampler gives
    each rank contiguous blocks of indices:
        - Rank 0: indices [0, bucket_size)
        - Rank 1: indices [bucket_size, 2*bucket_size)
        - etc.

    This aligns with our chunk assignment where:
        - Chunk 0 contains samples [0, samples_per_chunk)
        - Chunk 1 contains samples [samples_per_chunk, 2*samples_per_chunk)
        - etc.

    Args:
        directory: Path to directory containing chunk files (chunk_00000.jsonl, etc.)
        encode_fn: Function to encode samples (typically template.encode). Required unless pre_packed=True.
        samples_per_chunk: Expected samples per chunk file
        max_total_samples: Upper bound for __len__() - training stops via max_steps
        wait_poll_interval: Seconds between polls when waiting for chunks
        max_wait_time: Maximum seconds to wait for a chunk (0 = wait forever)
        mark_completed: Create .completed marker after fully reading a chunk
        chunk_pattern: Regex to extract chunk index from filename
        dp_rank: Override data parallel rank (default: auto-detect)
        dp_world_size: Override data parallel world size (default: auto-detect)
        pre_packed: If True, chunks contain pre-packed samples (from PackingProducer)
                   that don't need encoding. Default: False.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        encode_fn: Optional[Callable[[Dict[str, Any], bool], Optional[Dict[str, Any]]]] = None,
        samples_per_chunk: int = 1000,
        max_total_samples: int = 100_000_000,
        wait_poll_interval: float = 5.0,
        max_wait_time: float = 0,
        mark_completed: bool = True,
        chunk_pattern: str = r'chunk_(\d+)\.jsonl',
        dp_rank: Optional[int] = None,
        dp_world_size: Optional[int] = None,
        pre_packed: bool = False,
    ):
        self.directory = Path(directory)
        self.encode_fn = encode_fn
        self.pre_packed = pre_packed

        # Validate: need encode_fn unless pre_packed
        if not pre_packed and encode_fn is None:
            raise ValueError("encode_fn is required when pre_packed=False")
        self.samples_per_chunk = samples_per_chunk
        self.max_total_samples = max_total_samples
        self.wait_poll_interval = wait_poll_interval
        self.max_wait_time = max_wait_time
        self.mark_completed = mark_completed
        self.chunk_pattern = re.compile(chunk_pattern)

        # Get Megatron data parallel info
        if dp_rank is not None and dp_world_size is not None:
            self.dp_rank = dp_rank
            self.dp_world_size = dp_world_size
        else:
            self.dp_rank, self.dp_world_size = get_data_parallel_info()

        # Cache: chunk_idx -> list of encoded samples
        self._chunk_cache: Dict[int, List[Dict[str, Any]]] = {}

        # Track chunks we've fully processed
        self._completed_chunks: set = set()

        logger.info(
            f'[LazyShardedDataset] Initialized: dp_rank={self.dp_rank}/{self.dp_world_size}, '
            f'directory={self.directory}, samples_per_chunk={samples_per_chunk}, '
            f'pre_packed={pre_packed}'
        )

    def __len__(self) -> int:
        """Return max_total_samples.

        Training stops via --max_steps, not dataset exhaustion.
        This large number ensures BatchSamplerShard treats us as a sized dataset.
        """
        return self.max_total_samples

    def _is_my_chunk(self, chunk_idx: int) -> bool:
        """Check if this chunk is assigned to this DP rank."""
        return chunk_idx % self.dp_world_size == self.dp_rank

    def _chunk_path(self, chunk_idx: int) -> Path:
        """Get expected path for a chunk index."""
        return self.directory / f'chunk_{chunk_idx:05d}.jsonl'

    def _completed_marker_path(self, chunk_path: Path) -> Path:
        """Get .completed marker path for a chunk."""
        return chunk_path.with_suffix(chunk_path.suffix + '.completed')

    def _wait_for_chunk(self, chunk_idx: int) -> Optional[Path]:
        """Wait for a chunk file to appear on disk.

        Args:
            chunk_idx: The chunk index to wait for

        Returns:
            Path to the chunk file, or None if timeout/already completed
        """
        chunk_path = self._chunk_path(chunk_idx)
        completed_marker = self._completed_marker_path(chunk_path)
        start_time = time.time()
        logged_waiting = False

        while True:
            # Check if chunk exists
            if chunk_path.exists():
                # Brief delay to ensure file is fully written
                file_mtime = chunk_path.stat().st_mtime
                if time.time() - file_mtime > 1.0:
                    return chunk_path
                # File is very new, wait a bit
                time.sleep(1.0)
                continue

            # Check if already completed (was processed and deleted by producer)
            if completed_marker.exists():
                logger.debug(f'[LazyShardedDataset] Chunk {chunk_idx} already completed')
                return None

            # Check timeout
            elapsed = time.time() - start_time
            if self.max_wait_time > 0 and elapsed > self.max_wait_time:
                logger.warning(
                    f'[LazyShardedDataset] Rank {self.dp_rank}: Timeout after {elapsed:.1f}s '
                    f'waiting for chunk {chunk_idx}'
                )
                return None

            # Log waiting message (once)
            if not logged_waiting:
                logger.info(
                    f'[LazyShardedDataset] Rank {self.dp_rank}: Waiting for chunk {chunk_idx} '
                    f'({chunk_path.name})...'
                )
                logged_waiting = True

            time.sleep(self.wait_poll_interval)

    def _load_chunk(self, chunk_idx: int) -> List[Dict[str, Any]]:
        """Load and encode all samples from a chunk file.

        Args:
            chunk_idx: The chunk index to load

        Returns:
            List of encoded samples from the chunk
        """
        # Return cached if available
        if chunk_idx in self._chunk_cache:
            return self._chunk_cache[chunk_idx]

        # Verify this chunk belongs to this rank
        if not self._is_my_chunk(chunk_idx):
            logger.warning(
                f'[LazyShardedDataset] Rank {self.dp_rank} asked for chunk {chunk_idx} '
                f'but it belongs to rank {chunk_idx % self.dp_world_size}'
            )
            return []

        # Wait for chunk to appear
        chunk_path = self._wait_for_chunk(chunk_idx)
        if chunk_path is None:
            return []

        # Load samples (and encode if not pre-packed)
        samples = []
        errors = 0
        try:
            with open(chunk_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)

                        if self.pre_packed:
                            # Pre-packed format: use directly, add length field if missing
                            if 'length' not in raw and 'pack_length' in raw:
                                raw['length'] = raw['pack_length']
                            samples.append(raw)
                        else:
                            # Raw format: encode using template
                            # CRITICAL: return_length=True for Megatron's padding_free mode
                            encoded = self.encode_fn(raw, return_length=True)
                            if encoded is not None:
                                samples.append(encoded)

                    except json.JSONDecodeError as e:
                        errors += 1
                        if errors <= 3:
                            logger.warning(f'Invalid JSON at {chunk_path.name}:{line_num}: {e}')
                    except Exception as e:
                        errors += 1
                        if errors <= 3:
                            logger.warning(f'Encoding error at {chunk_path.name}:{line_num}: {e}')

            if errors > 3:
                logger.warning(f'[LazyShardedDataset] {errors} total errors in chunk {chunk_idx}')

            logger.info(
                f'[LazyShardedDataset] Rank {self.dp_rank}: Loaded chunk {chunk_idx} '
                f'({len(samples)} samples from {chunk_path.name})'
            )

            # Mark as completed so producer can delete this chunk
            if self.mark_completed and samples:
                try:
                    self._completed_marker_path(chunk_path).touch()
                    logger.debug(f'[LazyShardedDataset] Created completion marker for chunk {chunk_idx}')
                except OSError as e:
                    logger.warning(f'Failed to create completion marker for chunk {chunk_idx}: {e}')

        except FileNotFoundError:
            logger.warning(f'[LazyShardedDataset] Chunk {chunk_idx} disappeared while reading')
            return []
        except Exception as e:
            logger.error(f'[LazyShardedDataset] Error reading chunk {chunk_idx}: {e}')
            return []

        # Cache the loaded samples
        self._chunk_cache[chunk_idx] = samples
        return samples

    def _local_idx_to_chunk_and_offset(self, local_idx: int) -> tuple:
        """Map LOCAL sample index to (global_chunk_idx, sample_offset_in_chunk).

        LOCAL index interpretation (for producer-consumer sequential access):
            - Each rank treats indices as local to its assigned chunks
            - local_idx 0 to samples_per_chunk-1 → this rank's 1st chunk
            - local_idx samples_per_chunk to 2*samples_per_chunk-1 → this rank's 2nd chunk
            - etc.

        This rank's chunks (modulo assignment):
            - 1st chunk: global chunk index = dp_rank
            - 2nd chunk: global chunk index = dp_rank + dp_world_size
            - 3rd chunk: global chunk index = dp_rank + 2*dp_world_size
            - etc.

        Example with dp_world_size=8, dp_rank=3, samples_per_chunk=1000:
            local_idx 0-999   → global chunk 3  (rank's 1st chunk)
            local_idx 1000-1999 → global chunk 11 (rank's 2nd chunk)
            local_idx 2000-2999 → global chunk 19 (rank's 3rd chunk)

        Args:
            local_idx: Local sample index for this rank

        Returns:
            Tuple of (global_chunk_idx, sample_offset)
        """
        my_chunk_num = local_idx // self.samples_per_chunk  # Which of MY chunks (0, 1, 2, ...)
        sample_offset = local_idx % self.samples_per_chunk

        # Map to global chunk index: rank, rank+world_size, rank+2*world_size, ...
        global_chunk_idx = self.dp_rank + (my_chunk_num * self.dp_world_size)

        return global_chunk_idx, sample_offset

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index.

        IMPORTANT: For producer-consumer streaming, use --train_dataloader_shuffle false
        so that indices are sequential (0, 1, 2, ...). This ensures chunks are consumed
        in order as they're produced.

        The index is treated as LOCAL to this rank and mapped to this rank's assigned
        chunks via modulo assignment.

        Args:
            idx: Sample index (treated as local to this rank)

        Returns:
            Encoded sample dictionary
        """
        global_chunk_idx, sample_offset = self._local_idx_to_chunk_and_offset(idx)

        # Load chunk (waits if not yet available)
        samples = self._load_chunk(global_chunk_idx)

        if not samples:
            # Chunk not available yet - this shouldn't happen with sequential access
            # and a producer that writes chunks in order
            raise IndexError(
                f'[LazyShardedDataset] Rank {self.dp_rank}: No samples available for idx {idx} '
                f'(mapped to global chunk {global_chunk_idx}). '
                f'Ensure producer is writing chunks in order and use --train_dataloader_shuffle false.'
            )

        # Handle case where chunk has fewer samples than expected
        if sample_offset >= len(samples):
            sample_offset = sample_offset % len(samples)

        return samples[sample_offset]

    def evict_chunk(self, chunk_idx: int) -> None:
        """Remove chunk from cache to free memory.

        Call this after you're done with a chunk to reduce memory usage.

        Args:
            chunk_idx: The chunk index to evict
        """
        if chunk_idx in self._chunk_cache:
            del self._chunk_cache[chunk_idx]
            logger.debug(f'[LazyShardedDataset] Evicted chunk {chunk_idx} from cache')

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the chunk cache.

        Returns:
            Dictionary with cache statistics
        """
        return {
            'cached_chunks': len(self._chunk_cache),
            'cached_chunk_ids': list(self._chunk_cache.keys()),
            'total_cached_samples': sum(len(s) for s in self._chunk_cache.values()),
        }
