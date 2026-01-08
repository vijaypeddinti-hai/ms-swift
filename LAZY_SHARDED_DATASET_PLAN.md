# Implementation Plan: LazyShardedDataset

## Goal

Eliminate the rank 0 data loading bottleneck in streaming training by having each rank independently load its assigned chunk files using modulo-based assignment.

## Current Problem

```
Current Flow (streaming=True):

Producer → chunks/ → [Rank 0 reads ALL chunks]
                            ↓
                     DataLoaderDispatcher
                            ↓
                     scatter_object_list()
                            ↓
              ┌─────────────┼─────────────┐
              ↓             ↓             ↓
           Rank 0        Rank 1   ...  Rank N

Bottleneck: Rank 0 loads ~3MB audio features per sample, then sends over network
```

## Proposed Solution

```
New Flow (LazyShardedDataset):

Producer → chunks/ → Each rank reads ONLY its assigned chunks

           chunk_00000.jsonl → Rank 0 (0 % 8 == 0)
           chunk_00001.jsonl → Rank 1 (1 % 8 == 1)
           chunk_00002.jsonl → Rank 2 (2 % 8 == 2)
           ...
           chunk_00008.jsonl → Rank 0 (8 % 8 == 0)
           chunk_00009.jsonl → Rank 1 (9 % 8 == 1)
           ...

No scatter! Each rank reads directly from shared filesystem.
Uses BatchSamplerShard (efficient) instead of DataLoaderDispatcher (bottleneck).
```

## Key Insight

The decision between efficient vs bottleneck path is made in `get_train_dataloader()`:

```python
if hasattr(train_dataset, '__len__'):
    # Map-style → BatchSamplerShard (each rank reads own data) ✓
else:
    # IterableDataset → DataLoaderDispatcher (rank 0 reads all) ✗
```

By providing `__len__()`, we force the efficient path.

---

## Implementation Details

### 1. New File: `swift/llm/dataset/lazy_sharded.py`

```python
"""
Lazy Sharded Dataset for distributed training without rank 0 bottleneck.

Each rank independently loads only its assigned chunk files based on:
    chunk_idx % world_size == rank

This avoids the DataLoaderDispatcher scatter bottleneck by using a map-style
Dataset that works with BatchSamplerShard.
"""

import time
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch.distributed as dist
from torch.utils.data import Dataset

from swift.utils import get_logger

logger = get_logger()


class LazyShardedDataset(Dataset):
    """Map-style dataset where each rank reads only its assigned chunks.

    Chunks are assigned via modulo: chunk_idx % world_size == rank

    This enables parallel data loading without the rank 0 scatter bottleneck
    that occurs with IterableDataset + DataLoaderDispatcher.

    Args:
        directory: Path to directory containing chunk files.
        encode_fn: Function to encode samples (typically template.encode).
        chunk_pattern: Regex pattern to extract chunk index from filename.
            Must have a capture group for the index. Default: r'chunk_(\d+)\.jsonl'
        samples_per_chunk: Expected samples per chunk (for __len__ estimation).
        max_samples: Maximum samples to return (acts as upper bound for __len__).
        wait_poll_interval: Seconds between polls when waiting for chunks.
        max_wait_time: Maximum seconds to wait for a chunk (0 = infinite).
        mark_completed: Create .completed marker after reading chunk.
        world_size: Override world_size (default: from dist.get_world_size()).
        rank: Override rank (default: from dist.get_rank()).
    """

    def __init__(
        self,
        directory: Union[str, Path],
        encode_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        chunk_pattern: str = r'chunk_(\d+)\.jsonl',
        samples_per_chunk: int = 1000,
        max_samples: int = 10_000_000,
        wait_poll_interval: float = 5.0,
        max_wait_time: float = 0,
        mark_completed: bool = True,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self.directory = Path(directory)
        self.encode_fn = encode_fn
        self.chunk_pattern = re.compile(chunk_pattern)
        self.samples_per_chunk = samples_per_chunk
        self.max_samples = max_samples
        self.wait_poll_interval = wait_poll_interval
        self.max_wait_time = max_wait_time
        self.mark_completed = mark_completed

        # Get distributed info
        if world_size is not None:
            self.world_size = world_size
        elif dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        if rank is not None:
            self.rank = rank
        elif dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0

        # Cache for loaded chunks: chunk_idx -> List[encoded_samples]
        self._chunk_cache: Dict[int, List[Dict[str, Any]]] = {}

        # Track which chunks we've seen
        self._known_chunks: Dict[int, Path] = {}

        logger.info(
            f'[LazyShardedDataset] Initialized: rank={self.rank}/{self.world_size}, '
            f'directory={self.directory}, samples_per_chunk={self.samples_per_chunk}'
        )

    def __len__(self) -> int:
        """Return max_samples. Training stops via max_steps, not dataset exhaustion."""
        return self.max_samples

    def _scan_for_chunks(self) -> Dict[int, Path]:
        """Scan directory for chunk files and return {chunk_idx: path}."""
        chunks = {}
        if not self.directory.exists():
            return chunks

        for path in self.directory.glob('*.jsonl'):
            # Skip completed chunks
            if self.mark_completed and path.with_suffix(path.suffix + '.completed').exists():
                continue

            match = self.chunk_pattern.search(path.name)
            if match:
                chunk_idx = int(match.group(1))
                chunks[chunk_idx] = path

        return chunks

    def _is_my_chunk(self, chunk_idx: int) -> bool:
        """Check if this chunk is assigned to this rank."""
        return chunk_idx % self.world_size == self.rank

    def _get_chunk_path(self, chunk_idx: int) -> Path:
        """Get the expected path for a chunk index."""
        # Reconstruct filename from pattern (assumes pattern like 'chunk_(\d+)\.jsonl')
        return self.directory / f'chunk_{chunk_idx:05d}.jsonl'

    def _wait_for_chunk(self, chunk_idx: int) -> Optional[Path]:
        """Wait for a specific chunk file to appear."""
        chunk_path = self._get_chunk_path(chunk_idx)
        start_time = time.time()

        while not chunk_path.exists():
            elapsed = time.time() - start_time

            if self.max_wait_time > 0 and elapsed > self.max_wait_time:
                logger.warning(
                    f'[LazyShardedDataset] Rank {self.rank}: Timeout waiting for {chunk_path.name}'
                )
                return None

            # Also check if a .completed marker exists (chunk was already processed and deleted)
            if chunk_path.with_suffix(chunk_path.suffix + '.completed').exists():
                logger.debug(f'[LazyShardedDataset] Chunk {chunk_idx} already completed, skipping')
                return None

            logger.debug(
                f'[LazyShardedDataset] Rank {self.rank}: Waiting for {chunk_path.name}...'
            )
            time.sleep(self.wait_poll_interval)

        return chunk_path

    def _load_chunk(self, chunk_idx: int) -> List[Dict[str, Any]]:
        """Load and encode all samples from a chunk."""
        if chunk_idx in self._chunk_cache:
            return self._chunk_cache[chunk_idx]

        chunk_path = self._wait_for_chunk(chunk_idx)
        if chunk_path is None:
            return []

        samples = []
        try:
            with open(chunk_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_sample = json.loads(line)
                        # Encode with return_length=True for padding_free compatibility
                        encoded = self.encode_fn(raw_sample, return_length=True)
                        if encoded is not None:
                            samples.append(encoded)
                    except json.JSONDecodeError as e:
                        logger.warning(f'Invalid JSON at {chunk_path}:{line_num}: {e}')
                    except Exception as e:
                        logger.warning(f'Encoding error at {chunk_path}:{line_num}: {e}')

            logger.info(
                f'[LazyShardedDataset] Rank {self.rank}: Loaded chunk {chunk_idx} '
                f'({len(samples)} samples)'
            )

            # Mark as completed
            if self.mark_completed and samples:
                marker_path = chunk_path.with_suffix(chunk_path.suffix + '.completed')
                try:
                    marker_path.touch()
                except (OSError, IOError) as e:
                    logger.warning(f'Failed to create completion marker: {e}')

        except Exception as e:
            logger.error(f'[LazyShardedDataset] Error reading {chunk_path}: {e}')
            return []

        self._chunk_cache[chunk_idx] = samples
        return samples

    def _local_idx_to_chunk_and_offset(self, local_idx: int) -> tuple:
        """Map a local index to (chunk_idx, sample_offset).

        Local indices are relative to this rank's assigned chunks.

        Example with world_size=4, rank=1, samples_per_chunk=100:
            local_idx 0-99   → chunk 1 (1 % 4 == 1), offset 0-99
            local_idx 100-199 → chunk 5 (5 % 4 == 1), offset 0-99
            local_idx 200-299 → chunk 9 (9 % 4 == 1), offset 0-99
        """
        # Which of "my" chunks does this index fall into?
        my_chunk_num = local_idx // self.samples_per_chunk
        sample_offset = local_idx % self.samples_per_chunk

        # Map to global chunk index
        # My chunks are: rank, rank + world_size, rank + 2*world_size, ...
        chunk_idx = self.rank + (my_chunk_num * self.world_size)

        return chunk_idx, sample_offset

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index.

        The idx is a global index from BatchSamplerShard. We map it to our
        local chunk assignment.
        """
        # Map global idx to local idx for this rank
        # BatchSamplerShard gives us indices like: rank, rank+world_size, rank+2*world_size, ...
        # But it may also shuffle, so we can't assume ordering.
        #
        # Simpler approach: treat idx as a local sequential index into our samples.
        # BatchSamplerShard already handles the distribution.

        chunk_idx, sample_offset = self._local_idx_to_chunk_and_offset(idx)

        samples = self._load_chunk(chunk_idx)

        if not samples:
            # Chunk not available or empty - return a placeholder that will be filtered
            # Or we could block here waiting for more data
            logger.warning(
                f'[LazyShardedDataset] Rank {self.rank}: No samples for idx {idx} '
                f'(chunk {chunk_idx})'
            )
            # Try next chunk as fallback
            return self.__getitem__(idx + self.samples_per_chunk)

        # Handle case where chunk has fewer samples than expected
        if sample_offset >= len(samples):
            sample_offset = sample_offset % len(samples)

        return samples[sample_offset]

    def evict_chunk(self, chunk_idx: int) -> None:
        """Remove a chunk from cache to free memory."""
        if chunk_idx in self._chunk_cache:
            del self._chunk_cache[chunk_idx]
            logger.debug(f'[LazyShardedDataset] Evicted chunk {chunk_idx} from cache')
```

### 2. New Argument: `--sharded_lazy` in `data_args.py`

Add to `DataArguments`:

```python
sharded_lazy: bool = False
# When True, use LazyShardedDataset for distributed chunk loading.
# Each rank loads only chunks where chunk_idx % world_size == rank.
# Requires predictable chunk naming: chunk_00000.jsonl, chunk_00001.jsonl, ...
# Incompatible with streaming=True (uses map-style dataset instead).

sharded_lazy_samples_per_chunk: int = 1000
# Expected number of samples per chunk file for LazyShardedDataset.
```

### 3. Integration in `sft.py` `_post_process_datasets()`

```python
def _post_process_datasets(self, datasets: List) -> List:
    args = self.args
    template = self.template

    for i, dataset in enumerate(datasets):
        if dataset is None:
            continue

        # NEW: Handle sharded_lazy mode
        if args.sharded_lazy:
            from swift.llm.dataset.lazy_sharded import LazyShardedDataset

            # dataset should be a path string in this case
            dataset_path = args.dataset[i] if isinstance(args.dataset, list) else args.dataset

            dataset = LazyShardedDataset(
                directory=dataset_path,
                encode_fn=template.encode,
                samples_per_chunk=args.sharded_lazy_samples_per_chunk,
                max_samples=args.max_steps * args.per_device_train_batch_size * 10,
                mark_completed=True,
            )
            datasets[i] = dataset
            continue

        # ... existing code for streaming, packing, etc.
```

### 4. Modify Dataset Loading in `loader.py`

When `sharded_lazy=True`, skip normal dataset loading and just return the directory path:

```python
def load_dataset(...):
    if args.sharded_lazy:
        # Return path - LazyShardedDataset will handle loading
        return dataset_path
    # ... existing loading logic
```

---

## File Changes Summary

| File | Change |
|------|--------|
| `swift/llm/dataset/lazy_sharded.py` | **NEW** - LazyShardedDataset implementation |
| `swift/llm/dataset/__init__.py` | Export LazyShardedDataset |
| `swift/llm/argument/base_args/data_args.py` | Add `sharded_lazy`, `sharded_lazy_samples_per_chunk` |
| `swift/llm/train/sft.py` | Route to LazyShardedDataset in `_post_process_datasets` |
| `swift/llm/dataset/loader.py` | Handle sharded_lazy in load path |

---

## Usage

```bash
megatron sft \
    --model /path/to/Qwen3-Omni-30B-A3B-Instruct \
    --dataset /workspace/data/chunks \
    --sharded_lazy true \
    --sharded_lazy_samples_per_chunk 1000 \
    --packing false \
    --max_steps 20000 \
    ...
```

**Producer writes to**: `/workspace/data/chunks/chunk_00000.jsonl`, `chunk_00001.jsonl`, ...

**Each rank reads**: Only chunks where `chunk_idx % world_size == rank`

**No streaming flag needed** - LazyShardedDataset is map-style.

---

## Verification Checklist

- [ ] `hasattr(LazyShardedDataset, '__len__')` returns True → uses BatchSamplerShard
- [ ] Each rank only loads its assigned chunks (verify with logging)
- [ ] `encode_fn` called with `return_length=True` → padding_free works
- [ ] Completion markers created → producer can delete processed chunks
- [ ] Waiting for chunks works when producer is slower than training
- [ ] Memory: chunks evicted from cache after use

---

## Questions for DeepWiki Validation

1. Is `_post_process_datasets` the correct integration point for custom dataset wrappers?
2. Does BatchSamplerShard work correctly when `__len__` returns a large fixed number?
3. Are there any other places where streaming vs non-streaming is checked that we need to handle?
4. Does Megatron-SWIFT have any additional data loading hooks we need to consider?
