# Implementation Plan: LazyShardedDataset for Megatron-SWIFT

## Goal

Eliminate the rank 0 data loading bottleneck by using the efficient non-streaming path in Megatron-SWIFT.

## Current Problem (CONFIRMED)

```
streaming=True path:

MegatronSft.run()
    ↓
if args.streaming:
    train_dataset = build_streaming_dataloader(args, train_dataset, data_collator)
    ↓
build_streaming_dataloader() creates:
    DataLoader(dataset) → MegatronDataLoaderDispatcher → cyclic_iter
    ↓
MegatronDataLoaderDispatcher inherits DataLoaderDispatcher:
    def __iter__(self):
        if self.rank == 0:
            data = [next(base_iter) for _ in range(self.world_size)]  # RANK 0 READS ALL
            data = self._scatter_object_list(data)                      # THEN SCATTERS
        else:
            data = self._scatter_object_list(None)                      # OTHER RANKS RECEIVE

BOTTLENECK: Rank 0 loads ~3MB audio per sample, then sends over network to all ranks
```

## Efficient Path (ALREADY EXISTS)

```
streaming=False path:

MegatronSft.run()
    ↓
# Does NOT call build_streaming_dataloader()
self.trainer.train(train_dataset, val_dataset, data_collator)
    ↓
BaseMegatronTrainer.build_pretraining_data_loader()
    ↓
batch_sampler = MegatronPretrainingSampler(
    total_samples=len(dataset),
    data_parallel_rank=mpu.get_data_parallel_rank(),      # EACH RANK
    data_parallel_size=mpu.get_data_parallel_world_size(), # GETS OWN SHARD
)
    ↓
DataLoader(dataset, batch_sampler=batch_sampler)

EFFICIENT: Each rank reads only its assigned samples directly from disk!
```

## Solution: Use Non-Streaming Path with Lazy Loading

**Key insight**: Don't set `streaming=True`. Instead, create a map-style dataset that:
1. Provides `__len__()` so it's NOT treated as IterableDataset
2. Each rank only loads chunks assigned to it: `chunk_idx % world_size == rank`
3. Handles lazy file discovery (wait for chunks to appear)
4. Works with `MegatronPretrainingSampler` automatically

---

## Implementation Details

### 1. New File: `swift/llm/dataset/lazy_sharded.py`

```python
"""
Lazy Sharded Dataset for Megatron-SWIFT distributed training.

Each rank independently loads only its assigned chunk files:
    chunk_idx % data_parallel_size == data_parallel_rank

This uses the efficient non-streaming path (MegatronPretrainingSampler)
instead of the bottlenecked streaming path (MegatronDataLoaderDispatcher).
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


def get_data_parallel_info():
    """Get data parallel rank and world size, handling Megatron's mpu if available."""
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
    - Provides __len__() so it goes through MegatronPretrainingSampler (efficient)
    - NOT through MegatronDataLoaderDispatcher (bottleneck)
    - Each rank loads its assigned chunks lazily (waits if not yet available)
    - Encodes samples with return_length=True for padding_free compatibility

    Args:
        directory: Path to directory containing chunk files (chunk_00000.jsonl, etc.)
        encode_fn: Function to encode samples (typically template.encode)
        samples_per_chunk: Expected samples per chunk file
        max_total_samples: Upper bound for __len__() - training stops via max_steps
        wait_poll_interval: Seconds between polls when waiting for chunks
        max_wait_time: Maximum seconds to wait for a chunk (0 = wait forever)
        mark_completed: Create .completed marker after fully reading a chunk
        chunk_pattern: Regex to extract chunk index from filename
    """

    def __init__(
        self,
        directory: Union[str, Path],
        encode_fn: Callable[[Dict[str, Any], bool], Optional[Dict[str, Any]]],
        samples_per_chunk: int = 1000,
        max_total_samples: int = 100_000_000,
        wait_poll_interval: float = 5.0,
        max_wait_time: float = 0,
        mark_completed: bool = True,
        chunk_pattern: str = r'chunk_(\d+)\.jsonl',
    ):
        self.directory = Path(directory)
        self.encode_fn = encode_fn
        self.samples_per_chunk = samples_per_chunk
        self.max_total_samples = max_total_samples
        self.wait_poll_interval = wait_poll_interval
        self.max_wait_time = max_wait_time
        self.mark_completed = mark_completed
        self.chunk_pattern = re.compile(chunk_pattern)

        # Get Megatron data parallel info
        self.dp_rank, self.dp_world_size = get_data_parallel_info()

        # Cache: chunk_idx -> list of encoded samples
        self._chunk_cache: Dict[int, List[Dict[str, Any]]] = {}

        # Track chunks we've fully processed
        self._completed_chunks: set = set()

        logger.info(
            f'[LazyShardedDataset] dp_rank={self.dp_rank}/{self.dp_world_size}, '
            f'directory={self.directory}, samples_per_chunk={samples_per_chunk}'
        )

    def __len__(self) -> int:
        """Return max_total_samples. Training stops via --max_steps, not dataset exhaustion."""
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
        """Wait for a chunk file to appear on disk."""
        chunk_path = self._chunk_path(chunk_idx)
        start_time = time.time()

        while True:
            if chunk_path.exists():
                # Wait a bit for file to be fully written
                time.sleep(0.5)
                return chunk_path

            # Check if already completed (was processed and deleted)
            if self._completed_marker_path(chunk_path).exists():
                return None

            elapsed = time.time() - start_time
            if self.max_wait_time > 0 and elapsed > self.max_wait_time:
                logger.warning(f'[LazyShardedDataset] Timeout waiting for chunk {chunk_idx}')
                return None

            logger.debug(f'[LazyShardedDataset] Rank {self.dp_rank} waiting for chunk {chunk_idx}...')
            time.sleep(self.wait_poll_interval)

    def _load_chunk(self, chunk_idx: int) -> List[Dict[str, Any]]:
        """Load and encode all samples from a chunk file."""
        if chunk_idx in self._chunk_cache:
            return self._chunk_cache[chunk_idx]

        if not self._is_my_chunk(chunk_idx):
            logger.warning(f'[LazyShardedDataset] Rank {self.dp_rank} asked for chunk {chunk_idx} '
                          f'but it belongs to rank {chunk_idx % self.dp_world_size}')
            return []

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
                        raw = json.loads(line)
                        # CRITICAL: return_length=True for Megatron's padding_free mode
                        encoded = self.encode_fn(raw, return_length=True)
                        if encoded is not None:
                            samples.append(encoded)
                    except json.JSONDecodeError as e:
                        logger.warning(f'Invalid JSON at {chunk_path}:{line_num}: {e}')
                    except Exception as e:
                        logger.warning(f'Encoding error at {chunk_path}:{line_num}: {e}')

            logger.info(f'[LazyShardedDataset] Rank {self.dp_rank} loaded chunk {chunk_idx}: '
                       f'{len(samples)} samples')

            # Mark as completed so producer can delete
            if self.mark_completed and samples:
                try:
                    self._completed_marker_path(chunk_path).touch()
                except OSError as e:
                    logger.warning(f'Failed to create completion marker: {e}')

        except Exception as e:
            logger.error(f'[LazyShardedDataset] Error reading chunk {chunk_idx}: {e}')
            return []

        self._chunk_cache[chunk_idx] = samples
        return samples

    def _global_idx_to_local(self, global_idx: int) -> tuple:
        """Map global sample index to (chunk_idx, sample_offset_in_chunk).

        Global index space:
            Samples 0 to samples_per_chunk-1 are in chunk 0
            Samples samples_per_chunk to 2*samples_per_chunk-1 are in chunk 1
            etc.

        Each rank only loads chunks where chunk_idx % dp_world_size == dp_rank.
        """
        chunk_idx = global_idx // self.samples_per_chunk
        sample_offset = global_idx % self.samples_per_chunk
        return chunk_idx, sample_offset

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by global index.

        MegatronPretrainingSampler distributes indices across DP ranks.
        Each rank gets indices for samples that may be in various chunks.
        We load the chunk (if assigned to us) and return the sample.
        """
        chunk_idx, sample_offset = self._global_idx_to_local(idx)

        # Load chunk (waits if not yet available)
        samples = self._load_chunk(chunk_idx)

        if not samples:
            # Chunk not available or not ours - skip by returning from a nearby valid chunk
            # Find the nearest chunk that IS ours
            my_chunk_idx = (chunk_idx // self.dp_world_size) * self.dp_world_size + self.dp_rank
            if my_chunk_idx != chunk_idx:
                samples = self._load_chunk(my_chunk_idx)

        if not samples:
            # Still no samples - this shouldn't happen in normal operation
            raise IndexError(f'No samples available for idx {idx} (chunk {chunk_idx})')

        # Handle case where chunk has fewer samples than expected
        actual_offset = sample_offset % len(samples)
        return samples[actual_offset]

    def evict_chunk(self, chunk_idx: int) -> None:
        """Remove chunk from cache to free memory."""
        if chunk_idx in self._chunk_cache:
            del self._chunk_cache[chunk_idx]
```

### 2. New Arguments in `data_args.py`

```python
# In DataArguments class:

sharded_lazy: bool = False
"""Use LazyShardedDataset for distributed chunk loading.
Each DP rank loads only chunks where chunk_idx % dp_world_size == dp_rank.
Requires sequential chunk naming: chunk_00000.jsonl, chunk_00001.jsonl, ...
Does NOT require --streaming (uses efficient MegatronPretrainingSampler path).
"""

sharded_lazy_samples_per_chunk: int = 1000
"""Expected number of samples per chunk file for LazyShardedDataset."""
```

### 3. Integration in `swift/llm/train/sft.py`

In `_post_process_datasets()`, add handling BEFORE the streaming checks:

```python
def _post_process_datasets(self, datasets: List) -> List:
    args = self.args
    template = self.template

    for i, dataset in enumerate(datasets):
        if dataset is None:
            continue

        # NEW: Handle sharded_lazy mode - MUST be before streaming checks
        if getattr(args, 'sharded_lazy', False):
            from swift.llm.dataset.lazy_sharded import LazyShardedDataset

            # Get directory path
            dataset_path = args.dataset[i] if isinstance(args.dataset, list) else args.dataset

            dataset = LazyShardedDataset(
                directory=dataset_path,
                encode_fn=template.encode,
                samples_per_chunk=getattr(args, 'sharded_lazy_samples_per_chunk', 1000),
                max_total_samples=args.max_steps * args.per_device_train_batch_size * 100,
                mark_completed=True,
            )
            datasets[i] = dataset
            continue

        # ... existing streaming/packing logic ...
```

### 4. Ensure streaming=False in MegatronSft

When `sharded_lazy=True`, we must NOT use streaming mode:

```python
# In MegatronBaseArguments.__post_init__() or validation:

if self.sharded_lazy:
    if self.streaming:
        logger.warning('sharded_lazy=True is incompatible with streaming=True. '
                      'Setting streaming=False.')
        self.streaming = False
```

---

## File Changes Summary

| File | Change |
|------|--------|
| `swift/llm/dataset/lazy_sharded.py` | **NEW** - LazyShardedDataset implementation |
| `swift/llm/dataset/__init__.py` | Export LazyShardedDataset |
| `swift/llm/argument/base_args/data_args.py` | Add `sharded_lazy`, `sharded_lazy_samples_per_chunk` |
| `swift/llm/train/sft.py` | Route to LazyShardedDataset in `_post_process_datasets` |
| `swift/megatron/argument/megatron_base_args.py` | Ensure streaming=False when sharded_lazy=True |

---

## Usage

```bash
megatron sft \
    --model /path/to/Qwen3-Omni-30B-A3B-Instruct \
    --dataset /workspace/data/chunks \
    --sharded_lazy true \
    --sharded_lazy_samples_per_chunk 1000 \
    --max_steps 20000 \
    --packing false \
    ...
```

**Note**: No `--streaming true` needed! The dataset is map-style.

**Producer writes**: `chunk_00000.jsonl`, `chunk_00001.jsonl`, `chunk_00002.jsonl`, ...

**Rank 0 loads**: chunk_00000, chunk_00008, chunk_00016, ... (idx % 8 == 0)
**Rank 1 loads**: chunk_00001, chunk_00009, chunk_00017, ... (idx % 8 == 1)
...

**No scatter needed** - each rank reads directly from shared filesystem.

---

## Data Flow Comparison

### Before (streaming=True, BOTTLENECK):

```
Producer → chunks/ → [DP Rank 0 reads ALL via DynamicDirectoryDataset]
                            ↓
                     MegatronDataLoaderDispatcher
                            ↓
                     scatter_object_list() over network
                            ↓
              ┌─────────────┼─────────────┐
              ↓             ↓             ↓
           Rank 0        Rank 1   ...  Rank N

Network bandwidth: ~82MB per batch (8 ranks × ~10MB audio data)
```

### After (sharded_lazy=True, EFFICIENT):

```
Producer → chunks/ → Each DP rank reads ONLY its assigned chunks

           chunk_00000 ──→ Rank 0 (direct disk read)
           chunk_00001 ──→ Rank 1 (direct disk read)
           chunk_00002 ──→ Rank 2 (direct disk read)
           ...

Network bandwidth: 0 (each rank reads from shared filesystem)
```

---

## Verification Checklist

- [ ] `LazyShardedDataset` has `__len__()` → NOT treated as IterableDataset
- [ ] Goes through `build_pretraining_data_loader()` NOT `build_streaming_dataloader()`
- [ ] Uses `MegatronPretrainingSampler` NOT `MegatronDataLoaderDispatcher`
- [ ] Each rank only loads its assigned chunks (verify via logging)
- [ ] `encode_fn(..., return_length=True)` → padding_free works
- [ ] Completion markers created → producer can delete old chunks
- [ ] Waiting for chunks works when producer is slower than training

---

## Risk: Index Distribution

One potential issue: `MegatronPretrainingSampler` may distribute indices such that one rank gets indices spanning multiple chunks. Need to verify the sampler's behavior.

If the sampler gives rank 0 indices [0, 1, 2, ...] and rank 1 indices [1000, 1001, ...], then:
- Rank 0 needs chunk 0 (indices 0-999)
- Rank 1 needs chunk 1 (indices 1000-1999)
- This matches our modulo assignment!

If the sampler interleaves (rank 0 gets [0, 8, 16, ...], rank 1 gets [1, 9, 17, ...]):
- Both ranks might need chunk 0
- This would break our modulo assignment

**Need to verify `MegatronPretrainingSampler` behavior before implementation.**
