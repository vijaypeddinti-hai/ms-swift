# Implementation Plan: LazyShardedDataset for Megatron-SWIFT

## Status: COMPLETE

**Last Updated**: 2026-01-09

---

## Goal

Eliminate the rank 0 data loading bottleneck by using the efficient non-streaming path in Megatron-SWIFT.

---

## Confirmed Code Paths

### Streaming Path (streaming=True) - BOTTLENECK
```
MegatronSft.run() [swift/megatron/train/sft.py:71-72]
    │
    └── build_streaming_dataloader() [swift/megatron/train/utils.py:15-26]
            │
            └── MegatronDataLoaderDispatcher [swift/megatron/train/utils.py:8-12]
                    │
                    └── INHERITS FROM DataLoaderDispatcher [swift/llm/data_loader.py:100]
                            │
                            └── __iter__() [swift/llm/data_loader.py:132-148]
                                    │
                                    └── if rank == 0: scatter_object_list()  # BOTTLENECK
```

### Non-Streaming Path (streaming=False) - EFFICIENT
```
MegatronSft.run() [swift/megatron/train/sft.py:77]
    │
    └── trainer.train() → build_pretraining_data_loader() [swift/megatron/trainers/base.py:1107-1177]
            │
            └── MegatronPretrainingRandomSampler [swift/megatron/trainers/utils.py:392-485]
                    │
                    └── With data_sharding=True (DEFAULT):
                            start_idx = data_parallel_rank * bucket_size  # CONTIGUOUS BLOCKS!
```

**Key Setting**: `no_data_sharding: bool = False` in `swift/megatron/argument/megatron_args.py:602`
- This means `data_sharding=True` by default
- Each DP rank gets contiguous index blocks
- Aligns perfectly with modulo-based chunk assignment

---

## Implementation Checklist

### ALL COMPLETED:

- [x] **swift/llm/dataset/lazy_sharded.py** - NEW FILE
  - `LazyShardedDataset` class with `__len__()` and `__getitem__()`
  - Uses `get_data_parallel_info()` to get Megatron DP rank/size
  - Modulo-based chunk assignment: `chunk_idx % dp_world_size == dp_rank`
  - Lazy waiting for chunks with `_wait_for_chunk()`
  - Encodes with `return_length=True` for padding_free
  - Creates `.completed` markers for producer coordination
  - Chunk caching with `_chunk_cache`

- [x] **swift/llm/argument/base_args/data_args.py** - UPDATED
  - Added `sharded_lazy: bool = False`
  - Added `sharded_lazy_samples_per_chunk: int = 1000`
  - Added docstrings for both arguments
  - Added validation in `__post_init__` with detailed error messages

- [x] **swift/llm/dataset/__init__.py** - UPDATED
  - Exports `LazyShardedDataset`

- [x] **swift/llm/train/sft.py** - UPDATED
  - Added handling in `_post_process_datasets()` for sharded_lazy mode
  - Creates `LazyShardedDataset` with directory path and template.encode

- [x] **swift/megatron/argument/megatron_base_args.py** - UPDATED
  - Auto-fixes streaming=True → False when sharded_lazy=True
  - Auto-fixes rescan_files=True → False when sharded_lazy=True
  - Logs informative messages about the changes

---

## File Change Summary

| File | Status | Change |
|------|--------|--------|
| `swift/llm/dataset/lazy_sharded.py` | DONE | NEW - LazyShardedDataset class |
| `swift/llm/argument/base_args/data_args.py` | DONE | Add args + validation |
| `swift/llm/dataset/__init__.py` | DONE | Export LazyShardedDataset |
| `swift/llm/train/sft.py` | DONE | Route to LazyShardedDataset |
| `swift/megatron/argument/megatron_base_args.py` | DONE | Handle sharded_lazy |

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

**Note**: No `--streaming true` needed! Uses efficient non-streaming path.

**Producer writes**: `chunk_00000.jsonl`, `chunk_00001.jsonl`, ...
**Rank 0 loads**: chunk_00000, chunk_00008, chunk_00016, ... (idx % 8 == 0)
**Rank 1 loads**: chunk_00001, chunk_00009, chunk_00017, ... (idx % 8 == 1)

---

## Data Flow After Implementation

```
Producer → chunks/ → Each DP rank reads ONLY its assigned chunks

           chunk_00000 ──→ Rank 0 (direct disk read via __getitem__)
           chunk_00001 ──→ Rank 1 (direct disk read via __getitem__)
           chunk_00002 ──→ Rank 2 (direct disk read via __getitem__)
           ...

Goes through: build_pretraining_data_loader() → MegatronPretrainingRandomSampler
NOT through: build_streaming_dataloader() → MegatronDataLoaderDispatcher

Network bandwidth for data loading: 0 (each rank reads from shared filesystem)
```

---

## Key Code References

1. **Bottleneck location**: `swift/llm/data_loader.py:136-141`
   ```python
   if self.rank == 0:
       data = [next(base_iter) for _ in range(self.world_size)]
       data = self._scatter_object_list(data)
   ```

2. **Efficient sampler**: `swift/megatron/trainers/utils.py:451-459`
   ```python
   if self.data_sharding:
       bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
       start_idx = self.data_parallel_rank * bucket_size
       idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
   ```

3. **Default data_sharding**: `swift/megatron/argument/megatron_args.py:602`
   ```python
   no_data_sharding: bool = False  # means data_sharding=True by default
   ```

---

## Testing Required

1. Single-node test with 8 GPUs
2. Verify each rank only loads its assigned chunks (check logs)
3. Verify `.completed` markers are created
4. Verify training progresses without rank 0 bottleneck
5. Compare throughput with old streaming approach
