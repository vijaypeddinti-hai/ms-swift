# PR Justification: Dynamic Directory Streaming + Pre-Packing for Megatron-SWIFT

This document explains the explicit need for each component introduced in this PR.

---

## Component 1: LazyShardedDataset

**Problem it solves:** Dynamic directory streaming without rank 0 bottleneck

### Why existing solutions don't work

| Existing Option | Limitation |
|-----------------|------------|
| `--streaming true` | Rank 0 loads ALL data and scatters to other ranks. Bottleneck for large data. |
| `--streaming true --rescan_files true` | Still has rank 0 bottleneck. When file list is exhausted and rescanned, already-processed files are re-read. |
| Standard (non-streaming) | Requires all data upfront. Can't handle files appearing dynamically. |

### What LazyShardedDataset provides

- Each DP rank reads only its assigned chunks (`chunk_idx % dp_world_size == dp_rank`)
- No rank 0 bottleneck - direct disk reads per rank
- Waits for chunks that don't exist yet (producer-consumer pattern)
- Chunks consumed once, marked with `.completed` files
- Works with Megatron's `MegatronPretrainingRandomSampler` (non-streaming path)

### Use case

Training on data that arrives continuously (e.g., audio being transcribed and written to chunks over time). When the initial file list is exhausted, rescanning would re-read already-processed files. LazyShardedDataset avoids this by consuming chunks sequentially without rescanning.

---

## Component 2: PackingProducer

**Problem it solves:** Packing for audio/multimodal data on Megatron-SWIFT path

### Why existing `--packing true` doesn't work

1. **Audio/multimodal not supported**
   - MS-SWIFT FAQ states Qwen2Audio "does not support packing"
   - Source: https://swift.readthedocs.io/en/latest/Instruction/Frequently-asked-questions.html

2. **Megatron-SWIFT path differs from Swift Trainer**
   - Packing integration differs between training paths
   - Custom datasets like LazyShardedDataset may not integrate with existing packing collator

3. **No global view for bin-packing**
   - Runtime packing sees only current batch/buffer
   - Leads to suboptimal packing for highly variable audio lengths (0.5s to 5min)

4. **Memory fragmentation with runtime packing**
   - Users report GPU memory increases after each allocator flush, eventually causing OOM
   - Source: https://swift.readthedocs.io/en/latest/Instruction/Frequently-asked-questions.html

### What PackingProducer provides

- Offline tokenization + packing (amortized cost, done once)
- First-fit decreasing bin-packing with global sample visibility
- Fixed pack sizes → stable memory, no OOM surprises
- Deterministic pack composition → reproducible training
- Outputs pre-packed chunks for LazyShardedDataset consumption

---

## How They Work Together

```
Producer (offline):
  raw audio files → PackingProducer → pre-packed chunks (token IDs)

Training (online):
  pre-packed chunks → LazyShardedDataset → each rank reads its chunks → train
                      (no tokenization, no packing, no bottleneck)
```

### Training command

```bash
megatron sft \
    --dataset /path/to/packed_chunks \
    --sharded_lazy true \
    --sharded_lazy_pre_packed true \
    --packing false \
    --train_dataloader_shuffle false
```

**Flag explanations:**
- `--sharded_lazy true` - Use LazyShardedDataset
- `--sharded_lazy_pre_packed true` - Chunks contain pre-packed data (skip encoding)
- `--packing false` - Data is ALREADY packed by producer
- `--train_dataloader_shuffle false` - Sequential access for producer-consumer pattern

---

## Summary

| Component | Need | Existing Alternative | Why Alternative Fails |
|-----------|------|---------------------|----------------------|
| LazyShardedDataset | Dynamic streaming without bottleneck | `--streaming --rescan_files` | Rank 0 bottleneck; re-reads files on rescan |
| PackingProducer | Packing for audio/multimodal | `--packing true` | Not supported for audio; no global view; memory issues |

---

## References

- MS-SWIFT FAQ (packing limitations): https://swift.readthedocs.io/en/latest/Instruction/Frequently-asked-questions.html
- MS-SWIFT Command Line Parameters: https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Command-line-parameters.html
- GitHub Issue #5402 (packing + lazy_tokenize): https://github.com/modelscope/ms-swift/issues/5402
