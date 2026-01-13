# Sharded Lazy Loading for Megatron-SWIFT

## Overview

`--sharded_lazy` eliminates the rank 0 data loading bottleneck in distributed training by having each data parallel rank load only its assigned chunk files directly from disk.

**Problem it solves:** With `--streaming true`, rank 0 loads ALL data and scatters to other ranks via network. For large audio/multimodal data (~3MB per sample), this becomes a severe bottleneck.

**Solution:** Each rank independently loads chunks where `chunk_idx % dp_world_size == dp_rank`. No scatter, no bottleneck.

---

## Quick Start

```bash
megatron sft \
    --model /path/to/model \
    --dataset /workspace/data/chunks \
    --sharded_lazy true \
    --sharded_lazy_samples_per_chunk 1000 \
    --train_dataloader_shuffle false \
    --train_iters 20000
```

**IMPORTANT:** `--train_dataloader_shuffle false` is required for producer-consumer streaming.
This ensures chunks are consumed in order as they're produced.

---

## Pre-Packed Format (Recommended for Variable-Length Data)

For data with high length variance (e.g., audio ranging from 0.5s to 5min), use **pre-packed chunks** to avoid OOM and improve GPU utilization.

### Why Pre-Pack Instead of Runtime Packing?

For Megatron-SWIFT + LazyShardedDataset pipelines, runtime packing (`--packing true`) is insufficient due to two hard constraints:

**1. Runtime packing needs token lengths, but lazy pipelines delay tokenization.**
- Bin-packing requires a length signal to work
- With `lazy_tokenize` / multimodal processors, length is unknown until full preprocessing
- Forces either: tokenizing ahead (negates "lazy"), buffering many samples (RAM + latency), or poor packing efficiency

**2. At scale, runtime packing shifts the bottleneck from GPU to CPU/input.**
- Packing on every rank replicates expensive CPU work N times
- Packing centrally creates a rank 0 / input pipeline bottleneck
- Either way, training becomes input-bound, especially for multimodal/audio

**What Pre-Packing Provides:**

| Benefit | Description |
|---------|-------------|
| Amortized preprocessing | Tokenize once offline; training stays compute-bound |
| Preserved laziness | Training reads packed objects, no lookahead needed |
| Stable cardinality | Known pack count enables well-defined `max_steps` / resume |
| Sharding balance | Constant-volume packs reduce stragglers across DP ranks |
| Multimodal OOM control | Enforce overlength policy with exact preprocessing |
| Determinism | Pack composition is fixed and inspectable |

### Quick Start with Pre-Packing

**Step 1: Pre-pack raw samples**

```bash
python -m swift.llm.dataset.packing_producer \
    --model /path/to/model \
    --input /data/raw_samples/*.jsonl \
    --output /data/packed_chunks \
    --packing_length 8192 \
    --samples_per_chunk 100
```

**Step 2: Train with pre-packed chunks**

```bash
megatron sft \
    --model /path/to/model \
    --dataset /data/packed_chunks \
    --sharded_lazy true \
    --sharded_lazy_pre_packed true \
    --sharded_lazy_samples_per_chunk 100 \
    --packing false \
    --train_iters 20000
```

Note: `--packing false` because data is ALREADY packed by the producer.

### Pre-Packed Chunk Format

Each line in a pre-packed chunk contains:

```json
{
    "input_ids": [1, 2, 3, ..., 100, 200, 201, ...],
    "labels": [-100, -100, 3, ..., -100, 200, 201, ...],
    "position_ids": [0, 1, 2, ..., 0, 1, 2, ...],
    "lengths": [50, 60, 45],
    "pack_length": 155,
    "num_samples": 3
}
```

- `input_ids`: Concatenated tokens from multiple samples
- `labels`: Concatenated labels (-100 for non-loss positions)
- `position_ids`: Reset to 0 at each sample boundary (enables cu_seqlens derivation)
- `lengths`: Individual sample lengths in this pack
- `pack_length`: Total tokens in the pack

### Additional Arguments for Pre-Packing

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sharded_lazy_pre_packed` | bool | `false` | Chunks are pre-packed (skip encoding) |

---

## Requirements

### 1. Chunk File Naming

Producer must create chunks with **sequential zero-padded naming**:

```
/workspace/data/chunks/
├── chunk_00000.jsonl
├── chunk_00001.jsonl
├── chunk_00002.jsonl
├── chunk_00003.jsonl
└── ...
```

The pattern is: `chunk_{index:05d}.jsonl`

### 2. Chunk File Format

Each chunk is a JSONL file (one JSON object per line) in ms-swift message format:

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

For multimodal (audio/image):
```json
{"messages": [{"role": "user", "content": "<audio>What is being said?</audio>"}, {"role": "assistant", "content": "The transcript is..."}], "audios": ["base64_encoded_audio_data"]}
```

### 3. Shared Filesystem

All ranks must have access to the same chunk directory. This means:
- NFS mount, or
- Distributed filesystem (Lustre, GPFS, etc.), or
- All processes on same node

### 4. max_steps Required

Since the dataset size is unknown (chunks appear dynamically), you must specify `--train_iters`:

```bash
--train_iters 20000
```

---

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sharded_lazy` | bool | `false` | Enable sharded lazy loading |
| `--sharded_lazy_samples_per_chunk` | int | `1000` | Expected samples per chunk file |
| `--train_dataloader_shuffle` | bool | `true` | **Must be `false`** for producer-consumer streaming |
| `--dataset` | str | - | Path to directory containing chunks |
| `--train_iters` | int | - | Required when using sharded_lazy |

---

## How Chunk Assignment Works

With 8 data parallel ranks:

```
chunk_00000.jsonl  →  Rank 0  (0 % 8 == 0)
chunk_00001.jsonl  →  Rank 1  (1 % 8 == 1)
chunk_00002.jsonl  →  Rank 2  (2 % 8 == 2)
chunk_00003.jsonl  →  Rank 3  (3 % 8 == 3)
chunk_00004.jsonl  →  Rank 4  (4 % 8 == 4)
chunk_00005.jsonl  →  Rank 5  (5 % 8 == 5)
chunk_00006.jsonl  →  Rank 6  (6 % 8 == 6)
chunk_00007.jsonl  →  Rank 7  (7 % 8 == 7)
chunk_00008.jsonl  →  Rank 0  (8 % 8 == 0)
chunk_00009.jsonl  →  Rank 1  (9 % 8 == 1)
...
```

Each rank only reads its assigned chunks. No network transfer of training data.

---

## Lazy Loading Behavior

If a rank needs a chunk that doesn't exist yet, it **waits**:

```
[LazyShardedDataset] Rank 3: Waiting for chunk 3 (chunk_00003.jsonl)...
```

This enables a **producer-consumer pattern** where:
1. Producer writes chunks continuously
2. Training consumes chunks as they appear
3. No need to wait for all data to be ready before training

---

## Completion Markers

After a rank finishes reading a chunk, it creates a `.completed` marker:

```
chunk_00000.jsonl           # Original chunk
chunk_00000.jsonl.completed # Marker indicating chunk was consumed
```

**Purpose:** The producer can safely delete chunks that have `.completed` markers to free disk space.

---

## Producer-Consumer Example

### Producer Script

```python
import json
import os
import glob
import time

def producer(source_iterator, output_dir, samples_per_chunk=1000):
    """
    Write chunks continuously while training consumes them.

    Args:
        source_iterator: Iterator yielding samples in ms-swift format
        output_dir: Directory to write chunks (e.g., /workspace/data/chunks)
        samples_per_chunk: Samples per chunk file
    """
    os.makedirs(output_dir, exist_ok=True)

    chunk_idx = 0
    buffer = []

    for sample in source_iterator:
        buffer.append(sample)

        if len(buffer) >= samples_per_chunk:
            # Write chunk
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:05d}.jsonl")
            with open(chunk_path, 'w') as f:
                for s in buffer:
                    f.write(json.dumps(s) + '\n')

            print(f"Wrote {chunk_path} ({len(buffer)} samples)")
            buffer = []
            chunk_idx += 1

            # Clean up completed chunks to save disk
            cleanup_completed_chunks(output_dir)

    # Write remaining samples
    if buffer:
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:05d}.jsonl")
        with open(chunk_path, 'w') as f:
            for s in buffer:
                f.write(json.dumps(s) + '\n')
        print(f"Wrote {chunk_path} ({len(buffer)} samples)")


def cleanup_completed_chunks(output_dir, keep_recent=10):
    """Delete chunks that have been consumed by training."""
    completed = sorted(glob.glob(os.path.join(output_dir, "*.completed")))

    # Keep some recent completed chunks as buffer
    to_delete = completed[:-keep_recent] if len(completed) > keep_recent else []

    for marker in to_delete:
        chunk_file = marker.replace('.completed', '')
        try:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
            os.remove(marker)
            print(f"Cleaned up {chunk_file}")
        except OSError as e:
            print(f"Warning: Failed to delete {chunk_file}: {e}")
```

### Training Script

```bash
#!/bin/bash

# Start training - it will wait for chunks as needed
megatron sft \
    --model /path/to/Qwen3-Omni-30B-A3B-Instruct \
    --dataset /workspace/data/chunks \
    --sharded_lazy true \
    --sharded_lazy_samples_per_chunk 1000 \
    --train_iters 20000 \
    --packing false \
    --padding_free true \
    --micro_batch_size 1 \
    --global_batch_size 64 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --learning_rate 1e-5 \
    --output_dir /workspace/checkpoints/run1
```

### Running Together

```bash
# Terminal 1: Start producer
python producer.py --source s3://bucket/data --output /workspace/data/chunks

# Terminal 2: Start training (can start immediately, will wait for chunks)
./train.sh
```

---

## Comparison with Other Modes

| Mode | Flag | Rank 0 Bottleneck | Dynamic Files | Use Case |
|------|------|-------------------|---------------|----------|
| **Standard** | (none) | No | No | Pre-downloaded dataset |
| **Streaming** | `--streaming true` | **YES** | No | Large datasets, limited RAM |
| **Rescan Files** | `--streaming true --rescan_files true` | **YES** | Yes | Dynamic directory |
| **Sharded Lazy** | `--sharded_lazy true` | **No** | Yes | Large distributed training |

---

## Troubleshooting

### Training hangs waiting for chunk

```
[LazyShardedDataset] Rank 3: Waiting for chunk 3 (chunk_00003.jsonl)...
```

**Cause:** Producer hasn't written that chunk yet.

**Solution:**
- Ensure producer is running and writing chunks
- Check producer logs for errors
- Verify chunk naming is correct (`chunk_00003.jsonl` not `chunk_3.jsonl`)

### KeyError: 'length'

**Cause:** Encoding not including length field.

**Solution:** This should not happen with `LazyShardedDataset` as it calls `encode_fn(sample, return_length=True)`. If it does, check that you're using the latest code.

### Chunks not being cleaned up

**Cause:** `.completed` markers not being created or producer not checking them.

**Solution:**
- Check that training has read the chunks (look for log messages)
- Verify producer is calling `cleanup_completed_chunks()`

### Wrong samples_per_chunk

**Cause:** `--sharded_lazy_samples_per_chunk` doesn't match actual chunk size.

**Impact:** Index mapping will be off, potentially skipping or duplicating samples.

**Solution:** Set `--sharded_lazy_samples_per_chunk` to match your actual chunk size.

---

## Logs to Expect

Successful operation shows:

```
[LazyShardedDataset] Initialized: dp_rank=0/8, directory=/workspace/data/chunks, samples_per_chunk=1000
[LazyShardedDataset] Rank 0: Loaded chunk 0 (1000 samples from chunk_00000.jsonl)
[LazyShardedDataset] Rank 0: Loaded chunk 8 (1000 samples from chunk_00008.jsonl)
...
```

Each rank should only show chunks where `chunk_idx % world_size == rank`.

---

## Performance

| Scenario | Streaming Mode | Sharded Lazy Mode |
|----------|---------------|-------------------|
| 8x H200 single node | ~OK (NVLink) | Optimal |
| Multi-node | **Bottleneck** (~82MB/batch over network) | Optimal (0 network for data) |
| Large audio data | Severe bottleneck | No bottleneck |

---

## Migration from --streaming --rescan_files

If you were using:
```bash
--streaming true --rescan_files true --dataset /path/to/chunks
```

Change to:
```bash
--sharded_lazy true --dataset /path/to/chunks
```

The behavior is similar (dynamic file loading) but without the rank 0 bottleneck.
