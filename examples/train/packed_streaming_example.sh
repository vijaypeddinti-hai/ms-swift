#!/bin/bash
# Example: Pre-packed streaming training for Qwen3-Omni with 1M hours audio
#
# This example demonstrates the full flow:
# 1. Producer pre-packs raw samples into fixed-length chunks
# 2. Training consumes pre-packed chunks via LazyShardedDataset
#
# Benefits:
# - No OOM from variable sequence lengths (packing smooths everything)
# - No rank 0 bottleneck (each rank reads its own chunks)
# - Optimal bin-packing (producer sees all samples)
# - Dynamic/streaming compatible (producer writes, training consumes)

set -e

# Configuration
MODEL_PATH="/path/to/Qwen3-Omni-30B-A3B-Instruct"
RAW_DATA_DIR="/workspace/data/raw_audio_samples"
PACKED_CHUNKS_DIR="/workspace/data/packed_chunks"
PACKING_LENGTH=8192
SAMPLES_PER_CHUNK=100  # packs per chunk (not raw samples!)
BUFFER_SIZE=10000      # samples to buffer before bin-packing

# ============================================================================
# STEP 1: Pre-pack raw samples using PackingProducer
# ============================================================================
# This step tokenizes raw samples and bin-packs them into fixed-length packs.
# Run this on a CPU node or as a preprocessing job.

echo "Step 1: Pre-packing raw samples..."

python -m swift.llm.dataset.packing_producer \
    --model "${MODEL_PATH}" \
    --input "${RAW_DATA_DIR}/*.jsonl" \
    --output "${PACKED_CHUNKS_DIR}" \
    --packing_length ${PACKING_LENGTH} \
    --samples_per_chunk ${SAMPLES_PER_CHUNK} \
    --truncation_strategy delete \
    --buffer_size ${BUFFER_SIZE}

# Output format (each line in chunk is a pre-packed sequence):
# {
#   "input_ids": [...],      # Concatenated tokens from multiple samples
#   "labels": [...],         # Concatenated labels
#   "position_ids": [...],   # Position IDs (reset at sample boundaries)
#   "lengths": [L1, L2, ...],# Individual sample lengths
#   "pack_length": 8192      # Total tokens in this pack
# }

# ============================================================================
# STEP 2: Train with pre-packed chunks
# ============================================================================
# Training reads pre-packed chunks via LazyShardedDataset.
# Each DP rank reads only its assigned chunks (no rank 0 bottleneck).

echo "Step 2: Starting training with pre-packed chunks..."

megatron sft \
    --model "${MODEL_PATH}" \
    --dataset "${PACKED_CHUNKS_DIR}" \
    --sharded_lazy true \
    --sharded_lazy_pre_packed true \
    --sharded_lazy_samples_per_chunk ${SAMPLES_PER_CHUNK} \
    --train_iters 20000 \
    --micro_batch_size 4 \
    --packing false \
    --padding_free true \
    --attention_backend flash \
    --train_dataloader_shuffle false \
    --output_dir /workspace/output/qwen3-omni-packed

# Key flags explained:
# --sharded_lazy true              : Use LazyShardedDataset for distributed loading
# --sharded_lazy_pre_packed true   : Chunks are pre-packed (skip encoding)
# --sharded_lazy_samples_per_chunk : Packs per chunk (matches producer output)
# --packing false                  : Data is ALREADY packed by producer
# --padding_free true              : Use THD format (padding-free attention)
# --train_dataloader_shuffle false : Sequential access for producer-consumer

echo "Training complete!"

# ============================================================================
# ALTERNATIVE: Continuous producer-consumer streaming
# ============================================================================
# For continuous training where data arrives over time:
#
# Terminal 1 (Producer - runs continuously):
#   python continuous_producer.py \
#       --model "${MODEL_PATH}" \
#       --input-queue /data/incoming \
#       --output "${PACKED_CHUNKS_DIR}" \
#       --packing_length ${PACKING_LENGTH}
#
# Terminal 2 (Training - waits for chunks):
#   megatron sft \
#       --model "${MODEL_PATH}" \
#       --dataset "${PACKED_CHUNKS_DIR}" \
#       --sharded_lazy true \
#       --sharded_lazy_pre_packed true \
#       --train_iters 1000000 \
#       ...
#
# LazyShardedDataset will wait for chunks that don't exist yet,
# enabling true streaming training as data is produced.
