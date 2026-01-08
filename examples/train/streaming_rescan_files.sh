# Streaming training with dynamic file re-scanning
#
# This example demonstrates the --rescan_files feature which enables
# true streaming training from a directory where files are continuously
# added by a producer process (e.g., downloading from S3/R2).
#
# Use case:
#   - Producer process downloads data chunks to /path/to/buffer/
#   - Training reads from buffer with --streaming true --rescan_files true
#   - New files are automatically picked up on each iteration cycle
#   - Completed files are marked with .completed suffix
#   - Producer can safely delete files with .completed markers
#
# The standard --streaming true mode snapshots the file list at start
# and never re-scans. With --rescan_files true, the directory is
# re-scanned on each training cycle to pick up new files.

# First, create a buffer directory with some initial data
mkdir -p /tmp/streaming_buffer
cat > /tmp/streaming_buffer/chunk_001.jsonl << 'EOF'
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well!"}]}
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
EOF

# Run streaming training with rescan_files enabled
# Note: Use --max_steps since dataset length is unknown in streaming mode
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --train_type lora \
    --dataset /tmp/streaming_buffer \
    --streaming true \
    --rescan_files true \
    --torch_dtype bfloat16 \
    --max_steps 100 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --max_length 512 \
    --output_dir output/streaming_rescan \
    --model_author swift \
    --model_name swift-robot

# After training, you can see .completed markers for processed files:
# ls /tmp/streaming_buffer/
# chunk_001.jsonl
# chunk_001.jsonl.completed

# Clean up
rm -rf /tmp/streaming_buffer
