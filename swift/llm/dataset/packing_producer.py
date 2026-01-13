"""
Packing Producer for Megatron-SWIFT distributed training.

Pre-packs samples into fixed-length packs BEFORE training, enabling:
1. Optimal bin-packing with full visibility of samples
2. Consistent pack sizes (no OOM from variable lengths)
3. Compatibility with LazyShardedDataset for streaming consumption

The producer:
1. Reads raw samples from input files
2. Tokenizes using the model's template
3. Bin-packs samples into packs of `packing_length` tokens
4. Writes pre-packed chunks for LazyShardedDataset consumption

Pre-packed format (each line in chunk):
{
    "input_ids": [...],      # Concatenated token IDs
    "labels": [...],         # Concatenated labels (-100 for non-loss tokens)
    "position_ids": [...],   # Position IDs (reset to 0 at each sample boundary)
    "lengths": [L1, L2, ...],# Individual sample lengths in this pack
    "pack_length": N         # Total tokens in pack
}

Usage:
    from swift.llm.dataset.packing_producer import PackingProducer

    producer = PackingProducer(
        model_path="/path/to/Qwen3-Omni",
        packing_length=8192,
        output_dir="/workspace/data/packed_chunks",
        samples_per_chunk=100,  # packs per chunk
    )

    # Process input files
    for input_file in input_files:
        producer.process_file(input_file)

    # Flush remaining samples
    producer.flush()
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from swift.utils import get_logger

logger = get_logger()


@dataclass
class PackingStats:
    """Statistics for packing operations."""
    total_samples_read: int = 0
    total_samples_packed: int = 0
    total_packs_created: int = 0
    total_chunks_written: int = 0
    total_tokens_packed: int = 0
    samples_dropped_too_long: int = 0
    samples_dropped_encode_error: int = 0
    avg_samples_per_pack: float = 0.0
    avg_utilization: float = 0.0

    def update_averages(self):
        if self.total_packs_created > 0:
            self.avg_samples_per_pack = self.total_samples_packed / self.total_packs_created
        if self.total_packs_created > 0 and hasattr(self, '_packing_length'):
            self.avg_utilization = self.total_tokens_packed / (self.total_packs_created * self._packing_length)


@dataclass
class EncodedSample:
    """A tokenized sample ready for packing."""
    input_ids: List[int]
    labels: List[int]
    length: int
    raw_sample: Optional[Dict[str, Any]] = None  # Original sample for debugging


class BinPacker:
    """First-fit decreasing bin packing algorithm.

    Sorts samples by length (descending) and places each into the first bin
    that has enough remaining capacity. Creates new bins as needed.
    """

    def __init__(self, packing_length: int, padding_token_id: int = 0):
        self.packing_length = packing_length
        self.padding_token_id = padding_token_id

    def pack(self, samples: List[EncodedSample]) -> List[List[EncodedSample]]:
        """Pack samples into bins using first-fit decreasing.

        Args:
            samples: List of encoded samples to pack

        Returns:
            List of bins, where each bin is a list of samples that fit together
        """
        if not samples:
            return []

        # Sort by length descending (first-fit decreasing)
        sorted_samples = sorted(samples, key=lambda s: s.length, reverse=True)

        bins: List[List[EncodedSample]] = []
        bin_remaining: List[int] = []

        for sample in sorted_samples:
            if sample.length > self.packing_length:
                logger.warning(f"Sample length {sample.length} exceeds packing_length {self.packing_length}, skipping")
                continue

            # Find first bin with enough space
            placed = False
            for i, remaining in enumerate(bin_remaining):
                if remaining >= sample.length:
                    bins[i].append(sample)
                    bin_remaining[i] -= sample.length
                    placed = True
                    break

            # Create new bin if needed
            if not placed:
                bins.append([sample])
                bin_remaining.append(self.packing_length - sample.length)

        return bins

    def create_pack(self, samples: List[EncodedSample]) -> Dict[str, Any]:
        """Create a single pack from a list of samples.

        Args:
            samples: List of samples to combine into one pack

        Returns:
            Dict with concatenated input_ids, labels, position_ids, lengths
        """
        if not samples:
            return {}

        input_ids = []
        labels = []
        position_ids = []
        lengths = []

        for sample in samples:
            input_ids.extend(sample.input_ids)
            labels.extend(sample.labels)
            # Position IDs reset to 0 at each sample boundary
            position_ids.extend(list(range(sample.length)))
            lengths.append(sample.length)

        pack_length = len(input_ids)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'position_ids': position_ids,
            'lengths': lengths,
            'pack_length': pack_length,
            'num_samples': len(samples),
        }


class PackingProducer:
    """Produces pre-packed chunks for LazyShardedDataset consumption.

    This class handles:
    1. Reading raw samples from input files
    2. Tokenizing using the model's template
    3. Bin-packing samples into fixed-length packs
    4. Writing pre-packed chunks to output directory

    Args:
        model_path: Path to the model (for tokenizer/template)
        packing_length: Maximum tokens per pack
        output_dir: Directory to write packed chunks
        samples_per_chunk: Number of packs per output chunk file
        template_type: Template type override (default: auto-detect)
        truncation_strategy: How to handle overlong samples ('delete', 'left', 'right')
        buffer_size: Number of samples to buffer before packing (for better bin-packing)
        model_type: Model type override (default: auto-detect from model_path)
    """

    def __init__(
        self,
        model_path: str,
        packing_length: int = 8192,
        output_dir: str = "./packed_chunks",
        samples_per_chunk: int = 100,
        template_type: Optional[str] = None,
        truncation_strategy: str = 'delete',
        buffer_size: int = 10000,
        model_type: Optional[str] = None,
    ):
        self.model_path = model_path
        self.packing_length = packing_length
        self.output_dir = Path(output_dir)
        self.samples_per_chunk = samples_per_chunk
        self.truncation_strategy = truncation_strategy
        self.buffer_size = buffer_size

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize template and tokenizer
        self._init_template(model_path, template_type, model_type)

        # Initialize packer
        self.packer = BinPacker(
            packing_length=packing_length,
            padding_token_id=self.tokenizer.pad_token_id or 0
        )

        # Buffers
        self._sample_buffer: List[EncodedSample] = []
        self._pack_buffer: List[Dict[str, Any]] = []
        self._chunk_counter: int = 0

        # Stats
        self.stats = PackingStats()
        self.stats._packing_length = packing_length

        logger.info(
            f"[PackingProducer] Initialized: model={model_path}, "
            f"packing_length={packing_length}, output_dir={output_dir}, "
            f"samples_per_chunk={samples_per_chunk}, buffer_size={buffer_size}"
        )

    def _init_template(
        self,
        model_path: str,
        template_type: Optional[str],
        model_type: Optional[str]
    ):
        """Initialize the tokenizer and template from model."""
        from swift.llm import get_model_tokenizer, get_template

        # Get tokenizer
        self.model, self.tokenizer = get_model_tokenizer(
            model_path,
            model_type=model_type,
            load_model=False,  # Don't load model weights, just tokenizer
        )

        # Map 'delete' to 'raise' for template - we'll catch MaxLengthError ourselves
        # Template only accepts: 'raise', 'left', 'right', 'split'
        template_truncation = 'raise' if self.truncation_strategy == 'delete' else self.truncation_strategy

        # Get template
        self.template = get_template(
            template_type or self.model.model_meta.template,
            self.tokenizer,
            default_system=None,
            max_length=self.packing_length,
            truncation_strategy=template_truncation,
        )
        self.template.set_mode('train')

        logger.info(f"[PackingProducer] Template initialized: {self.template.__class__.__name__}")

    def encode_sample(self, raw_sample: Dict[str, Any]) -> Optional[EncodedSample]:
        """Encode a raw sample using the template.

        Args:
            raw_sample: Raw sample dict (e.g., {"messages": [...]})

        Returns:
            EncodedSample or None if encoding fails/sample too long
        """
        from swift.llm.template.base import MaxLengthError

        try:
            encoded = self.template.encode(raw_sample, return_length=True)

            # Handle truncation_strategy='split' which returns a list
            if isinstance(encoded, list):
                # For split, return the first part (or handle differently)
                encoded = encoded[0] if encoded else None
                if encoded is None:
                    return None

            length = encoded.get('length', len(encoded.get('input_ids', [])))

            # Handle overlong samples (shouldn't happen with left/right truncation)
            if length > self.packing_length:
                if self.truncation_strategy == 'delete':
                    self.stats.samples_dropped_too_long += 1
                    return None
                # left/right truncation should have been handled by template

            return EncodedSample(
                input_ids=encoded['input_ids'],
                labels=encoded.get('labels', [-100] * len(encoded['input_ids'])),
                length=length,
                raw_sample=raw_sample,
            )

        except MaxLengthError:
            # Sample exceeded max_length with truncation_strategy='delete' (mapped to 'raise')
            self.stats.samples_dropped_too_long += 1
            return None
        except Exception as e:
            logger.warning(f"[PackingProducer] Encoding error: {e}")
            self.stats.samples_dropped_encode_error += 1
            return None

    def add_sample(self, raw_sample: Dict[str, Any]) -> int:
        """Add a raw sample to the buffer.

        Automatically triggers packing when buffer is full.

        Args:
            raw_sample: Raw sample dict

        Returns:
            Number of chunks written (0 or more)
        """
        self.stats.total_samples_read += 1

        encoded = self.encode_sample(raw_sample)
        if encoded is None:
            return 0

        self._sample_buffer.append(encoded)

        # Pack when buffer is full
        chunks_written = 0
        if len(self._sample_buffer) >= self.buffer_size:
            chunks_written = self._pack_and_flush()

        return chunks_written

    def _pack_and_flush(self) -> int:
        """Pack buffered samples and write chunks.

        Returns:
            Number of chunks written
        """
        if not self._sample_buffer:
            return 0

        # Bin-pack the samples
        bins = self.packer.pack(self._sample_buffer)

        # Create packs from bins
        for bin_samples in bins:
            pack = self.packer.create_pack(bin_samples)
            if pack:
                self._pack_buffer.append(pack)
                self.stats.total_packs_created += 1
                self.stats.total_samples_packed += len(bin_samples)
                self.stats.total_tokens_packed += pack['pack_length']

        # Clear sample buffer
        self._sample_buffer.clear()

        # Write chunks when we have enough packs
        chunks_written = 0
        while len(self._pack_buffer) >= self.samples_per_chunk:
            self._write_chunk(self._pack_buffer[:self.samples_per_chunk])
            self._pack_buffer = self._pack_buffer[self.samples_per_chunk:]
            chunks_written += 1

        return chunks_written

    def _write_chunk(self, packs: List[Dict[str, Any]]):
        """Write a chunk file with the given packs.

        Args:
            packs: List of pack dicts to write
        """
        chunk_path = self.output_dir / f"chunk_{self._chunk_counter:05d}.jsonl"

        with open(chunk_path, 'w', encoding='utf-8') as f:
            for pack in packs:
                f.write(json.dumps(pack) + '\n')

        self.stats.total_chunks_written += 1
        self._chunk_counter += 1

        logger.info(
            f"[PackingProducer] Wrote {chunk_path.name}: "
            f"{len(packs)} packs, "
            f"avg {sum(p['num_samples'] for p in packs) / len(packs):.1f} samples/pack"
        )

    def process_file(self, input_path: Union[str, Path]) -> int:
        """Process an input file containing raw samples.

        Args:
            input_path: Path to JSONL file with raw samples

        Returns:
            Number of chunks written
        """
        input_path = Path(input_path)
        logger.info(f"[PackingProducer] Processing {input_path}")

        chunks_written = 0
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_sample = json.loads(line)
                    chunks_written += self.add_sample(raw_sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"[PackingProducer] Invalid JSON at {input_path.name}:{line_num}: {e}")

        return chunks_written

    def process_iterator(self, samples: Iterator[Dict[str, Any]]) -> int:
        """Process samples from an iterator.

        Args:
            samples: Iterator yielding raw sample dicts

        Returns:
            Number of chunks written
        """
        chunks_written = 0
        for sample in samples:
            chunks_written += self.add_sample(sample)
        return chunks_written

    def flush(self) -> int:
        """Flush remaining samples and packs to disk.

        Call this after processing all input to ensure all data is written.

        Returns:
            Number of chunks written
        """
        chunks_written = 0

        # Pack remaining samples
        if self._sample_buffer:
            chunks_written += self._pack_and_flush()

        # Write remaining packs (even if less than samples_per_chunk)
        if self._pack_buffer:
            self._write_chunk(self._pack_buffer)
            self._pack_buffer.clear()
            chunks_written += 1

        self.stats.update_averages()

        logger.info(f"[PackingProducer] Flush complete. Stats: {self.get_stats_summary()}")

        return chunks_written

    def get_stats_summary(self) -> str:
        """Get a summary of packing statistics."""
        self.stats.update_averages()
        return (
            f"samples_read={self.stats.total_samples_read}, "
            f"samples_packed={self.stats.total_samples_packed}, "
            f"packs={self.stats.total_packs_created}, "
            f"chunks={self.stats.total_chunks_written}, "
            f"dropped_long={self.stats.samples_dropped_too_long}, "
            f"dropped_error={self.stats.samples_dropped_encode_error}, "
            f"avg_samples_per_pack={self.stats.avg_samples_per_pack:.2f}, "
            f"avg_utilization={self.stats.avg_utilization:.1%}"
        )


def main():
    """CLI entry point for packing producer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-pack samples for Megatron-SWIFT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Pack a single file
    python -m swift.llm.dataset.packing_producer \\
        --model /path/to/Qwen3-Omni \\
        --input /data/raw_samples.jsonl \\
        --output /data/packed_chunks \\
        --packing_length 8192

    # Pack multiple files
    python -m swift.llm.dataset.packing_producer \\
        --model /path/to/Qwen3-Omni \\
        --input /data/raw/*.jsonl \\
        --output /data/packed_chunks \\
        --packing_length 8192 \\
        --samples_per_chunk 100
        """
    )

    parser.add_argument('--model', required=True, help='Path to model (for tokenizer/template)')
    parser.add_argument('--input', required=True, nargs='+', help='Input JSONL file(s) with raw samples')
    parser.add_argument('--output', required=True, help='Output directory for packed chunks')
    parser.add_argument('--packing_length', type=int, default=8192, help='Maximum tokens per pack')
    parser.add_argument('--samples_per_chunk', type=int, default=100, help='Number of packs per chunk file')
    parser.add_argument('--truncation_strategy', default='delete', choices=['delete', 'left', 'right'],
                       help='How to handle overlong samples')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Samples to buffer before packing')
    parser.add_argument('--model_type', default=None, help='Model type override')
    parser.add_argument('--template_type', default=None, help='Template type override')

    args = parser.parse_args()

    # Initialize producer
    producer = PackingProducer(
        model_path=args.model,
        packing_length=args.packing_length,
        output_dir=args.output,
        samples_per_chunk=args.samples_per_chunk,
        truncation_strategy=args.truncation_strategy,
        buffer_size=args.buffer_size,
        model_type=args.model_type,
        template_type=args.template_type,
    )

    # Process input files
    import glob
    input_files = []
    for pattern in args.input:
        input_files.extend(glob.glob(pattern))

    if not input_files:
        logger.error(f"No input files found matching: {args.input}")
        return 1

    logger.info(f"[PackingProducer] Processing {len(input_files)} input file(s)")

    for input_file in sorted(input_files):
        producer.process_file(input_file)

    # Flush remaining
    producer.flush()

    logger.info(f"[PackingProducer] Done! Output directory: {args.output}")
    return 0


if __name__ == '__main__':
    exit(main())
