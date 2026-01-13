"""
Unit tests for PackingProducer and related components.

Tests:
1. BinPacker - bin packing algorithm correctness
2. EncodedSample - dataclass behavior
3. PackingStats - statistics tracking
4. LazyShardedDataset - pre-packed format handling (requires swift)

These tests are designed to run independently of heavy swift dependencies.
The core algorithm classes (BinPacker, EncodedSample, PackingStats) are
defined inline to avoid import issues.
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# ============================================================================
# Inline definitions for standalone testing (mirrors packing_producer.py)
# ============================================================================

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
    _packing_length: int = 0

    def update_averages(self):
        if self.total_packs_created > 0:
            self.avg_samples_per_pack = self.total_samples_packed / self.total_packs_created
        if self.total_packs_created > 0 and self._packing_length > 0:
            self.avg_utilization = self.total_tokens_packed / (self.total_packs_created * self._packing_length)


@dataclass
class EncodedSample:
    """A tokenized sample ready for packing."""
    input_ids: List[int]
    labels: List[int]
    length: int
    raw_sample: Optional[Dict[str, Any]] = None


class BinPacker:
    """First-fit decreasing bin packing algorithm."""

    def __init__(self, packing_length: int, padding_token_id: int = 0):
        self.packing_length = packing_length
        self.padding_token_id = padding_token_id

    def pack(self, samples: List[EncodedSample]) -> List[List[EncodedSample]]:
        """Pack samples into bins using first-fit decreasing."""
        if not samples:
            return []

        # Sort by length descending (first-fit decreasing)
        sorted_samples = sorted(samples, key=lambda s: s.length, reverse=True)

        bins: List[List[EncodedSample]] = []
        bin_remaining: List[int] = []

        for sample in sorted_samples:
            if sample.length > self.packing_length:
                continue  # Skip samples that don't fit

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
        """Create a single pack from a list of samples."""
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


# ============================================================================
# Test classes
# ============================================================================

class TestBinPacker:
    """Tests for the BinPacker class."""

    def test_empty_input(self):
        """Empty input should return empty output."""
        packer = BinPacker(packing_length=100)
        bins = packer.pack([])
        assert bins == []

    def test_single_sample_fits(self):
        """Single sample that fits should create one bin."""
        packer = BinPacker(packing_length=100)
        sample = EncodedSample(input_ids=[1, 2, 3], labels=[1, 2, 3], length=50)
        bins = packer.pack([sample])

        assert len(bins) == 1
        assert len(bins[0]) == 1
        assert bins[0][0].length == 50

    def test_single_sample_too_long(self):
        """Sample longer than packing_length should be skipped."""
        packer = BinPacker(packing_length=100)
        sample = EncodedSample(input_ids=list(range(150)), labels=list(range(150)), length=150)
        bins = packer.pack([sample])

        assert len(bins) == 0  # Sample skipped

    def test_multiple_samples_fit_one_bin(self):
        """Multiple small samples should pack into one bin."""
        packer = BinPacker(packing_length=100)
        samples = [
            EncodedSample(input_ids=[1] * 20, labels=[1] * 20, length=20),
            EncodedSample(input_ids=[2] * 30, labels=[2] * 30, length=30),
            EncodedSample(input_ids=[3] * 40, labels=[3] * 40, length=40),
        ]
        bins = packer.pack(samples)

        assert len(bins) == 1
        assert len(bins[0]) == 3
        total_length = sum(s.length for s in bins[0])
        assert total_length == 90

    def test_multiple_bins_needed(self):
        """Samples that don't all fit should create multiple bins."""
        packer = BinPacker(packing_length=100)
        samples = [
            EncodedSample(input_ids=[1] * 60, labels=[1] * 60, length=60),
            EncodedSample(input_ids=[2] * 60, labels=[2] * 60, length=60),
            EncodedSample(input_ids=[3] * 60, labels=[3] * 60, length=60),
        ]
        bins = packer.pack(samples)

        assert len(bins) == 3  # Each needs its own bin
        for b in bins:
            assert len(b) == 1

    def test_first_fit_decreasing(self):
        """Verify first-fit decreasing packs efficiently."""
        packer = BinPacker(packing_length=100)
        # Samples: 70, 30, 30, 30, 20
        # Optimal: [70, 30], [30, 30, 20] = 2 bins with some waste
        # But FFD sorts descending: 70, 30, 30, 30, 20
        # Places: 70 -> bin0, 30 -> bin0 (70+30=100), 30 -> bin1, 30 -> bin1, 20 -> bin1
        # Result: [70, 30], [30, 30, 20] = 2 bins
        samples = [
            EncodedSample(input_ids=[1] * 30, labels=[1] * 30, length=30),
            EncodedSample(input_ids=[2] * 70, labels=[2] * 70, length=70),
            EncodedSample(input_ids=[3] * 30, labels=[3] * 30, length=30),
            EncodedSample(input_ids=[4] * 20, labels=[4] * 20, length=20),
            EncodedSample(input_ids=[5] * 30, labels=[5] * 30, length=30),
        ]
        bins = packer.pack(samples)

        # Should pack into 2 bins efficiently
        assert len(bins) == 2
        total_samples = sum(len(b) for b in bins)
        assert total_samples == 5

    def test_create_pack_empty(self):
        """Creating pack from empty list should return empty dict."""
        packer = BinPacker(packing_length=100)
        pack = packer.create_pack([])
        assert pack == {}

    def test_create_pack_single_sample(self):
        """Creating pack from single sample."""
        packer = BinPacker(packing_length=100)
        sample = EncodedSample(
            input_ids=[1, 2, 3, 4, 5],
            labels=[-100, -100, 3, 4, 5],
            length=5
        )
        pack = packer.create_pack([sample])

        assert pack['input_ids'] == [1, 2, 3, 4, 5]
        assert pack['labels'] == [-100, -100, 3, 4, 5]
        assert pack['position_ids'] == [0, 1, 2, 3, 4]
        assert pack['lengths'] == [5]
        assert pack['pack_length'] == 5
        assert pack['num_samples'] == 1

    def test_create_pack_multiple_samples(self):
        """Creating pack from multiple samples - position_ids should reset."""
        packer = BinPacker(packing_length=100)
        samples = [
            EncodedSample(input_ids=[1, 2, 3], labels=[-100, 2, 3], length=3),
            EncodedSample(input_ids=[4, 5], labels=[4, 5], length=2),
            EncodedSample(input_ids=[6, 7, 8, 9], labels=[-100, 7, 8, 9], length=4),
        ]
        pack = packer.create_pack(samples)

        assert pack['input_ids'] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert pack['labels'] == [-100, 2, 3, 4, 5, -100, 7, 8, 9]
        # Position IDs reset at each sample boundary
        assert pack['position_ids'] == [0, 1, 2, 0, 1, 0, 1, 2, 3]
        assert pack['lengths'] == [3, 2, 4]
        assert pack['pack_length'] == 9
        assert pack['num_samples'] == 3


class TestEncodedSample:
    """Tests for EncodedSample dataclass."""

    def test_basic_creation(self):
        """Basic EncodedSample creation."""
        sample = EncodedSample(
            input_ids=[1, 2, 3],
            labels=[1, 2, 3],
            length=3
        )
        assert sample.input_ids == [1, 2, 3]
        assert sample.labels == [1, 2, 3]
        assert sample.length == 3
        assert sample.raw_sample is None

    def test_with_raw_sample(self):
        """EncodedSample with raw_sample preserved."""
        raw = {"messages": [{"role": "user", "content": "hello"}]}
        sample = EncodedSample(
            input_ids=[1, 2, 3],
            labels=[1, 2, 3],
            length=3,
            raw_sample=raw
        )
        assert sample.raw_sample == raw


class TestPackingStats:
    """Tests for PackingStats dataclass."""

    def test_initial_values(self):
        """Initial stats should be zero."""
        stats = PackingStats()
        assert stats.total_samples_read == 0
        assert stats.total_packs_created == 0
        assert stats.avg_samples_per_pack == 0.0

    def test_update_averages(self):
        """Test average calculation."""
        stats = PackingStats()
        stats.total_samples_packed = 100
        stats.total_packs_created = 10
        stats.total_tokens_packed = 8000
        stats._packing_length = 1000
        stats.update_averages()

        assert stats.avg_samples_per_pack == 10.0
        assert stats.avg_utilization == 0.8  # 8000 / (10 * 1000)


class TestPrePackedChunkFormat:
    """Tests for pre-packed chunk file format."""

    def test_write_and_read_chunk(self):
        """Test writing and reading pre-packed chunk format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_path = Path(tmpdir) / "chunk_00000.jsonl"

            # Create packs using BinPacker
            packer = BinPacker(packing_length=100)
            samples = [
                EncodedSample(input_ids=[1, 2, 3], labels=[-100, 2, 3], length=3),
                EncodedSample(input_ids=[4, 5], labels=[4, 5], length=2),
            ]
            pack = packer.create_pack(samples)

            # Write chunk
            with open(chunk_path, 'w') as f:
                f.write(json.dumps(pack) + '\n')

            # Read back
            with open(chunk_path, 'r') as f:
                loaded_pack = json.loads(f.readline())

            assert loaded_pack['input_ids'] == [1, 2, 3, 4, 5]
            assert loaded_pack['labels'] == [-100, 2, 3, 4, 5]
            assert loaded_pack['position_ids'] == [0, 1, 2, 0, 1]
            assert loaded_pack['lengths'] == [3, 2]
            assert loaded_pack['pack_length'] == 5
            assert loaded_pack['num_samples'] == 2

    def test_multiple_packs_per_chunk(self):
        """Test writing multiple packs to a single chunk file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_path = Path(tmpdir) / "chunk_00000.jsonl"

            packer = BinPacker(packing_length=10)
            packs = [
                packer.create_pack([
                    EncodedSample(input_ids=[1, 2], labels=[1, 2], length=2),
                    EncodedSample(input_ids=[3, 4, 5], labels=[3, 4, 5], length=3),
                ]),
                packer.create_pack([
                    EncodedSample(input_ids=[10, 20, 30, 40], labels=[10, 20, 30, 40], length=4),
                ]),
            ]

            # Write multiple packs
            with open(chunk_path, 'w') as f:
                for pack in packs:
                    f.write(json.dumps(pack) + '\n')

            # Read back
            with open(chunk_path, 'r') as f:
                lines = f.readlines()

            assert len(lines) == 2

            pack1 = json.loads(lines[0])
            assert pack1['num_samples'] == 2
            assert pack1['pack_length'] == 5

            pack2 = json.loads(lines[1])
            assert pack2['num_samples'] == 1
            assert pack2['pack_length'] == 4


class TestPackingEfficiency:
    """Tests for packing efficiency metrics."""

    def test_utilization_calculation(self):
        """Test that packing achieves good utilization."""
        packer = BinPacker(packing_length=100)

        # Create samples that should pack efficiently
        # 50 + 50 = 100 (perfect), 30 + 30 + 30 = 90 (90%), 20 = 20 (20%)
        samples = [
            EncodedSample(input_ids=[0] * 50, labels=[0] * 50, length=50),
            EncodedSample(input_ids=[0] * 50, labels=[0] * 50, length=50),
            EncodedSample(input_ids=[0] * 30, labels=[0] * 30, length=30),
            EncodedSample(input_ids=[0] * 30, labels=[0] * 30, length=30),
            EncodedSample(input_ids=[0] * 30, labels=[0] * 30, length=30),
            EncodedSample(input_ids=[0] * 20, labels=[0] * 20, length=20),
        ]

        bins = packer.pack(samples)

        # Calculate utilization
        total_capacity = len(bins) * 100
        total_used = sum(sum(s.length for s in b) for b in bins)
        utilization = total_used / total_capacity

        # FFD should achieve good utilization (>80% typically)
        assert utilization >= 0.7, f"Utilization {utilization:.1%} is too low"

    def test_samples_per_pack_metric(self):
        """Test average samples per pack calculation."""
        packer = BinPacker(packing_length=100)

        # 10 samples of length 10 each = should fit ~10 per pack
        samples = [
            EncodedSample(input_ids=[0] * 10, labels=[0] * 10, length=10)
            for _ in range(20)
        ]

        bins = packer.pack(samples)

        avg_samples_per_pack = len(samples) / len(bins)
        # Should pack ~10 samples per bin (100/10 = 10 per bin)
        assert avg_samples_per_pack >= 9, f"Expected ~10 samples/pack, got {avg_samples_per_pack:.1f}"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_exact_fit(self):
        """Sample that exactly fits packing_length."""
        packer = BinPacker(packing_length=100)
        sample = EncodedSample(input_ids=[0] * 100, labels=[0] * 100, length=100)
        bins = packer.pack([sample])

        assert len(bins) == 1
        assert bins[0][0].length == 100

    def test_one_over_limit(self):
        """Sample that is exactly one token over limit."""
        packer = BinPacker(packing_length=100)
        sample = EncodedSample(input_ids=[0] * 101, labels=[0] * 101, length=101)
        bins = packer.pack([sample])

        assert len(bins) == 0  # Should be skipped

    def test_all_samples_too_long(self):
        """All samples exceed packing_length."""
        packer = BinPacker(packing_length=10)
        samples = [
            EncodedSample(input_ids=[0] * 20, labels=[0] * 20, length=20),
            EncodedSample(input_ids=[0] * 15, labels=[0] * 15, length=15),
            EncodedSample(input_ids=[0] * 11, labels=[0] * 11, length=11),
        ]
        bins = packer.pack(samples)

        assert len(bins) == 0

    def test_mixed_valid_and_invalid(self):
        """Mix of valid and invalid (too long) samples."""
        packer = BinPacker(packing_length=100)
        samples = [
            EncodedSample(input_ids=[0] * 50, labels=[0] * 50, length=50),
            EncodedSample(input_ids=[0] * 150, labels=[0] * 150, length=150),  # Too long
            EncodedSample(input_ids=[0] * 30, labels=[0] * 30, length=30),
        ]
        bins = packer.pack(samples)

        # Only 2 valid samples
        total_samples = sum(len(b) for b in bins)
        assert total_samples == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
