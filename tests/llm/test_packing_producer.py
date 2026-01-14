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


# ============================================================================
# LazyShardedDataset tests (standalone implementation for testing)
# ============================================================================

import re
import threading
import time
from typing import Callable


class MockLazyShardedDataset:
    """Standalone mock of LazyShardedDataset for testing without swift dependencies.

    This mirrors the core logic of the real LazyShardedDataset.
    """

    def __init__(
        self,
        directory: str,
        encode_fn: Callable = None,
        samples_per_chunk: int = 1000,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        pre_packed: bool = False,
        max_wait_time: float = 1.0,
        wait_poll_interval: float = 0.1,
        mark_completed: bool = True,
    ):
        self.directory = Path(directory)
        self.encode_fn = encode_fn
        self.samples_per_chunk = samples_per_chunk
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.pre_packed = pre_packed
        self.max_wait_time = max_wait_time
        self.wait_poll_interval = wait_poll_interval
        self.mark_completed = mark_completed
        self.chunk_pattern = re.compile(r'chunk_(\d+)\.jsonl')

        self._chunk_cache: Dict[int, List[Dict[str, Any]]] = {}

        if not pre_packed and encode_fn is None:
            raise ValueError("encode_fn is required when pre_packed=False")

    def _is_my_chunk(self, chunk_idx: int) -> bool:
        return chunk_idx % self.dp_world_size == self.dp_rank

    def _chunk_path(self, chunk_idx: int) -> Path:
        return self.directory / f'chunk_{chunk_idx:05d}.jsonl'

    def _completed_marker_path(self, chunk_path: Path) -> Path:
        return chunk_path.with_suffix(chunk_path.suffix + '.completed')

    def _wait_for_chunk(self, chunk_idx: int) -> Optional[Path]:
        chunk_path = self._chunk_path(chunk_idx)
        start_time = time.time()

        while True:
            if chunk_path.exists():
                return chunk_path

            elapsed = time.time() - start_time
            if self.max_wait_time > 0 and elapsed > self.max_wait_time:
                return None

            time.sleep(self.wait_poll_interval)

    def _load_chunk(self, chunk_idx: int) -> List[Dict[str, Any]]:
        if chunk_idx in self._chunk_cache:
            return self._chunk_cache[chunk_idx]

        if not self._is_my_chunk(chunk_idx):
            return []

        chunk_path = self._wait_for_chunk(chunk_idx)
        if chunk_path is None:
            return []

        samples = []
        with open(chunk_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Skip corrupt lines

                if self.pre_packed:
                    if 'length' not in raw and 'pack_length' in raw:
                        raw['length'] = raw['pack_length']
                    samples.append(raw)
                else:
                    encoded = self.encode_fn(raw)
                    if encoded is not None:
                        samples.append(encoded)

        if self.mark_completed and samples:
            self._completed_marker_path(chunk_path).touch()

        self._chunk_cache[chunk_idx] = samples
        return samples

    def _local_idx_to_chunk_and_offset(self, local_idx: int) -> tuple:
        my_chunk_num = local_idx // self.samples_per_chunk
        sample_offset = local_idx % self.samples_per_chunk
        global_chunk_idx = self.dp_rank + (my_chunk_num * self.dp_world_size)
        return global_chunk_idx, sample_offset

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_chunk_idx, sample_offset = self._local_idx_to_chunk_and_offset(idx)
        samples = self._load_chunk(global_chunk_idx)

        if not samples:
            raise IndexError(f'No samples for idx {idx} (chunk {global_chunk_idx})')

        if sample_offset >= len(samples):
            sample_offset = sample_offset % len(samples)

        return samples[sample_offset]

    def __len__(self) -> int:
        return 100_000_000  # Large number, training stops via max_steps


class TestLazyShardedDatasetChunkAssignment:
    """Tests for chunk assignment logic (modulo distribution)."""

    def test_single_rank(self):
        """Single rank should get all chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                encode_fn=lambda x: x,
                dp_rank=0,
                dp_world_size=1,
            )

            # All chunks belong to rank 0
            assert dataset._is_my_chunk(0) == True
            assert dataset._is_my_chunk(1) == True
            assert dataset._is_my_chunk(100) == True

    def test_two_ranks(self):
        """Two ranks should split chunks evenly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_rank0 = MockLazyShardedDataset(
                directory=tmpdir,
                encode_fn=lambda x: x,
                dp_rank=0,
                dp_world_size=2,
            )
            dataset_rank1 = MockLazyShardedDataset(
                directory=tmpdir,
                encode_fn=lambda x: x,
                dp_rank=1,
                dp_world_size=2,
            )

            # Rank 0 gets even chunks
            assert dataset_rank0._is_my_chunk(0) == True
            assert dataset_rank0._is_my_chunk(1) == False
            assert dataset_rank0._is_my_chunk(2) == True
            assert dataset_rank0._is_my_chunk(3) == False

            # Rank 1 gets odd chunks
            assert dataset_rank1._is_my_chunk(0) == False
            assert dataset_rank1._is_my_chunk(1) == True
            assert dataset_rank1._is_my_chunk(2) == False
            assert dataset_rank1._is_my_chunk(3) == True

    def test_eight_ranks(self):
        """Eight ranks should distribute chunks correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for rank in range(8):
                dataset = MockLazyShardedDataset(
                    directory=tmpdir,
                    encode_fn=lambda x: x,
                    dp_rank=rank,
                    dp_world_size=8,
                )

                # Each rank gets chunks where chunk_idx % 8 == rank
                for chunk_idx in range(24):
                    expected = (chunk_idx % 8 == rank)
                    assert dataset._is_my_chunk(chunk_idx) == expected, \
                        f"Rank {rank}, chunk {chunk_idx}: expected {expected}"


class TestLazyShardedDatasetIndexMapping:
    """Tests for local index to chunk mapping."""

    def test_local_idx_single_rank(self):
        """Local index mapping for single rank."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                encode_fn=lambda x: x,
                samples_per_chunk=100,
                dp_rank=0,
                dp_world_size=1,
            )

            # idx 0-99 -> chunk 0
            assert dataset._local_idx_to_chunk_and_offset(0) == (0, 0)
            assert dataset._local_idx_to_chunk_and_offset(50) == (0, 50)
            assert dataset._local_idx_to_chunk_and_offset(99) == (0, 99)

            # idx 100-199 -> chunk 1
            assert dataset._local_idx_to_chunk_and_offset(100) == (1, 0)
            assert dataset._local_idx_to_chunk_and_offset(150) == (1, 50)

    def test_local_idx_multi_rank(self):
        """Local index mapping for multiple ranks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Rank 3 of 8
            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                encode_fn=lambda x: x,
                samples_per_chunk=1000,
                dp_rank=3,
                dp_world_size=8,
            )

            # idx 0-999 -> rank's 1st chunk = global chunk 3
            assert dataset._local_idx_to_chunk_and_offset(0) == (3, 0)
            assert dataset._local_idx_to_chunk_and_offset(500) == (3, 500)

            # idx 1000-1999 -> rank's 2nd chunk = global chunk 11 (3 + 8)
            assert dataset._local_idx_to_chunk_and_offset(1000) == (11, 0)
            assert dataset._local_idx_to_chunk_and_offset(1500) == (11, 500)

            # idx 2000-2999 -> rank's 3rd chunk = global chunk 19 (3 + 16)
            assert dataset._local_idx_to_chunk_and_offset(2000) == (19, 0)


class TestLazyShardedDatasetPrePacked:
    """Tests for pre-packed chunk loading."""

    def test_load_prepacked_chunk(self):
        """Load pre-packed chunk directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write pre-packed chunk
            chunk_path = Path(tmpdir) / "chunk_00000.jsonl"
            packs = [
                {
                    "input_ids": [1, 2, 3, 4, 5],
                    "labels": [-100, 2, 3, 4, 5],
                    "position_ids": [0, 1, 2, 0, 1],
                    "lengths": [3, 2],
                    "pack_length": 5,
                    "num_samples": 2,
                },
                {
                    "input_ids": [10, 20, 30],
                    "labels": [10, 20, 30],
                    "position_ids": [0, 1, 2],
                    "lengths": [3],
                    "pack_length": 3,
                    "num_samples": 1,
                },
            ]
            with open(chunk_path, 'w') as f:
                for pack in packs:
                    f.write(json.dumps(pack) + '\n')

            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                samples_per_chunk=2,
                pre_packed=True,
                dp_rank=0,
                dp_world_size=1,
            )

            # Access samples
            sample0 = dataset[0]
            assert sample0['input_ids'] == [1, 2, 3, 4, 5]
            assert sample0['length'] == 5  # Added from pack_length

            sample1 = dataset[1]
            assert sample1['input_ids'] == [10, 20, 30]

    def test_prepacked_adds_length_field(self):
        """Pre-packed should add length from pack_length if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_path = Path(tmpdir) / "chunk_00000.jsonl"
            pack = {
                "input_ids": [1, 2, 3],
                "labels": [1, 2, 3],
                "pack_length": 3,
                # No 'length' field
            }
            with open(chunk_path, 'w') as f:
                f.write(json.dumps(pack) + '\n')

            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                samples_per_chunk=1,
                pre_packed=True,
                dp_rank=0,
                dp_world_size=1,
            )

            sample = dataset[0]
            assert sample['length'] == 3  # Should be added


class TestLazyShardedDatasetRawFormat:
    """Tests for raw format chunk loading with encoding."""

    def test_load_raw_chunk_with_encode(self):
        """Load raw chunk and encode samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_path = Path(tmpdir) / "chunk_00000.jsonl"
            raw_samples = [
                {"messages": [{"role": "user", "content": "hello"}]},
                {"messages": [{"role": "user", "content": "world"}]},
            ]
            with open(chunk_path, 'w') as f:
                for sample in raw_samples:
                    f.write(json.dumps(sample) + '\n')

            # Mock encode function
            def mock_encode(sample):
                content = sample['messages'][0]['content']
                tokens = [ord(c) for c in content]
                return {
                    'input_ids': tokens,
                    'labels': tokens,
                    'length': len(tokens),
                }

            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                encode_fn=mock_encode,
                samples_per_chunk=2,
                pre_packed=False,
                dp_rank=0,
                dp_world_size=1,
            )

            sample0 = dataset[0]
            assert sample0['input_ids'] == [ord(c) for c in "hello"]
            assert sample0['length'] == 5

            sample1 = dataset[1]
            assert sample1['input_ids'] == [ord(c) for c in "world"]

    def test_encode_fn_required_when_not_prepacked(self):
        """Should raise error if encode_fn missing for raw format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="encode_fn is required"):
                MockLazyShardedDataset(
                    directory=tmpdir,
                    encode_fn=None,  # Missing!
                    pre_packed=False,
                )


class TestLazyShardedDatasetCompletionMarkers:
    """Tests for completion marker behavior."""

    def test_completion_marker_created(self):
        """Completion marker should be created after loading chunk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_path = Path(tmpdir) / "chunk_00000.jsonl"
            with open(chunk_path, 'w') as f:
                f.write(json.dumps({"input_ids": [1], "pack_length": 1}) + '\n')

            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                samples_per_chunk=1,
                pre_packed=True,
                mark_completed=True,
                dp_rank=0,
                dp_world_size=1,
            )

            # Marker shouldn't exist yet
            marker_path = chunk_path.with_suffix('.jsonl.completed')
            assert not marker_path.exists()

            # Load sample (triggers chunk load)
            _ = dataset[0]

            # Marker should exist now
            assert marker_path.exists()

    def test_completion_marker_disabled(self):
        """Completion marker should not be created when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_path = Path(tmpdir) / "chunk_00000.jsonl"
            with open(chunk_path, 'w') as f:
                f.write(json.dumps({"input_ids": [1], "pack_length": 1}) + '\n')

            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                samples_per_chunk=1,
                pre_packed=True,
                mark_completed=False,  # Disabled
                dp_rank=0,
                dp_world_size=1,
            )

            _ = dataset[0]

            marker_path = chunk_path.with_suffix('.jsonl.completed')
            assert not marker_path.exists()


class TestLazyShardedDatasetWaiting:
    """Tests for chunk waiting behavior."""

    def test_timeout_on_missing_chunk(self):
        """Should timeout and raise error for missing chunk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                samples_per_chunk=1,
                pre_packed=True,
                max_wait_time=0.2,  # Short timeout
                wait_poll_interval=0.05,
                dp_rank=0,
                dp_world_size=1,
            )

            with pytest.raises(IndexError):
                _ = dataset[0]  # Chunk doesn't exist

    def test_waits_for_chunk_to_appear(self):
        """Should wait for chunk to appear within timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                samples_per_chunk=1,
                pre_packed=True,
                max_wait_time=2.0,
                wait_poll_interval=0.1,
                dp_rank=0,
                dp_world_size=1,
            )

            # Write chunk after a delay in another thread
            def write_chunk_delayed():
                time.sleep(0.3)
                chunk_path = Path(tmpdir) / "chunk_00000.jsonl"
                with open(chunk_path, 'w') as f:
                    f.write(json.dumps({"input_ids": [42], "pack_length": 1}) + '\n')

            writer_thread = threading.Thread(target=write_chunk_delayed)
            writer_thread.start()

            # This should wait and then succeed
            sample = dataset[0]
            assert sample['input_ids'] == [42]

            writer_thread.join()


class TestLazyShardedDatasetCaching:
    """Tests for chunk caching behavior."""

    def test_chunk_cached_after_load(self):
        """Chunk should be cached after first load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_path = Path(tmpdir) / "chunk_00000.jsonl"
            with open(chunk_path, 'w') as f:
                f.write(json.dumps({"input_ids": [1], "pack_length": 1}) + '\n')
                f.write(json.dumps({"input_ids": [2], "pack_length": 1}) + '\n')

            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                samples_per_chunk=2,
                pre_packed=True,
                dp_rank=0,
                dp_world_size=1,
            )

            assert len(dataset._chunk_cache) == 0

            _ = dataset[0]  # Load first sample
            assert 0 in dataset._chunk_cache
            assert len(dataset._chunk_cache[0]) == 2

            _ = dataset[1]  # Should use cache
            assert len(dataset._chunk_cache) == 1  # Still just one chunk cached


class TestEndToEndFlow:
    """Tests for complete producer-to-dataset flow."""

    def test_packer_to_dataset_flow(self):
        """Test packing samples and reading via dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Create packed chunks using BinPacker
            packer = BinPacker(packing_length=100)

            samples_batch1 = [
                EncodedSample(input_ids=[1, 2, 3], labels=[1, 2, 3], length=3),
                EncodedSample(input_ids=[4, 5], labels=[4, 5], length=2),
            ]
            samples_batch2 = [
                EncodedSample(input_ids=[10, 20, 30, 40], labels=[10, 20, 30, 40], length=4),
            ]

            pack1 = packer.create_pack(samples_batch1)
            pack2 = packer.create_pack(samples_batch2)

            # Write chunks
            chunk0_path = Path(tmpdir) / "chunk_00000.jsonl"
            with open(chunk0_path, 'w') as f:
                f.write(json.dumps(pack1) + '\n')
                f.write(json.dumps(pack2) + '\n')

            # Step 2: Read via dataset
            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                samples_per_chunk=2,
                pre_packed=True,
                dp_rank=0,
                dp_world_size=1,
            )

            sample0 = dataset[0]
            assert sample0['input_ids'] == [1, 2, 3, 4, 5]
            assert sample0['position_ids'] == [0, 1, 2, 0, 1]
            assert sample0['lengths'] == [3, 2]

            sample1 = dataset[1]
            assert sample1['input_ids'] == [10, 20, 30, 40]

    def test_multi_rank_distribution(self):
        """Test that multiple ranks read different chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 4 chunks
            for i in range(4):
                chunk_path = Path(tmpdir) / f"chunk_{i:05d}.jsonl"
                pack = {
                    "input_ids": [i * 100 + j for j in range(5)],
                    "labels": [i * 100 + j for j in range(5)],
                    "pack_length": 5,
                    "chunk_id": i,  # For verification
                }
                with open(chunk_path, 'w') as f:
                    f.write(json.dumps(pack) + '\n')

            # Create datasets for 2 ranks
            dataset_rank0 = MockLazyShardedDataset(
                directory=tmpdir,
                samples_per_chunk=1,
                pre_packed=True,
                dp_rank=0,
                dp_world_size=2,
            )
            dataset_rank1 = MockLazyShardedDataset(
                directory=tmpdir,
                samples_per_chunk=1,
                pre_packed=True,
                dp_rank=1,
                dp_world_size=2,
            )

            # Rank 0 should get chunks 0, 2
            sample_r0_0 = dataset_rank0[0]  # -> chunk 0
            assert sample_r0_0['chunk_id'] == 0

            sample_r0_1 = dataset_rank0[1]  # -> chunk 2
            assert sample_r0_1['chunk_id'] == 2

            # Rank 1 should get chunks 1, 3
            sample_r1_0 = dataset_rank1[0]  # -> chunk 1
            assert sample_r1_0['chunk_id'] == 1

            sample_r1_1 = dataset_rank1[1]  # -> chunk 3
            assert sample_r1_1['chunk_id'] == 3


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_corrupt_json_skipped(self):
        """Corrupt JSON lines should be skipped gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_path = Path(tmpdir) / "chunk_00000.jsonl"
            with open(chunk_path, 'w') as f:
                f.write(json.dumps({"input_ids": [1], "pack_length": 1}) + '\n')
                f.write("not valid json\n")  # Corrupt line
                f.write(json.dumps({"input_ids": [2], "pack_length": 1}) + '\n')

            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                samples_per_chunk=3,
                pre_packed=True,
                dp_rank=0,
                dp_world_size=1,
            )

            # Should load 2 valid samples, skip corrupt one
            samples = dataset._load_chunk(0)
            assert len(samples) == 2
            assert samples[0]['input_ids'] == [1]
            assert samples[1]['input_ids'] == [2]

    def test_empty_chunk_file(self):
        """Empty chunk file should return empty samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_path = Path(tmpdir) / "chunk_00000.jsonl"
            chunk_path.touch()  # Empty file

            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                samples_per_chunk=1,
                pre_packed=True,
                dp_rank=0,
                dp_world_size=1,
            )

            samples = dataset._load_chunk(0)
            assert samples == []

    def test_encode_fn_returns_none(self):
        """Samples where encode_fn returns None should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_path = Path(tmpdir) / "chunk_00000.jsonl"
            with open(chunk_path, 'w') as f:
                f.write(json.dumps({"valid": True}) + '\n')
                f.write(json.dumps({"valid": False}) + '\n')
                f.write(json.dumps({"valid": True}) + '\n')

            def selective_encode(sample):
                if sample.get('valid'):
                    return {'input_ids': [1], 'length': 1}
                return None  # Skip invalid

            dataset = MockLazyShardedDataset(
                directory=tmpdir,
                encode_fn=selective_encode,
                samples_per_chunk=3,
                pre_packed=False,
                dp_rank=0,
                dp_world_size=1,
            )

            samples = dataset._load_chunk(0)
            assert len(samples) == 2  # Only valid ones


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
