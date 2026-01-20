"""
Unit tests for PackingProducer, BinPacker, and LazyShardedDataset.

Tests the packing producer pipeline for pre-packing training data,
and the lazy sharded dataset for distributed chunk loading.

Run with: python tests/llm/test_packing_producer.py
Or: pytest tests/llm/test_packing_producer.py -v
"""

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from swift.llm.dataset.lazy_sharded import LazyShardedDataset
from swift.llm.dataset.packing_producer import BinPacker, EncodedSample, PackingStats
from swift.utils import get_logger

logger = get_logger()


class TestBinPacker(unittest.TestCase):
    """Tests for the BinPacker class."""

    def test_empty_input(self):
        """Empty input should return empty output."""
        packer = BinPacker(packing_length=100)
        bins = packer.pack([])
        self.assertEqual(bins, [])

    def test_single_sample_fits(self):
        """Single sample that fits should create one bin."""
        packer = BinPacker(packing_length=100)
        sample = EncodedSample(input_ids=[1, 2, 3], labels=[1, 2, 3], length=50)
        bins = packer.pack([sample])

        self.assertEqual(len(bins), 1)
        self.assertEqual(len(bins[0]), 1)
        self.assertEqual(bins[0][0].length, 50)

    def test_single_sample_too_long(self):
        """Sample longer than packing_length should be skipped."""
        packer = BinPacker(packing_length=100)
        sample = EncodedSample(input_ids=list(range(150)), labels=list(range(150)), length=150)
        bins = packer.pack([sample])

        self.assertEqual(len(bins), 0)  # Sample skipped

    def test_multiple_samples_fit_one_bin(self):
        """Multiple small samples should pack into one bin."""
        packer = BinPacker(packing_length=100)
        samples = [
            EncodedSample(input_ids=[1] * 20, labels=[1] * 20, length=20),
            EncodedSample(input_ids=[2] * 30, labels=[2] * 30, length=30),
            EncodedSample(input_ids=[3] * 40, labels=[3] * 40, length=40),
        ]
        bins = packer.pack(samples)

        self.assertEqual(len(bins), 1)
        self.assertEqual(len(bins[0]), 3)
        total_length = sum(s.length for s in bins[0])
        self.assertEqual(total_length, 90)

    def test_multiple_bins_needed(self):
        """Samples that don't all fit should create multiple bins."""
        packer = BinPacker(packing_length=100)
        samples = [
            EncodedSample(input_ids=[1] * 60, labels=[1] * 60, length=60),
            EncodedSample(input_ids=[2] * 60, labels=[2] * 60, length=60),
            EncodedSample(input_ids=[3] * 60, labels=[3] * 60, length=60),
        ]
        bins = packer.pack(samples)

        self.assertEqual(len(bins), 3)  # Each needs its own bin
        for b in bins:
            self.assertEqual(len(b), 1)

    def test_first_fit_decreasing(self):
        """Verify first-fit decreasing packs efficiently."""
        packer = BinPacker(packing_length=100)
        # Samples: 70, 30, 30, 30, 20
        # FFD sorts descending: 70, 30, 30, 30, 20
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
        self.assertEqual(len(bins), 2)
        total_packed = sum(s.length for b in bins for s in b)
        self.assertEqual(total_packed, 180)

    def test_create_pack_concatenates_correctly(self):
        """Verify create_pack concatenates input_ids, labels, and position_ids."""
        packer = BinPacker(packing_length=100)
        samples = [
            EncodedSample(input_ids=[1, 2, 3], labels=[-100, 2, 3], length=3),
            EncodedSample(input_ids=[4, 5], labels=[4, 5], length=2),
        ]
        pack = packer.create_pack(samples)

        self.assertEqual(pack['input_ids'], [1, 2, 3, 4, 5])
        self.assertEqual(pack['labels'], [-100, 2, 3, 4, 5])
        self.assertEqual(pack['position_ids'], [0, 1, 2, 0, 1])  # Reset at boundary
        self.assertEqual(pack['lengths'], [3, 2])
        self.assertEqual(pack['pack_length'], 5)
        self.assertEqual(pack['num_samples'], 2)

    def test_create_pack_empty(self):
        """Empty input should return empty dict."""
        packer = BinPacker(packing_length=100)
        pack = packer.create_pack([])
        self.assertEqual(pack, {})


class TestEncodedSample(unittest.TestCase):
    """Tests for the EncodedSample dataclass."""

    def test_creation(self):
        """Test basic creation."""
        sample = EncodedSample(
            input_ids=[1, 2, 3],
            labels=[1, 2, 3],
            length=3,
            raw_sample={'messages': [{'role': 'user', 'content': 'test'}]},
        )
        self.assertEqual(sample.input_ids, [1, 2, 3])
        self.assertEqual(sample.labels, [1, 2, 3])
        self.assertEqual(sample.length, 3)
        self.assertIsNotNone(sample.raw_sample)

    def test_optional_raw_sample(self):
        """raw_sample should be optional."""
        sample = EncodedSample(input_ids=[1], labels=[1], length=1)
        self.assertIsNone(sample.raw_sample)


class TestPackingStats(unittest.TestCase):
    """Tests for the PackingStats dataclass."""

    def test_initial_values(self):
        """Test default initial values."""
        stats = PackingStats()
        self.assertEqual(stats.total_samples_read, 0)
        self.assertEqual(stats.total_samples_packed, 0)
        self.assertEqual(stats.total_packs_created, 0)

    def test_update_averages(self):
        """Test average calculations."""
        stats = PackingStats(
            total_samples_packed=100, total_packs_created=10, total_tokens_packed=8000
        )
        stats._packing_length = 1000  # Set as attribute, not init param
        stats.update_averages()
        self.assertEqual(stats.avg_samples_per_pack, 10.0)
        self.assertEqual(stats.avg_utilization, 0.8)


class TestLazyShardedDatasetChunkAssignment(unittest.TestCase):
    """Tests for chunk assignment logic (modulo distribution)."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name
        logger.info(f'self.tmp_dir: {self.tmp_dir}')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_single_rank(self):
        """Single rank should get all chunks."""
        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=100,
            pre_packed=True,
            dp_rank=0,
            dp_world_size=1,
            max_wait_time=0.1,
        )

        # All chunks belong to rank 0
        self.assertTrue(dataset._is_my_chunk(0))
        self.assertTrue(dataset._is_my_chunk(1))
        self.assertTrue(dataset._is_my_chunk(100))

    def test_two_ranks(self):
        """Two ranks should split chunks evenly."""
        dataset_rank0 = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=100,
            pre_packed=True,
            dp_rank=0,
            dp_world_size=2,
            max_wait_time=0.1,
        )
        dataset_rank1 = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=100,
            pre_packed=True,
            dp_rank=1,
            dp_world_size=2,
            max_wait_time=0.1,
        )

        # Rank 0 gets even chunks
        self.assertTrue(dataset_rank0._is_my_chunk(0))
        self.assertFalse(dataset_rank0._is_my_chunk(1))
        self.assertTrue(dataset_rank0._is_my_chunk(2))

        # Rank 1 gets odd chunks
        self.assertFalse(dataset_rank1._is_my_chunk(0))
        self.assertTrue(dataset_rank1._is_my_chunk(1))
        self.assertFalse(dataset_rank1._is_my_chunk(2))

    def test_eight_ranks(self):
        """Eight ranks should distribute chunks correctly."""
        for rank in range(8):
            dataset = LazyShardedDataset(
                directory=self.tmp_dir,
                samples_per_chunk=100,
                pre_packed=True,
                dp_rank=rank,
                dp_world_size=8,
                max_wait_time=0.1,
            )

            # Each rank gets chunks where chunk_idx % 8 == rank
            for chunk_idx in range(24):
                expected = (chunk_idx % 8 == rank)
                self.assertEqual(
                    dataset._is_my_chunk(chunk_idx), expected, f'Rank {rank}, chunk {chunk_idx}: expected {expected}'
                )


class TestLazyShardedDatasetIndexMapping(unittest.TestCase):
    """Tests for local index to chunk mapping."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_local_idx_single_rank(self):
        """Local index mapping for single rank."""
        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=100,
            pre_packed=True,
            dp_rank=0,
            dp_world_size=1,
            max_wait_time=0.1,
        )

        # idx 0-99 -> chunk 0
        self.assertEqual(dataset._local_idx_to_chunk_and_offset(0), (0, 0))
        self.assertEqual(dataset._local_idx_to_chunk_and_offset(50), (0, 50))
        self.assertEqual(dataset._local_idx_to_chunk_and_offset(99), (0, 99))

        # idx 100-199 -> chunk 1
        self.assertEqual(dataset._local_idx_to_chunk_and_offset(100), (1, 0))
        self.assertEqual(dataset._local_idx_to_chunk_and_offset(150), (1, 50))

    def test_local_idx_multi_rank(self):
        """Local index mapping for multiple ranks."""
        # Rank 3 of 8
        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=1000,
            pre_packed=True,
            dp_rank=3,
            dp_world_size=8,
            max_wait_time=0.1,
        )

        # idx 0-999 -> rank's 1st chunk = global chunk 3
        self.assertEqual(dataset._local_idx_to_chunk_and_offset(0), (3, 0))
        self.assertEqual(dataset._local_idx_to_chunk_and_offset(500), (3, 500))

        # idx 1000-1999 -> rank's 2nd chunk = global chunk 11 (3 + 8)
        self.assertEqual(dataset._local_idx_to_chunk_and_offset(1000), (11, 0))
        self.assertEqual(dataset._local_idx_to_chunk_and_offset(1500), (11, 500))

        # idx 2000-2999 -> rank's 3rd chunk = global chunk 19 (3 + 16)
        self.assertEqual(dataset._local_idx_to_chunk_and_offset(2000), (19, 0))


class TestLazyShardedDatasetPrePacked(unittest.TestCase):
    """Tests for pre-packed chunk loading."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_prepacked_chunk(self):
        """Load pre-packed chunk directly."""
        # Write pre-packed chunk
        chunk_path = Path(self.tmp_dir) / 'chunk_00000.jsonl'
        packs = [
            {
                'input_ids': [1, 2, 3, 4, 5],
                'labels': [-100, 2, 3, 4, 5],
                'position_ids': [0, 1, 2, 0, 1],
                'lengths': [3, 2],
                'pack_length': 5,
                'num_samples': 2,
            },
            {
                'input_ids': [10, 20, 30],
                'labels': [10, 20, 30],
                'position_ids': [0, 1, 2],
                'lengths': [3],
                'pack_length': 3,
                'num_samples': 1,
            },
        ]
        with open(chunk_path, 'w') as f:
            for pack in packs:
                f.write(json.dumps(pack) + '\n')

        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=2,
            pre_packed=True,
            dp_rank=0,
            dp_world_size=1,
            max_wait_time=1.0,
        )

        # Access samples
        sample0 = dataset[0]
        self.assertEqual(sample0['input_ids'], [1, 2, 3, 4, 5])
        self.assertEqual(sample0['length'], 5)  # Added from pack_length

        sample1 = dataset[1]
        self.assertEqual(sample1['input_ids'], [10, 20, 30])

    def test_prepacked_adds_length_field(self):
        """Pre-packed should add length from pack_length if missing."""
        chunk_path = Path(self.tmp_dir) / 'chunk_00000.jsonl'
        pack = {
            'input_ids': [1, 2, 3],
            'labels': [1, 2, 3],
            'pack_length': 3,
            # No 'length' field
        }
        with open(chunk_path, 'w') as f:
            f.write(json.dumps(pack) + '\n')

        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=1,
            pre_packed=True,
            dp_rank=0,
            dp_world_size=1,
            max_wait_time=1.0,
        )

        sample = dataset[0]
        self.assertEqual(sample['length'], 3)  # Should be added


class TestLazyShardedDatasetRawFormat(unittest.TestCase):
    """Tests for raw format chunk loading with encoding."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_raw_chunk_with_encode(self):
        """Load raw chunk and encode samples."""
        chunk_path = Path(self.tmp_dir) / 'chunk_00000.jsonl'
        raw_samples = [
            {'messages': [{'role': 'user', 'content': 'hello'}]},
            {'messages': [{'role': 'user', 'content': 'world'}]},
        ]
        with open(chunk_path, 'w') as f:
            for sample in raw_samples:
                f.write(json.dumps(sample) + '\n')

        # Mock encode function
        def mock_encode(sample, return_length=False):
            content = sample['messages'][0]['content']
            tokens = [ord(c) for c in content]
            return {
                'input_ids': tokens,
                'labels': tokens,
                'length': len(tokens),
            }

        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
            encode_fn=mock_encode,
            samples_per_chunk=2,
            pre_packed=False,
            dp_rank=0,
            dp_world_size=1,
            max_wait_time=1.0,
        )

        sample0 = dataset[0]
        self.assertEqual(sample0['input_ids'], [ord(c) for c in 'hello'])
        self.assertEqual(sample0['length'], 5)

        sample1 = dataset[1]
        self.assertEqual(sample1['input_ids'], [ord(c) for c in 'world'])

    def test_encode_fn_required_when_not_prepacked(self):
        """Should raise error if encode_fn missing for raw format."""
        with self.assertRaises(ValueError):
            LazyShardedDataset(
                directory=self.tmp_dir,
                encode_fn=None,  # Missing!
                pre_packed=False,
            )


class TestLazyShardedDatasetCompletionMarkers(unittest.TestCase):
    """Tests for completion marker behavior."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_completion_marker_created(self):
        """Completion marker should be created after loading chunk."""
        chunk_path = Path(self.tmp_dir) / 'chunk_00000.jsonl'
        with open(chunk_path, 'w') as f:
            f.write(json.dumps({'input_ids': [1], 'pack_length': 1}) + '\n')

        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=1,
            pre_packed=True,
            mark_completed=True,
            dp_rank=0,
            dp_world_size=1,
            max_wait_time=1.0,
        )

        # Marker shouldn't exist yet
        marker_path = chunk_path.with_suffix('.jsonl.completed')
        self.assertFalse(marker_path.exists())

        # Load sample (triggers chunk load)
        _ = dataset[0]

        # Marker should exist now
        self.assertTrue(marker_path.exists())

    def test_completion_marker_disabled(self):
        """Completion marker should not be created when disabled."""
        chunk_path = Path(self.tmp_dir) / 'chunk_00000.jsonl'
        with open(chunk_path, 'w') as f:
            f.write(json.dumps({'input_ids': [1], 'pack_length': 1}) + '\n')

        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=1,
            pre_packed=True,
            mark_completed=False,  # Disabled
            dp_rank=0,
            dp_world_size=1,
            max_wait_time=1.0,
        )

        _ = dataset[0]

        marker_path = chunk_path.with_suffix('.jsonl.completed')
        self.assertFalse(marker_path.exists())


class TestLazyShardedDatasetWaiting(unittest.TestCase):
    """Tests for chunk waiting behavior."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_timeout_on_missing_chunk(self):
        """Should timeout and raise error for missing chunk."""
        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=1,
            pre_packed=True,
            max_wait_time=0.2,  # Short timeout
            wait_poll_interval=0.05,
            dp_rank=0,
            dp_world_size=1,
        )

        with self.assertRaises(IndexError):
            _ = dataset[0]  # Chunk doesn't exist

    def test_waits_for_chunk_to_appear(self):
        """Should wait for chunk to appear within timeout."""
        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
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
            chunk_path = Path(self.tmp_dir) / 'chunk_00000.jsonl'
            with open(chunk_path, 'w') as f:
                f.write(json.dumps({'input_ids': [42], 'pack_length': 1}) + '\n')

        writer_thread = threading.Thread(target=write_chunk_delayed)
        writer_thread.start()

        # This should wait and then succeed
        sample = dataset[0]
        self.assertEqual(sample['input_ids'], [42])

        writer_thread.join()


class TestLazyShardedDatasetCaching(unittest.TestCase):
    """Tests for chunk caching behavior."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_chunk_cached_after_load(self):
        """Chunk should be cached after first load."""
        chunk_path = Path(self.tmp_dir) / 'chunk_00000.jsonl'
        with open(chunk_path, 'w') as f:
            f.write(json.dumps({'input_ids': [1], 'pack_length': 1}) + '\n')
            f.write(json.dumps({'input_ids': [2], 'pack_length': 1}) + '\n')

        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=2,
            pre_packed=True,
            dp_rank=0,
            dp_world_size=1,
            max_wait_time=1.0,
        )

        self.assertEqual(len(dataset._chunk_cache), 0)

        _ = dataset[0]  # Load first sample
        self.assertIn(0, dataset._chunk_cache)
        self.assertEqual(len(dataset._chunk_cache[0]), 2)

        _ = dataset[1]  # Should use cache
        self.assertEqual(len(dataset._chunk_cache), 1)  # Still just one chunk cached


class TestEndToEndFlow(unittest.TestCase):
    """Tests for complete producer-to-dataset flow."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_packer_to_dataset_flow(self):
        """Test packing samples and reading via dataset."""
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
        chunk0_path = Path(self.tmp_dir) / 'chunk_00000.jsonl'
        with open(chunk0_path, 'w') as f:
            f.write(json.dumps(pack1) + '\n')
            f.write(json.dumps(pack2) + '\n')

        # Step 2: Read via dataset
        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=2,
            pre_packed=True,
            dp_rank=0,
            dp_world_size=1,
            max_wait_time=1.0,
        )

        sample0 = dataset[0]
        self.assertEqual(sample0['input_ids'], [1, 2, 3, 4, 5])
        self.assertEqual(sample0['position_ids'], [0, 1, 2, 0, 1])
        self.assertEqual(sample0['lengths'], [3, 2])

        sample1 = dataset[1]
        self.assertEqual(sample1['input_ids'], [10, 20, 30, 40])

    def test_multi_rank_distribution(self):
        """Test that multiple ranks read different chunks."""
        # Create 4 chunks
        for i in range(4):
            chunk_path = Path(self.tmp_dir) / f'chunk_{i:05d}.jsonl'
            pack = {
                'input_ids': [i * 100 + j for j in range(5)],
                'labels': [i * 100 + j for j in range(5)],
                'pack_length': 5,
                'chunk_id': i,  # For verification
            }
            with open(chunk_path, 'w') as f:
                f.write(json.dumps(pack) + '\n')

        # Create datasets for 2 ranks
        dataset_rank0 = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=1,
            pre_packed=True,
            dp_rank=0,
            dp_world_size=2,
            max_wait_time=1.0,
        )
        dataset_rank1 = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=1,
            pre_packed=True,
            dp_rank=1,
            dp_world_size=2,
            max_wait_time=1.0,
        )

        # Rank 0 should get chunks 0, 2
        sample_r0_0 = dataset_rank0[0]  # -> chunk 0
        self.assertEqual(sample_r0_0['chunk_id'], 0)

        sample_r0_1 = dataset_rank0[1]  # -> chunk 2
        self.assertEqual(sample_r0_1['chunk_id'], 2)

        # Rank 1 should get chunks 1, 3
        sample_r1_0 = dataset_rank1[0]  # -> chunk 1
        self.assertEqual(sample_r1_0['chunk_id'], 1)

        sample_r1_1 = dataset_rank1[1]  # -> chunk 3
        self.assertEqual(sample_r1_1['chunk_id'], 3)


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling scenarios."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_corrupt_json_skipped(self):
        """Corrupt JSON lines should be skipped gracefully."""
        chunk_path = Path(self.tmp_dir) / 'chunk_00000.jsonl'
        with open(chunk_path, 'w') as f:
            f.write(json.dumps({'input_ids': [1], 'pack_length': 1}) + '\n')
            f.write('not valid json\n')  # Corrupt line
            f.write(json.dumps({'input_ids': [2], 'pack_length': 1}) + '\n')

        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=3,
            pre_packed=True,
            dp_rank=0,
            dp_world_size=1,
            max_wait_time=1.0,
        )

        # Should load 2 valid samples, skip corrupt one
        samples = dataset._load_chunk(0)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]['input_ids'], [1])
        self.assertEqual(samples[1]['input_ids'], [2])

    def test_empty_chunk_file(self):
        """Empty chunk file should return empty samples."""
        chunk_path = Path(self.tmp_dir) / 'chunk_00000.jsonl'
        chunk_path.touch()  # Empty file

        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
            samples_per_chunk=1,
            pre_packed=True,
            dp_rank=0,
            dp_world_size=1,
            max_wait_time=1.0,
        )

        samples = dataset._load_chunk(0)
        self.assertEqual(samples, [])

    def test_encode_fn_returns_none(self):
        """Samples where encode_fn returns None should be skipped."""
        chunk_path = Path(self.tmp_dir) / 'chunk_00000.jsonl'
        with open(chunk_path, 'w') as f:
            f.write(json.dumps({'valid': True}) + '\n')
            f.write(json.dumps({'valid': False}) + '\n')
            f.write(json.dumps({'valid': True}) + '\n')

        def selective_encode(sample, return_length=False):
            if sample.get('valid'):
                return {'input_ids': [1], 'length': 1}
            return None  # Skip invalid

        dataset = LazyShardedDataset(
            directory=self.tmp_dir,
            encode_fn=selective_encode,
            samples_per_chunk=3,
            pre_packed=False,
            dp_rank=0,
            dp_world_size=1,
            max_wait_time=1.0,
        )

        samples = dataset._load_chunk(0)
        self.assertEqual(len(samples), 2)  # Only valid ones


if __name__ == '__main__':
    unittest.main()
