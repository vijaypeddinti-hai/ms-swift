import tempfile
import unittest
from pathlib import Path

import json

from swift.llm import load_dataset
from swift.llm.dataset import DynamicDirectoryDataset


class TestDataset(unittest.TestCase):

    def test_load_v_dataset(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return

        for ds in ['m3it#1000', 'mantis-instruct#1000', 'llava-med-zh-instruct#1000']:
            ds = load_dataset(ds)
            assert len(ds[0]) > 800


class TestDynamicDirectoryDataset(unittest.TestCase):
    """Tests for DynamicDirectoryDataset with rescan_files support."""

    def test_basic_iteration(self):
        """Test basic reading from JSONL files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create test file
            with open(tmpdir / 'chunk_001.jsonl', 'w') as f:
                for i in range(3):
                    f.write(json.dumps({'id': i, 'text': f'sample {i}'}) + '\n')

            ds = DynamicDirectoryDataset(
                directory=tmpdir,
                shuffle=False,
                wait_for_files=False,
                min_file_age_seconds=0,
            )
            samples = list(ds)
            self.assertEqual(len(samples), 3)
            self.assertEqual(samples[0]['id'], 0)

    def test_rescan_picks_up_new_files(self):
        """Test that new files are detected on subsequent iterations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create first file
            with open(tmpdir / 'chunk_001.jsonl', 'w') as f:
                for i in range(3):
                    f.write(json.dumps({'id': f'file1_{i}'}) + '\n')

            ds = DynamicDirectoryDataset(
                directory=tmpdir,
                shuffle=False,
                wait_for_files=False,
                min_file_age_seconds=0,
                mark_completed=False,  # Disable markers for this test
            )

            # First iteration
            samples1 = list(ds)
            self.assertEqual(len(samples1), 3)

            # Add new file
            with open(tmpdir / 'chunk_002.jsonl', 'w') as f:
                for i in range(2):
                    f.write(json.dumps({'id': f'file2_{i}'}) + '\n')

            # Second iteration should see all files
            samples2 = list(ds)
            self.assertEqual(len(samples2), 5)  # 3 + 2

    def test_completed_markers(self):
        """Test that .completed markers prevent re-reading files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create two files
            for i in range(2):
                with open(tmpdir / f'chunk_{i:03d}.jsonl', 'w') as f:
                    for j in range(3):
                        f.write(json.dumps({'id': f'file{i}_{j}'}) + '\n')

            ds = DynamicDirectoryDataset(
                directory=tmpdir,
                shuffle=False,
                wait_for_files=False,
                min_file_age_seconds=0,
                mark_completed=True,  # Enable markers
            )

            # First iteration - read all 6 samples
            samples1 = list(ds)
            self.assertEqual(len(samples1), 6)

            # Check markers were created
            markers = list(tmpdir.glob('*.completed'))
            self.assertEqual(len(markers), 2)

            # Add new file
            with open(tmpdir / 'chunk_002.jsonl', 'w') as f:
                for j in range(2):
                    f.write(json.dumps({'id': f'file2_{j}'}) + '\n')

            # Second iteration should only see new file
            samples2 = list(ds)
            self.assertEqual(len(samples2), 2)  # Only new file

    def test_shuffle_buffer(self):
        """Test that shuffle buffer works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with open(tmpdir / 'chunk_001.jsonl', 'w') as f:
                for i in range(100):
                    f.write(json.dumps({'id': i}) + '\n')

            ds = DynamicDirectoryDataset(
                directory=tmpdir,
                shuffle=True,
                shuffle_buffer_size=10,
                wait_for_files=False,
                min_file_age_seconds=0,
                seed=42,
            )

            samples = list(ds)
            self.assertEqual(len(samples), 100)
            # Check that order is shuffled (not sequential)
            ids = [s['id'] for s in samples]
            self.assertNotEqual(ids, list(range(100)))


if __name__ == '__main__':
    unittest.main()
