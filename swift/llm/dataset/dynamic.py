"""
Dynamic Directory Streaming Dataset

This module provides an IterableDataset implementation that re-scans a directory
for new files on each iteration cycle. This is essential for streaming training
scenarios where a producer process continuously writes new data files while
training consumes them.

Use Case:
    - Producer thread downloads data from remote storage (S3, R2, etc.)
    - Converts to JSONL and writes to a local buffer directory
    - Training reads from the buffer using --streaming true --rescan_files true
    - As new files appear, training automatically picks them up

The standard HuggingFace load_dataset(..., streaming=True) snapshots the file
list at construction time and never re-scans. This class solves that problem.

Example:
    >>> from swift.llm.dataset.dynamic import DynamicDirectoryDataset
    >>> dataset = DynamicDirectoryDataset(
    ...     directory="/workspace/data/buffer",
    ...     file_pattern="*.jsonl",
    ...     shuffle=True,
    ... )
    >>> for sample in dataset:
    ...     # Will see new files as they're added
    ...     print(sample)
"""
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Union

from torch.utils.data import IterableDataset

from swift.utils import get_logger

logger = get_logger()


class DynamicDirectoryDataset(IterableDataset):
    """An IterableDataset that re-scans a directory for files on each iteration cycle.

    Unlike HuggingFace's streaming datasets which snapshot file lists at construction,
    this dataset re-scans the directory each time iteration restarts. This enables
    true streaming scenarios where files are continuously added to the directory.

    Args:
        directory: Path to the directory containing data files.
        file_pattern: Glob pattern for matching files (default: "*.jsonl").
        file_type: Type of files to read. Supported: "jsonl", "json", "parquet".
        shuffle: Whether to shuffle files and samples within each cycle.
        shuffle_buffer_size: Size of the shuffle buffer for samples (default: 1000).
        wait_for_files: If True, wait indefinitely when no files found.
            If False, raise StopIteration when directory is empty.
        wait_poll_interval: Seconds to wait between polls when wait_for_files=True.
        min_file_age_seconds: Only read files older than this (avoids partial writes).
        transform: Optional function to transform each sample dict.
        seed: Random seed for shuffling.
        mark_completed: If True, create a .completed marker file after reading each
            data file. Files with markers are skipped on subsequent scans. This enables
            producer-consumer coordination: producer can safely delete files that have
            .completed markers. Default: True.

    Example:
        >>> dataset = DynamicDirectoryDataset(
        ...     directory="/data/buffer",
        ...     file_pattern="*.jsonl",
        ...     shuffle=True,
        ... )
        >>> for sample in dataset:
        ...     process(sample)
    """

    def __init__(
        self,
        directory: Union[str, Path],
        file_pattern: str = "*.jsonl",
        file_type: Literal["jsonl", "json", "parquet"] = "jsonl",
        shuffle: bool = True,
        shuffle_buffer_size: int = 1000,
        wait_for_files: bool = True,
        wait_poll_interval: float = 5.0,
        min_file_age_seconds: float = 1.0,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        mark_completed: bool = True,
    ):
        self.directory = Path(directory)
        self.file_pattern = file_pattern
        self.file_type = file_type
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.wait_for_files = wait_for_files
        self.wait_poll_interval = wait_poll_interval
        self.min_file_age_seconds = min_file_age_seconds
        self.transform = transform
        self.seed = seed
        self.mark_completed = mark_completed

        self._rng = random.Random(seed)
        self._cycle = 0

    def _get_completed_marker_path(self, data_file: Path) -> Path:
        """Get the path to the .completed marker file for a data file."""
        return data_file.with_suffix(data_file.suffix + '.completed')

    def _is_completed(self, data_file: Path) -> bool:
        """Check if a data file has been marked as completed."""
        return self._get_completed_marker_path(data_file).exists()

    def _mark_as_completed(self, data_file: Path) -> None:
        """Create a .completed marker file for a data file."""
        marker_path = self._get_completed_marker_path(data_file)
        try:
            marker_path.touch()
            logger.debug(f"Marked as completed: {data_file.name}")
        except (OSError, IOError) as e:
            logger.warning(f"Failed to create completion marker for {data_file}: {e}")

    def _scan_directory(self) -> List[Path]:
        """Scan directory for matching files, skipping completed ones."""
        now = time.time()
        files = []
        if not self.directory.exists():
            return files
        for path in self.directory.glob(self.file_pattern):
            if path.is_file():
                # Skip files that have been marked as completed
                if self.mark_completed and self._is_completed(path):
                    continue
                # Skip files that are too new (might still be written)
                try:
                    file_age = now - path.stat().st_mtime
                    if file_age >= self.min_file_age_seconds:
                        files.append(path)
                except (OSError, IOError):
                    # File may have been deleted
                    continue
        return sorted(files, key=lambda p: p.stat().st_mtime)

    def _wait_for_files(self) -> List[Path]:
        """Wait until files are available in the directory."""
        while True:
            files = self._scan_directory()
            if files:
                return files
            if not self.wait_for_files:
                return []
            logger.info(
                f"[DynamicDirectoryDataset] No files in {self.directory}, "
                f"waiting {self.wait_poll_interval}s..."
            )
            time.sleep(self.wait_poll_interval)

    def _read_jsonl_file(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Read samples from a JSONL file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at {path}:{line_num}: {e}")
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")

    def _read_json_file(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Read samples from a JSON file (expects list of dicts or single dict)."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                yield from data
            elif isinstance(data, dict):
                yield data
            else:
                logger.warning(f"Unexpected JSON structure in {path}")
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")

    def _read_parquet_file(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Read samples from a Parquet file."""
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            for batch in table.to_batches():
                for row in batch.to_pylist():
                    yield row
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")

    def _read_file(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Read samples from a file based on file_type."""
        if self.file_type == "jsonl":
            yield from self._read_jsonl_file(path)
        elif self.file_type == "json":
            yield from self._read_json_file(path)
        elif self.file_type == "parquet":
            yield from self._read_parquet_file(path)
        else:
            raise ValueError(f"Unsupported file_type: {self.file_type}")

    def _shuffle_buffer_iter(
        self,
        sample_iter: Iterator[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Apply shuffle buffer to sample iterator."""
        buffer: List[Dict[str, Any]] = []

        for sample in sample_iter:
            buffer.append(sample)
            if len(buffer) >= self.shuffle_buffer_size:
                idx = self._rng.randint(0, len(buffer) - 1)
                yield buffer.pop(idx)

        # Yield remaining samples in random order
        self._rng.shuffle(buffer)
        yield from buffer

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples, re-scanning directory on each cycle.

        This is the key method that differs from standard HuggingFace datasets.
        Each time this is called (i.e., when the DataLoader starts a new epoch
        or when iteration restarts), it re-scans the directory for files.
        """
        self._cycle += 1
        cycle = self._cycle

        # Re-scan directory for files
        files = self._wait_for_files()
        if not files:
            logger.info("[DynamicDirectoryDataset] No files found, stopping iteration")
            return

        logger.info(
            f"[DynamicDirectoryDataset] Cycle {cycle}: Found {len(files)} files"
        )

        # Shuffle file order if requested
        if self.shuffle:
            files = list(files)
            self._rng.shuffle(files)

        # Create sample iterator over all files
        def sample_generator():
            for file_path in files:
                sample_count_in_file = 0
                for sample in self._read_file(file_path):
                    if self.transform:
                        sample = self.transform(sample)
                        if sample is None:
                            continue
                    sample_count_in_file += 1
                    yield sample
                # Mark file as completed after reading all samples
                if self.mark_completed and sample_count_in_file > 0:
                    self._mark_as_completed(file_path)
                    logger.info(f"[DynamicDirectoryDataset] Completed: {file_path.name} ({sample_count_in_file} samples)")

        # Apply shuffle buffer if requested
        if self.shuffle:
            sample_iter = self._shuffle_buffer_iter(sample_generator())
        else:
            sample_iter = sample_generator()

        # Yield all samples
        sample_count = 0
        for sample in sample_iter:
            sample_count += 1
            yield sample

        logger.info(
            f"[DynamicDirectoryDataset] Cycle {cycle} complete: "
            f"yielded {sample_count} samples from {len(files)} files"
        )


def load_dynamic_directory_dataset(
    directory: Union[str, Path],
    file_pattern: str = "*.jsonl",
    file_type: Literal["jsonl", "json", "parquet"] = "jsonl",
    shuffle: bool = True,
    shuffle_buffer_size: int = 1000,
    wait_for_files: bool = True,
    seed: Optional[int] = None,
    mark_completed: bool = True,
    **kwargs,
) -> DynamicDirectoryDataset:
    """Load a dynamic streaming dataset from a directory.

    This is the main entry point for creating a dynamic streaming dataset
    that re-scans the directory for new files on each iteration cycle.

    Args:
        directory: Path to directory containing data files.
        file_pattern: Glob pattern for files (default: "*.jsonl").
        file_type: Type of files ("jsonl", "json", "parquet").
        shuffle: Whether to shuffle files and samples.
        shuffle_buffer_size: Size of shuffle buffer.
        wait_for_files: Wait for files if directory is empty.
        seed: Random seed for shuffling.
        mark_completed: Create .completed marker after reading each file.
            Producer can safely delete files with markers. Default: True.
        **kwargs: Additional arguments passed to DynamicDirectoryDataset.

    Returns:
        DynamicDirectoryDataset instance.

    Example:
        >>> dataset = load_dynamic_directory_dataset(
        ...     "/workspace/data/buffer",
        ...     file_pattern="*.jsonl",
        ...     shuffle=True,
        ... )
    """
    return DynamicDirectoryDataset(
        directory=directory,
        file_pattern=file_pattern,
        file_type=file_type,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        wait_for_files=wait_for_files,
        seed=seed,
        mark_completed=mark_completed,
        **kwargs,
    )
