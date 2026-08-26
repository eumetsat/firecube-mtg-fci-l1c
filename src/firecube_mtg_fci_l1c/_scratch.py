# Copyright 2025-2026 EUMETSAT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-batch scratch workspace for temporary ZIP extraction.

``BatchScratch`` creates one temp root per batch, extracts each ZIP into a
numbered subdirectory, and removes the root on context exit. Extraction rejects
archive members that would escape the destination directory.
"""

from __future__ import annotations

import atexit
import threading
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

from firecube.core.api import extract_all_from_zips


_pending_cleanup_threads: list[threading.Thread] = []
_pending_lock = threading.Lock()


def register_cleanup_thread(thread: threading.Thread) -> None:
    with _pending_lock:
        _pending_cleanup_threads.append(thread)


def _await_pending_cleanups() -> None:
    with _pending_lock:
        threads = list(_pending_cleanup_threads)
    for thread in threads:
        thread.join(timeout=30)


atexit.register(_await_pending_cleanups)


class BatchScratch:
    """Per-batch scratch directory with numbered ZIP extract dirs."""

    def __init__(
        self,
        base_dir: str | None,
        scratch_id: str,
        *,
        prefix: str = "firecube",
    ) -> None:
        if base_dir is not None:
            Path(base_dir).mkdir(parents=True, exist_ok=True)
        self._tmp = tempfile.TemporaryDirectory(
            dir=base_dir,
            prefix=f"{prefix}_{scratch_id}_",
        )
        self._scratch_root = Path(self._tmp.name)
        # Serial by contract: extract_all_from_zips resolves destination
        # directories on the calling thread before any extraction starts.
        self._extract_counter = 0

    @property
    def scratch_root(self) -> Path:
        """Path to this batch's scratch root directory."""
        return self._scratch_root

    def _next_extract_dir(self, zip_path: Path) -> Path:
        """Allocate the next numbered extraction directory for *zip_path*."""
        self._extract_counter += 1
        return self._scratch_root / f"{Path(zip_path).stem}-{self._extract_counter}"

    def extract_zip(self, zip_path: Path) -> Path:
        """Extract *zip_path* into a numbered subdirectory and return its path.

        Raises ``ValueError`` when the archive is invalid or contains an
        unsafe member name.
        """
        path = Path(zip_path)
        extracted, failures = extract_all_from_zips([path], self._next_extract_dir)
        if failures:
            raise ValueError(failures[path])
        return extracted[path]

    def extract_zips_parallel(
        self,
        zip_paths: Sequence[Path],
        *,
        max_workers: int = 4,
    ) -> tuple[dict[Path, Path], dict[Path, str]]:
        """Extract several ZIPs concurrently into numbered subdirectories.

        Thin wrapper over :func:`firecube.core.api.extract_all_from_zips`
        supplying this batch's numbered directories; see that function for
        the full contract (zip-slip guard, per-archive failure isolation).

        Args:
            zip_paths: Archives to extract.
            max_workers: Upper bound on concurrent extractions.

        Returns:
            The core function's ``(extracted, failures)`` pair; every input
            path appears in exactly one of the two mappings.
        """
        return extract_all_from_zips(
            [Path(p) for p in zip_paths],
            self._next_extract_dir,
            workers=max_workers,
        )

    def cleanup(self) -> None:
        """Remove the scratch root and all contents; safe to call repeatedly."""
        self._tmp.cleanup()

    def __enter__(self) -> BatchScratch:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.cleanup()


__all__ = ["BatchScratch"]
