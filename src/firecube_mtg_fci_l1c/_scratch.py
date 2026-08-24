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
import zipfile
from pathlib import Path
from types import TracebackType


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


def _safe_extractall(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract *zf* into *dest*, rejecting any member that escapes *dest*."""
    dest_resolved = dest.resolve()
    for member in zf.namelist():
        target = (dest_resolved / member).resolve()
        if target != dest_resolved and dest_resolved not in target.parents:
            raise ValueError(f"Unsafe path in archive (zip-slip): {member!r}")
    zf.extractall(dest_resolved)


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
        self._extract_counter = 0

    @property
    def scratch_root(self) -> Path:
        """Path to this batch's scratch root directory."""
        return self._scratch_root

    def extract_zip(self, zip_path: Path) -> Path:
        """Extract *zip_path* into a numbered subdirectory and return its path."""
        self._extract_counter += 1
        extract_dir = (
            self._scratch_root / f"{Path(zip_path).stem}-{self._extract_counter}"
        )
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extractall(zf, extract_dir)
        return extract_dir

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
