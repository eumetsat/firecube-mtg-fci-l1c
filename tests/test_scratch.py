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

"""Unit tests for the plugin-local ``BatchScratch`` ZIP extraction helper."""

from __future__ import annotations

import threading
import time
import zipfile
from pathlib import Path

import h5netcdf  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import pytest

from firecube_mtg_fci_l1c._scratch import BatchScratch
from firecube_mtg_fci_l1c._decode import (  # pyright: ignore[reportMissingImports]
    SharedNcPartReader,
)

pytestmark = pytest.mark.unit


def _make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _write_minimal_nc_part(path: Path) -> None:
    with h5netcdf.File(path, "w") as ds:
        data_group = ds.create_group("data")
        measured = data_group.create_group("vis_04").create_group("measured")
        measured.dimensions["y"] = 2
        measured.dimensions["x"] = 3
        radiance = measured.create_variable(
            "effective_radiance",
            ("y", "x"),
            data=np.full((2, 3), 7, dtype=np.uint16),
        )
        radiance.attrs["start_position_row"] = 1
        radiance.attrs["end_position_row"] = 2
        radiance.attrs["scale_factor"] = 0.5
        radiance.attrs["add_offset"] = 1.5


def test_extract_zip_returns_numbered_dirs_and_contents(tmp_path: Path):
    zip_path = _make_zip(tmp_path / "a.zip", {"body/part.nc": b"hello"})
    with BatchScratch(str(tmp_path / "scratch"), "run-batch_0000") as scratch:
        d1 = scratch.extract_zip(zip_path)
        d2 = scratch.extract_zip(zip_path)
        assert (d1 / "body" / "part.nc").read_bytes() == b"hello"
        # Distinct numbered subdirs under one batch root.
        assert d1 != d2
        assert d1.parent == scratch.scratch_root
        assert d2.parent == scratch.scratch_root


def test_cleanup_on_context_exit(tmp_path: Path):
    zip_path = _make_zip(tmp_path / "a.zip", {"x.nc": b"x"})
    with BatchScratch(str(tmp_path / "scratch"), "run-batch_0000") as scratch:
        root = scratch.scratch_root
        scratch.extract_zip(zip_path)
        assert root.exists()
    assert not root.exists()  # removed on exit


def test_close_hands_removal_to_a_daemon_thread(tmp_path: Path):
    scratch = BatchScratch(str(tmp_path / "scratch"), "run-batch_0000")
    root = scratch.scratch_root
    assert root.exists()

    scratch.close()

    deadline = time.monotonic() + 5
    while root.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not root.exists()


def test_scratch_id_in_root_name(tmp_path: Path):
    with BatchScratch(str(tmp_path / "scratch"), "run-batch_0007") as scratch:
        assert "batch_0007" in scratch.scratch_root.name


def test_base_dir_created_if_missing(tmp_path: Path):
    missing = tmp_path / "does" / "not" / "exist"
    with BatchScratch(str(missing), "run-batch_0000") as scratch:
        assert scratch.scratch_root.exists()
        assert str(scratch.scratch_root).startswith(str(missing))


def test_extract_zips_parallel_maps_every_archive(tmp_path: Path):
    # >2 archives exercises the thread-pool path; each input must land in
    # exactly one of the two result mappings, in a distinct directory.
    zips = [
        _make_zip(tmp_path / f"cycle-{i}.zip", {f"part-{i}.nc": bytes([i]) * 8})
        for i in range(5)
    ]
    with BatchScratch(str(tmp_path / "scratch"), "run-batch_0000") as scratch:
        extracted, failures = scratch.extract_zips_parallel(zips)

        assert failures == {}
        assert sorted(extracted) == sorted(zips)
        dirs = list(extracted.values())
        assert len(set(dirs)) == len(dirs), "extraction dirs must be unique"
        for zip_path, extract_dir in extracted.items():
            member = extract_dir / f"part-{zips.index(zip_path)}.nc"
            assert member.is_file()


def test_extract_zips_parallel_isolates_failures(tmp_path: Path):
    # A corrupt archive and a zip-slip archive fail alone; good ones extract.
    good = [_make_zip(tmp_path / f"good-{i}.zip", {"a.nc": b"ok"}) for i in range(2)]
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"this is not a zip archive")
    evil = _make_zip(tmp_path / "evil.zip", {"../escape.nc": b"pwned"})

    with BatchScratch(str(tmp_path / "scratch"), "run-batch_0000") as scratch:
        extracted, failures = scratch.extract_zips_parallel([*good, corrupt, evil])

        assert sorted(extracted) == sorted(good)
        assert set(failures) == {corrupt, evil}
        assert "Unsafe ZIP member path" in failures[evil]
    assert not (tmp_path / "escape.nc").exists()


def test_extract_zips_parallel_serial_small_batches(tmp_path: Path):
    # Small batches yield the same result semantics as large ones.
    one = _make_zip(tmp_path / "one.zip", {"a.nc": b"x"})
    corrupt = tmp_path / "bad.zip"
    corrupt.write_bytes(b"nope")
    with BatchScratch(str(tmp_path / "scratch"), "run-batch_0000") as scratch:
        extracted, failures = scratch.extract_zips_parallel([one, corrupt])

        assert list(extracted) == [one]
        assert (extracted[one] / "a.nc").read_bytes() == b"x"
        assert list(failures) == [corrupt]


def test_zip_slip_member_rejected(tmp_path: Path):
    # A member that escapes the extract dir must be refused, not written outside.
    evil = _make_zip(tmp_path / "evil.zip", {"../escape.nc": b"pwned"})
    with BatchScratch(str(tmp_path / "scratch"), "run-batch_0000") as scratch:
        with pytest.raises(ValueError, match="Unsafe ZIP member path"):
            scratch.extract_zip(evil)
    assert not (tmp_path / "escape.nc").exists()


def test_shared_reader_closes_when_exception_raised_mid_batch(tmp_path: Path):
    # Mirrors the ingestor's nested lifecycle so a mid-batch failure cannot
    # leak nc_part file handles or the scratch root.
    part = tmp_path / "body.nc"
    _write_minimal_nc_part(part)

    shared = SharedNcPartReader()

    with pytest.raises(RuntimeError, match="mid-batch failure"):
        with BatchScratch(str(tmp_path / "scratch"), "run-batch_0000") as scratch:
            root = scratch.scratch_root
            with shared:
                shared.decode_channel(part, "vis_04")
                cached_reader = shared._readers[Path(part)]
                assert cached_reader._ds is not None
                raise RuntimeError("mid-batch failure")

    assert cached_reader._ds is None
    assert shared._readers == {}
    assert not root.exists()


def test_register_cleanup_thread_prunes_finished_threads():
    # One thread is registered per batch; without pruning the registry would
    # grow by one entry for every batch the process ever handles.
    from firecube_mtg_fci_l1c import _scratch

    with _scratch._pending_lock:
        _scratch._pending_cleanup_threads.clear()

    for _ in range(5):
        thread = threading.Thread(target=lambda: None)
        thread.start()
        thread.join()  # finished before the next registration prunes it
        _scratch.register_cleanup_thread(thread)

    with _scratch._pending_lock:
        pending = list(_scratch._pending_cleanup_threads)

    # Only the most recent registration survives; the four completed threads
    # ahead of it were pruned rather than accumulating.
    assert len(pending) == 1

    _scratch._await_pending_cleanups()
    with _scratch._pending_lock:
        assert _scratch._pending_cleanup_threads == []


def test_register_cleanup_thread_keeps_running_threads():
    # Pruning must not drop a thread that is still doing work.
    from firecube_mtg_fci_l1c import _scratch

    with _scratch._pending_lock:
        _scratch._pending_cleanup_threads.clear()

    release = threading.Event()
    slow = threading.Thread(target=release.wait)
    slow.start()
    _scratch.register_cleanup_thread(slow)

    finished = threading.Thread(target=lambda: None)
    finished.start()
    finished.join()
    _scratch.register_cleanup_thread(finished)

    try:
        with _scratch._pending_lock:
            pending = list(_scratch._pending_cleanup_threads)
        assert slow in pending
    finally:
        release.set()
        slow.join(timeout=5)

    _scratch._await_pending_cleanups()
