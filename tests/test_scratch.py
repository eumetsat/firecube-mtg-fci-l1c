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


def test_scratch_id_in_root_name(tmp_path: Path):
    with BatchScratch(str(tmp_path / "scratch"), "run-batch_0007") as scratch:
        assert "batch_0007" in scratch.scratch_root.name


def test_base_dir_created_if_missing(tmp_path: Path):
    missing = tmp_path / "does" / "not" / "exist"
    with BatchScratch(str(missing), "run-batch_0000") as scratch:
        assert scratch.scratch_root.exists()
        assert str(scratch.scratch_root).startswith(str(missing))


def test_zip_slip_member_rejected(tmp_path: Path):
    # A member that escapes the extract dir must be refused, not written outside.
    evil = _make_zip(tmp_path / "evil.zip", {"../escape.nc": b"pwned"})
    with BatchScratch(str(tmp_path / "scratch"), "run-batch_0000") as scratch:
        with pytest.raises(ValueError, match="zip-slip"):
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
