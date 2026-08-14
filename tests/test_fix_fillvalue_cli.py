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

"""Tests for the ``fix-fillvalue`` plugin CLI subcommand.

The command is an offline post-ingestion workaround for GitHub issue #2:
Zarr 3 does not stamp a user ``_FillValue`` attribute when arrays are
created with ``fill_value=<x>``, which breaks xarray tooling that reads
``_FillValue`` via ``mask_and_scale=True``. The ``fix-fillvalue`` CLI walks
a preallocated store and stamps ``_FillValue`` on the variables that
declare a non-None fill in the plugin schema.

These tests are the RED phase of TDD: they will fail until the CLI
subcommand is implemented in :mod:`firecube_mtg_fci_l1c.plugin_cli`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from click.testing import CliRunner

from firecube_mtg_fci_l1c.plugin_cli import cli


def _build_synthetic_store(
    tmp_path: Path,
    *,
    with_counts: bool = True,
    with_x: bool = True,
    with_time: bool = False,
    fillvalue_on_counts: int | None = None,
) -> Path:
    """Build a minimal on-disk Zarr store shaped like a preallocated FCI cube.

    ``counts`` is created with ``fill_value=65535`` but *without* the
    ``_FillValue`` user attribute, reproducing the issue #2 symptom.
    Optional flags let each test build only the arrays it cares about.
    """
    store_path = tmp_path / "store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    grp = root.require_group("data_1km")
    if with_counts:
        arr = grp.create_array(
            "counts",
            shape=(1, 10, 10, 1),
            dtype=np.uint16,
            fill_value=65535,
        )
        if fillvalue_on_counts is not None:
            arr.attrs.update({"_FillValue": fillvalue_on_counts})
    if with_x:
        grp.create_array("x", shape=(10,), dtype=np.float64, fill_value=None)
    if with_time:
        grp.create_array("time", shape=(1,), dtype="datetime64[s]")
    return store_path


@pytest.mark.unit
def test_fix_fillvalue_dry_run_by_default(tmp_path: Path) -> None:
    store_path = _build_synthetic_store(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["fix-fillvalue", "--store", str(store_path)])

    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output.lower()
    assert "--yes-i-really-mean-it" in result.output

    arr = zarr.open_array(str(store_path / "data_1km/counts"), mode="r")
    assert "_FillValue" not in dict(arr.attrs)


@pytest.mark.unit
def test_fix_fillvalue_apply_stamps_fillvalue(tmp_path: Path) -> None:
    store_path = _build_synthetic_store(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fix-fillvalue", "--store", str(store_path), "--yes-i-really-mean-it"],
    )

    assert result.exit_code == 0, result.output

    arr = zarr.open_array(str(store_path / "data_1km/counts"), mode="r")
    fill = arr.attrs.get("_FillValue")
    assert fill is not None
    assert fill == 65535


@pytest.mark.unit
def test_fix_fillvalue_idempotent_on_matching_value(tmp_path: Path) -> None:
    store_path = _build_synthetic_store(tmp_path)
    runner = CliRunner()

    first = runner.invoke(
        cli,
        ["fix-fillvalue", "--store", str(store_path), "--yes-i-really-mean-it"],
    )
    assert first.exit_code == 0, first.output

    second = runner.invoke(
        cli,
        ["fix-fillvalue", "--store", str(store_path), "--yes-i-really-mean-it"],
    )
    assert second.exit_code == 0, second.output

    arr = zarr.open_array(str(store_path / "data_1km/counts"), mode="r")
    fill = arr.attrs.get("_FillValue")
    assert fill is not None
    assert fill == 65535


@pytest.mark.unit
def test_fix_fillvalue_errors_on_conflicting_existing_value(tmp_path: Path) -> None:
    store_path = _build_synthetic_store(tmp_path, fillvalue_on_counts=12345)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fix-fillvalue", "--store", str(store_path), "--yes-i-really-mean-it"],
    )

    assert result.exit_code != 0, result.output
    assert "12345" in result.output
    assert "65535" in result.output

    arr = zarr.open_array(str(store_path / "data_1km/counts"), mode="r")
    assert arr.attrs["_FillValue"] == 12345


@pytest.mark.unit
def test_fix_fillvalue_skips_datetime_arrays(tmp_path: Path) -> None:
    store_path = _build_synthetic_store(tmp_path, with_time=True)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fix-fillvalue", "--store", str(store_path), "--yes-i-really-mean-it"],
    )
    assert result.exit_code == 0, result.output

    counts = zarr.open_array(str(store_path / "data_1km/counts"), mode="r")
    assert counts.attrs.get("_FillValue") is not None

    time_arr = zarr.open_array(str(store_path / "data_1km/time"), mode="r")
    assert "_FillValue" not in dict(time_arr.attrs)


@pytest.mark.unit
def test_fix_fillvalue_skips_arrays_with_none_fillvalue(tmp_path: Path) -> None:
    store_path = _build_synthetic_store(tmp_path, with_x=True)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fix-fillvalue", "--store", str(store_path), "--yes-i-really-mean-it"],
    )
    assert result.exit_code == 0, result.output

    counts = zarr.open_array(str(store_path / "data_1km/counts"), mode="r")
    assert counts.attrs.get("_FillValue") is not None

    x_arr = zarr.open_array(str(store_path / "data_1km/x"), mode="r")
    assert "_FillValue" not in dict(x_arr.attrs)


@pytest.mark.unit
def test_fix_fillvalue_errors_on_nonexistent_store() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "fix-fillvalue",
            "--store",
            "/nonexistent/path.zarr",
            "--yes-i-really-mean-it",
        ],
    )

    assert result.exit_code != 0 or result.exception is not None


@pytest.mark.unit
def test_fix_fillvalue_help_text() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["fix-fillvalue", "--help"])

    assert result.exit_code == 0, result.output
    assert "--store" in result.output
    assert "--yes-i-really-mean-it" in result.output
    assert "ingestion" in result.output.lower()


@pytest.mark.unit
def test_fix_fillvalue_empty_store(tmp_path: Path) -> None:
    store_path = tmp_path / "empty.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.require_group("data_1km")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fix-fillvalue", "--store", str(store_path), "--yes-i-really-mean-it"],
    )
    assert result.exit_code == 0, result.output


@pytest.mark.unit
def test_fix_fillvalue_partial_store_with_missing_arrays(tmp_path: Path) -> None:
    store_path = _build_synthetic_store(
        tmp_path, with_counts=True, with_x=False, with_time=False
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fix-fillvalue", "--store", str(store_path), "--yes-i-really-mean-it"],
    )
    assert result.exit_code == 0, result.output

    counts = zarr.open_array(str(store_path / "data_1km/counts"), mode="r")
    fill = counts.attrs.get("_FillValue")
    assert fill is not None
    assert fill == 65535


@pytest.mark.unit
def test_fix_fillvalue_uses_on_disk_dtype_for_int_pixel_time(tmp_path: Path) -> None:
    store_path = tmp_path / "int_pixel_time.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    grp = root.require_group("data_1km")
    int32_sentinel = int(np.iinfo(np.int32).max)
    grp.create_array(
        "pixel_time",
        shape=(1, 10, 10, 1),
        dtype=np.int32,
        fill_value=int32_sentinel,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fix-fillvalue", "--store", str(store_path), "--yes-i-really-mean-it"],
    )
    assert result.exit_code == 0, result.output

    stamped = zarr.open_array(str(store_path / "data_1km/pixel_time"), mode="r")
    fill_attr = stamped.attrs.get("_FillValue")
    assert isinstance(fill_attr, int), (
        f"expected integer _FillValue for int32 array, got {type(fill_attr).__name__}: {fill_attr!r}"
    )
    assert fill_attr == int32_sentinel


@pytest.mark.unit
def test_fix_fillvalue_conflict_writes_no_partial_state(tmp_path: Path) -> None:
    store_path = tmp_path / "conflict_partial.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    grp = root.require_group("data_1km")
    grp.create_array(
        "counts",
        shape=(1, 10, 10, 1),
        dtype=np.uint16,
        fill_value=65535,
    )
    pq = grp.create_array(
        "pixel_quality",
        shape=(1, 10, 10, 1),
        dtype=np.uint8,
        fill_value=0,
    )
    pq.attrs.update({"_FillValue": 42})

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fix-fillvalue", "--store", str(store_path), "--yes-i-really-mean-it"],
    )
    assert result.exit_code == 1, result.output

    counts_after = zarr.open_array(str(store_path / "data_1km/counts"), mode="r")
    assert "_FillValue" not in dict(counts_after.attrs), (
        "no-partial-state violated: counts was stamped despite conflict on pixel_quality"
    )
    pq_after = zarr.open_array(str(store_path / "data_1km/pixel_quality"), mode="r")
    assert pq_after.attrs.get("_FillValue") == 42, (
        "conflicting existing _FillValue on pixel_quality was overwritten"
    )


@pytest.mark.integration
@pytest.mark.plugin
def test_fix_fillvalue_end_to_end(tmp_path: Path, fdhsi_zip: Path) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from test_integration import _run_ingest  # noqa: PLC0415

    store_path = _run_ingest(fdhsi_zip.parent, tmp_path, options={})

    counts_before = zarr.open_array(str(store_path / "data_1km/counts"), mode="r")
    assert "_FillValue" not in dict(counts_before.attrs), (
        "Baseline precondition failed: counts already carries _FillValue; "
        "the RED test cannot demonstrate the fix."
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fix-fillvalue", "--store", str(store_path), "--yes-i-really-mean-it"],
    )
    assert result.exit_code == 0, result.output

    import xarray as xr  # noqa: PLC0415

    ds: Any = xr.open_zarr(
        str(store_path / "data_1km"),
        consolidated=False,
        mask_and_scale=True,
    )
    try:
        assert ds.counts.encoding.get("_FillValue") == 65535
    finally:
        ds.close()
