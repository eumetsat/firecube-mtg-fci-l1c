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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import zarr
from click.testing import CliRunner

from _store_files import store_files
from firecube_mtg_fci_l1c.plugin_cli import cli
from test_integration import _run_ingest


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
    store_path = _run_ingest(fdhsi_zip.parent, tmp_path, options={})

    # Ingest stamps _FillValue on counts. Strip it from the arrays the command
    # covers so the command has something to stamp.
    counts_before = zarr.open_array(str(store_path / "data_1km/counts"), mode="r+")
    assert dict(counts_before.attrs).get("_FillValue") == 65535, (
        "Precondition failed: ingest does not stamp _FillValue on counts."
    )
    for arr_name in ("counts", "pixel_quality", "spatial_ref"):
        arr = zarr.open_array(str(store_path / f"data_1km/{arr_name}"), mode="r+")
        if "_FillValue" in dict(arr.attrs):
            del arr.attrs["_FillValue"]
    counts_stripped = zarr.open_array(str(store_path / "data_1km/counts"), mode="r")
    assert "_FillValue" not in dict(counts_stripped.attrs)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fix-fillvalue", "--store", str(store_path), "--yes-i-really-mean-it"],
    )
    assert result.exit_code == 0, result.output

    ds: Any = xr.open_zarr(
        str(store_path / "data_1km"),
        consolidated=False,
        mask_and_scale=True,
    )
    try:
        assert ds.counts.encoding.get("_FillValue") == 65535
    finally:
        ds.close()


def _flat_store_without_fillvalue_attrs(
    zip_path: Path, workspace: Path, resolution: str
) -> Path:
    """Ingest *zip_path* into a flat store and strip every ``_FillValue`` attr
    so the command has something to stamp.
    """
    store_path = _run_ingest(
        zip_path.parent,
        workspace,
        options={"resolutions": resolution, "flat_store": True},
    )
    root = zarr.open_group(str(store_path), mode="r+")
    assert not list(root.group_keys()), "flat ingest should create no groups"
    for name in root.array_keys():
        attrs = root[name].attrs
        if "_FillValue" in dict(attrs):
            del attrs["_FillValue"]
    return store_path


@pytest.mark.integration
@pytest.mark.plugin
def test_fix_fillvalue_dry_run_then_apply_on_flat_store(
    tmp_path: Path, fdhsi_zip: Path
) -> None:
    store_path = _flat_store_without_fillvalue_attrs(fdhsi_zip, tmp_path, "1km")
    before = store_files(store_path, skip_control_plane=False)

    dry_run = CliRunner().invoke(cli, ["fix-fillvalue", "--store", str(store_path)])

    assert dry_run.exit_code == 0, dry_run.output
    assert "Mode:  dry-run" in dry_run.output
    assert "Would stamp _FillValue on 8 array(s):" in dry_run.output
    assert "  /counts: _FillValue = 65535" in dry_run.output
    assert "  /pixel_quality: _FillValue = 0" in dry_run.output
    assert "/x:" not in dry_run.output
    assert "--yes-i-really-mean-it" in dry_run.output
    assert store_files(store_path, skip_control_plane=False) == before

    result = CliRunner().invoke(
        cli,
        ["fix-fillvalue", "--store", str(store_path), "--yes-i-really-mean-it"],
    )
    assert result.exit_code == 0, result.output
    assert "  /counts: _FillValue = 65535" in result.output
    assert "data_" not in result.output

    root = zarr.open_group(str(store_path), mode="r")
    assert root["counts"].attrs["_FillValue"] == 65535
    assert root["pixel_quality"].attrs["_FillValue"] == 0
    assert root["spatial_ref"].attrs["_FillValue"] == 0
    # NaN float fills are stored base64-encoded, as xarray expects for Zarr v3.
    assert root["latitude"].attrs["_FillValue"] == "AAAAAAAA+H8="
    assert root["slope"].attrs["_FillValue"] == "AAAAAAAA+H8="
    # No schema fill (x, y) or non-numeric dtype (time, channel_name): untouched.
    for name in ("x", "y", "time", "channel_name"):
        assert "_FillValue" not in dict(root[name].attrs), name

    ds: Any = xr.open_zarr(str(store_path), consolidated=False, mask_and_scale=True)
    try:
        assert ds.counts.encoding.get("_FillValue") == 65535
    finally:
        ds.close()


def _empty_store(store_path: Path) -> None:
    zarr.open_group(str(store_path), mode="w")


def _store_with_unrelated_arrays(store_path: Path) -> None:
    root = zarr.open_group(str(store_path), mode="w")
    root.create_array("radiance", shape=(4,), dtype=np.uint16, fill_value=65535)
    root.require_group("level2").create_array(
        "counts", shape=(4,), dtype=np.uint16, fill_value=65535
    )


def _mixed_store(store_path: Path) -> None:
    root = zarr.open_group(str(store_path), mode="w")
    root.create_array("counts", shape=(4,), dtype=np.uint16, fill_value=65535)
    root.require_group("data_1km").create_array(
        "counts", shape=(4,), dtype=np.uint16, fill_value=65535
    )


_NO_LAYOUT = "neither plugin arrays at the root nor data_<res> groups"
_BOTH_LAYOUTS = "both plugin arrays at the root and data_<res> groups"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("build_store", "reason"),
    [
        (_empty_store, _NO_LAYOUT),
        (_store_with_unrelated_arrays, _NO_LAYOUT),
        (_mixed_store, _BOTH_LAYOUTS),
    ],
    ids=["empty", "unrelated-arrays", "mixed"],
)
def test_fix_fillvalue_refuses_store_without_a_single_layout(
    tmp_path: Path, build_store: Any, reason: str
) -> None:
    store_path = tmp_path / "store.zarr"
    build_store(store_path)
    before = store_files(store_path, skip_control_plane=False)

    result = CliRunner().invoke(
        cli, ["fix-fillvalue", "--store", str(store_path), "--yes-i-really-mean-it"]
    )

    assert result.exit_code == 1, result.output
    assert reason in result.output
    assert "No writes performed." in result.output
    assert "_FillValue =" not in result.output
    assert store_files(store_path, skip_control_plane=False) == before
