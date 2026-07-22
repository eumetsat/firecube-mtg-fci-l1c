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

from __future__ import annotations

# pyright: reportMissingImports=false

import copy
import zipfile
from pathlib import Path

import h5netcdf
import numpy as np
import pytest

from firecube_mtg_fci_l1c._streaming import (
    NCPartReader,
    TimeMapAccumulator,
    expand_pixel_time,
    list_fci_nc_parts,
)  # pyright: ignore[reportMissingImports]
from firecube_mtg_fci_l1c._constants import (  # pyright: ignore[reportMissingImports]
    PRODUCT_TYPE_FDHSI,
    get_nc_part_prefix,
)


def _write_nc_part(
    path: Path,
    *,
    channel_specs: dict[str, tuple[int, int, int]],
    index_values: np.ndarray | None = None,
    time_values: np.ndarray | None = None,
    include_time_map: bool = True,
    scale_factor: float = 0.5,
    add_offset: float = 1.5,
) -> None:
    index_values_arr = (
        np.asarray(index_values, dtype=np.int64)
        if index_values is not None
        else np.asarray([0, 1, 2], dtype=np.int64)
    )
    time_values_arr = (
        np.asarray(time_values, dtype=np.float64)
        if time_values is not None
        else np.asarray([0.0, 60.0, 120.0], dtype=np.float64)
    )

    with h5netcdf.File(path, "w") as ds:
        if include_time_map:
            ds.dimensions["n_time"] = len(index_values_arr)
            ds.create_variable("index", ("n_time",), data=index_values_arr)
            ds.create_variable("time", ("n_time",), data=time_values_arr)

        data_group = ds.create_group("data")
        for channel, (start_row, end_row, base) in channel_specs.items():
            measured = data_group.create_group(channel).create_group("measured")
            measured.dimensions["y"] = 2
            measured.dimensions["x"] = 3

            radiance = measured.create_variable(
                "effective_radiance",
                ("y", "x"),
                data=np.full((2, 3), base, dtype=np.uint16),
            )
            radiance.attrs["start_position_row"] = start_row
            radiance.attrs["end_position_row"] = end_row
            radiance.attrs["scale_factor"] = scale_factor
            radiance.attrs["add_offset"] = add_offset

            measured.create_variable(
                "pixel_quality",
                ("y", "x"),
                data=np.full((2, 3), base + 1, dtype=np.uint8),
            )
            measured.create_variable(
                "index_map",
                ("y", "x"),
                data=np.asarray([[0, 1, 99], [2, -1, 1]], dtype=np.int32),
            )


@pytest.fixture
def _small_fci_constants(monkeypatch):
    from firecube_mtg_fci_l1c import _constants as const_mod  # pyright: ignore[reportMissingImports]

    backup = copy.deepcopy(const_mod.CONSTANTS)
    const_mod.CONSTANTS[PRODUCT_TYPE_FDHSI] = {
        "1km": {"channels": ["vis_04"], "dimsize": 2, "nc_channels": ["vis_04"]},
        "2km": {"channels": ["ir_38"], "dimsize": 2, "nc_channels": ["ir_38"]},
    }
    yield
    const_mod.CONSTANTS.clear()
    const_mod.CONSTANTS.update(backup)


@pytest.fixture
def fci_test_zip(tmp_path: Path, _small_fci_constants) -> Path:
    prefix = get_nc_part_prefix(PRODUCT_TYPE_FDHSI)
    trail_prefix = prefix.replace("BODY", "TRAIL")

    src = tmp_path / "src"
    src.mkdir()
    body_0001 = src / "body_0001.nc"
    body_0002 = src / "body_0002.nc"
    trail_0001 = src / "trail_0001.nc"
    other = src / "not_a_part.nc"

    _write_nc_part(
        body_0001,
        channel_specs={
            "vis_04": (1, 4, 11),
            "ir_38": (5, 8, 21),
        },
    )
    _write_nc_part(
        body_0002,
        channel_specs={
            "vis_04": (9, 12, 31),
            "ir_38": (13, 16, 41),
        },
    )
    _write_nc_part(
        trail_0001,
        channel_specs={
            "vis_04": (17, 20, 51),
            "ir_38": (21, 24, 61),
        },
    )
    _write_nc_part(other, channel_specs={"vis_04": (1, 2, 99)})

    zip_path = tmp_path / "W_XX-FCI-1C-RRAD-FDHSI-FD-20240101000000-END.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(body_0002, arcname=f"{prefix}0002.nc")
        zf.write(trail_0001, arcname=f"{trail_prefix}0001.nc")
        zf.write(body_0001, arcname=f"{prefix}0001.nc")
        zf.write(other, arcname="nested/not_a_part.nc")
    return zip_path


@pytest.mark.unit
def test_nc_part_reader_channel_data_and_row_ranges(
    fci_test_zip: Path, tmp_path: Path
):
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    with zipfile.ZipFile(fci_test_zip) as zf:
        zf.extractall(extract_dir)

    first_body = list_fci_nc_parts(extract_dir)[0]

    with NCPartReader(first_body) as reader:
        assert reader.read_row_range("1km") == (0, 4)
        assert reader.read_row_range("2km") == (4, 8)
        assert reader.available_channels() == ["ir_38", "vis_04"]

        effective_radiance, pixel_quality, index_map = reader.read_channel_data(
            "vis_04"
        )
        assert effective_radiance.shape == (2, 3)
        assert pixel_quality.shape == (2, 3)
        assert index_map.shape == (2, 3)
        assert effective_radiance[0, 0] == 11
        assert pixel_quality[0, 0] == 12
        assert index_map[0, 2] == 99


@pytest.mark.unit
def test_time_map_accumulator_merges_and_overrides(tmp_path: Path):
    part1 = tmp_path / "W_XX-FCI-1C-RRAD-FDHSI-FD-body-0001.nc"
    part2 = tmp_path / "W_XX-FCI-1C-RRAD-FDHSI-FD-body-0002.nc"

    _write_nc_part(
        part1,
        channel_specs={"vis_04": (1, 2, 1)},
        index_values=np.asarray([0, 1], dtype=np.int64),
        time_values=np.asarray([0.0, 60.0], dtype=np.float64),
    )
    _write_nc_part(
        part2,
        channel_specs={"vis_04": (1, 2, 1)},
        index_values=np.asarray([1, 2], dtype=np.int64),
        time_values=np.asarray([61.0, 120.0], dtype=np.float64),
    )

    accumulator = TimeMapAccumulator()
    with NCPartReader(part1) as reader1:
        accumulator.accumulate(reader1)
    with NCPartReader(part2) as reader2:
        accumulator.accumulate(reader2)

    assert accumulator.build_index2time() == {0: 0.0, 1: 61.0, 2: 120.0}


@pytest.mark.unit
def test_time_map_accumulator_skips_parts_without_root_time_map(tmp_path: Path):
    body = tmp_path / "W_XX-FCI-1C-RRAD-FDHSI-FD-body-0001.nc"
    trail = tmp_path / "W_XX-FCI-1C-RRAD-FDHSI-FD-trail-0002.nc"

    _write_nc_part(
        body,
        channel_specs={"vis_04": (1, 2, 1)},
        index_values=np.asarray([5, 6], dtype=np.int64),
        time_values=np.asarray([50.0, 60.0], dtype=np.float64),
    )
    _write_nc_part(
        trail,
        channel_specs={"vis_04": (1, 2, 1)},
        include_time_map=False,
    )

    accumulator = TimeMapAccumulator()
    with NCPartReader(body) as body_reader:
        accumulator.accumulate(body_reader)
    with NCPartReader(trail) as trail_reader:
        accumulator.accumulate(trail_reader)

    assert accumulator.build_index2time() == {5: 50.0, 6: 60.0}


@pytest.mark.unit
def test_nc_part_reader_reads_calibration_from_radiance_attrs(tmp_path: Path):
    part = tmp_path / "W_XX-FCI-1C-RRAD-FDHSI-FD-body-0001.nc"
    _write_nc_part(
        part,
        channel_specs={"vis_04": (1, 2, 1)},
        scale_factor=0.25,
        add_offset=3.5,
    )

    with NCPartReader(part) as reader:
        assert reader.read_calibration("vis_04") == pytest.approx((0.25, 3.5))


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_expand_pixel_time_dtype_and_nan_for_unmapped(dtype):
    index_map = np.asarray([[0, 1, 3], [2, -1, 99]], dtype=np.int32)
    index2time = {0: 0.0, 1: 10.5, 2: 20.25}

    expanded = expand_pixel_time(
        index_map=index_map, index2time=index2time, dtype=dtype
    )

    assert expanded.dtype == np.dtype(dtype)
    assert expanded[0, 0] == pytest.approx(0.0)
    assert expanded[0, 1] == pytest.approx(10.5)
    assert expanded[1, 0] == pytest.approx(20.25)
    assert np.isnan(expanded[0, 2])
    assert np.isnan(expanded[1, 1])
    assert np.isnan(expanded[1, 2])
