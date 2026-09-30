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

"""End-to-end round trip of the ``flat_store`` layout.

A real ingest with ``flat_store=true`` and one resolution must produce a store
that ``xr.open_zarr(store)`` opens without ``group=`` and that holds the same
arrays as the ``data_<res>`` group of the default layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import zarr
from firecube.core.api import RESERVED_ARRAY_ATTRS

from test_integration import _run_ingest

pytestmark = [pytest.mark.integration, pytest.mark.plugin]

EXPECTED_DATA_VARS = {
    "counts",
    "pixel_quality",
    "pixel_time",
    "slope",
    "offset",
    "channel_name",
    "spatial_ref",
}
EXPECTED_COORDS = {"time", "y", "x", "latitude", "longitude"}


@pytest.mark.parametrize(
    (
        "zip_fixture",
        "resolution",
        "expected_counts",
        "expected_slope",
        "expected_offset",
        "expected_channels",
    ),
    [
        pytest.param(
            "fdhsi_zip",
            "1km",
            [1, 2],
            [1.0, 2.0],
            [0.0, 1.0],
            [b"vis_04", b"vis_06"],
            id="fdhsi-1km",
        ),
        pytest.param(
            "hrfi_zip", "500m", [1], [1.0], [0.0], [b"vis_06"], id="hrfi-500m"
        ),
    ],
)
def test_flat_store_opens_without_group_and_holds_the_ingested_data(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    zip_fixture: str,
    resolution: str,
    expected_counts: list[int],
    expected_slope: list[float],
    expected_offset: list[float],
    expected_channels: list[bytes],
) -> None:
    source_zip: Path = request.getfixturevalue(zip_fixture)

    store = _run_ingest(
        source_zip.parent,
        tmp_path,
        options={"resolutions": resolution, "flat_store": True},
    )

    root = zarr.open_group(str(store), mode="r")
    assert sorted(root.group_keys()) == []

    ds = xr.open_zarr(str(store), consolidated=False)
    try:
        assert set(ds.data_vars) == EXPECTED_DATA_VARS
        assert set(ds.coords) == EXPECTED_COORDS

        n_channels = len(expected_counts)
        assert ds["counts"].dims == ("time", "y", "x", "channel")
        assert ds["counts"].shape == (1, 4, 4, n_channels)
        counts = ds["counts"].values
        for channel_index, value in enumerate(expected_counts):
            np.testing.assert_array_equal(
                counts[0, :, :, channel_index], np.full((4, 4), value)
            )
        np.testing.assert_array_equal(ds["slope"].values, [expected_slope])
        np.testing.assert_array_equal(ds["offset"].values, [expected_offset])
        assert ds["channel_name"].values.tolist() == expected_channels

        np.testing.assert_array_equal(
            ds["time"].values, np.array(["2024-01-01T00:00:00"], dtype="datetime64[ns]")
        )

        for name in ("x", "y"):
            assert ds[name].shape == (4,)
            assert np.all(np.isfinite(ds[name].values)), name
        for name in ("latitude", "longitude"):
            values = ds[name].values
            assert values.shape == (4, 4)
            # The small test grid puts a space pixel (NaN) at [0, 0] and 0.0 elsewhere.
            assert np.isnan(values[0, 0]), name
            assert np.all(values.ravel()[1:] == 0.0), name

        assert ds.attrs["Conventions"] == "CF-1.8"
        assert ds.attrs["institution"] == "EUMETSAT"
        assert ds.attrs["title"] == (
            f"MTG FCI Level 1C effective radiances ({resolution})"
        )
    finally:
        ds.close()


def _plugin_attrs(array: Any) -> dict[str, Any]:
    return {k: v for k, v in array.attrs.items() if k not in RESERVED_ARRAY_ATTRS}


def test_flat_root_arrays_equal_the_grouped_arrays_from_the_same_input(
    tmp_path: Path, fdhsi_zip: Path
) -> None:
    flat_store = _run_ingest(
        fdhsi_zip.parent,
        tmp_path / "flat",
        options={"resolutions": "1km", "flat_store": True},
    )
    grouped_store = _run_ingest(
        fdhsi_zip.parent, tmp_path / "grouped", options={"resolutions": "1km"}
    )

    flat_root: Any = zarr.open_group(str(flat_store), mode="r")
    grouped_root: Any = zarr.open_group(str(grouped_store), mode="r")
    assert sorted(grouped_root.group_keys()) == ["data_1km"]
    grouped = grouped_root["data_1km"]

    assert sorted(flat_root.array_keys()) == sorted(grouped.array_keys())
    assert len(list(flat_root.array_keys())) == 12

    for name in sorted(flat_root.array_keys()):
        flat_array = flat_root[name]
        grouped_array = grouped[name]
        assert flat_array.shape == grouped_array.shape, name
        assert flat_array.dtype == grouped_array.dtype, name
        assert flat_array.chunks == grouped_array.chunks, name
        assert flat_array.shards == grouped_array.shards, name
        assert flat_array.fill_value == grouped_array.fill_value or (
            np.isnan(flat_array.fill_value) and np.isnan(grouped_array.fill_value)
        ), name
        assert (
            flat_array.metadata.dimension_names
            == grouped_array.metadata.dimension_names
        ), name
        assert _plugin_attrs(flat_array) == _plugin_attrs(grouped_array), name
        np.testing.assert_array_equal(flat_array[...], grouped_array[...], err_msg=name)
