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

"""Tests for FciGrids class."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from firecube_mtg_fci_l1c.geolocation.grids import FciGrids


def _make_test_npz(path: Path, resolutions: list[str]) -> None:
    """Write a minimal test NPZ file."""
    arrays = {}
    metadata = {"resolutions": resolutions, "sub_satellite_lon": 0.0}
    for res in resolutions:
        dimsize = {"500m": 4, "1km": 4, "2km": 4}[res]
        lat = np.zeros((dimsize, dimsize), dtype=np.float32)
        lon = np.zeros((dimsize, dimsize), dtype=np.float32)
        lat[0, 0] = np.nan
        arrays[f"{res}_lat"] = lat
        arrays[f"{res}_lon"] = lon
    arrays["_metadata"] = np.array([json.dumps(metadata)], dtype="U10000")
    np.savez_compressed(path, **arrays)


@pytest.mark.unit
def test_available_resolutions(tmp_path: Path) -> None:
    npz = tmp_path / "grids.npz"
    _make_test_npz(npz, ["1km", "2km"])
    loader = FciGrids(npz)
    assert loader.available_resolutions() == [1000, 2000]


@pytest.mark.unit
def test_get_coordinates_1km(tmp_path: Path) -> None:
    npz = tmp_path / "grids.npz"
    _make_test_npz(npz, ["1km", "2km"])
    loader = FciGrids(npz)
    lat, lon = loader.get_coordinates(1000)
    assert lat.shape == (4, 4)
    assert lon.shape == (4, 4)
    assert lat.dtype == np.float32
    assert np.isnan(lat[0, 0])


@pytest.mark.unit
def test_missing_resolution_raises(tmp_path: Path) -> None:
    npz = tmp_path / "grids.npz"
    _make_test_npz(npz, ["2km"])
    loader = FciGrids(npz)
    with pytest.raises(ValueError, match="1000m not available"):
        loader.get_coordinates(1000)


@pytest.mark.unit
def test_file_not_found_raises(tmp_path: Path) -> None:
    loader = FciGrids(tmp_path / "nonexistent.npz")
    with pytest.raises(FileNotFoundError, match="geo generate"):
        loader.get_coordinates(2000)


@pytest.mark.unit
def test_metadata_round_trip(tmp_path: Path) -> None:
    npz = tmp_path / "grids.npz"
    _make_test_npz(npz, ["1km"])
    loader = FciGrids(npz)
    meta = loader.get_metadata()
    assert meta["resolutions"] == ["1km"]
    assert meta["sub_satellite_lon"] == 0.0
