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

import importlib
import inspect
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from firecube_mtg_fci_l1c.geolocation.provider import LatLonProvider
from firecube_mtg_fci_l1c.geolocation.projection import compute_latlon


@pytest.fixture(scope="module")
def geo_2km() -> tuple[np.ndarray, np.ndarray]:
    return compute_latlon(2000)


@pytest.mark.unit
def test_compute_latlon_2km_shape(geo_2km):
    lat, _lon = geo_2km
    assert lat.shape == (5568, 5568)


@pytest.mark.unit
def test_compute_latlon_1km_shape():
    lat, _lon = compute_latlon(1000)
    assert lat.shape == (11136, 11136)
    assert lat.dtype == np.float32


@pytest.mark.unit
@pytest.mark.slow
def test_compute_latlon_500m_shape():
    lat, _lon = compute_latlon(500)
    assert lat.shape == (22272, 22272)
    assert lat.dtype == np.float32


@pytest.mark.unit
def test_compute_latlon_dtype(geo_2km):
    lat, lon = geo_2km
    assert lat.dtype == np.float32
    assert lon.dtype == np.float32


@pytest.mark.unit
def test_compute_latlon_off_earth_nan(geo_2km):
    lat, lon = geo_2km
    corners = [(0, 0), (0, 5567), (5567, 0), (5567, 5567)]
    for y, x in corners:
        assert np.isnan(lat[y, x])
        assert np.isnan(lon[y, x])


@pytest.mark.unit
def test_compute_latlon_center_near_zero(geo_2km):
    lat, lon = geo_2km
    assert abs(float(lat[2784, 2784])) <= 0.1
    assert abs(float(lon[2784, 2784])) <= 0.1


@pytest.mark.unit
def test_compute_latlon_ns_symmetry(geo_2km):
    lat, _lon = geo_2km
    dim = lat.shape[0]
    y, x = 1500, 3000
    y2 = dim - 1 - y
    if np.isfinite(lat[y, x]) and np.isfinite(lat[y2, x]):
        np.testing.assert_allclose(lat[y, x], -lat[y2, x], atol=1e-5)


@pytest.mark.unit
def test_compute_latlon_invalid_resolution():
    with pytest.raises(ValueError):
        compute_latlon(300)


@pytest.mark.unit
def test_no_io_imports():
    module = importlib.import_module("firecube_mtg_fci_l1c.geolocation.projection")
    source = inspect.getsource(module)
    assert "import pyproj" not in source
    assert "import xarray" not in source
    assert "import zarr" not in source
    assert "pyproj" not in sys.modules


@pytest.mark.unit
def test_lat_lon_provider_reuses_npz_loader_for_repeated_resolution(monkeypatch):
    import firecube_mtg_fci_l1c.geolocation.grids as grids_mod

    created: list[str] = []

    class FakeFciGrids:
        def __init__(self, grids_file: str):
            created.append(grids_file)

        def get_coordinates(self, _resolution_m: int):
            grid = np.zeros((2, 2), dtype=np.float32)
            return grid, grid

    monkeypatch.setattr(grids_mod, "FciGrids", FakeFciGrids)

    logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None)
    provider = LatLonProvider(logger)

    provider.get_lat_lon("grids.npz", 1000)
    provider.get_lat_lon("grids.npz", 1000)

    assert created == ["grids.npz"]
