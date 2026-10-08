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


def _quiet_logger() -> SimpleNamespace:
    return SimpleNamespace(warning=lambda *_args, **_kwargs: None)


def _cached_arrays(provider: LatLonProvider) -> list[np.ndarray]:
    return [array for pair in provider._cache.values() for array in pair]


@pytest.mark.unit
def test_lat_lon_rows_equal_the_full_disk_rows_and_cache_only_them(geo_2km):
    lat_full, lon_full = geo_2km
    provider = LatLonProvider(_quiet_logger())

    lat, lon = provider.get_lat_lon(None, 2000, rows=(4309, 5568))

    np.testing.assert_array_equal(lat, lat_full[4309:5568])
    np.testing.assert_array_equal(lon, lon_full[4309:5568])
    assert lat.dtype == lon.dtype == np.float32
    cached = _cached_arrays(provider)
    assert len(cached) == 2
    for array in cached:
        assert array.shape == (1259, 5568)
        # Owns its memory: no view keeping a full-disk base alive.
        assert array.base is None
    assert sum(array.nbytes for array in cached) == 2 * 1259 * 5568 * 4


@pytest.mark.unit
def test_lat_lon_rows_release_the_full_disk_pair(monkeypatch):
    import gc
    import weakref

    import firecube_mtg_fci_l1c.geolocation.provider as provider_mod

    refs: list[weakref.ref] = []

    def fake_compute_latlon(_resolution_m: int) -> tuple[np.ndarray, np.ndarray]:
        lat = np.arange(48, dtype=np.float32).reshape(8, 6)
        lon = -lat
        refs.extend([weakref.ref(lat), weakref.ref(lon)])
        return lat, lon

    monkeypatch.setattr(provider_mod, "compute_latlon", fake_compute_latlon)
    provider = LatLonProvider(_quiet_logger())

    lat, lon = provider.get_lat_lon(None, 2000, rows=(2, 5))
    gc.collect()

    np.testing.assert_array_equal(
        lat, np.arange(12, 30, dtype=np.float32).reshape(3, 6)
    )
    np.testing.assert_array_equal(lon, -lat)
    assert len(refs) == 2
    assert all(ref() is None for ref in refs), "full-disk pair still referenced"
    assert [array.shape for array in _cached_arrays(provider)] == [(3, 6), (3, 6)]


@pytest.mark.unit
def test_lat_lon_rows_from_a_grids_file(tmp_path):
    grids_file = tmp_path / "grids.npz"
    lat_full = np.arange(24, dtype=np.float32).reshape(6, 4)
    np.savez_compressed(grids_file, **{"2km_lat": lat_full, "2km_lon": lat_full + 100})
    provider = LatLonProvider(_quiet_logger())

    lat, lon = provider.get_lat_lon(str(grids_file), 2000, rows=(2, 5))

    np.testing.assert_array_equal(lat, lat_full[2:5])
    np.testing.assert_array_equal(lon, lat_full[2:5] + 100)
    assert [array.shape for array in _cached_arrays(provider)] == [(3, 4), (3, 4)]


@pytest.mark.unit
def test_lat_lon_rows_reuse_a_cached_full_disk_pair(monkeypatch):
    import firecube_mtg_fci_l1c.geolocation.provider as provider_mod

    calls: list[int] = []

    def fake_compute_latlon(resolution_m: int) -> tuple[np.ndarray, np.ndarray]:
        calls.append(resolution_m)
        lat = np.arange(20, dtype=np.float32).reshape(5, 4)
        return lat, lat + 1

    monkeypatch.setattr(provider_mod, "compute_latlon", fake_compute_latlon)
    provider = LatLonProvider(_quiet_logger())

    full_lat, _full_lon = provider.get_lat_lon(None, 2000)
    lat, _lon = provider.get_lat_lon(None, 2000, rows=(1, 3))
    provider.get_lat_lon(None, 2000, rows=(1, 3))

    assert calls == [2000]
    np.testing.assert_array_equal(lat, full_lat[1:3])


@pytest.mark.unit
@pytest.mark.parametrize("rows", [(0, 9), (3, 3), (-1, 2), (4, 2)])
def test_lat_lon_rows_outside_the_grid_raise(monkeypatch, rows):
    import firecube_mtg_fci_l1c.geolocation.provider as provider_mod

    def fake_compute_latlon(_resolution_m: int) -> tuple[np.ndarray, np.ndarray]:
        lat = np.zeros((8, 3), dtype=np.float32)
        return lat, lat

    monkeypatch.setattr(provider_mod, "compute_latlon", fake_compute_latlon)
    provider = LatLonProvider(_quiet_logger())

    with pytest.raises(ValueError, match="rows"):
        provider.get_lat_lon(None, 2000, rows=rows)
    assert provider._cache == {}
