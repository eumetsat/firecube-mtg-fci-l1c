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

"""CRS-oracle tests for issue #1.

Uses ``pyproj`` as the external CRS authority to verify that projection
coordinates produced by the plugin's ``_projection_x_source`` agree with the
expected east-positive geostationary convention. This is NOT a mirror test —
``pyproj`` is an independent library.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyproj")

from pyproj import CRS, Transformer  # noqa: E402  (pyproj is optional)

from firecube_mtg_fci_l1c._schema import VariableContext  # noqa: E402
from firecube_mtg_fci_l1c.config import MtgFciL1cConfig  # noqa: E402
from firecube_mtg_fci_l1c._variables import (  # noqa: E402
    _MTG_GEOS_WKT,
    _projection_x_source,
    _projection_y_source,
)


_MTG_SATELLITE_HEIGHT_M: float = 35786400.0


def _build_ctx(group: str, dimsize: int, config: MtgFciL1cConfig) -> VariableContext:
    """Construct a minimal static-phase ``VariableContext`` for x/y sources."""
    return VariableContext(
        group=group,
        product_type="FDHSI",
        config=config,
        dimsize=dimsize,
        n_channels=8,
        logical_channels=(),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("group", "dimsize"),
    [
        ("data_500m", 22272),
        ("data_1km", 11136),
        ("data_2km", 5568),
    ],
)
def test_meter_mode_x_monotonically_increasing_all_resolutions(
    group: str, dimsize: int
) -> None:
    """After the fix, ``x`` is east-positive (monotonically increasing)."""
    ctx = _build_ctx(group, dimsize, MtgFciL1cConfig())
    x_arr = _projection_x_source(ctx)
    assert x_arr is not None, f"_projection_x_source returned None for {group}"
    assert x_arr.shape == (dimsize,)
    diffs = np.diff(x_arr)
    assert np.all(diffs > 0), (
        f"x must be monotonically increasing (east-positive) for {group}; "
        f"first diff={diffs[0]!r}, last diff={diffs[-1]!r}"
    )


@pytest.mark.unit
def test_pyproj_lon_correlates_positively_with_x() -> None:
    """pyproj GEOS→WGS84 must yield lon strongly positively correlated with x.

    Independent oracle: parse the plugin's own ``_MTG_GEOS_WKT`` via pyproj,
    build a transformer to EPSG:4326, and check that increasing ``x`` yields
    increasing longitude along a mid-latitude row. ``always_xy=True`` forces
    pyproj to return ``(lon, lat)`` regardless of the CRS's declared axis
    order, which is mandatory for this comparison.
    """
    dimsize = 11136
    ctx = _build_ctx("data_1km", dimsize, MtgFciL1cConfig())
    x = _projection_x_source(ctx)
    y = _projection_y_source(ctx)
    assert x is not None and y is not None

    crs_geos = CRS.from_wkt(_MTG_GEOS_WKT)
    to_lonlat = Transformer.from_crs(crs_geos, CRS.from_epsg(4326), always_xy=True)

    cols = np.linspace(0, dimsize - 1, 300).astype(int)
    row = dimsize // 2
    lon_proj, _lat_proj = to_lonlat.transform(x[cols], np.full(300, y[row]))

    mask = np.isfinite(lon_proj)
    assert mask.sum() >= 10, (
        f"pyproj returned too few finite longitudes ({int(mask.sum())}/300); "
        "the x-values likely fall outside the disk projection."
    )

    corr = float(np.corrcoef(x[cols][mask], lon_proj[mask])[0, 1])
    assert corr > 0.99, (
        f"expected strong positive correlation between x and pyproj-derived "
        f"lon (east-positive convention), got corr={corr:.4f}"
    )


@pytest.mark.unit
def test_radian_mode_after_scaling_matches_meter_mode() -> None:
    """Radian-mode x, scaled by satellite height, must equal meter-mode x."""
    dimsize = 11136
    x_meter = _projection_x_source(_build_ctx("data_1km", dimsize, MtgFciL1cConfig()))
    x_rad = _projection_x_source(
        _build_ctx(
            "data_1km",
            dimsize,
            MtgFciL1cConfig(projection_units="radian"),
        )
    )
    assert x_meter is not None and x_rad is not None
    assert np.allclose(x_rad * _MTG_SATELLITE_HEIGHT_M, x_meter), (
        "radian-mode x scaled by MTG perspective-point height must equal "
        "meter-mode x element-wise."
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("group", "dimsize"),
    [
        ("data_500m", 22272),
        ("data_1km", 11136),
        ("data_2km", 5568),
    ],
)
def test_centre_pixels_straddle_nadir_all_resolutions(group: str, dimsize: int) -> None:
    """Issue #8: index 0 must be pixel centre ``-(dimsize/2 - 0.5)`` steps from nadir.

    The FCI fixed grid has an even number of pixels per axis, so nadir lies on
    the boundary between the two central pixels: their centres sit at
    ``-scale/2`` and ``+scale/2``. This pins the offset per resolution without
    reference to any constant copied from a file.
    """
    from firecube_mtg_fci_l1c._constants import FCI_PROJ_SCALE_RAD_PER_INDEX

    scale = FCI_PROJ_SCALE_RAD_PER_INDEX[group.removeprefix("data_")]
    ctx = _build_ctx(group, dimsize, MtgFciL1cConfig(projection_units="radian"))
    for arr in (_projection_x_source(ctx), _projection_y_source(ctx)):
        assert arr is not None
        half = dimsize // 2
        assert arr[half - 1] == pytest.approx(-scale / 2, abs=1e-15)
        assert arr[half] == pytest.approx(scale / 2, abs=1e-15)
        assert arr[0] == pytest.approx(-arr[-1], abs=1e-15)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("group", "resolution_m", "dimsize"),
    [
        ("data_1km", 1000, 11136),
        ("data_2km", 2000, 5568),
    ],
)
def test_x_y_land_on_latlon_pixel_centres(
    group: str, resolution_m: int, dimsize: int
) -> None:
    """Issue #8: ``x[col]``/``y[row]`` must be the geos coordinates of the pixel
    whose ``latitude``/``longitude`` the plugin writes for ``(row, col)``.

    Projects a sample of ``compute_latlon`` pixels through the plugin's own
    ``spatial_ref`` WKT with pyproj and compares against ``x``/``y`` in pixel
    units. Before the fix this was off by a constant -1.00 px (1 km) and
    -0.75 px (2 km) on both axes. 500 m is covered by the centre-pixel test
    above; its full grid is too heavy for a unit test.
    """
    from firecube_mtg_fci_l1c._constants import (
        FCI_PROJ_SCALE_RAD_PER_INDEX,
        MTG_PERSPECTIVE_POINT_HEIGHT_M,
    )
    from firecube_mtg_fci_l1c.geolocation.projection import compute_latlon

    ctx = _build_ctx(group, dimsize, MtgFciL1cConfig())
    x = _projection_x_source(ctx)
    y = _projection_y_source(ctx)
    assert x is not None and y is not None
    lat, lon = compute_latlon(resolution_m)

    idx = np.arange(dimsize // 10, dimsize - dimsize // 10, dimsize // 40)
    rows, cols = np.meshgrid(idx, idx, indexing="ij")
    to_geos = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_wkt(_MTG_GEOS_WKT), always_xy=True
    )
    gx, gy = to_geos.transform(
        lon[rows, cols].astype(np.float64), lat[rows, cols].astype(np.float64)
    )
    ok = np.isfinite(gx) & np.isfinite(gy)
    assert ok.sum() > 100

    px_m = (
        FCI_PROJ_SCALE_RAD_PER_INDEX[group.removeprefix("data_")]
        * MTG_PERSPECTIVE_POINT_HEIGHT_M
    )
    dx = (x[cols] - gx)[ok] / px_m
    dy = (y[rows] - gy)[ok] / px_m
    # float32 lat/lon limits agreement to ~1e-2 px; a 1-based/offset error is >= 0.25 px.
    assert abs(float(dx.mean())) < 0.05, f"x offset {dx.mean():+.3f} px"
    assert abs(float(dy.mean())) < 0.05, f"y offset {dy.mean():+.3f} px"
    assert float(np.abs(dx).max()) < 0.1
    assert float(np.abs(dy).max()) < 0.1
