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

from firecube_mtg_fci_l1c._variable import VariableContext  # noqa: E402
from firecube_mtg_fci_l1c.config import MtgFciL1cConfig  # noqa: E402
from firecube_mtg_fci_l1c.schema import (  # noqa: E402
    _MTG_GEOS_WKT,
    _projection_x_source,
    _projection_y_source,
)


_MTG_SATELLITE_HEIGHT_M: float = 35786400.0


def _build_ctx(
    group: str, dimsize: int, config: MtgFciL1cConfig
) -> VariableContext:
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
    to_lonlat = Transformer.from_crs(
        crs_geos, CRS.from_epsg(4326), always_xy=True
    )

    cols = np.linspace(0, dimsize - 1, 300).astype(int)
    row = dimsize // 2
    lon_proj, _lat_proj = to_lonlat.transform(
        x[cols], np.full(300, y[row])
    )

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
    x_meter = _projection_x_source(
        _build_ctx("data_1km", dimsize, MtgFciL1cConfig())
    )
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
