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

"""Declarative variable schema for MTG FCI L1C.

THIS FILE is the only one you need to edit to add or change a Zarr variable.

To add a variable:
  1. Define a module-level source function (below the existing ones)
  2. Append one ``Variable(...)`` entry to :data:`VARIABLES`
  3. That is all — the ingestor dispatches by ``dims`` shape automatically

Source functions MUST be module-level (never lambdas) so the list stays
picklable for ``ProcessPoolExecutor`` workers.

See ``docs/guides/add-zarr-variable.md`` for the decision table and examples.
"""

from __future__ import annotations

import numpy as np  # pyright: ignore[reportMissingImports]

from . import _schema
from ._constants import (
    FCI_PROJ_SCALE_RAD_PER_INDEX,
    MTG_PERSPECTIVE_POINT_HEIGHT_M,
)
from .config import MtgFciL1cConfig
from ._schema import Variable, VariableContext, variable_enabled

# Re-exported so that existing ``from ._variables import Variable, VariableContext``
# imports continue to work without change.
__all__ = [
    "Variable",
    "VariableContext",
    "variable_enabled",
    "build_specs",
    "build_all_specs",
    "VARIABLES",
    "TIME_COORD_NAME",
]

# Time coordinate name. Wired to the ingestor's time_dim_name. Change here only.
TIME_COORD_NAME = "time"

# WKT 2 (ISO 19162) string for the MTG geostationary CRS.
# Written to both ``crs_wkt`` (read by rioxarray < GDAL 4) and ``spatial_ref``
# (read by GDAL 4+ / modern rioxarray) attributes of the ``spatial_ref``
# grid-mapping container so the projection round-trips through rasterio/GDAL.
_MTG_GEOS_WKT: str = (
    'PROJCRS["MTG Geostationary",'
    'BASEGEOGCRS["WGS 84",'
    'DATUM["World Geodetic System 1984",'
    'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]]],'
    'CONVERSION["MTG Geostationary",'
    'METHOD["Geostationary Satellite (Sweep Y)"],'
    'PARAMETER["Longitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433]],'
    'PARAMETER["Satellite Height",35786400,LENGTHUNIT["metre",1]]],'
    "CS[Cartesian,2],"
    'AXIS["easting (X)",east,ORDER[1],LENGTHUNIT["metre",1]],'
    'AXIS["northing (Y)",north,ORDER[2],LENGTHUNIT["metre",1]]]'
)

assert f'Satellite Height",{int(MTG_PERSPECTIVE_POINT_HEIGHT_M)},' in _MTG_GEOS_WKT, (
    "WKT satellite height does not match MTG_PERSPECTIVE_POINT_HEIGHT_M"
)


# ---------------------------------------------------------------------------
# Source functions
# ---------------------------------------------------------------------------
# Each source function receives a VariableContext and returns an ndarray (or
# None to skip). They MUST be module-level — no lambdas — for pickle-safety.
#
# When you add a variable, define its source function here first, then
# reference it in the Variable(...) entry in VARIABLES below.
# ---------------------------------------------------------------------------


def _counts_source(ctx: VariableContext) -> np.ndarray | None:
    return ctx.channel_payload.counts if ctx.channel_payload else None


def _pixel_quality_source(ctx: VariableContext) -> np.ndarray | None:
    return ctx.channel_payload.pixel_quality if ctx.channel_payload else None


def _pixel_time_source(ctx: VariableContext) -> np.ndarray | None:
    return ctx.channel_payload.pixel_time if ctx.channel_payload else None


def _geo_coordinates_resolver(config: MtgFciL1cConfig) -> dict[str, str]:
    if config.include_geolocation:
        return {"coordinates": "latitude longitude"}
    return {}


def _slope_source(ctx: VariableContext) -> np.ndarray | None:
    if ctx.calibration_table is None:
        return None
    vec = np.zeros(ctx.n_channels, dtype=np.float64)
    found = False
    for i, ch in enumerate(ctx.nc_channels):
        cal = ctx.calibration_table.get(ch)
        if cal is not None:
            vec[i] = cal[0]
            found = True
    return vec if found else None


def _offset_source(ctx: VariableContext) -> np.ndarray | None:
    if ctx.calibration_table is None:
        return None
    vec = np.zeros(ctx.n_channels, dtype=np.float64)
    found = False
    for i, ch in enumerate(ctx.nc_channels):
        cal = ctx.calibration_table.get(ch)
        if cal is not None:
            vec[i] = cal[1]
            found = True
    return vec if found else None


def _latitude_source(ctx: VariableContext) -> np.ndarray | None:
    if ctx.geo_provider is None:
        return None
    res_m = ctx.geo_provider.resolution_m_for_group(ctx.group)
    if res_m is None:
        return None
    lat, _lon = ctx.geo_provider.get_lat_lon(ctx.config.fci_grids_file, res_m)
    return lat


def _longitude_source(ctx: VariableContext) -> np.ndarray | None:
    if ctx.geo_provider is None:
        return None
    res_m = ctx.geo_provider.resolution_m_for_group(ctx.group)
    if res_m is None:
        return None
    _lat, lon = ctx.geo_provider.get_lat_lon(ctx.config.fci_grids_file, res_m)
    return lon


def _projection_angle_source(ctx: VariableContext) -> np.ndarray | None:
    """Return the geos scan angle (radians) or distance (metres) of each pixel centre.

    Shared by ``x`` and ``y``: the FCI fixed grid is square and symmetric around
    the sub-satellite point, so index ``i`` on either axis sits at
    ``(i - (dimsize / 2 - 0.5)) * scale``. This equals the L1C files' own
    ``x``/``y`` for packed column/row ``i + 1`` (their ``add_offset`` is
    ``(dimsize / 2 + 0.5) * |scale|`` per resolution), negated for ``x``
    because the files store ``x`` positive-westward while the cube is
    east-positive.
    """
    res = ctx.group.removeprefix("data_")
    if res not in FCI_PROJ_SCALE_RAD_PER_INDEX:
        return None
    scale = FCI_PROJ_SCALE_RAD_PER_INDEX[res]
    centre = ctx.dimsize / 2 - 0.5
    rad = (np.arange(ctx.dimsize, dtype=np.float64) - centre) * scale
    if ctx.config.projection_units in ("meter", "metre"):
        return rad * MTG_PERSPECTIVE_POINT_HEIGHT_M
    return rad


def _projection_x_source(ctx: VariableContext) -> np.ndarray | None:
    return _projection_angle_source(ctx)


def _projection_y_source(ctx: VariableContext) -> np.ndarray | None:
    return _projection_angle_source(ctx)


def _time_source(ctx: VariableContext) -> None:
    # Timestamp writes handled by WriteIntent kind="timestamp"; source returns None.
    return None


def _channel_name_source(ctx: VariableContext) -> np.ndarray:
    """Return channel names as a fixed-width byte string array."""
    return np.asarray(ctx.logical_channels, dtype="S16")


def _projection_x_attrs(config: MtgFciL1cConfig) -> dict[str, str]:
    if config.projection_units in ("meter", "metre"):
        return {
            "standard_name": "projection_x_coordinate",
            "long_name": "x coordinate of projection",
            "units": "m",
        }
    return {
        "standard_name": "projection_x_angular_coordinate",
        "long_name": "MTG geostationary projection x angle",
        "units": "radian",
    }


def _projection_y_attrs(config: MtgFciL1cConfig) -> dict[str, str]:
    if config.projection_units in ("meter", "metre"):
        return {
            "standard_name": "projection_y_coordinate",
            "long_name": "y coordinate of projection",
            "units": "m",
        }
    return {
        "standard_name": "projection_y_angular_coordinate",
        "long_name": "MTG geostationary projection y angle",
        "units": "radian",
    }


# ---------------------------------------------------------------------------
# EXAMPLE (commented out): add a new time-channel variable
# ---------------------------------------------------------------------------
# def _calibration_coefficients_source(ctx: VariableContext) -> np.ndarray | None:
#     """Combined slope+offset as a single (channel, 2) array per timestamp."""
#     if ctx.calibration_table is None:
#         return None
#     vec = np.zeros((ctx.n_channels, 2), dtype=np.float64)
#     for i, ch in enumerate(ctx.nc_channels):
#         cal = ctx.calibration_table.get(ch)
#         if cal is not None:
#             vec[i, 0] = cal[0]  # slope
#             vec[i, 1] = cal[1]  # offset
#     return vec
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# VARIABLES — the declarative registry
# ---------------------------------------------------------------------------
# Append one Variable(...) entry here to add a new array to the cube.
# Remove an entry (or set enabled_by=...) to disable one.
#
# Field reference:
#   name        str                  — Zarr array name inside the group
#   dims        tuple[str, ...]      — determines shape + dispatch:
#                  ("time","y","x","channel")  → 4-D spatial (sharded)
#                  ("time","channel")          → 2-D calibration scalar
#                  ("time",)                   → 1-D timestamp coord
#                  ("y","x")                   → 2-D static grid (lat/lon)
#                  ("channel",)                → 1-D static coord
#                  ()                          → 0-D scalar (attrs only)
#   dtype       np.dtype | str       — Zarr storage dtype
#   fill_value  Any                  — written for missing/masked elements
#   attrs       dict | None          — CF-convention metadata (optional)
#   source      Callable | None      — module-level function returning ndarray
#   enabled_by  str | None           — name of a MtgFciL1cConfig bool flag
# ---------------------------------------------------------------------------

# VARIABLE ATTRIBUTES
# Keep array metadata close to each Variable declaration. Avoid reserved Zarr /
# Firecube keys such as _ARRAY_DIMENSIONS, _FillValue, firecube_run_id,
# firecube_span_id, and firecube_internal; those are owned by the writer.

VARIABLES: list[Variable] = [
    Variable(
        name="counts",
        dims=(TIME_COORD_NAME, "y", "x", "channel"),
        dtype=np.uint16,
        fill_value=np.iinfo(np.uint16).max,
        attrs={
            "units": "1",
            "long_name": "FCI raw detector counts",
            "grid_mapping": "spatial_ref",
            "ancillary_variables": "pixel_quality pixel_time",
        },
        source=_counts_source,
        attrs_resolver=_geo_coordinates_resolver,
    ),
    Variable(
        name="pixel_quality",
        dims=(TIME_COORD_NAME, "y", "x", "channel"),
        dtype=np.uint8,
        fill_value=0,
        attrs={
            "units": "1",
            "long_name": "FCI pixel quality flags",
            "grid_mapping": "spatial_ref",
            "flag_masks": [1, 2, 4, 8, 16, 32, 64, 128],
            "flag_meanings": (
                "missing_warning radiometric_warning noise_warning "
                "geolocation_warning saturation_warning "
                "straylight_correction_warning extended_dynamic_range_warning "
                "encoding_saturation_warning"
            ),
        },
        source=_pixel_quality_source,
        enabled_by="include_pixel_quality",
        attrs_resolver=_geo_coordinates_resolver,
    ),
    Variable(
        name="pixel_time",
        dims=(TIME_COORD_NAME, "y", "x", "channel"),
        dtype=np.float64,
        fill_value=np.nan,
        attrs={
            "units": "seconds since 2000-01-01",
            "long_name": "FCI per-pixel observation time",
            "grid_mapping": "spatial_ref",
            "standard_name": "time",
            "calendar": "standard",
        },
        source=_pixel_time_source,
        enabled_by="include_pixel_time",
        attrs_resolver=_geo_coordinates_resolver,
    ),
    Variable(
        name="slope",
        dims=(TIME_COORD_NAME, "channel"),
        dtype=np.float64,
        fill_value=np.nan,
        attrs={"units": "mW m-2 sr-1 (cm-1)-1", "long_name": "FCI calibration slope"},
        source=_slope_source,
        enabled_by="include_calibration",
    ),
    Variable(
        name="offset",
        dims=(TIME_COORD_NAME, "channel"),
        dtype=np.float64,
        fill_value=np.nan,
        attrs={"units": "mW m-2 sr-1 (cm-1)-1", "long_name": "FCI calibration offset"},
        source=_offset_source,
        enabled_by="include_calibration",
    ),
    Variable(
        name="latitude",
        dims=("y", "x"),
        dtype=np.float32,
        fill_value=np.float32(np.nan),
        attrs={
            "units": "degrees_north",
            "standard_name": "latitude",
            "long_name": "latitude",
        },
        source=_latitude_source,
        enabled_by="include_geolocation",
    ),
    Variable(
        name="longitude",
        dims=("y", "x"),
        dtype=np.float32,
        fill_value=np.float32(np.nan),
        attrs={
            "units": "degrees_east",
            "standard_name": "longitude",
            "long_name": "longitude",
        },
        source=_longitude_source,
        enabled_by="include_geolocation",
    ),
    Variable(
        name="x",
        dims=("x",),
        dtype=np.float64,
        fill_value=None,
        attrs={"axis": "X"},
        source=_projection_x_source,
        attrs_resolver=_projection_x_attrs,
    ),
    Variable(
        name="y",
        dims=("y",),
        dtype=np.float64,
        fill_value=None,
        attrs={"axis": "Y"},
        source=_projection_y_source,
        attrs_resolver=_projection_y_attrs,
    ),
    Variable(
        name=TIME_COORD_NAME,
        dims=(TIME_COORD_NAME,),
        dtype="datetime64[s]",
        fill_value=np.datetime64("NaT", "s"),
        attrs={
            "standard_name": "time",
            "long_name": "observation time",
            "axis": "T",
            # NOTE (issue #3): 'units' and 'calendar' intentionally NOT set here.
            # The on-disk dtype is native datetime64[s]; xarray manages CF units and
            # calendar through the encoding channel, not attrs. Writing them into attrs
            # causes a collision on xr.open_zarr(store).to_zarr(new_store) because
            # xarray refuses to overwrite existing attrs with encoding values.
        },
        source=_time_source,
    ),
    Variable(
        name="channel_name",
        dims=("channel",),
        dtype="S16",
        fill_value=b"",
        attrs={
            "long_name": "FCI logical channel name",
            # CF-1.8 §3.1: units required on data variables; "1" is the
            # conventional placeholder for dimensionless / label quantities.
            "units": "1",
        },
        source=_channel_name_source,
    ),
    Variable(
        name="spatial_ref",
        dims=(),
        dtype=np.int8,
        fill_value=0,
        attrs={
            "long_name": "MTG geostationary projection",
            "grid_mapping_name": "geostationary",
            "latitude_of_projection_origin": 0.0,
            "longitude_of_projection_origin": 0.0,
            "perspective_point_height": MTG_PERSPECTIVE_POINT_HEIGHT_M,
            "sweep_angle_axis": "y",
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.31424518,
            "inverse_flattening": 298.25722356301,
            "crs_wkt": _MTG_GEOS_WKT,
            "spatial_ref": _MTG_GEOS_WKT,
        },
        # Scalar grid-mapping container: CF attrs only, no data payload.
    ),
    # -----------------------------------------------------------------------
    # EXAMPLE (commented out): combined calibration coefficients
    # -----------------------------------------------------------------------
    # Variable(
    #     name="calibration_coefficients",
    #     dims=(TIME_COORD_NAME, "channel"),
    #     dtype=np.float64,
    #     fill_value=np.nan,
    #     attrs={
    #         "units": "mW m-2 sr-1 (cm-1)-1",
    #         "long_name": "slope and offset coefficients",
    #     },
    #     source=_calibration_coefficients_source,
    #     enabled_by="include_calibration",
    # ),
    # -----------------------------------------------------------------------
]

build_specs = build_all_specs = _schema.build_specs
