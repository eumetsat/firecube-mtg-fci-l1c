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

"""Constants for MTG FCI Level 1C data processing.

FCI operates in two modes:
- FDHSI (Full Disk High Spectral Imagery): 16 channels, 10-min repeat cycle,
  1km VIS/NIR + 2km IR/WV.  Collection EO:EUM:DAT:0662.
- HRFI (High Resolution Fast Imagery): 4 high-resolution channels,
  500m VIS/NIR + 1km IR.  Collection EO:EUM:DAT:0665.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypedDict


class ResolutionInfo(TypedDict):
    """Per-resolution channel and grid configuration entry."""

    channels: list[str]
    dimsize: int
    nc_channels: list[str]


# String identifiers embedded in EUMETSAT ZIP filenames.
PRODUCT_TYPE_FDHSI = "FDHSI"
PRODUCT_TYPE_HRFI = "HRFI"


def get_nc_part_prefix(product_type: str) -> str:
    """Return the WMO-style filename prefix for BODY nc_parts."""
    return (
        "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-"
        f"{product_type}-FD--CHK-BODY---"
    )


# Per-product-type channel and grid configuration.
#
# Structure: CONSTANTS[product_type][resolution] -> dict with:
#   "channels"    - logical channel names used as coordinate labels in Zarr
#   "nc_channels" - NetCDF group names inside the FCI nc_part files.
#                   For HRFI these carry a "_hr" suffix (e.g. "vis_06_hr")
#                   because EUMETSAT uses separate group names for the
#                   high-resolution variants of the same spectral band.
#   "dimsize"     - full-disk detector dimension (pixels per side)
CONSTANTS: dict[str, dict[str, ResolutionInfo]] = {
    PRODUCT_TYPE_FDHSI: {
        "1km": {
            "channels": [
                "vis_04",
                "vis_05",
                "vis_06",
                "vis_08",
                "vis_09",
                "nir_13",
                "nir_16",
                "nir_22",
            ],
            "dimsize": 11136,
            "nc_channels": [
                "vis_04",
                "vis_05",
                "vis_06",
                "vis_08",
                "vis_09",
                "nir_13",
                "nir_16",
                "nir_22",
            ],
        },
        "2km": {
            "channels": [
                "ir_38",
                "wv_63",
                "wv_73",
                "ir_87",
                "ir_97",
                "ir_105",
                "ir_123",
                "ir_133",
            ],
            "dimsize": 5568,
            "nc_channels": [
                "ir_38",
                "wv_63",
                "wv_73",
                "ir_87",
                "ir_97",
                "ir_105",
                "ir_123",
                "ir_133",
            ],
        },
    },
    PRODUCT_TYPE_HRFI: {
        "500m": {
            "channels": ["vis_06", "nir_22"],
            "dimsize": 22272,
            "nc_channels": ["vis_06_hr", "nir_22_hr"],
        },
        "1km": {
            "channels": ["ir_38", "ir_105"],
            "dimsize": 11136,
            "nc_channels": ["ir_38_hr", "ir_105_hr"],
        },
    },
}

# Valid output resolutions per product type.
# FDHSI: 1km (VIS/NIR) + 2km (IR/WV).  HRFI: 500m (VIS/NIR) + 1km (IR).
VALID_RESOLUTIONS = {
    PRODUCT_TYPE_FDHSI: ["1km", "2km"],
    PRODUCT_TYPE_HRFI: ["500m", "1km"],
}

# Default Zarr chunk sizes per resolution (Y-dimension nc_part-aligned).
# These are the nc_part row-count defaults used in streaming.
CHUNK_DEFAULTS_BY_RESOLUTION: dict[str, int] = {
    "500m": 556,
    "1km": 278,
    "2km": 139,
}

# Number of BODY chunks (``..._CHK-BODY---`` nc_parts) in a full-disk product.
BODY_CHUNK_COUNT: dict[str, int] = {
    PRODUCT_TYPE_FDHSI: 40,
    PRODUCT_TYPE_HRFI: 40,
}

# Row span of each BODY chunk in the full-disk grid.
#
# BODY_CHUNK_ROWS[product_type][resolution][chunk - 1] -> (r0, r1): 0-based,
# half-open row range that BODY chunk number ``chunk`` (1-based) covers. Spans
# tile 0..dimsize without gap or overlap. Chunk heights are not uniform (the
# real products have a few taller and shorter chunks), so they cannot be derived
# from a fixed rows-per-chunk. Measured from real FCI L1C products: three FDHSI
# dates and one HRFI date.
BODY_CHUNK_ROWS: dict[str, dict[str, tuple[tuple[int, int], ...]]] = {
    "FDHSI": {
        "1km": (
            (0, 278),
            (278, 556),
            (556, 835),
            (835, 1113),
            (1113, 1392),
            (1392, 1670),
            (1670, 1948),
            (1948, 2227),
            (2227, 2505),
            (2505, 2784),
            (2784, 3062),
            (3062, 3340),
            (3340, 3619),
            (3619, 3897),
            (3897, 4176),
            (4176, 4454),
            (4454, 4732),
            (4732, 5011),
            (5011, 5289),
            (5289, 5568),
            (5568, 5846),
            (5846, 6124),
            (6124, 6403),
            (6403, 6681),
            (6681, 6960),
            (6960, 7258),
            (7258, 7556),
            (7556, 7856),
            (7856, 8133),
            (8133, 8391),
            (8391, 8649),
            (8649, 8908),
            (8908, 9187),
            (9187, 9465),
            (9465, 9744),
            (9744, 10022),
            (10022, 10300),
            (10300, 10579),
            (10579, 10857),
            (10857, 11136),
        ),
        "2km": (
            (0, 139),
            (139, 278),
            (278, 417),
            (417, 556),
            (556, 696),
            (696, 835),
            (835, 974),
            (974, 1113),
            (1113, 1252),
            (1252, 1392),
            (1392, 1531),
            (1531, 1670),
            (1670, 1809),
            (1809, 1948),
            (1948, 2088),
            (2088, 2227),
            (2227, 2366),
            (2366, 2505),
            (2505, 2644),
            (2644, 2784),
            (2784, 2923),
            (2923, 3062),
            (3062, 3201),
            (3201, 3340),
            (3340, 3480),
            (3480, 3629),
            (3629, 3778),
            (3778, 3928),
            (3928, 4066),
            (4066, 4195),
            (4195, 4324),
            (4324, 4454),
            (4454, 4593),
            (4593, 4732),
            (4732, 4872),
            (4872, 5011),
            (5011, 5150),
            (5150, 5289),
            (5289, 5428),
            (5428, 5568),
        ),
    },
    "HRFI": {
        "500m": (
            (0, 557),
            (557, 1113),
            (1113, 1671),
            (1671, 2227),
            (2227, 2785),
            (2785, 3341),
            (3341, 3897),
            (3897, 4455),
            (4455, 5011),
            (5011, 5569),
            (5569, 6125),
            (6125, 6681),
            (6681, 7239),
            (7239, 7795),
            (7795, 8353),
            (8353, 8909),
            (8909, 9465),
            (9465, 10023),
            (10023, 10579),
            (10579, 11137),
            (11137, 11693),
            (11693, 12249),
            (12249, 12807),
            (12807, 13363),
            (13363, 13921),
            (13921, 14517),
            (14517, 15113),
            (15113, 15714),
            (15714, 16267),
            (16267, 16783),
            (16783, 17299),
            (17299, 17817),
            (17817, 18375),
            (18375, 18931),
            (18931, 19489),
            (19489, 20045),
            (20045, 20601),
            (20601, 21159),
            (21159, 21715),
            (21715, 22272),
        ),
        "1km": (
            (0, 278),
            (278, 556),
            (556, 835),
            (835, 1113),
            (1113, 1392),
            (1392, 1670),
            (1670, 1948),
            (1948, 2227),
            (2227, 2505),
            (2505, 2784),
            (2784, 3062),
            (3062, 3340),
            (3340, 3619),
            (3619, 3897),
            (3897, 4176),
            (4176, 4454),
            (4454, 4732),
            (4732, 5011),
            (5011, 5289),
            (5289, 5568),
            (5568, 5846),
            (5846, 6124),
            (6124, 6403),
            (6403, 6681),
            (6681, 6960),
            (6960, 7258),
            (7258, 7556),
            (7556, 7856),
            (7856, 8133),
            (8133, 8391),
            (8391, 8649),
            (8649, 8908),
            (8908, 9187),
            (9187, 9465),
            (9465, 9744),
            (9744, 10022),
            (10022, 10300),
            (10300, 10579),
            (10579, 10857),
            (10857, 11136),
        ),
    },
}


def stripe_rows(
    product_type: str, resolution: str, first: int, last: int
) -> tuple[int, int]:
    """Return the ``(r0, r1)`` row span of BODY chunks ``first``..``last``.

    Chunk numbers are 1-based and inclusive; the result is 0-based half-open.
    Raises ``ValueError`` for an unknown product/resolution or a range that is
    reversed or outside ``1..BODY_CHUNK_COUNT[product_type]``.
    """
    by_resolution = BODY_CHUNK_ROWS.get(product_type)
    if by_resolution is None:
        raise ValueError(
            f"Unknown product type: {product_type!r}. "
            f"Expected one of {sorted(BODY_CHUNK_ROWS)}"
        )
    rows = by_resolution.get(resolution)
    if rows is None:
        raise ValueError(
            f"Unknown resolution {resolution!r} for {product_type}. "
            f"Expected one of {sorted(by_resolution)}"
        )
    if not 1 <= first <= last <= len(rows):
        raise ValueError(
            f"BODY chunk range [{first}, {last}] must satisfy "
            f"1 <= first <= last <= {len(rows)}"
        )
    return rows[first - 1][0], rows[last - 1][1]


# FCI geostationary projection angular sampling geometry.
#
# Derived from source NetCDF data/<channel>/measured/x and y coordinate
# attributes (scale_factor magnitude). Constants store POSITIVE magnitudes.
# _variables.py builds both axes as (index - (dimsize / 2 - 0.5)) * scale, which
# is symmetric around the sub-satellite point and matches the L1C files' own
# x/y for the corresponding 1-based packed column/row (the files use
# add_offset = (dimsize / 2 + 0.5) * |scale| per resolution). x is
# east-positive, y is north-positive. Multiply by
# MTG_PERSPECTIVE_POINT_HEIGHT_M to convert to projection metres (default
# projection_units="meter" mode).
FCI_PROJ_SCALE_RAD_PER_INDEX: dict[str, float] = {
    "500m": 1.39717881617e-05,
    "1km": 2.79435763233999e-05,
    "2km": 5.58871526468e-05,
}
# MTG geostationary perspective point height in metres.
# Matches the source NetCDF perspective_point_height attribute on MTG L1C.
# Static (not read per-scene) to avoid I/O; the value is fixed for MTG-I1.
# Used to convert projection angles (radians) to metres for CF-compliant
# projection_x_coordinate / projection_y_coordinate (default projection_units mode).
MTG_PERSPECTIVE_POINT_HEIGHT_M: float = 35786400.0
FCI_PROJ_SWEEP_AXIS: str = "y"

# Satellite position, read per product from index-dimensioned tables (one row
# per ~1 s of the repeat cycle) and stored as one (time,) value per slot: the
# mean over the repeat cycle. Within a cycle they vary by < 0.03 deg and 520 m.
# Maps output variable name -> netCDF path.
SLOT_GEOMETRY_SOURCES: dict[str, str] = {
    "subsatellite_latitude": "state/platform/subsatellite_latitude",
    "subsatellite_longitude": "state/platform/subsatellite_longitude",
    "platform_altitude": "state/platform/platform_altitude",
}

# Radiance conversion constants read per product from the float32 scalars
# data/<channel>/measured/<name>. Each becomes a (time, channel) array.
FCI_CONVERSION_CONSTANT_NAMES: tuple[str, ...] = (
    "radiance_unit_conversion_coefficient",
    "radiance_to_bt_conversion_coefficient_wavenumber",
    "radiance_to_bt_conversion_coefficient_a",
    "radiance_to_bt_conversion_coefficient_b",
    "radiance_to_bt_conversion_constant_c1",
    "radiance_to_bt_conversion_constant_c2",
    "channel_effective_solar_irradiance",
)

# Logical channels with dual-gain (cold/warm) calibration. Only these carry
# meaningful warm_scale_factor/warm_add_offset; the L1C files set them to 0.0
# for every other channel.
DUAL_GAIN_CHANNELS: frozenset[str] = frozenset({"ir_38"})


@lru_cache(maxsize=None)
def logical_channel_resolution_map(product_type: str) -> dict[str, str]:
    """Map LOGICAL channel name to resolution. User-facing (_variables.py, config.py)."""
    result: dict[str, str] = {}
    for resolution, info in CONSTANTS[product_type].items():
        for channel in info["channels"]:
            result[channel] = resolution
    return result


@lru_cache(maxsize=None)
def nc_channel_resolution_map(product_type: str) -> dict[str, str]:
    """Map NetCDF channel alias to resolution. Reader-facing (_decode.py)."""
    result: dict[str, str] = {}
    for resolution, info in CONSTANTS[product_type].items():
        for nc_channel in info["nc_channels"]:
            result[nc_channel] = resolution
    return result


def dimsize_for(product_type: str, resolution: str) -> int:
    """Return the detector dimension for a product/resolution, or 0 if unknown."""
    res_map = CONSTANTS.get(product_type)
    if res_map is None or resolution not in res_map:
        return 0
    return res_map[resolution]["dimsize"]


# EUMETSAT Data Store collection identifiers used for data download/discovery.
FCI_COLLECTION_IDS = {
    PRODUCT_TYPE_FDHSI: "EO:EUM:DAT:0662",
    PRODUCT_TYPE_HRFI: "EO:EUM:DAT:0665",
}

# FCI nominal repeat-cycle schedule. FDHSI (0662) and HRFI (0665) share it:
# a 10-minute full-disk cycle, 144 cycles per day, resetting at 00:00 UTC.
# EUMETSAT's own ``repeatCycleIdentifier`` is ``hour*6 + minute//10 + 1`` (1-based,
# 1..144); the plugin uses the 0-based form for Zarr slot indexing.
REPEAT_CYCLE_MINUTES = 10
REPEAT_CYCLES_PER_DAY = 144

# Earliest FCI L1C availability in the EUMETSAT Data Store (verified: zero
# products before this date for both collections). Used as the default anchor
# (day 0) for the deterministic time-slot index so every product maps to a
# stable, non-negative index regardless of ingest order or which pod writes it.
FCI_DATA_EPOCH = "2024-09-24"
