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

# FCI geostationary projection angular sampling geometry.
#
# These values are derived from the source NetCDF `data/<channel>/measured/x`
# and `y` coordinate attributes (`scale_factor`, `add_offset`) in the FCI L1C
# `nc_part` files. The constants intentionally store POSITIVE magnitudes only:
# the x scale is applied as NEGATIVE (column 0 east-of-nadir), the y scale is
# POSITIVE, and the offsets are mirrored (x offset positive, y offset
# negative). Sign application happens in schema.py when the projection inputs
# are assembled.
FCI_PROJ_SCALE_RAD_PER_INDEX: dict[str, float] = {
    "500m": 1.39717881617e-05,
    "1km": 2.79435763233999e-05,
    "2km": 5.58871526468e-05,
}
FCI_PROJ_OFFSET_RAD: float = 0.1556038047568524
FCI_PROJ_SWEEP_AXIS: str = "y"


@lru_cache(maxsize=None)
def logical_channel_resolution_map(product_type: str) -> dict[str, str]:
    """Map LOGICAL channel name to resolution. User-facing (schema.py, config.py)."""
    result: dict[str, str] = {}
    for resolution, info in CONSTANTS[product_type].items():
        for channel in info["channels"]:
            result[channel] = resolution
    return result


@lru_cache(maxsize=None)
def nc_channel_resolution_map(product_type: str) -> dict[str, str]:
    """Map NetCDF channel alias to resolution. Reader-facing (_streaming.py)."""
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
