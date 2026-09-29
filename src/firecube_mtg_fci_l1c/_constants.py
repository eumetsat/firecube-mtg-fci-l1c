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

# Per-channel radiance conversion constants, keyed by LOGICAL channel name.
#
# Copied from the float32 scalars data/<channel>/measured/<name> of MTI1 FCI
# L1C products. They are static over time: the FDHSI products of 2026-09-03
# and 2026-09-28 carry identical values, and HRFI ``*_hr`` channels carry the
# same values as the FDHSI channel of the same band. Values are the exact
# float64 widening of the float32 source values. ``None`` marks a constant
# the source sets to _FillValue because it does not apply to the channel.
FCI_CONVERSION_CONSTANT_NAMES: tuple[str, ...] = (
    "radiance_unit_conversion_coefficient",
    "radiance_to_bt_conversion_coefficient_wavenumber",
    "radiance_to_bt_conversion_coefficient_a",
    "radiance_to_bt_conversion_coefficient_b",
    "radiance_to_bt_conversion_constant_c1",
    "radiance_to_bt_conversion_constant_c2",
    "channel_effective_solar_irradiance",
)

_BT_C1 = 1.1910429748240858e-05
_BT_C2 = 1.4387749433517456


def _vnir(unit_conversion: float, solar_irradiance: float) -> dict[str, float | None]:
    return {
        "radiance_unit_conversion_coefficient": unit_conversion,
        "radiance_to_bt_conversion_coefficient_wavenumber": None,
        "radiance_to_bt_conversion_coefficient_a": None,
        "radiance_to_bt_conversion_coefficient_b": None,
        "radiance_to_bt_conversion_constant_c1": None,
        "radiance_to_bt_conversion_constant_c2": None,
        "channel_effective_solar_irradiance": solar_irradiance,
    }


def _ir(
    unit_conversion: float,
    wavenumber: float,
    a: float,
    b: float,
    solar_irradiance: float | None = None,
) -> dict[str, float | None]:
    return {
        "radiance_unit_conversion_coefficient": unit_conversion,
        "radiance_to_bt_conversion_coefficient_wavenumber": wavenumber,
        "radiance_to_bt_conversion_coefficient_a": a,
        "radiance_to_bt_conversion_coefficient_b": b,
        "radiance_to_bt_conversion_constant_c1": _BT_C1,
        "radiance_to_bt_conversion_constant_c2": _BT_C2,
        "channel_effective_solar_irradiance": solar_irradiance,
    }


FCI_CONVERSION_CONSTANTS: dict[str, dict[str, float | None]] = {
    "vis_04": _vnir(50.265708923339844, 38.09846115112305),
    "vis_05": _vnir(38.74919891357422, 49.66482162475586),
    "vis_06": _vnir(24.666019439697266, 65.727783203125),
    "vis_08": _vnir(13.517550468444824, 71.42111206054688),
    "vis_09": _vnir(11.984620094299316, 72.5357437133789),
    "nir_13": _vnir(5.261388778686523, 67.4814453125),
    "nir_16": _vnir(3.8494789600372314, 61.78792190551758),
    "nir_22": _vnir(1.9657750129699707, 37.786476135253906),
    "ir_38": _ir(
        0.6971830129623413,
        2644.3291015625,
        0.9966715574264526,
        2.258397102355957,
        solar_irradiance=15.373283386230469,
    ),
    "wv_63": _ir(
        0.25366830825805664, 1603.5819091796875, 0.993518590927124, 3.4607369899749756
    ),
    "wv_73": _ir(
        0.18172499537467957, 1349.646484375, 0.9991138577461243, 0.4422863721847534
    ),
    "ir_87": _ir(
        0.13067220151424408, 1143.9576416015625, 0.9995022416114807, 0.2201579511165619
    ),
    "ir_97": _ir(
        0.10713479667901993,
        1035.3956298828125,
        0.9997828006744385,
        0.08755578100681305,
    ),
    "ir_105": _ir(
        0.08997274190187454, 949.9734497070312, 0.999032199382782, 0.36442750692367554
    ),
    "ir_123": _ir(
        0.06602230668067932, 813.1241455078125, 0.9995448589324951, 0.14765778183937073
    ),
    "ir_133": _ir(
        0.056650541722774506,
        753.2736206054688,
        0.9994872808456421,
        0.15411429107189178,
    ),
}


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
