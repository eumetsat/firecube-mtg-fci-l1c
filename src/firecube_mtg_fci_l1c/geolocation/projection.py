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

"""FCI geolocation: compute latitude/longitude grids from pixel positions.

Implements the inverse geostationary (GEOS) satellite projection for the
MTG FCI instrument. Converts (row, column) pixel indices on the FCI detector
grid into geographic (latitude, longitude) coordinates on Earth's surface.

The math follows the formulas in the FCI L1 Dataset User Guide [FCIL1DUG],
using refined constants from the EUMETSAT CONV document.

Note: This module is FCI-specific. The grid sampling angles, scan origin
offsets, and detector dimensions are all specific to the FCI instrument
on Meteosat Third Generation (MTG) satellites. Other instruments (e.g.
MSG/SEVIRI) use pre-computed lat/lon lookup tables instead.

Coordinate expectations:
    - Input pixel coordinates follow the FCI fixed-grid convention with 1-based
      detector indices used by the projection formulas.
    - Returned arrays use the FCI native ``[row, col]`` order: row 0 is the
      southern edge and col 0 is the western edge of the full-disk image,
      matching the row order of the L1C files and of the cube's ``y``/``x``.
    - Longitudes are normalized to ``[-180, 180)`` after optional
      ``sub_satellite_lon`` shifting.

Reference:
    EUMETSAT (2020), "MTG FCI Level 1c Data Format Familiarisation",
    https://www-cdn.eumetsat.int/files/2020-04/fci_level_1c_format_familiarisation.pdf
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# FCI detector grid constants (from EUMETSAT CONV document)
#
# Each FCI channel observes a fixed-size grid of pixels covering the full
# Earth disk. The detector scans East-to-West (columns) and South-to-North
# (rows). These constants define the angular spacing and starting position
# of the grid for each spatial sampling distance (SSD).
#
# SSD "0.5" = 500m channels, "1" = 1km channels, "2" = 2km channels.
# ---------------------------------------------------------------------------

# Angular step between adjacent pixels (degrees per pixel).
# Smaller values = finer resolution = more pixels covering the same angular range.
GRID_SAMPLING_DEG: dict[str, float] = {
    "0.5": 0.00080053344196,  # 500m: ~0.0008 deg/pixel
    "1": 0.0016010668837,  # 1km:  ~0.0016 deg/pixel
    "2": 0.0032021337647,  # 2km:  ~0.0032 deg/pixel
}

# Scan start angle for columns (East-West), in degrees.
# This is the angular offset of column 1 from the sub-satellite point.
# Positive = East of sub-satellite point.
COLUMN_ORIGIN_DEG: dict[str, float] = {
    "0.5": 8.9143401430,
    "1": 8.9139398750,
    "2": 8.9131393340,
}

# Scan start angle for rows (South-North), in degrees.
# This is the angular offset of row 1 from the sub-satellite point.
# Negative = South of sub-satellite point (scan goes upward).
ROW_ORIGIN_DEG: dict[str, float] = {
    "0.5": -8.9143401430,
    "1": -8.9139398750,
    "2": -8.9131393340,
}

# Full-disk detector size (pixels) per resolution.
# The FCI detector is square: dimsize x dimsize pixels.
DIMSIZE_BY_RESOLUTION: dict[int, int] = {
    500: 22272,  # 500m: 22272 x 22272 = ~496 million pixels
    1000: 11136,  # 1km:  11136 x 11136 = ~124 million pixels
    2000: 5568,  # 2km:   5568 x  5568 =  ~31 million pixels
}

# Maps resolution in meters to the SSD key used in the grid constant dicts.
_SSD_KEY: dict[int, str] = {500: "0.5", 1000: "1", 2000: "2"}


def compute_latlon(
    resolution_m: int,
    sub_satellite_lon: float = 0.0,
    h: float = 42164.537,
    r_eq: float = 6378.137,
    r_p: float = 6356.752314,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute full-disk FCI latitude/longitude arrays for a given resolution.

    Uses the inverse GEOS (geostationary) projection to convert each pixel
    position on the FCI detector into a (latitude, longitude) coordinate.

    The full disk is symmetric around the sub-satellite point, so we only
    compute one quadrant (top-right) and mirror it to the other three.
    This reduces both computation time and peak memory usage by 4x.

    Args:
        resolution_m: Pixel resolution in meters. Must be 500, 1000, or 2000.
        sub_satellite_lon: Sub-satellite longitude in degrees (0.0 for MTG-I1
            at 0 deg, will be different for MTG-I2 or other positions).
        h: Distance from Earth center to satellite in km (orbital radius).
            Default 42164.537 km for geostationary orbit.
        r_eq: Earth equatorial radius in km (WGS84: 6378.137).
        r_p: Earth polar radius in km (WGS84: 6356.752314).

    Returns:
        Tuple of (latitude, longitude) as float32 numpy arrays with shape
        (dimsize, dimsize). Off-Earth pixels are set to NaN.

        Arrays are indexed as ``[row, col]`` in FCI native order (south-to-north,
        west-to-east). Longitudes are in degrees east and wrapped to
        ``[-180, 180)``.

    Raises:
        ValueError: If resolution_m is not 500, 1000, or 2000.

    Example:
        >>> lat, lon = compute_latlon(2000)
        >>> lat.shape
        (5568, 5568)
        >>> lat[2784, 2784]  # Center pixel ~ sub-satellite point
        0.009...  # approximately 0 degrees latitude
    """
    if resolution_m not in DIMSIZE_BY_RESOLUTION:
        msg = f"resolution_m must be one of {sorted(DIMSIZE_BY_RESOLUTION)}, got {resolution_m}"
        raise ValueError(msg)

    dimsize = DIMSIZE_BY_RESOLUTION[resolution_m]
    ssd = _SSD_KEY[resolution_m]
    half = dimsize // 2

    # Convert grid constants from degrees to radians for trig operations
    grid_sampling_rad = np.deg2rad(GRID_SAMPLING_DEG[ssd])
    col_origin_rad = np.deg2rad(COLUMN_ORIGIN_DEG[ssd])
    row_origin_rad = np.deg2rad(ROW_ORIGIN_DEG[ssd])

    # Build 1D coordinate arrays for the top-right quadrant.
    # We compute rows 1..half (top half) and columns half+1..dimsize (right half).
    # The GEOS projection formulas use 1-based pixel indices.
    rows = np.arange(1, half + 1, dtype=np.float64)
    cols = np.arange(half + 1, dimsize + 1, dtype=np.float64)

    # Convert pixel indices to GEOS scanning angles (radians).
    # lambda_s = East-West angle for each column (decreases with column index)
    # phi_s = North-South angle for each row (increases with row index)
    lambda_s = col_origin_rad - (cols - 1.0) * grid_sampling_rad
    phi_s = row_origin_rad + (rows - 1.0) * grid_sampling_rad

    # --- Vectorized GEOS inverse projection ---
    # Broadcast to 2D: rows along axis 0, cols along axis 1
    cos_lambda = np.cos(lambda_s)[None, :]
    sin_lambda = np.sin(lambda_s)[None, :]
    cos_phi = np.cos(phi_s)[:, None]
    sin_phi = np.sin(phi_s)[:, None]

    cos_lambda_sq = cos_lambda * cos_lambda
    cos_phi_sq = cos_phi * cos_phi
    sin_phi_sq = sin_phi * sin_phi

    # Ellipsoid flattening ratio and orbit geometry
    s_4 = (r_eq / r_p) ** 2  # ~1.00674 — accounts for Earth's oblateness
    s_5 = h**2 - r_eq**2  # distance^2 from satellite to Earth equatorial surface

    # The discriminant determines whether a pixel sees Earth or empty space.
    # disc < 0 means the line of sight misses Earth entirely (off-disk pixel).
    denom = cos_phi_sq + s_4 * sin_phi_sq
    disc = (h**2) * (cos_lambda_sq * cos_phi_sq) - denom * s_5
    valid = disc >= 0.0

    # Clamp disc to 0 for off-Earth pixels to avoid sqrt of negative numbers.
    s_d = np.sqrt(np.maximum(disc, 0.0))

    # s_n = distance from satellite to the point on Earth's surface
    s_n = (h * cos_lambda * cos_phi - s_d) / denom

    # Cartesian coordinates of the surface point (satellite-centered frame)
    s_1 = h - s_n * cos_lambda * cos_phi  # X: along satellite-Earth line
    s_2 = -s_n * sin_lambda * cos_phi  # Y: East-West
    s_3 = s_n * sin_phi  # Z: North-South
    s_xy = np.sqrt(s_1 * s_1 + s_2 * s_2)  # horizontal distance in X-Y plane

    # Convert from Cartesian to geographic coordinates
    lon_q = np.degrees(np.arctan2(s_2, s_1))
    lat_q = np.degrees(np.arctan(s_4 * s_3 / s_xy))

    # Downcast to float32 and mask off-Earth pixels
    lon_q = lon_q.astype(np.float32, copy=False)
    lat_q = lat_q.astype(np.float32, copy=False)
    lon_q[~valid] = np.nan
    lat_q[~valid] = np.nan

    # --- Mirror the quadrant to build the full disk ---
    # The FCI full disk is symmetric around the sub-satellite point:
    #   - Top-right quadrant: computed above (lat_q, lon_q)
    #   - Top-left:  same latitudes, negated longitudes (mirror columns)
    #   - Bottom-right: negated latitudes, same longitudes (mirror rows)
    #   - Bottom-left: negated both (mirror rows and columns)
    latitude = np.empty((dimsize, dimsize), dtype=np.float32)
    longitude = np.empty((dimsize, dimsize), dtype=np.float32)

    latitude[:half, half:] = lat_q
    latitude[:half, :half] = lat_q[:, ::-1]
    latitude[half:, half:] = -lat_q[::-1, :]
    latitude[half:, :half] = -lat_q[::-1, ::-1]

    longitude[:half, half:] = lon_q
    longitude[:half, :half] = -lon_q[:, ::-1]
    longitude[half:, half:] = lon_q[::-1, :]
    longitude[half:, :half] = -lon_q[::-1, ::-1]

    # Shift longitudes if satellite is not at 0 degrees
    if sub_satellite_lon != 0.0:
        valid_lon = ~np.isnan(longitude)
        longitude[valid_lon] += np.float32(sub_satellite_lon)
        longitude[valid_lon] = ((longitude[valid_lon] + 180.0) % 360.0) - 180.0

    return latitude, longitude
