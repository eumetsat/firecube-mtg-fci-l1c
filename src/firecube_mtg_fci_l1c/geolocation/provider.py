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

"""Latitude/longitude grid provider for static geolocation intents."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np  # pyright: ignore[reportMissingImports]

from .projection import compute_latlon

# Resolution group suffix -> spatial sampling distance in metres.
_RES_TO_M: dict[str, int] = {"500m": 500, "1km": 1000, "2km": 2000}


class LatLonProvider:
    """Stateless lookup with per-(grids_file, resolution_m) cache.

    Concurrency: a single threading.Lock guards cache mutation.  The heavy
    NPZ load runs OUTSIDE the lock to keep contention short.
    """

    def __init__(self, logger: Any) -> None:
        self._log = logger
        self._cache: dict[tuple[str | None, int], tuple[np.ndarray, np.ndarray]] = {}
        self._lock = threading.Lock()

    def resolution_m_for_group(self, group: str) -> int | None:
        """Map a ``data_<res>`` group name to its sampling distance in metres."""
        return _RES_TO_M.get(group.removeprefix("data_"))

    def get_lat_lon(
        self, grids_file: str | None, resolution_m: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return float32 ``(lat, lon)`` grids; cached per input key.

        The heavy NPZ or compute step runs only once per unique key.
        """
        key = (grids_file, resolution_m)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Heavy work outside the lock — avoids holding it during I/O.
        if grids_file:
            from .grids import FciGrids

            loader = FciGrids(grids_file)
            try:
                lat, lon = loader.get_coordinates(resolution_m)
            except (FileNotFoundError, ValueError) as exc:
                self._log.warning(
                    "Failed to load grids from %s: %s. Falling back to on-the-fly "
                    "computation.",
                    grids_file,
                    exc,
                )
                lat, lon = compute_latlon(resolution_m)
        else:
            lat, lon = compute_latlon(resolution_m)

        lat32 = np.asarray(lat, dtype=np.float32)
        lon32 = np.asarray(lon, dtype=np.float32)
        with self._lock:
            # Tolerate concurrent population — last writer wins, same data.
            self._cache.setdefault(key, (lat32, lon32))
            return self._cache[key]
