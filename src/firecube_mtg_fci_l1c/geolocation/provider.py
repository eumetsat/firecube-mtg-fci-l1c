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

# Resolution -> spatial sampling distance in metres.
_RES_TO_M: dict[str, int] = {"500m": 500, "1km": 1000, "2km": 2000}


class LatLonProvider:
    """Stateless lookup with per-(grids_file, resolution_m, rows) cache.

    Concurrency: a single threading.Lock guards cache mutation.  The heavy
    NPZ load runs OUTSIDE the lock to keep contention short.
    """

    def __init__(self, logger: Any) -> None:
        self._log = logger
        self._cache: dict[
            tuple[str | None, int, tuple[int, int] | None],
            tuple[np.ndarray, np.ndarray],
        ] = {}
        self._lock = threading.Lock()

    def resolution_m(self, resolution: str) -> int | None:
        """Map a resolution such as ``"1km"`` to its sampling distance in metres."""
        return _RES_TO_M.get(resolution)

    def get_lat_lon(
        self,
        grids_file: str | None,
        resolution_m: int,
        rows: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return float32 ``(lat, lon)`` grids; cached per input key.

        ``rows=(start, stop)`` returns full-disk rows ``start:stop`` (all
        columns) and caches only those rows: the full-disk pair is loaded or
        computed, copied from, and released, unless a full-disk call already
        cached it. Without ``rows`` the full-disk pair is returned and cached.
        The heavy NPZ or compute step runs only once per unique key.
        """
        key = (grids_file, resolution_m, rows)
        with self._lock:
            cached = self._cache.get(key)
            full = self._cache.get((grids_file, resolution_m, None))
        if cached is not None:
            return cached
        if full is None:
            full = self._load_full(grids_file, resolution_m)
        if rows is None:
            pair = full
        else:
            start, stop = rows
            n_rows = full[0].shape[0]
            if not 0 <= start < stop <= n_rows:
                raise ValueError(
                    f"rows {rows!r} must satisfy 0 <= start < stop <= {n_rows}"
                )
            # Copies own their memory. Each full grid is dropped right after
            # its rows are copied, so at most one slice coexists with the
            # full pair, and nothing full-disk outlives this call.
            lat_full, lon_full = full
            del full
            lat = lat_full[start:stop].copy()
            del lat_full
            lon = lon_full[start:stop].copy()
            del lon_full
            pair = (lat, lon)
        with self._lock:
            # Tolerate concurrent population — last writer wins, same data.
            self._cache.setdefault(key, pair)
            return self._cache[key]

    def _load_full(
        self, grids_file: str | None, resolution_m: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load or compute the full-disk float32 ``(lat, lon)`` pair; not cached."""
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

        return np.asarray(lat, dtype=np.float32), np.asarray(lon, dtype=np.float32)
