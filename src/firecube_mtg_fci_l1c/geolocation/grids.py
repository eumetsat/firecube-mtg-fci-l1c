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

"""Loader for pre-generated FCI latitude/longitude grids in ``.npz`` files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Resolution in meters -> NPZ array key prefix.
_RES_KEY = {500: "500m", 1000: "1km", 2000: "2km"}


class FciGrids:
    """Lazy NPZ-backed loader for full-disk FCI coordinate grids."""

    def __init__(self, grids_file: str | Path) -> None:
        """Initialize from a file produced by ``geo generate``."""
        self.grids_file = Path(grids_file)
        self._data: np.lib.npyio.NpzFile | None = None
        self._metadata: dict | None = None
        # Cache decompressed arrays by resolution_m so repeated calls in the
        # same process don't re-decompress the zip each time.
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _load(self) -> np.lib.npyio.NpzFile:
        """Open the NPZ archive and parse optional metadata once."""
        data = self._data
        if data is None:
            if not self.grids_file.exists():
                raise FileNotFoundError(
                    f"FCI grids file not found: {self.grids_file}\n"
                    "Generate it first with:\n"
                    "  firecube plugins mtg_fci_l1c geo generate "
                    f"--output {self.grids_file}"
                )
            data = np.load(self.grids_file)
            self._data = data
            if "_metadata" in data:
                self._metadata = json.loads(str(data["_metadata"][0]))
            else:
                self._metadata = {}
        return data

    def available_resolutions(self) -> list[int]:
        """Return list of resolution values (in meters) available in the file."""
        data = self._load()
        return sorted(res_m for res_m, key in _RES_KEY.items() if f"{key}_lat" in data)

    def get_coordinates(self, resolution_m: int) -> tuple[np.ndarray, np.ndarray]:
        """Return cached ``(lat, lon)`` arrays for a resolution in meters."""
        if resolution_m in self._cache:
            return self._cache[resolution_m]
        data = self._load()
        key = _RES_KEY.get(resolution_m)
        if key is None or f"{key}_lat" not in data:
            available = self.available_resolutions()
            raise ValueError(
                f"Resolution {resolution_m}m not available in {self.grids_file}. "
                f"Available: {[str(r) + 'm' for r in available]}. "
                "Regenerate with the desired --resolutions."
            )
        lat, lon = data[f"{key}_lat"], data[f"{key}_lon"]
        self._cache[resolution_m] = (lat, lon)
        return lat, lon

    def get_metadata(self) -> dict:
        """Return metadata dict stored in the NPZ file."""
        self._load()
        return dict(self._metadata or {})
