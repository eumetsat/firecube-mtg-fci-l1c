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

"""Streaming helpers for DirectZarr nc_part processing.

This module provides FCI-specific helpers used by the streaming ingest path.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import h5netcdf  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]

from ._constants import CONSTANTS, get_nc_part_prefix, nc_channel_resolution_map

_PART_NUMBER_RE = re.compile(r"(\d+)\.nc$", re.IGNORECASE)

class NCPartReader:
    """Read one extracted nc_part file lazily and per-channel."""

    def __init__(self, nc_part_path: Path):
        """Store the source file path.

        Args:
            nc_part_path: Path to one extracted nc_part NetCDF file.
        """
        self._path = Path(nc_part_path)
        self._ds: h5netcdf.File | None = None

    def __enter__(self) -> NCPartReader:
        """Open the NetCDF file and return this reader instance."""
        self._open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close the NetCDF file when leaving context manager scope."""
        self.close()

    def _open(self) -> h5netcdf.File:
        """Open the underlying file on first access.

        Returns:
            Open ``h5netcdf.File`` handle.
        """
        if self._ds is None:
            self._ds = h5netcdf.File(self._path, mode="r")
        return self._ds

    @property
    def path(self) -> Path:
        """Return the path backing this reader."""
        return self._path

    def available_channels(self) -> list[str]:
        """List channel group names present in this nc_part.

        Returns:
            Sorted list of channel names under root ``data`` group.
        """
        ds = self._open()
        return sorted(ds["data"].groups)

    def _channel_resolution_map(self) -> dict[str, str]:
        """Build nc-channel-to-resolution mapping from cached constants helpers."""
        mapping: dict[str, str] = {}
        for product_type in CONSTANTS:
            for nc_channel, resolution in nc_channel_resolution_map(product_type).items():
                mapping.setdefault(nc_channel, resolution)
        return mapping

    @staticmethod
    def _as_int(value: Any) -> int:
        """Normalize scalar-like HDF5 values to ``int``."""
        return int(np.asarray(value).item())

    @classmethod
    def _read_row_bounds(cls, measured_group: Any) -> tuple[int, int]:
        """Return zero-based/exclusive row bounds from attrs or fallback datasets."""
        effective_radiance = measured_group["effective_radiance"]
        attrs = effective_radiance.attrs
        if "start_position_row" in attrs and "end_position_row" in attrs:
            start_row = cls._as_int(attrs["start_position_row"]) - 1
            end_row = cls._as_int(attrs["end_position_row"])
            return start_row, end_row

        if (
            "start_position_row" in measured_group
            and "end_position_row" in measured_group
        ):
            start_row = cls._as_int(measured_group["start_position_row"][...]) - 1
            end_row = cls._as_int(measured_group["end_position_row"][...])
            return start_row, end_row

        raise KeyError("No start/end row metadata found in measured group")

    def read_row_range(self, resolution: str) -> tuple[int, int]:
        """Return zero-based/exclusive row bounds for one resolution."""
        ds = self._open()
        channel_to_res = self._channel_resolution_map()
        for channel in self.available_channels():
            if channel_to_res.get(channel) != resolution:
                continue
            return self._read_row_bounds(ds[f"data/{channel}/measured"])
        raise KeyError(
            f"No channel found for resolution {resolution!r} in {self._path.name}"
        )

    def read_channel_data(
        self, channel: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(effective_radiance, pixel_quality, index_map)`` arrays."""
        ds = self._open()
        measured = ds[f"data/{channel}/measured"]
        effective_radiance = np.asarray(measured["effective_radiance"][...])
        pixel_quality = np.asarray(measured["pixel_quality"][...])
        index_map = np.asarray(measured["index_map"][...])
        return effective_radiance, pixel_quality, index_map

    def has_time_map(self) -> bool:
        """Return whether this nc_part exposes root-level ``index``/``time``."""
        ds = self._open()
        return "index" in ds.variables and "time" in ds.variables

    def read_time_map(self) -> dict[int, float]:
        """Build root-level index-to-time mapping.

        Returns:
            Mapping from integer ``index`` values to ``time`` values.
            Returns an empty mapping for trailer-like nc_parts that do not
            expose root-level time tables.
        """
        if not self.has_time_map():
            return {}

        ds = self._open()
        index_values = np.asarray(ds["index"][:])
        time_values = np.asarray(ds["time"][:])
        return {
            int(index): float(time)
            for index, time in zip(index_values, time_values, strict=True)
        }

    def read_calibration(self, channel: str) -> tuple[float, float] | None:
        """Return ``(scale_factor, add_offset)`` for a channel, if present."""
        ds = self._open()
        if "data" not in ds.groups or channel not in ds["data"].groups:
            return None

        ch_group = ds[f"data/{channel}"]
        if "measured" not in ch_group.groups:
            return None

        measured = ch_group["measured"]
        if "effective_radiance" not in measured.variables:
            return None

        radiance = measured["effective_radiance"]
        attrs = radiance.attrs
        if "scale_factor" not in attrs or "add_offset" not in attrs:
            return None

        return (
            float(np.asarray(attrs["scale_factor"]).item()),
            float(np.asarray(attrs["add_offset"]).item()),
        )

    def close(self) -> None:
        """Close the underlying NetCDF file if currently open."""
        if self._ds is not None:
            self._ds.close()
            self._ds = None


@dataclasses.dataclass(frozen=True)
class ChannelSlicePayload:
    """Pre-loaded arrays for one (nc_part, nc_channel) slice.

    ``pixel_time`` is None when ``include_pixel_time`` is False at config
    time. When present, it is already cast to the configured dtype.
    """

    counts: np.ndarray
    pixel_quality: np.ndarray
    pixel_time: np.ndarray | None


def load_channel_slice(
    reader: NCPartReader,
    nc_channel: str,
    index2time: dict[int, float] | None,
    pixel_time_dtype: np.dtype,
) -> ChannelSlicePayload:
    """Load one (nc_part, nc_channel) slice into a ChannelSlicePayload.

    Calls ``read_channel_data`` exactly once. ``pixel_time`` is expanded
    from the index_map only when ``index2time`` is non-empty.
    """
    counts, quality, index_map = reader.read_channel_data(nc_channel)
    pixel_time: np.ndarray | None = None
    if index2time:
        pixel_time = expand_pixel_time(index_map, index2time, pixel_time_dtype)
    return ChannelSlicePayload(counts=counts, pixel_quality=quality, pixel_time=pixel_time)


class TimeMapAccumulator:
    """Accumulate root-level ``index``/``time`` values across nc_parts."""

    def __init__(self) -> None:
        """Initialize empty index-to-time mapping state."""
        self._index2time: dict[int, float] = {}

    def accumulate(self, reader: NCPartReader) -> None:
        """Merge one nc_part root-level time mapping into accumulator state."""
        self._index2time.update(reader.read_time_map())

    def build_index2time(self) -> dict[int, float]:
        """Return a copy of the accumulated ``index -> time`` mapping."""
        return dict(self._index2time)
def expand_pixel_time(
    index_map: np.ndarray,
    index2time: dict[int, float],
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Expand integer ``index_map`` to float pixel times, using NaN for misses."""
    output_dtype = np.dtype(dtype)
    fill_value = output_dtype.type(np.nan)
    pixel_time = np.full(index_map.shape, fill_value, dtype=output_dtype)

    if not index2time:
        return pixel_time

    max_index = max(index2time)
    lookup = np.full(max_index + 1, fill_value, dtype=output_dtype)
    mapping_indices = np.fromiter(
        index2time.keys(), dtype=np.int64, count=len(index2time)
    )
    mapping_values = np.fromiter(
        index2time.values(),
        dtype=output_dtype,
        count=len(index2time),
    )
    lookup[mapping_indices] = mapping_values

    valid = (index_map >= 0) & (index_map <= max_index)
    if np.any(valid):
        pixel_time[valid] = lookup[index_map[valid]]
    return pixel_time


def list_fci_nc_parts(extracted_dir: Path) -> list[Path]:
    """List FCI BODY/TRAIL nc_parts in an extracted ZIP directory.

    BODY parts come first (sorted by numeric part number, then by filename),
    followed by TRAIL parts. Only files matching the FCI BODY/TRAIL naming
    convention (built from ``get_nc_part_prefix(product_type)`` and its TRAIL
    counterpart) are returned; other ``.nc`` members are ignored.
    """
    prefix_pairs: list[tuple[str, str]] = []
    for product_type in CONSTANTS:
        body = get_nc_part_prefix(product_type)
        trail = body.replace("BODY", "TRAIL")
        prefix_pairs.append((body, trail))

    parts: list[tuple[Path, int]] = []
    for path in Path(extracted_dir).rglob("*.nc"):
        name = path.name
        classification: int | None = None
        for body_prefix, trail_prefix in prefix_pairs:
            if name.startswith(body_prefix):
                classification = 0
                break
            if name.startswith(trail_prefix):
                classification = 1
                break

        if classification is None:
            continue
        parts.append((path, classification))

    def _part_number(path: Path) -> int:
        match = _PART_NUMBER_RE.search(path.name)
        if match is None:
            return 0
        return int(match.group(1))

    sorted_parts = sorted(
        parts,
        key=lambda item: (item[1], _part_number(item[0]), item[0].name),
    )
    return [item[0] for item in sorted_parts]
