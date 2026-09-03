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
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5netcdf  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]

from ._constants import CONSTANTS, get_nc_part_prefix, nc_channel_resolution_map

_PART_NUMBER_RE = re.compile(r"(\d+)\.nc$", re.IGNORECASE)
_BOUNDED_UNSIGNED_DTYPES = frozenset(
    {np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32")}
)


def _read_variable_array(variable: Any) -> np.ndarray:
    """Read a h5netcdf variable into one NumPy allocation when possible."""
    out = np.empty(variable.shape, dtype=variable.dtype)
    h5_dataset = getattr(variable, "_h5ds", None)
    if h5_dataset is not None and hasattr(h5_dataset, "read_direct"):
        h5_dataset.read_direct(out)
        return out
    return np.asarray(variable)


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
            for nc_channel, resolution in nc_channel_resolution_map(
                product_type
            ).items():
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
        effective_radiance = _read_variable_array(measured["effective_radiance"])
        pixel_quality = _read_variable_array(measured["pixel_quality"])
        index_map = _read_variable_array(measured["index_map"])
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

    Read-only contract: a cached payload may be returned to multiple
    consumers within a single batch. Consumers MUST treat arrays as
    read-only. Mutating a returned array in place is undefined behavior
    and may silently corrupt other consumers reading the same cached
    slice.
    """

    counts: np.ndarray
    pixel_quality: np.ndarray
    pixel_time: np.ndarray | None

    def __post_init__(self) -> None:
        """Enforce the read-only payload contract at construction time."""
        self.counts.flags.writeable = False
        self.pixel_quality.flags.writeable = False
        if self.pixel_time is not None:
            self.pixel_time.flags.writeable = False


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
    return ChannelSlicePayload(
        counts=counts, pixel_quality=quality, pixel_time=pixel_time
    )


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


class SharedNcPartReader:
    """Batch-scoped shared reader for FCI nc_part files.

    Caches one :class:`NCPartReader` per nc_part path for the duration of a
    single batch. All decode calls for the same ``item`` reuse the same
    underlying open file handle, so each nc_part file is opened at most
    once per batch instead of once per phase (time-map / row-range /
    calibration / spatial) as in the earlier context-manager-per-phase
    pattern.

    The reader is a pure refactor of the existing eager-decode surface:

    - :meth:`decode_spatial` mirrors the per-channel spatial payload used by
      :meth:`MtgFciL1cIngestor._emit_spatial_intents` (delegates to
      :func:`load_channel_slice`).
    - :meth:`decode_channel` mirrors the per-channel calibration lookup used
      by the ``calibration_table`` build in
      :meth:`MtgFciL1cIngestor.build_write_intents` (delegates to
      :meth:`NCPartReader.read_calibration`).

    Lifetime is caller-owned and batch-scoped. Use it as a context manager,
    or call :meth:`close` explicitly at the end of the batch. Do NOT cache
    the same instance across batches: per-batch lifecycle prevents
    cross-batch file-handle leaks and lines up with the ``BatchScratch``
    extraction lifetime that owns the on-disk nc_part files.

    Spatial payload caching is owned by :class:`ChunkOwnedAssembler`.
    ``decode_spatial`` only reuses the open nc_part file handle.
    """

    def __init__(self) -> None:
        """Initialize an empty file-handle cache."""
        self._readers: dict[Path, NCPartReader] = {}

    def _get_reader(self, nc_part_path: Path) -> NCPartReader:
        """Return the cached :class:`NCPartReader` for ``nc_part_path``.

        Creates a new reader on first access. The underlying NetCDF file
        opens lazily on the first ``read_*`` call, then stays open for the
        remaining lifetime of this shared reader.
        """
        path = Path(nc_part_path)
        reader = self._readers.get(path)
        if reader is None:
            reader = NCPartReader(path)
            self._readers[path] = reader
        return reader

    def decode_spatial(
        self,
        item: Path | str,
        nc_channel: str,
        index2time: dict[int, float] | None,
        pixel_time_dtype: np.dtype,
    ) -> ChannelSlicePayload:
        """Decode one (nc_part, nc_channel) spatial slice."""
        reader = self._get_reader(Path(item))
        return load_channel_slice(reader, nc_channel, index2time, pixel_time_dtype)

    def decode_channel(
        self,
        item: Path,
        nc_channel: str,
    ) -> tuple[float, float] | None:
        """Decode per-channel ``(scale_factor, add_offset)`` calibration.

        Matches the eager-decode surface of the ``calibration_table`` build
        in :meth:`MtgFciL1cIngestor.build_write_intents` (feeding
        :meth:`_emit_time_channel_intents`). Returns ``None`` if the channel
        or its calibration attributes are absent, matching
        :meth:`NCPartReader.read_calibration`.
        """
        reader = self._get_reader(item)
        return reader.read_calibration(nc_channel)

    def has_time_map(self, item: Path) -> bool:
        """Return whether the nc_part carries a root-level time map."""
        return self._get_reader(Path(item)).has_time_map()

    def reader_for(self, item: Path) -> NCPartReader:
        """Return the shared per-batch reader for ``item``.

        The returned reader's file handle is owned by this
        :class:`SharedNcPartReader`; callers must not close it.
        """
        return self._get_reader(Path(item))

    def read_row_range(self, item: Path, resolution: str) -> tuple[int, int]:
        """Read the row range for ``resolution``; raises ``KeyError`` if absent."""
        return self._get_reader(Path(item)).read_row_range(resolution)

    def close(self) -> None:
        """Close all cached file handles and empty the cache.

        Safe to call multiple times. After ``close()``, further decode
        calls will re-open the underlying files (and re-cache), so callers
        should treat the instance as spent for the batch and construct a
        fresh :class:`SharedNcPartReader` for the next batch instead.
        """
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> SharedNcPartReader:
        """Return ``self`` for context-manager usage."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close all cached file handles on context-manager exit."""
        self.close()


class AssemblyPreconditionError(Exception):
    """Raised when chunk-owned assembly preconditions are violated."""


class _IdentityRef:
    """Wrap an object for identity hashing while holding a strong reference.

    Using bare ``id(obj)`` as a cache key is unsafe: after the referent is
    garbage-collected a new object can be allocated at the same address,
    producing a false cache hit. ``_IdentityRef`` holds the referent alive
    for as long as the key exists, making id-reuse impossible for live entries.
    """

    __slots__ = ("obj",)

    def __init__(self, obj: object) -> None:
        self.obj = obj

    def __hash__(self) -> int:
        return id(self.obj)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _IdentityRef) and self.obj is other.obj

    def __repr__(self) -> str:
        return f"_IdentityRef({self.obj!r})"


@dataclasses.dataclass(frozen=True)
class _SourceKey:
    part: Path
    channel: str
    index2time_ref: _IdentityRef
    pixel_time_dtype: np.dtype


@dataclasses.dataclass(frozen=True)
class _AssembledKey:
    group: str
    slot_key: Any
    channel: str
    y_range: tuple[int, int]
    variable_set: frozenset[str]
    index2time_ref: _IdentityRef
    pixel_time_dtype: np.dtype


class ChunkOwnedAssembler:
    """Assemble output chunks from at most two nc_parts with bounded caches."""

    def __init__(self, shared_reader: SharedNcPartReader) -> None:
        self._reader = shared_reader
        self._source_cache: OrderedDict[_SourceKey, ChannelSlicePayload] = OrderedDict()
        self._assembled_cache: OrderedDict[_AssembledKey, ChannelSlicePayload] = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    def assemble(
        self,
        nc_parts: list[Path],
        nc_channel: str,
        index2time: dict[int, float] | None,
        pixel_time_dtype: np.dtype,
        group: str,
        slot_key: Any,
        y_range: tuple[int, int],
        variable_set: frozenset[str],
        part_row_ranges: dict[Path, tuple[int, int]] | None = None,
    ) -> ChannelSlicePayload:
        """Assemble one output chunk from at most two nc_parts."""
        if len(nc_parts) > 2:
            raise AssemblyPreconditionError(
                f"Output chunk y={y_range} intersects {len(nc_parts)} nc_parts; "
                "max supported is 2. Reduce zarr_chunk_y or file an issue."
            )

        index2time_ref = _IdentityRef(index2time)
        assembled_key = _AssembledKey(
            group=group,
            slot_key=slot_key,
            channel=nc_channel,
            y_range=y_range,
            variable_set=variable_set,
            index2time_ref=index2time_ref,
            pixel_time_dtype=pixel_time_dtype,
        )

        with self._lock:
            cached = self._assembled_cache.get(assembled_key)
            if cached is not None:
                return cached

        payloads: list[ChannelSlicePayload] = []
        for part in nc_parts:
            source_key = _SourceKey(
                part=Path(part),
                channel=nc_channel,
                index2time_ref=index2time_ref,
                pixel_time_dtype=pixel_time_dtype,
            )
            with self._lock:
                payload = self._source_cache.get(source_key)
                if payload is None:
                    payload = self._reader.decode_spatial(
                        part,
                        nc_channel,
                        index2time,
                        pixel_time_dtype,
                    )
                    self._source_cache[source_key] = payload
                    if len(self._source_cache) > 2:
                        self._source_cache.popitem(last=False)
            if part_row_ranges is not None:
                payload = self._slice_payload(
                    payload,
                    part_row_ranges[Path(part)],
                    y_range,
                )
            payloads.append(payload)

        assembled = self._assemble_payloads(payloads)

        with self._lock:
            self._assembled_cache[assembled_key] = assembled
            if len(self._assembled_cache) > 1:
                self._assembled_cache.popitem(last=False)

        return assembled

    @staticmethod
    def _assemble_payloads(payloads: list[ChannelSlicePayload]) -> ChannelSlicePayload:
        if len(payloads) == 1:
            return payloads[0]

        counts = np.concatenate([payload.counts for payload in payloads], axis=0)
        quality = np.concatenate(
            [payload.pixel_quality for payload in payloads], axis=0
        )
        pixel_time = None
        if all(payload.pixel_time is not None for payload in payloads):
            pixel_time = np.concatenate(
                [
                    payload.pixel_time
                    for payload in payloads
                    if payload.pixel_time is not None
                ],
                axis=0,
            )
        return ChannelSlicePayload(
            counts=counts,
            pixel_quality=quality,
            pixel_time=pixel_time,
        )

    @staticmethod
    def _slice_payload(
        payload: ChannelSlicePayload,
        part_range: tuple[int, int],
        y_range: tuple[int, int],
    ) -> ChannelSlicePayload:
        part_start, part_end = part_range
        y_start, y_end = y_range
        slice_start = max(y_start, part_start) - part_start
        slice_end = min(y_end, part_end) - part_start
        pixel_time = None
        if payload.pixel_time is not None:
            pixel_time = payload.pixel_time[slice_start:slice_end]
        return ChannelSlicePayload(
            counts=payload.counts[slice_start:slice_end],
            pixel_quality=payload.pixel_quality[slice_start:slice_end],
            pixel_time=pixel_time,
        )

    def close(self) -> None:
        """Clear source and assembled payload caches."""
        with self._lock:
            self._source_cache.clear()
            self._assembled_cache.clear()


def expand_pixel_time(
    index_map: np.ndarray,
    index2time: dict[int, float],
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Expand integer ``index_map`` to float pixel times, using NaN for misses."""
    output_dtype = np.dtype(dtype)
    fill_value = output_dtype.type(np.nan)

    if not index2time:
        return np.full(index_map.shape, fill_value, dtype=output_dtype)

    if index_map.dtype in _BOUNDED_UNSIGNED_DTYPES:
        return _expand_pixel_time_direct(
            index_map=index_map,
            index2time=index2time,
            output_dtype=output_dtype,
            fill_value=fill_value,
        )

    return _expand_pixel_time_masked(
        index_map=index_map,
        index2time=index2time,
        output_dtype=output_dtype,
        fill_value=fill_value,
    )


def _expand_pixel_time_direct(
    index_map: np.ndarray,
    index2time: dict[int, float],
    output_dtype: np.dtype[Any],
    fill_value: Any,
) -> np.ndarray:
    """Direct lookup for bounded unsigned maps, with dtype max as NaN sentinel."""
    max_val = np.iinfo(index_map.dtype).max
    lookup = np.full(max_val + 1, fill_value, dtype=output_dtype)

    for idx, time_value in index2time.items():
        if 0 <= idx < max_val:
            lookup[idx] = time_value

    out = np.empty(index_map.shape, dtype=output_dtype)
    np.take(lookup, index_map, out=out)
    out.flags.writeable = False
    return out


def _expand_pixel_time_masked(
    index_map: np.ndarray,
    index2time: dict[int, float],
    output_dtype: np.dtype[Any],
    fill_value: Any,
) -> np.ndarray:
    """Mask-based fallback for signed or otherwise unbounded index maps."""
    pixel_time = np.full(index_map.shape, fill_value, dtype=output_dtype)

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
