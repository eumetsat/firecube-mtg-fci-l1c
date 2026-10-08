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

"""Input-data helpers for MTG FCI L1C ZIP products.

This module identifies product type, observation time, and valid source files.
It does not read nc_parts or build arrays; streaming data reads live in
``_decode.py``.
"""

from __future__ import annotations

import datetime
import posixpath
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ._constants import PRODUCT_TYPE_FDHSI, PRODUCT_TYPE_HRFI


def extract_timestamp_from_path(path: Path) -> datetime.datetime | None:
    """Return the observation-start timestamp from an FCI filename, if present."""
    timestamps = re.findall(r"(\d{14})", path.name)
    if not timestamps:
        return None
    # FCI filenames usually end with observation-start then observation-end.
    ts_str = timestamps[-2] if len(timestamps) >= 2 else timestamps[0]
    try:
        return datetime.datetime.strptime(ts_str, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def extract_slot_time_from_path(path: Path) -> datetime.datetime | None:
    """Return the ``time`` coordinate label of an FCI product, if present.

    This is the observation start floored to the full minute, for example
    ``12:20:06`` becomes ``12:20:00``, so a slot can be selected by its round
    time. The exact acquisition times stay available in ``pixel_time``.
    """
    timestamp = extract_timestamp_from_path(path)
    if timestamp is None:
        return None
    return timestamp.replace(second=0, microsecond=0)


def detect_product_type(path_or_name: str | Path) -> str:
    """Return ``FDHSI`` or ``HRFI`` from the ZIP filename."""
    name = Path(path_or_name).name
    if PRODUCT_TYPE_FDHSI in name:
        return PRODUCT_TYPE_FDHSI
    if PRODUCT_TYPE_HRFI in name:
        return PRODUCT_TYPE_HRFI
    raise ValueError(
        "Cannot detect product type from filename: "
        f"{name!r}. Expected 'FDHSI' or 'HRFI' in name."
    )


def is_valid_fci_zip(path: Path) -> bool:
    """Return True for ZIP filenames that look like FCI L1C RRAD products."""
    if path.suffix != ".zip":
        return False
    if "FCI-1C-RRAD" not in path.name:
        return False
    return extract_timestamp_from_path(path) is not None


# --- Unpacked chunk input -------------------------------------------------

_CHUNK_TAIL = re.compile(
    r"_(?P<disseminated>\d{14})_[A-Z]+_[A-Z]+_"
    r"(?P<start>\d{14})_(?P<end>\d{14})_N__[A-Z]_"
    r"(?P<cycle>\d{4})_(?P<chunk>\d{4})\.nc$"
)
_CYCLES_PER_DAY = 144
_CYCLE_MINUTES = 10
_CYCLE_START_TOLERANCE = datetime.timedelta(minutes=5)


def _item_name(item: str | Path) -> str:
    """Return the final ``/``-separated component of a local path or URI string.

    Plain string handling on purpose: ``Path`` would collapse ``s3://b/x`` into
    ``s3:/b/x``, and inputs may be remote URIs.
    """
    return posixpath.basename(str(item))


def is_valid_fci_chunk(path: str | Path) -> bool:
    """Return True for unpacked FCI L1C RRAD ``CHK-BODY``/``CHK-TRAIL`` ``.nc`` files.

    Accepts local paths and remote URI strings; only the basename is inspected.
    """
    name = _item_name(path)
    return (
        name.endswith(".nc")
        and "FCI-1C-RRAD" in name
        and ("CHK-BODY" in name or "CHK-TRAIL" in name)
    )


@dataclass(frozen=True)
class ChunkName:
    """Fields parsed from an unpacked FCI chunk file name."""

    product_type: str
    sensing_start: datetime.datetime
    repeat_cycle: int
    chunk_number: int
    is_trail: bool


def parse_chunk_name(path: str | Path) -> ChunkName:
    """Parse an unpacked chunk file name; raise ``ValueError`` if it is invalid."""
    name = _item_name(path)
    if not is_valid_fci_chunk(name):
        raise ValueError(f"Not a valid FCI L1C chunk file name: {name!r}")
    match = _CHUNK_TAIL.search(name)
    if match is None:
        raise ValueError(f"Cannot parse FCI chunk file name: {name!r}")
    try:
        sensing_start = datetime.datetime.strptime(match["start"], "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise ValueError(f"Invalid sensing start in chunk name {name!r}") from exc
    return ChunkName(
        product_type=detect_product_type(name),
        sensing_start=sensing_start,
        repeat_cycle=int(match["cycle"]),
        chunk_number=int(match["chunk"]),
        is_trail="CHK-TRAIL" in name,
    )


def nominal_cycle_start(
    sensing_start: datetime.datetime, repeat_cycle: int
) -> datetime.datetime:
    """Return the nominal start of a 10-minute repeat cycle.

    The cycle number counts from midnight of the sensing day. A chunk of the
    last cycle may be sensed just after midnight; the nominal start would then
    lie in the future, so it belongs to the previous day.
    """
    if not 1 <= repeat_cycle <= _CYCLES_PER_DAY:
        raise ValueError(
            f"Repeat cycle must be in 1..{_CYCLES_PER_DAY}, got {repeat_cycle}"
        )
    midnight = sensing_start.replace(hour=0, minute=0, second=0, microsecond=0)
    nominal = midnight + datetime.timedelta(minutes=(repeat_cycle - 1) * _CYCLE_MINUTES)
    if nominal > sensing_start + _CYCLE_START_TOLERANCE:
        nominal -= datetime.timedelta(days=1)
    return nominal


@dataclass(frozen=True)
class SceneBundle:
    """Chunks of one repeat cycle, identified by product type and nominal start.

    ``members`` hold the original item strings (local paths or remote URIs),
    unchanged, so they can be handed to the materialiser verbatim.
    """

    product_type: str
    coordinate: datetime.datetime
    members: tuple[str, ...]

    @property
    def uri(self) -> str:
        return f"fci-scene://{self.product_type}/{self.coordinate:%Y%m%d%H%M%S}"

    def __str__(self) -> str:
        return self.uri


def group_chunks_into_bundles(items: Iterable[str | Path]) -> list[SceneBundle]:
    """Group chunk files into per-cycle bundles, ordered by (product, coordinate)."""
    groups: dict[tuple[str, datetime.datetime], dict[int, tuple[ChunkName, str]]]
    groups = {}
    for item in items:
        member = str(item)
        chunk = parse_chunk_name(member)
        key = (
            chunk.product_type,
            nominal_cycle_start(chunk.sensing_start, chunk.repeat_cycle),
        )
        members = groups.setdefault(key, {})
        if chunk.chunk_number in members:
            raise ValueError(
                f"Duplicate chunk number {chunk.chunk_number:04d} in scene "
                f"{key[0]}/{key[1]:%Y%m%d%H%M%S}: "
                f"{_item_name(members[chunk.chunk_number][1])!r} and {_item_name(member)!r}"
            )
        members[chunk.chunk_number] = (chunk, member)

    bundles: list[SceneBundle] = []
    for (product_type, coordinate), members in sorted(groups.items()):
        ordered = sorted(
            members.values(),
            key=lambda m: (m[0].is_trail, m[0].chunk_number, _item_name(m[1])),
        )
        bundles.append(
            SceneBundle(product_type, coordinate, tuple(p for _, p in ordered))
        )
    return bundles


def classify_items(
    items: Iterable[str | Path],
) -> tuple[list[str | Path], list[str | Path], list[str | Path]]:
    """Split inputs into ``(zips, chunks, ignored)``; ZIPs and chunks cannot mix.

    The original item objects are returned untouched (remote URIs included).
    """
    zips: list[str | Path] = []
    chunks: list[str | Path] = []
    ignored: list[str | Path] = []
    for item in items:
        if is_valid_fci_zip(Path(_item_name(item))):
            zips.append(item)
        elif is_valid_fci_chunk(item):
            chunks.append(item)
        else:
            ignored.append(item)
    if zips and chunks:
        raise ValueError(
            "Mixed ZIP and unpacked chunk input: "
            f"{len(zips)} ZIP file(s) and {len(chunks)} chunk file(s). "
            "Provide one input kind at a time."
        )
    return zips, chunks, ignored
