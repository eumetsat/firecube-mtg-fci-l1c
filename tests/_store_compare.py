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

"""Comparators for two Zarr stores written by this plugin.

``assert_stores_bitwise_equal`` is the strictest statement two ingests can
satisfy: every file outside ``.firecube/`` carries the same bytes. ``zarr.json``
files are compared as parsed JSON, so key order and whitespace do not matter.
Two ingests of identical input into separate stores were inspected and differ
in no file at all, not even in ``zarr.json``, so no key is dropped from them:
a key that changed between two runs would be a real difference to report.

``assert_rows_equal`` compares one array of a full-disk store with the same
array of a store that holds only a band of rows, decoded value by decoded
value, and never loads a whole array.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from _store_files import store_files
from firecube.core.api import compare_zarr_stores

# Largest array compared without slabs (no ``y`` axis): platform tables, the
# channel coordinate and similar.
_MAX_WHOLE_ARRAY_BYTES = 64 * 1024 * 1024


def _first_json_difference(a: Any, b: Any, path: str = "") -> str | None:
    """Return the JSON path of the first difference between two documents."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}/{key}"
            found = _first_json_difference(a[key], b[key], f"{path}/{key}")
            if found is not None:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            found = _first_json_difference(left, right, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    # json.dumps keeps NaN comparable (NaN != NaN as floats) and tells 1 from 1.0.
    return None if json.dumps(a) == json.dumps(b) else path or "/"


def assert_stores_bitwise_equal(a: Path, b: Path) -> None:
    """Assert that two local stores hold the same bytes outside ``.firecube/``.

    The failure message names the first differing relative path. Afterwards the
    stores are compared once more by core's value-level comparison, which reads
    the arrays through Zarr instead of the files.
    """
    files_a = store_files(a)
    files_b = store_files(b)
    for rel in sorted(set(files_a) | set(files_b)):
        if rel not in files_a:
            raise AssertionError(f"{rel}: only in the second store")
        if rel not in files_b:
            raise AssertionError(f"{rel}: only in the first store")
        if files_a[rel] == files_b[rel]:
            continue
        if rel.endswith("zarr.json"):
            where = _first_json_difference(
                json.loads(files_a[rel]), json.loads(files_b[rel])
            )
            if where is None:
                continue  # formatting only
            raise AssertionError(f"{rel}: zarr.json differs at {where}")
        first = next(
            (i for i, (x, y) in enumerate(zip(files_a[rel], files_b[rel])) if x != y),
            min(len(files_a[rel]), len(files_b[rel])),
        )
        raise AssertionError(
            f"{rel}: bytes differ ({len(files_a[rel])} vs {len(files_b[rel])} "
            f"bytes, first difference at offset {first})"
        )

    report = compare_zarr_stores(
        a.resolve().as_uri(),
        b.resolve().as_uri(),
        storage_type="local",
        storage_driver="fsspec",
    )
    assert report.equivalent, "value-level comparison found differences: " + "; ".join(
        report.mismatches
    )


def _dimension_names(arr: Any) -> list[str]:
    return list(arr.metadata.dimension_names or [])


def _attrs_json(arr: Any) -> str:
    return json.dumps(dict(arr.attrs), sort_keys=True)


def _same_values(left: np.ndarray, right: np.ndarray) -> bool:
    """Equal decoded values; numeric arrays are compared on their bytes."""
    if left.dtype.kind in "biufcmM":
        return left.tobytes() == right.tobytes()
    return left.tolist() == right.tolist()


def _slab_starts(length: int, step: int) -> range:
    return range(0, length, max(step, 1))


def assert_rows_equal(
    full_arr: Any, stripe_arr: Any, y0: int, *, name: str = ""
) -> None:
    """Assert that ``stripe_arr`` holds the rows ``[y0, y0 + ny)`` of ``full_arr``.

    Both arguments are opened Zarr arrays. Compared: dtype, fill value,
    dimension names, attributes and the decoded values, not the chunk or shard
    layout. The ``y`` axis is the one named ``y`` in the dimension names: axis 1
    of a ``(time, y, x, ...)`` array and axis 0 of ``(y, x)`` or ``(y,)``.
    Arrays without a ``y`` axis are compared whole and must be small.

    Values are compared slab by slab, ``chunks`` of the stripe array along
    ``y`` and ``time`` (axis 0 when it is not ``y``), so memory stays at one
    slab per side.
    """
    label = name or str(getattr(stripe_arr, "path", "array"))
    assert full_arr.dtype == stripe_arr.dtype, (
        f"{label}: dtype {full_arr.dtype} vs {stripe_arr.dtype}"
    )
    assert repr(full_arr.fill_value) == repr(stripe_arr.fill_value), (
        f"{label}: fill_value {full_arr.fill_value!r} vs {stripe_arr.fill_value!r}"
    )
    dims = _dimension_names(full_arr)
    assert dims == _dimension_names(stripe_arr), (
        f"{label}: dimension_names {dims} vs {_dimension_names(stripe_arr)}"
    )
    assert _attrs_json(full_arr) == _attrs_json(stripe_arr), (
        f"{label}: attrs differ: {dict(full_arr.attrs)} vs {dict(stripe_arr.attrs)}"
    )
    assert len(full_arr.shape) == len(stripe_arr.shape), f"{label}: rank differs"

    y_axis = dims.index("y") if "y" in dims else None
    if y_axis is None:
        assert full_arr.shape == stripe_arr.shape, (
            f"{label}: shape {full_arr.shape} vs {stripe_arr.shape}"
        )
        nbytes = int(np.prod(full_arr.shape, dtype=np.int64)) * full_arr.dtype.itemsize
        assert nbytes <= _MAX_WHOLE_ARRAY_BYTES, (
            f"{label}: {nbytes} bytes is too large to compare whole; it has no y axis"
        )
        assert _same_values(np.asarray(full_arr[...]), np.asarray(stripe_arr[...])), (
            f"{label}: values differ"
        )
        return

    ny = stripe_arr.shape[y_axis]
    assert 0 <= y0 and y0 + ny <= full_arr.shape[y_axis], (
        f"{label}: rows [{y0}, {y0 + ny}) are outside the full array "
        f"({full_arr.shape[y_axis]} rows)"
    )
    for axis, (left, right) in enumerate(zip(full_arr.shape, stripe_arr.shape)):
        assert axis == y_axis or left == right, (
            f"{label}: shape {full_arr.shape} vs {stripe_arr.shape} differs off y"
        )

    # Slabs along y and, when axis 0 is not y, along axis 0; other axes whole.
    chunks = stripe_arr.chunks
    ranges: list[range] = []
    for axis, length in enumerate(stripe_arr.shape):
        if axis == y_axis or axis == 0:
            ranges.append(_slab_starts(length, chunks[axis]))
        else:
            ranges.append(range(0, 1))
    for starts in product(*ranges):
        stripe_index: list[slice] = []
        full_index: list[slice] = []
        for axis, start in enumerate(starts):
            if axis == y_axis or axis == 0:
                stop = min(start + max(chunks[axis], 1), stripe_arr.shape[axis])
                stripe_index.append(slice(start, stop))
                offset = y0 if axis == y_axis else 0
                full_index.append(slice(start + offset, stop + offset))
            else:
                stripe_index.append(slice(None))
                full_index.append(slice(None))
        got = np.asarray(stripe_arr[tuple(stripe_index)])
        want = np.asarray(full_arr[tuple(full_index)])
        assert _same_values(want, got), (
            f"{label}: values differ in stripe slab "
            f"{tuple((s.start, s.stop) for s in stripe_index)} "
            f"(full rows from {y0})"
        )
