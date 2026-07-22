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

"""Golden snapshot test for schema stability.

Generates a hash of the serialised ZarrGroupSpec output of build_specs()
for 2 product types × flag-combo matrix. A mismatch means an accidental
schema change slipped through without explicit regeneration.

To regenerate snapshots:
    rm tests/golden/schema_snapshots.json
    uv run pytest tests/test_golden_schema.py -v
    uv run pytest tests/test_golden_schema.py -v  # second run verifies
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import pathlib
from typing import Any

import numpy as np
import pytest

from firecube_mtg_fci_l1c.config import MtgFciL1cConfig
from firecube_mtg_fci_l1c.schema import build_specs

_PRODUCT_TYPES = ["FDHSI", "HRFI"]
_FLAG_NAMES = (
    "include_pixel_quality",
    "include_pixel_time",
    "include_calibration",
    "include_geolocation",
)
_FLAG_COMBOS: list[dict[str, bool]] = [
    dict[str, bool](zip(_FLAG_NAMES, values, strict=True))
    for values in itertools.product((True, False), repeat=len(_FLAG_NAMES))
]

GOLDEN_FILE = pathlib.Path(__file__).parent / "golden" / "schema_snapshots.json"


def _json_value(value: Any) -> Any:
    """Convert numpy scalars/arrays and dtypes to stable JSON-friendly values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="backslashreplace")
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, type) and issubclass(value, np.generic):
        return np.dtype(value).name
    if isinstance(value, np.dtype):
        return value.name
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(val) for key, val in sorted(value.items())}
    return value


def _normalize(specs: list[Any]) -> list[dict[str, Any]]:
    """Serialize ZarrGroupSpec list to a JSON-stable structure."""
    result = []
    for group_spec in specs:
        arrays = []
        for arr in group_spec.arrays:
            arrays.append(
                {
                    "name": arr.name,
                    "dtype": _json_value(arr.dtype),
                    "shape": list(arr.shape) if arr.shape is not None else None,
                    "chunks": list(arr.chunks) if arr.chunks is not None else None,
                    "shards": list(arr.shards) if arr.shards is not None else None,
                    "fill_value": _json_value(arr.fill_value),
                    "time_indexed": arr.time_indexed,
                    "expected_time_count": arr.expected_time_count,
                    "dimension_names": (
                        list(arr.dimension_names)
                        if arr.dimension_names is not None
                        else None
                    ),
                    "attrs": _json_value(dict(arr.attrs or {})),
                }
            )
        result.append(
            {
                "group": group_spec.group,
                "attrs": _json_value(dict(group_spec.attrs or {})),
                "arrays": sorted(arrays, key=lambda item: item["name"]),
                "coord_names": sorted(group_spec.coord_names),
            }
        )
    return sorted(result, key=lambda item: item["group"])


def _hash(normalized: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, default=str).encode()
    ).hexdigest()


def _combo_key(product_type: str, flags: dict[str, bool]) -> str:
    return f"{product_type}__" + "_".join(
        f"{key}={value}" for key, value in sorted(flags.items())
    )


def _config(flags: dict[str, bool]) -> MtgFciL1cConfig:
    return MtgFciL1cConfig(
        include_pixel_quality=flags["include_pixel_quality"],
        include_pixel_time=flags["include_pixel_time"],
        include_calibration=flags["include_calibration"],
        include_geolocation=flags["include_geolocation"],
    )


def _build_snapshots() -> dict[str, dict[str, Any]]:
    snapshots = {}
    for product_type in _PRODUCT_TYPES:
        for flags in _FLAG_COMBOS:
            specs = build_specs(_config(flags), product_type)
            normalized = _normalize(specs)
            snapshots[_combo_key(product_type, flags)] = {
                "hash": _hash(normalized),
                "normalized_spec": normalized,
            }
    return snapshots


def test_golden_schema() -> None:
    if not GOLDEN_FILE.exists():
        GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_FILE.write_text(
            json.dumps(_build_snapshots(), indent=2, sort_keys=True, default=str) + "\n"
        )
        pytest.skip("snapshots generated; rerun to verify")

    snapshots = json.loads(GOLDEN_FILE.read_text())
    current_snapshots = _build_snapshots()
    failures = []

    for key, snapshot in current_snapshots.items():
        if key not in snapshots:
            failures.append(f"  {key}: MISSING from golden file")
            continue
        expected_hash = snapshots[key]["hash"]
        current_hash = snapshot["hash"]
        if current_hash != expected_hash:
            failures.append(
                f"  {key}: expected {expected_hash[:12]}..., got {current_hash[:12]}..."
            )
        if snapshot["normalized_spec"] != snapshots[key].get("normalized_spec"):
            failures.append(f"  {key}: normalized spec mismatch")

    for key in sorted(set(snapshots) - set(current_snapshots)):
        failures.append(f"  {key}: unexpected in golden file")

    if failures:
        pytest.fail("Schema drift:\n" + "\n".join(failures))
