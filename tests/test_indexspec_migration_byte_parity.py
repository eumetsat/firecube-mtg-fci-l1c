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

"""Byte-parity guard for the IndexSpec migration (W2 test-first, RED before W3-W5).

This test locks in the persisted Zarr *structure* produced by a full FCI ingest
against a committed baseline (``tests/golden/indexspec_migration_baseline.json``).
The load-bearing invariant it protects is:

    Replacing ``slot_index_model()`` with ``index_spec(ctx) + inspect_item(item)``
    (firecube v0.1.6 clean cut) MUST NOT change the store layout or metadata
    that downstream Zarr consumers observe.

Scope (per operator direction, task W2):

* group names present
* array names present
* array shapes, chunks, dtype, dimension names, fill value
* stable Zarr metadata (attrs, minus known volatile keys)
* NOT raw chunk bytes — structure parity is sufficient (chunk ordering is not
  a documented invariant of the writer, so byte-level chunk equality would be
  fragile theatre rather than evidence).

Lifecycle:

* **RED phase** (before W3-W5): running this test writes the baseline JSON on
  first invocation and passes. Subsequent runs against pre-migration code
  continue to pass — the baseline is the pre-migration ground truth.
* **GREEN phase** (after W3-W5 land): the baseline is committed. The same test
  now re-runs the ingest through ``index_spec()`` code and asserts the
  captured structure equals the baseline. Any divergence is a migration bug.

Regeneration (deliberate, e.g. after an intentional store-format change)::

    rm tests/golden/indexspec_migration_baseline.json
    uv run --frozen python -m pytest tests/test_indexspec_migration_byte_parity.py -v
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).parent))
from test_integration import (  # noqa: E402
    _make_fdhsi_zip_at,
    _make_zip_with_nc_part,
    _run_ingest,
)
from firecube_mtg_fci_l1c._constants import PRODUCT_TYPE_HRFI  # noqa: E402

BASELINE_FILE = Path(__file__).parent / "golden" / "indexspec_migration_baseline.json"

# Volatile / environment-dependent attributes that must be excluded from the
# structure snapshot so the invariant survives clock drift, run ids, etc.
_VOLATILE_ATTR_KEYS = frozenset(
    {
        "creation_time",
        "created",
        "created_at",
        "updated_at",
        "firecube_run_id",
        "firecube_ingest_run_id",
        "firecube_slot_index_model",  # naming change is expected during migration
    }
)


def _json_value(value: Any) -> Any:
    """Convert Zarr attribute values into stable JSON-safe primitives."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="backslashreplace")
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(val) for key, val in sorted(value.items())}
    return value


def _stable_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop volatile / private-encoding attrs from the snapshot."""
    return {
        str(key): _json_value(value)
        for key, value in sorted(attrs.items())
        if str(key) not in _VOLATILE_ATTR_KEYS and not str(key).startswith("_NC")
    }


def _dimension_names(arr: Any) -> list[str]:
    names = getattr(arr, "dimension_names", None)
    if names is None:
        names = getattr(getattr(arr, "metadata", None), "dimension_names", None)
    if names is None:
        names = arr.attrs.get("_ARRAY_DIMENSIONS", [])
    return list(names or [])


def _capture_zarr_structure(zarr_path: Path) -> dict[str, Any]:
    """Snapshot the store layout and per-array metadata.

    Deliberately excludes raw chunk bytes: the migration contract is layout +
    metadata parity, not writer chunk-ordering parity.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    groups: dict[str, Any] = {}

    for group_name, group in sorted(root.groups(), key=lambda item: item[0]):
        arrays: dict[str, Any] = {}
        for array_name, arr in sorted(group.arrays(), key=lambda item: item[0]):
            arrays[array_name] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "chunks": list(arr.chunks),
                "dimension_names": _dimension_names(arr),
                "fill_value": _json_value(arr.fill_value),
                "attrs": _stable_attrs(arr.attrs),
            }
        groups[group_name] = {
            "attrs": _stable_attrs(group.attrs),
            "arrays": arrays,
        }

    return {"attrs": _stable_attrs(root.attrs), "groups": groups}


def _hash(data: Any) -> str:
    """Stable digest of the snapshot dict (order-independent via sort_keys)."""
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _hrfi_zip_at(src_dir: Path, ts_str: str) -> Path:
    """Mirror ``_make_fdhsi_zip_at`` for HRFI so we cover both product paths."""
    src_dir.mkdir(parents=True, exist_ok=True)
    zip_path = src_dir / f"W_XX-FCI-1C-RRAD-HRFI-FD-{ts_str}-END.zip"
    return _make_zip_with_nc_part(
        zip_path, PRODUCT_TYPE_HRFI, ["vis_06_hr", "ir_38_hr"], dimsize=4
    )


def _capture_case(
    tmp_path: Path,
    case_name: str,
    src_dir: Path,
) -> dict[str, Any]:
    """Run one ingest and return its structure snapshot + hash."""
    workspace = tmp_path / case_name
    workspace.mkdir()
    out = _run_ingest(src_dir, workspace)
    structure = _capture_zarr_structure(out)
    return {"hash": _hash(structure), "structure": structure}


def _build_snapshots(
    tmp_path: Path, small_fci_layout: list[int]
) -> dict[str, dict[str, Any]]:
    """Produce baselines for both FCI product families in a single session."""
    del small_fci_layout  # fixture side-effect (test constants) already active

    fdhsi_src = tmp_path / "fdhsi_src"
    _make_fdhsi_zip_at(fdhsi_src, "20240101000000")

    hrfi_src = tmp_path / "hrfi_src"
    _hrfi_zip_at(hrfi_src, "20240101000000")

    return {
        "fdhsi_defaults": _capture_case(tmp_path, "fdhsi_defaults", fdhsi_src),
        "hrfi_defaults": _capture_case(tmp_path, "hrfi_defaults", hrfi_src),
    }


def _write_baseline(snapshots: dict[str, dict[str, Any]]) -> None:
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_FILE.write_text(
        json.dumps(snapshots, indent=2, sort_keys=True, default=str) + "\n"
    )


def _diff_snapshots(
    current: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    for key, snapshot in current.items():
        if key not in expected:
            failures.append(f"  Missing baseline entry: {key}")
            continue
        want = expected[key]
        if snapshot["hash"] != want.get("hash"):
            failures.append(
                f"  Structure hash drift for {key}: "
                f"expected {want.get('hash', '')[:12]}..., "
                f"got {snapshot['hash'][:12]}..."
            )
        if snapshot["structure"] != want.get("structure"):
            failures.append(f"  Structure content drift for {key}")
    for key in sorted(set(expected) - set(current)):
        failures.append(f"  Unexpected baseline entry (missing in current run): {key}")
    return failures


@pytest.mark.integration
@pytest.mark.plugin
def test_indexspec_migration_preserves_zarr_structure(
    tmp_path: Path, small_fci_layout: list[int]
) -> None:
    """Ingest structure must equal the pre-migration baseline (FDHSI + HRFI).

    RED phase: on the very first run, the baseline JSON does not exist. The
    test writes it from the current (pre-migration) ingest and passes — that
    committed baseline IS the pre-migration ground truth.

    GREEN phase (after W3-W5): the baseline exists. The current ingest goes
    through ``index_spec()`` instead of ``slot_index_model()``, and the
    captured structure must equal the committed baseline. Any drift means the
    migration broke the persisted-format contract.
    """
    current = _build_snapshots(tmp_path, small_fci_layout)

    if not BASELINE_FILE.exists():
        _write_baseline(current)
        # First-run self-consistency: the just-written file must round-trip
        # equal to what we captured. This ensures the JSON encoder does not
        # silently lose fidelity we later depend on for comparison.
        expected = json.loads(BASELINE_FILE.read_text())
        failures = _diff_snapshots(current, expected)
        assert not failures, (
            "Freshly captured baseline failed self-consistency:\n" + "\n".join(failures)
        )
        return

    expected = json.loads(BASELINE_FILE.read_text())
    failures = _diff_snapshots(current, expected)
    assert not failures, "IndexSpec migration byte-parity guard failed:\n" + "\n".join(
        failures
    )
