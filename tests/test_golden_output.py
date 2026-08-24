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

"""Golden snapshot test for Zarr output written by end-to-end ingests.

To regenerate snapshots:
    rm tests/golden/output_snapshots.json
    uv run pytest tests/test_golden_output.py -v -s
    uv run pytest tests/test_golden_output.py -v
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

from firecube_mtg_fci_l1c.config import MtgFciL1cConfig
from firecube_mtg_fci_l1c._variables import build_specs

sys.path.insert(0, str(Path(__file__).parent))
from test_integration import _run_ingest  # noqa: E402

GOLDEN_FILE = Path(__file__).parent / "golden" / "output_snapshots.json"


def _json_value(value: Any) -> Any:
    """Convert attributes into stable JSON-friendly values."""
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
    """Keep deterministic metadata while excluding run-specific attrs."""
    volatile = {"creation_time", "created", "created_at", "updated_at"}
    return {
        str(key): _json_value(value)
        for key, value in sorted(attrs.items())
        if str(key) not in volatile and not str(key).startswith("_NC")
    }


def _dimension_names(arr: Any) -> list[str]:
    names = getattr(arr, "dimension_names", None)
    if names is None:
        names = getattr(getattr(arr, "metadata", None), "dimension_names", None)
    if names is None:
        names = arr.attrs.get("_ARRAY_DIMENSIONS", [])
    return list(names or [])


def _capture_zarr_structure(zarr_path: Path) -> dict[str, Any]:
    """Walk a Zarr store and capture structure/metadata, never raw data."""
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
                "has_static_marker": arr.attrs.get("firecube_static_written") is True,
            }
        groups[group_name] = {
            "attrs": _stable_attrs(group.attrs),
            "arrays": arrays,
        }

    return {"attrs": _stable_attrs(root.attrs), "groups": groups}


def _hash(data: Any) -> str:
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _expected_topology(
    config: MtgFciL1cConfig, product_type: str
) -> dict[str, list[str]]:
    specs = build_specs(config, product_type)
    return {
        spec.group: sorted(array.name for array in spec.arrays)
        for spec in sorted(specs, key=lambda item: item.group)
    }


def _assert_matches_declared_topology(
    captured: dict[str, Any], expected: dict[str, list[str]]
) -> None:
    groups = captured["groups"]
    assert sorted(groups) == sorted(expected)
    for group_name, array_names in expected.items():
        assert sorted(groups[group_name]["arrays"]) == array_names


def _snapshot_case(
    tmp_path: Path,
    name: str,
    source_dir: Path,
    product_type: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = options or {}
    config = MtgFciL1cConfig(**options)
    expected = _expected_topology(config, product_type)

    workspace = tmp_path / name
    workspace.mkdir()
    out = _run_ingest(source_dir, workspace, options=options)
    structure = _capture_zarr_structure(out)
    _assert_matches_declared_topology(structure, expected)

    return {
        "product_type": product_type,
        "options": dict(sorted(options.items())),
        "hash": _hash(structure),
        "structure": structure,
    }


def _build_snapshots(
    tmp_path: Path,
    fdhsi_zip: Path,
    hrfi_zip: Path,  # noqa: F811
) -> dict[str, dict[str, Any]]:
    return {
        "fdhsi_defaults": _snapshot_case(
            tmp_path, "fdhsi_defaults", fdhsi_zip.parent, "FDHSI"
        ),
        "hrfi_defaults": _snapshot_case(
            tmp_path, "hrfi_defaults", hrfi_zip.parent, "HRFI"
        ),
    }


@pytest.mark.integration
@pytest.mark.plugin
def test_golden_output_snapshots(
    tmp_path: Path, fdhsi_zip: Path, hrfi_zip: Path
) -> None:
    """Output structure/metadata must match the committed golden snapshot."""
    current = _build_snapshots(tmp_path, fdhsi_zip, hrfi_zip)

    if not GOLDEN_FILE.exists():
        GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_FILE.write_text(
            json.dumps(current, indent=2, sort_keys=True, default=str) + "\n"
        )
        pytest.skip(f"Generated golden file {GOLDEN_FILE}. Re-run to verify.")

    expected = json.loads(GOLDEN_FILE.read_text())
    failures: list[str] = []

    for key, snapshot in current.items():
        if key not in expected:
            failures.append(f"  Missing snapshot key: {key}")
            continue
        if snapshot["hash"] != expected[key].get("hash"):
            failures.append(
                f"  Hash mismatch for {key}: "
                f"expected {expected[key].get('hash', '')[:12]}... "
                f"got {snapshot['hash'][:12]}..."
            )
        if snapshot["structure"] != expected[key].get("structure"):
            failures.append(f"  Output structure mismatch for {key}")

    for key in sorted(set(expected) - set(current)):
        failures.append(f"  Unexpected snapshot key: {key}")

    if failures:
        pytest.fail("Output structure drift detected:\n" + "\n".join(failures))
