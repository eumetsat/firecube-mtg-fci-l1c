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

"""End-of-run drift check of ``scripts/fci-ingest.sh`` against real Zarr stores.

The check is the ``DRIFT_CHECK`` heredoc of the script. Bash ends a heredoc at
the first line equal to its delimiter, so the program is exactly the lines
between the line opening ``<<'DRIFT_CHECK'`` and the line ``DRIFT_CHECK``. The
tests extract it by that rule and run it with the script's arguments
(target, plan groups, plan source), so they exercise the bytes the script runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fci-ingest.sh"
STATIC = ("latitude", "longitude", "x", "y")

pytestmark = [pytest.mark.plugin]

# Store spec: location ("" = root) -> {array name: carries the static marker}.
StoreSpec = dict[str, dict[str, bool]]


def _all_marked(*names: str) -> dict[str, bool]:
    return dict.fromkeys(names, True)


def _drift_check_program() -> str:
    lines = SCRIPT_PATH.read_text(encoding="utf-8").splitlines()
    (opening,) = [i for i, line in enumerate(lines) if "<<'DRIFT_CHECK'" in line]
    closing = lines.index("DRIFT_CHECK", opening + 1)
    return "\n".join(lines[opening + 1 : closing]) + "\n"


TIME_INDEXED = ("counts", "slope")


def _write_store(path: Path, spec: StoreSpec) -> None:
    root = zarr.open_group(str(path), mode="w")
    for location, arrays in spec.items():
        group = root.require_group(location) if location else root
        for name, marked in arrays.items():
            dims = ("time",) if name in TIME_INDEXED else (name,)
            array = group.create_array(
                name, shape=(2,), dtype="float32", dimension_names=dims
            )
            array[:] = np.zeros(2, dtype="float32")
            if marked:
                array.attrs["firecube_static_written"] = True


def _run_check(
    target: Path, plan_groups: str, plan_source: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-", f"file://{target}", plan_groups, plan_source],
        input=_drift_check_program(),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


CASES = [
    pytest.param(
        {"": {**_all_marked(*STATIC), "counts": False, "slope": False}},
        "",
        "slots",
        0,
        "DRIFT-CHECK OK: flat layout",
        id="flat-complete",
    ),
    pytest.param(
        {"": {**_all_marked(*STATIC), "channel_name": False}},
        "",
        "slots",
        1,
        "missing firecube_static_written on: /channel_name",
        id="flat-unmarked-channel-name",
    ),
    pytest.param(
        {"": _all_marked("x", "y", "counts")},
        "",
        "slots",
        0,
        "DRIFT-CHECK OK: flat layout",
        id="flat-without-geolocation",
    ),
    pytest.param(
        {"": {**_all_marked("latitude", "longitude", "y"), "x": False}},
        "",
        "slots",
        1,
        "missing firecube_static_written on: /x",
        id="flat-unmarked-x",
    ),
    pytest.param(
        {"data_1km": _all_marked(*STATIC), "data_2km": _all_marked(*STATIC)},
        "data_1km,data_2km",
        "slots",
        0,
        "DRIFT-CHECK OK: grouped layout, static markers present in data_1km, data_2km; "
        "store groups match the zarr slots plan",
        id="grouped-complete",
    ),
    pytest.param(
        {
            "data_1km": _all_marked(*STATIC),
            "data_2km": _all_marked("latitude", "longitude"),
        },
        "data_1km,data_2km",
        "slots",
        1,
        "data_2km/x (absent), data_2km/y (absent)",
        id="grouped-without-xy",
    ),
    pytest.param(
        {}, "data_1km", "fallback", 1, "no FCI layout found at", id="empty-store"
    ),
    pytest.param(
        {"": _all_marked("x", "y"), "data_1km": _all_marked(*STATIC)},
        "data_1km",
        "fallback",
        1,
        "mixed layout at",
        id="mixed-store",
    ),
    pytest.param(
        {"": _all_marked(*STATIC)},
        "data_1km",
        "slots",
        1,
        "store groups [/ (root)] differ from planned groups [data_1km]",
        id="slots-plan-grouped-store-flat",
    ),
    pytest.param(
        {"data_1km": _all_marked(*STATIC)},
        "data_1km,data_2km",
        "fallback",
        0,
        "DRIFT-CHECK OK: grouped layout, static markers present in data_1km; "
        "plan cross-check skipped (fallback plan)",
        id="fallback-plan-group-missing-from-store",
    ),
]


@pytest.mark.parametrize(
    ("spec", "plan_groups", "plan_source", "exit_code", "message"), CASES
)
def test_drift_check_verdict(
    tmp_path: Path,
    spec: StoreSpec,
    plan_groups: str,
    plan_source: str,
    exit_code: int,
    message: str,
) -> None:
    target = tmp_path / "store.zarr"
    _write_store(target, spec)

    result = _run_check(target, plan_groups, plan_source)

    output = result.stdout + result.stderr
    assert result.returncode == exit_code, output
    assert message in output
    if exit_code:
        assert result.stderr.startswith("DRIFT-CHECK FAIL: "), output
        assert "DRIFT-CHECK OK" not in output
