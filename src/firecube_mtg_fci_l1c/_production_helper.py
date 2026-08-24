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

"""Pure helpers used by the production fan-out script."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

STATIC_ARRAY_NAMES = ("latitude", "longitude", "x", "y")


@dataclass(frozen=True)
class PodResult:
    """Result for one fan-out pod."""

    index: int
    slot_start: int
    slot_end: int
    emit_static_variables: bool
    returncode: int
    log_path: str


def select_static_writer(
    fanout_plan: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    """Select exactly one static writer from a fan-out plan.

    The first entry in plan order owns static variables. This is deliberately
    independent of slot number so partial retries remain valid.
    """
    if not fanout_plan:
        raise ValueError("Fan-out plan is empty; cannot select a static writer.")

    result: list[dict[str, Any]] = []
    for index, entry in enumerate(fanout_plan):
        result.append({**entry, "emit_static_variables": index == 0})
    validate_single_static_writer(result)
    return 0, result


def validate_single_static_writer(fanout_plan: Iterable[Mapping[str, Any]]) -> None:
    """Fail if a fan-out plan does not contain exactly one static writer."""
    owners = [
        entry for entry in fanout_plan if entry.get("emit_static_variables") is True
    ]
    if len(owners) != 1:
        raise ValueError(
            f"Fan-out plan must contain exactly one static writer; found {len(owners)}."
        )


def aggregate_pod_failures(results: Iterable[PodResult]) -> None:
    """Raise a clear failure if any fan-out pod failed."""
    failed = [result for result in results if result.returncode != 0]
    if not failed:
        return

    static_failures = [result for result in failed if result.emit_static_variables]
    if static_failures:
        result = static_failures[0]
        raise RuntimeError(
            "Static writer pod failed: "
            f"pod {result.index} [{result.slot_start},{result.slot_end}) -> {result.log_path}"
        )

    details = ", ".join(
        f"pod {result.index} [{result.slot_start},{result.slot_end}) -> {result.log_path}"
        for result in failed
    )
    raise RuntimeError(f"Fan-out pod(s) failed: {details}")


def verify_static_markers(root: Any) -> list[str]:
    """Return static arrays that are missing the Firecube static-write marker."""
    missing: list[str] = []
    for group_name in root.group_keys():
        group = root[group_name]
        for array_name in STATIC_ARRAY_NAMES:
            if array_name in group:
                array = group[array_name]
                if "firecube_static_written" not in array.attrs:
                    missing.append(f"{group_name}/{array_name}")
    return missing


def verify_zarr_static_markers(target: str) -> None:
    """Open a Zarr store and fail if static arrays lack write markers."""
    import zarr

    root = zarr.open_group(target.replace("file://", ""), mode="r")
    missing = verify_static_markers(root)
    if missing:
        raise RuntimeError(
            f"DRIFT-CHECK FAIL: missing firecube_static_written on: {missing}"
        )
