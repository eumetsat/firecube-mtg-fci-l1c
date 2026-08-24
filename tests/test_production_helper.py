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

"""Tests for production fan-out helper logic."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from firecube_mtg_fci_l1c._production_helper import (
    PodResult,
    aggregate_pod_failures,
    select_static_writer,
    validate_single_static_writer,
    verify_zarr_static_markers,
)


def test_helper_selects_exactly_one_static_writer_by_plan_order() -> None:
    plan = [
        {"slot_start": 12, "slot_end": 18},
        {"slot_start": 0, "slot_end": 6},
        {"slot_start": 6, "slot_end": 12},
        {"slot_start": 18, "slot_end": 24},
    ]

    static_writer_index, selected = select_static_writer(plan)

    assert static_writer_index == 0
    assert [entry["emit_static_variables"] for entry in selected] == [
        True,
        False,
        False,
        False,
    ]


def test_helper_passes_emit_static_variables_false_to_non_owners() -> None:
    _, selected = select_static_writer(
        [
            {"slot_start": 4, "slot_end": 5},
            {"slot_start": 5, "slot_end": 6},
            {"slot_start": 6, "slot_end": 7},
        ]
    )

    for entry in selected[1:]:
        assert entry["emit_static_variables"] is False


def test_helper_aggregates_static_writer_pod_failure() -> None:
    results = [
        PodResult(0, 4, 10, True, 1, "/logs/pod_4_10.log"),
        PodResult(1, 10, 16, False, 0, "/logs/pod_10_16.log"),
    ]

    with pytest.raises(RuntimeError, match="Static writer pod failed: pod 0"):
        aggregate_pod_failures(results)


def test_helper_verifies_static_markers_after_success() -> None:
    class FakeRoot(dict[str, dict[str, SimpleNamespace]]):
        def group_keys(self) -> list[str]:
            return list(self)

    data_1km = {
        "latitude": SimpleNamespace(attrs={"firecube_static_written": True}),
        "longitude": SimpleNamespace(attrs={"firecube_static_written": True}),
        "x": SimpleNamespace(attrs={"firecube_static_written": True}),
        "y": SimpleNamespace(attrs={"firecube_static_written": True}),
    }
    root_group = FakeRoot({"data_1km": data_1km})

    with patch("zarr.open_group", return_value=root_group) as open_group:
        verify_zarr_static_markers("file:///tmp/fci.zarr")

    open_group.assert_called_once_with("/tmp/fci.zarr", mode="r")


def test_helper_partial_retry_still_selects_exactly_one_static_writer_in_current_fanout_plan() -> (
    None
):
    plan = [
        {"slot_start": 4, "slot_end": 10},
        {"slot_start": 10, "slot_end": 16},
    ]

    static_writer_index, selected = select_static_writer(plan)

    assert static_writer_index == 0
    assert selected[0]["slot_start"] == 4
    assert selected[0]["emit_static_variables"] is True
    assert selected[1]["emit_static_variables"] is False


def test_helper_fails_locally_if_fanout_plan_has_zero_static_writers() -> None:
    with pytest.raises(ValueError, match="Fan-out plan is empty"):
        select_static_writer([])


def test_helper_fails_locally_if_fanout_plan_has_multiple_static_writers() -> None:
    corrupt_plan = [
        {"slot_start": 0, "slot_end": 6, "emit_static_variables": True},
        {"slot_start": 6, "slot_end": 12, "emit_static_variables": True},
    ]

    with pytest.raises(ValueError, match="exactly one static writer"):
        validate_single_static_writer(corrupt_plan)
