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

"""Unit tests for the declared EUMETSAT repeat-cycle time axis.

These cover the plugin's ``index_spec()`` declaration in isolation (no Zarr
I/O): timestamp -> slot mapping through the engine-resolved index, total-slot
accounting for preallocation, and filename -> slot mapping via
``inspect_item``. Slot filtering itself is engine-owned
(``firecube`` resolves ``inspect_item`` coordinates against the declared
axis); these tests pin the declaration the engine consumes. End-to-end
behavior (single-pod determinism, append idempotency, attr stamping) lives in
``test_integration.py``.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest
from firecube.core.api import resolve_index_spec

from firecube_mtg_fci_l1c import MtgFciL1cIngestor
from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig


def _ingestor(**config_kwargs) -> MtgFciL1cIngestor:
    ing = MtgFciL1cIngestor()
    ing.plugin_config = MtgFciL1cConfig(**config_kwargs)
    return ing


def _dt(s: str) -> datetime.datetime:
    return datetime.datetime.strptime(s, "%Y%m%d%H%M%S")


def _resolved(ing: MtgFciL1cIngestor):
    spec = ing.index_spec(_ctx())
    assert spec is not None, "index_spec() returned None; pass time_slots/time_end"
    return resolve_index_spec(spec, time_dim_name=MtgFciL1cIngestor.time_dim_name)


def _position(ing: MtgFciL1cIngestor, group: str, value) -> int:
    return _resolved(ing).position(group, value)


class TestDeclaredAxisPosition:
    def test_within_day_matches_eumetsat_repeat_cycle(self):
        # EUMETSAT repeatCycleIdentifier is 1-based (12:00 -> 0073); the axis
        # uses the 0-based form, so 12:00 on the epoch day -> 72.
        ing = _ingestor(time_epoch="2026-03-15", time_slots=1008)
        assert _position(ing, "data_1km", _dt("20260315000000")) == 0
        assert _position(ing, "data_1km", _dt("20260315001000")) == 1
        assert _position(ing, "data_1km", _dt("20260315120000")) == 72
        assert _position(ing, "data_1km", _dt("20260315235000")) == 143

    def test_seconds_jitter_does_not_shift_slot(self):
        # Real sensing start HH:M0:07 must land in the HH:M0:00 bin
        # (mode="floor" in the declared axis).
        ing = _ingestor(time_epoch="2026-03-15", time_slots=1008)
        assert _position(ing, "data_1km", _dt("20260315120007")) == 72
        assert _position(ing, "data_1km", _dt("20260315120923")) == 72

    def test_day_offset_accumulates(self):
        ing = _ingestor(time_epoch="2026-03-15", time_slots=1008)
        # next day, first cycle -> 144; next day 12:00 -> 144 + 72.
        assert _position(ing, "data_1km", _dt("20260316000000")) == 144
        assert _position(ing, "data_1km", _dt("20260316120000")) == 216

    def test_both_fci_types_share_one_index(self):
        # FDHSI and HRFI share the schedule, so the same acquisition time maps to
        # the same slot regardless of product type / group.
        fdhsi = _ingestor(
            time_epoch="2026-03-15", product_type="FDHSI", time_slots=1008
        )
        hrfi = _ingestor(time_epoch="2026-03-15", product_type="HRFI", time_slots=1008)
        t = _dt("20260315120000")
        assert _position(fdhsi, "data_1km", t) == 72
        assert _position(hrfi, "data_500m", t) == 72

    def test_group_does_not_change_index(self):
        ing = _ingestor(time_epoch="2026-03-15", time_slots=1008)
        t = _dt("20260315120000")
        assert _position(ing, "data_1km", t) == _position(ing, "data_2km", t)

    def test_accepts_numpy_datetime64(self):
        ing = _ingestor(time_epoch="2026-03-15", time_slots=1008)
        assert _position(ing, "data_1km", np.datetime64("2026-03-15T12:00:07")) == 72

    def test_rejects_pre_epoch_acquisition(self):
        ing = _ingestor(time_epoch="2026-03-15", time_slots=1008)
        with pytest.raises(ValueError, match="predates epoch"):
            _position(ing, "data_1km", _dt("20260314235000"))

    def test_default_epoch_is_dataset_start(self):
        # Default anchor is the verified first FCI L1C availability date.
        ing = _ingestor(time_slots=144)
        assert _position(ing, "data_1km", _dt("20240924000000")) == 0


class TestDeclaredAxisExtent:
    def test_from_time_slots(self):
        ing = _ingestor(product_type="FDHSI", time_slots=288)
        resolved = _resolved(ing)
        assert {g: resolved.size(g) for g in resolved.groups} == {
            "data_1km": 288,
            "data_2km": 288,
        }

    def test_from_time_end(self):
        # 2 days after the epoch -> 2 * 144 slots.
        ing = _ingestor(
            product_type="FDHSI", time_epoch="2026-03-15", time_end="2026-03-17"
        )
        resolved = _resolved(ing)
        assert set(resolved.groups) == {"data_1km", "data_2km"}
        assert all(resolved.size(g) == 288 for g in resolved.groups)

    def test_time_slots_takes_precedence_over_time_end(self):
        ing = _ingestor(
            product_type="FDHSI",
            time_epoch="2026-03-15",
            time_end="2026-03-17",
            time_slots=10,
        )
        resolved = _resolved(ing)
        assert all(resolved.size(g) == 10 for g in resolved.groups)

    def test_no_extent_declares_unbounded_axis_and_still_resolves_positions(self):
        # Without time_end/time_slots the plugin declares an unbounded axis
        # (slot_count=None); the engine's parallel gate refuses slot-range
        # flags loudly for unbounded axes, and serial ingestion maps
        # timestamps through the same declared axis.
        ing = _ingestor(product_type="FDHSI", time_epoch="2026-03-15")
        spec = ing.index_spec(_ctx())
        assert spec is not None
        assert all(axis.slot_count is None for axis in spec.groups.values())
        resolved = resolve_index_spec(
            spec, time_dim_name=MtgFciL1cIngestor.time_dim_name
        )
        assert resolved.position("data_1km", _dt("20260315120000")) == 72

    def test_non_positive_extent_raises(self):
        ing = _ingestor(
            product_type="FDHSI", time_epoch="2026-03-15", time_end="2026-03-15"
        )
        with pytest.raises(ValueError, match="not after time_epoch"):
            ing.index_spec(_ctx())


class TestFilenameToSlot:
    def test_inspect_item_coordinate_maps_to_expected_slot(self):
        # Slot filtering is engine-owned: the engine calls inspect_item and
        # resolves the coordinate against the declared axis. Pin that chain
        # for FCI-style filenames at 12:00 (slot 72) and 12:10 (slot 73).
        ing = _ingestor(time_epoch="2026-03-15", time_slots=1008)
        names = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--x-x---x_C_"
            "EUMT_20260315120235_IDPFI_OPE_20260315120007_20260315120923_N__O_0073_0000.zip",
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--x-x---x_C_"
            "EUMT_20260315121235_IDPFI_OPE_20260315121007_20260315121923_N__O_0074_0000.zip",
        ]
        items = [Path("/data") / n for n in names]
        slots = []
        for item in items:
            info = ing.inspect_item(item, _ctx())
            assert info is not None
            slots.append(_position(ing, "data_1km", info.coordinate))
        assert slots == [72, 73]


def _ctx():
    from firecube.ingestor.api import IngestContext

    return IngestContext(
        source="/tmp", target="/tmp/out.zarr", output_format="zarr", options={}
    )
