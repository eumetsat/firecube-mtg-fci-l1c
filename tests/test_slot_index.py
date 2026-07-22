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

"""Unit tests for the deterministic EUMETSAT repeat-cycle slot index.

These cover the slot-range parallelism contract surface in isolation (no Zarr
I/O): the timestamp -> ts_index mapping, total-slot accounting for preallocation,
and source-item filtering by slot range. End-to-end behavior (single-pod
determinism, append idempotency, attr stamping) lives in ``test_integration.py``.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest

from firecube_mtg_fci_l1c import MtgFciL1cIngestor
from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig


def _ingestor(**config_kwargs) -> MtgFciL1cIngestor:
    ing = MtgFciL1cIngestor()
    ing.plugin_config = MtgFciL1cConfig(**config_kwargs)
    return ing


def _dt(s: str) -> datetime.datetime:
    return datetime.datetime.strptime(s, "%Y%m%d%H%M%S")


class TestTimestampToTsIndex:
    def test_within_day_matches_eumetsat_repeat_cycle(self):
        # EUMETSAT repeatCycleIdentifier is 1-based (12:00 -> 0073); the plugin
        # uses the 0-based form, so 12:00 on the epoch day -> 72.
        ing = _ingestor(time_epoch="2026-03-15")
        assert ing.timestamp_to_ts_index("data_1km", _dt("20260315000000")) == 0
        assert ing.timestamp_to_ts_index("data_1km", _dt("20260315001000")) == 1
        assert ing.timestamp_to_ts_index("data_1km", _dt("20260315120000")) == 72
        assert ing.timestamp_to_ts_index("data_1km", _dt("20260315235000")) == 143

    def test_seconds_jitter_does_not_shift_slot(self):
        # Real sensing start HH:M0:07 must land in the HH:M0:00 bin.
        ing = _ingestor(time_epoch="2026-03-15")
        assert ing.timestamp_to_ts_index("data_1km", _dt("20260315120007")) == 72
        assert ing.timestamp_to_ts_index("data_1km", _dt("20260315120923")) == 72

    def test_day_offset_accumulates(self):
        ing = _ingestor(time_epoch="2026-03-15")
        # next day, first cycle -> 144; next day 12:00 -> 144 + 72.
        assert ing.timestamp_to_ts_index("data_1km", _dt("20260316000000")) == 144
        assert ing.timestamp_to_ts_index("data_1km", _dt("20260316120000")) == 216

    def test_both_fci_types_share_one_index(self):
        # FDHSI and HRFI share the schedule, so the same acquisition time maps to
        # the same slot regardless of product type / group.
        fdhsi = _ingestor(time_epoch="2026-03-15", product_type="FDHSI")
        hrfi = _ingestor(time_epoch="2026-03-15", product_type="HRFI")
        t = _dt("20260315120000")
        assert fdhsi.timestamp_to_ts_index("data_1km", t) == 72
        assert hrfi.timestamp_to_ts_index("data_500m", t) == 72

    def test_group_does_not_change_index(self):
        ing = _ingestor(time_epoch="2026-03-15")
        t = _dt("20260315120000")
        assert ing.timestamp_to_ts_index("data_1km", t) == ing.timestamp_to_ts_index(
            "data_2km", t
        )

    def test_accepts_numpy_datetime64(self):
        ing = _ingestor(time_epoch="2026-03-15")
        assert (
            ing.timestamp_to_ts_index("data_1km", np.datetime64("2026-03-15T12:00:07"))
            == 72
        )

    def test_rejects_pre_epoch_acquisition(self):
        ing = _ingestor(time_epoch="2026-03-15")
        with pytest.raises(ValueError, match="precedes time_epoch"):
            ing.timestamp_to_ts_index("data_1km", _dt("20260314235000"))

    def test_default_epoch_is_dataset_start(self):
        # Default anchor is the verified first FCI L1C availability date.
        ing = _ingestor()
        assert ing.timestamp_to_ts_index("data_1km", _dt("20240924000000")) == 0


class TestGlobalExpectedTimeCount:
    def test_from_time_slots(self, monkeypatch):
        ing = _ingestor(product_type="FDHSI", time_slots=288)
        counts = ing.global_expected_time_count(_ctx())
        assert counts == {"data_1km": 288, "data_2km": 288}

    def test_from_time_end(self):
        # 2 days after the epoch -> 2 * 144 slots.
        ing = _ingestor(product_type="FDHSI", time_epoch="2026-03-15", time_end="2026-03-17")
        counts = ing.global_expected_time_count(_ctx())
        assert set(counts) == {"data_1km", "data_2km"}
        assert all(v == 288 for v in counts.values())

    def test_time_slots_takes_precedence_over_time_end(self):
        ing = _ingestor(
            product_type="FDHSI",
            time_epoch="2026-03-15",
            time_end="2026-03-17",
            time_slots=10,
        )
        assert all(v == 10 for v in ing.global_expected_time_count(_ctx()).values())

    def test_requires_extent(self):
        ing = _ingestor(product_type="FDHSI")
        with pytest.raises(ValueError, match="time_end.*time_slots|time_slots"):
            ing.global_expected_time_count(_ctx())

    def test_rejects_non_positive(self):
        ing = _ingestor(product_type="FDHSI", time_epoch="2026-03-15", time_end="2026-03-15")
        with pytest.raises(ValueError):
            ing.global_expected_time_count(_ctx())


class TestFilterItemsToSlotRange:
    def test_keeps_only_items_in_range(self):
        ing = _ingestor(time_epoch="2026-03-15")
        # Build FCI-style filenames at 12:00 (slot 72) and 12:10 (slot 73).
        names = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--x-x---x_C_"
            "EUMT_20260315120235_IDPFI_OPE_20260315120007_20260315120923_N__O_0073_0000.zip",
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--x-x---x_C_"
            "EUMT_20260315121235_IDPFI_OPE_20260315121007_20260315121923_N__O_0074_0000.zip",
        ]
        items = [Path("/data") / n for n in names]
        kept = ing.filter_items_to_slot_range(items, 72, 73, _ctx())
        assert kept == [items[0]]  # slot 72 in [72,73); slot 73 excluded


def _ctx():
    from firecube.ingestor.api import IngestContext

    return IngestContext(
        source="/tmp", target="/tmp/out.zarr", output_format="zarr", options={}
    )
