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

"""Verify slot-index model works on both parallel and non-parallel paths (core T2 alignment)."""

from __future__ import annotations

import datetime

from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig, MtgFciL1cIngestor


def test_supports_slot_range_parallelism():
    assert getattr(MtgFciL1cIngestor, "SUPPORTS_SLOT_RANGE_PARALLELISM", False) is True


def test_slot_index_formula():
    """Formula: (date - epoch).days * 144 + hour * 6 + minute // 10."""
    ingestor = object.__new__(MtgFciL1cIngestor)
    ingestor.plugin_config = MtgFciL1cConfig(time_epoch="2000-01-01")

    # epoch is 2000-01-01
    test_dt = datetime.datetime(2000, 1, 1, 0, 10, 0)  # day 0, 00:10 → slot 1
    result = ingestor.timestamp_to_ts_index("data_1km", test_dt)
    assert result == 0 * 144 + 0 * 6 + 10 // 10  # == 1

    # Another test case: day 1, 06:20 → 144 + 6*6 + 2 = 144 + 36 + 2 = 182
    test_dt2 = datetime.datetime(2000, 1, 2, 6, 20, 0)
    result2 = ingestor.timestamp_to_ts_index("data_1km", test_dt2)
    assert result2 == 1 * 144 + 6 * 6 + 20 // 10  # == 182
