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

"""Pin the repeat-cycle slot formula against the declared axis with a non-default epoch."""

from __future__ import annotations

import datetime

from firecube.core.api import resolve_index_spec
from firecube.ingestor.api import IngestContext

from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig, MtgFciL1cIngestor


def test_slot_index_formula():
    """Formula: (date - epoch).days * 144 + hour * 6 + minute // 10."""
    ingestor = MtgFciL1cIngestor()
    ingestor.plugin_config = MtgFciL1cConfig(time_epoch="2000-01-01", time_slots=1008)
    ctx = IngestContext(
        source="/tmp", target="/tmp/out.zarr", output_format="zarr", options={}
    )
    spec = ingestor.index_spec(ctx)
    assert spec is not None
    resolved = resolve_index_spec(spec, time_dim_name=MtgFciL1cIngestor.time_dim_name)

    # epoch is 2000-01-01: day 0, 00:10 -> slot 1
    assert resolved.position("data_1km", datetime.datetime(2000, 1, 1, 0, 10, 0)) == 1

    # day 1, 06:20 -> 144 + 6*6 + 2 = 182
    assert resolved.position("data_1km", datetime.datetime(2000, 1, 2, 6, 20, 0)) == 182
