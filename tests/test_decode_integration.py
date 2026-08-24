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

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from firecube_mtg_fci_l1c import MtgFciL1cIngestor
from firecube_mtg_fci_l1c._constants import PRODUCT_TYPE_FDHSI, PRODUCT_TYPE_HRFI
from firecube_mtg_fci_l1c._data import extract_timestamp_from_path
from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

from firecube.ingestor.api import IngestContext


@pytest.fixture
def small_fci_layout_fdhsi(monkeypatch):
    from firecube_mtg_fci_l1c import _constants as const_mod

    constants_backup = copy.deepcopy(const_mod.CONSTANTS)
    const_mod.CONSTANTS[PRODUCT_TYPE_FDHSI] = {
        "1km": {
            "channels": ["vis_04", "vis_06"],
            "dimsize": 4,
            "nc_channels": ["vis_04", "vis_06"],
        },
        "2km": {"channels": ["ir_38"], "dimsize": 4, "nc_channels": ["ir_38"]},
    }
    const_mod.CONSTANTS[PRODUCT_TYPE_HRFI] = {
        "500m": {"channels": ["vis_06"], "dimsize": 4, "nc_channels": ["vis_06_hr"]},
        "1km": {"channels": ["ir_38"], "dimsize": 4, "nc_channels": ["ir_38_hr"]},
    }

    yield

    const_mod.CONSTANTS.clear()
    const_mod.CONSTANTS.update(constants_backup)


class TestStreamingIngestorConfig:
    def test_ingestor_instantiation(self):
        ingestor = MtgFciL1cIngestor()
        assert ingestor is not None
        assert ingestor.name == "mtg_fci_l1c"

    def test_slice_meta_includes_streaming_fields(self):
        ingestor = MtgFciL1cIngestor()
        config = MtgFciL1cConfig()
        ingestor.plugin_config = config

        ctx = IngestContext(
            source="/tmp",
            target="/tmp/out.zarr",
            output_format="zarr",
            options={},
        )
        meta = ingestor.slice_meta(ctx)  # pyright: ignore[reportArgumentType]
        assert "pixel_time_dtype" in meta
        assert "scratch_dir" in meta
        assert "zarr_chunk_y" in meta

    def test_slice_meta_streaming_defaults(self):
        ingestor = MtgFciL1cIngestor()
        config = MtgFciL1cConfig()
        ingestor.plugin_config = config

        ctx = IngestContext(
            source="/tmp",
            target="/tmp/out.zarr",
            output_format="zarr",
            options={},
        )
        meta = ingestor.slice_meta(ctx)  # pyright: ignore[reportArgumentType]
        assert meta["pixel_time_dtype"] == "float64"
        assert meta["scratch_dir"] is None
        assert meta["zarr_chunk_y"] is None

    def test_streaming_chunk_shape_defaults_match_nc_part_stripes(self):
        config = MtgFciL1cConfig()
        assert config.get_group_chunk_shape("data_500m") == (1, 556, 22272, 1)
        assert config.get_group_chunk_shape("data_1km") == (1, 278, 11136, 1)
        assert config.get_group_chunk_shape("data_2km") == (1, 139, 5568, 1)


class TestBatchGroupSelection:
    def test_get_batch_groups_fdhsi(self, small_fci_layout_fdhsi):
        ingestor = MtgFciL1cIngestor()
        config = MtgFciL1cConfig(product_type="FDHSI")
        ingestor.plugin_config = config

        # Canonical hook: get_batch_groups(items, ctx); product type from config.
        groups = ingestor.get_batch_groups([], None)  # pyright: ignore[reportArgumentType]
        assert "data_1km" in groups
        assert "data_2km" in groups

    def test_get_batch_groups_hrfi(self, small_fci_layout_fdhsi):
        ingestor = MtgFciL1cIngestor()
        config = MtgFciL1cConfig(product_type="HRFI")
        ingestor.plugin_config = config

        groups = ingestor.get_batch_groups([], None)  # pyright: ignore[reportArgumentType]
        assert "data_500m" in groups
        assert "data_1km" in groups


class TestTimestampHandling:
    def test_extract_timestamp_from_path(self):
        import datetime

        zip_path = "W_XX-FCI-1C-RRAD-FDHSI-FD-20241001005154-END.zip"
        ts = extract_timestamp_from_path(Path(zip_path))
        assert ts == datetime.datetime(2024, 10, 1, 0, 51, 54)
