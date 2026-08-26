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

"""Tests for MTG FCI L1C ingestor."""

from typing import Any
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import sys
import types

import numpy as np
import pytest

from firecube.core.api import IndexSpec, ItemInfo, RegularTimeAxis


class TestMtgFciL1cIngestorImport:
    def test_import(self):
        from firecube_mtg_fci_l1c import MtgFciL1cIngestor

        assert MtgFciL1cIngestor is not None

    def test_instantiate(self):
        from firecube_mtg_fci_l1c import MtgFciL1cIngestor

        ingestor = MtgFciL1cIngestor()
        assert ingestor is not None
        assert getattr(ingestor, "name", None) == "mtg_fci_l1c"


class TestMtgFciL1cConfig:
    def test_default_resolutions(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        config = MtgFciL1cConfig()
        assert config.get_resolutions() == ["1km", "2km"]

    def test_default_hrfi_resolutions(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        config = MtgFciL1cConfig()
        assert config.get_resolutions("HRFI") == ["500m", "1km"]

    def test_single_resolution(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        config = MtgFciL1cConfig(resolutions="1km")
        assert config.get_resolutions() == ["1km"]

    def test_multiple_resolutions(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        config = MtgFciL1cConfig(resolutions="2km,1km")
        assert config.get_resolutions() == ["2km", "1km"]

    def test_invalid_resolution_filtered(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        config = MtgFciL1cConfig(resolutions="500m,1km,2km")
        assert config.get_resolutions("FDHSI") == ["1km", "2km"]
        assert config.get_resolutions("HRFI") == ["500m", "1km"]

    def test_default_output_options(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        config = MtgFciL1cConfig()
        assert config.include_pixel_quality is True
        assert config.include_pixel_time is True
        assert config.include_calibration is True
        assert config.include_geolocation is True

    def test_default_channels_none(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        config = MtgFciL1cConfig()
        assert config.channels is None
        assert config.get_channels() is None

    def test_get_channels_parses_per_resolution(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        config = MtgFciL1cConfig(channels="vis_06,ir_105")
        assert config.get_channels("FDHSI") == {
            "1km": ["vis_06"],
            "2km": ["ir_105"],
        }

    def test_get_channels_invalid_raises(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        config = MtgFciL1cConfig(channels="invalid")
        with pytest.raises(ValueError, match="invalid"):
            config.get_channels("FDHSI")

    def test_get_channels_rejects_hrfi_nc_aliases(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        config = MtgFciL1cConfig(channels="vis_06_hr", product_type="HRFI")
        with pytest.raises(ValueError, match="vis_06_hr"):
            config.get_channels("HRFI")

    def test_fci_grids_file_config(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        cfg = MtgFciL1cConfig(fci_grids_file="/tmp/grids.npz")
        assert cfg.fci_grids_file == "/tmp/grids.npz"

    def test_fci_grids_file_default_none(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        cfg = MtgFciL1cConfig()
        assert cfg.fci_grids_file is None


class TestFilterItem:
    def test_valid_fci_filename(self):
        from firecube_mtg_fci_l1c._data import is_valid_fci_zip

        # Timestamp must be surrounded by dashes: -YYYYMMDDHHMMSS-
        path = Path(
            "W_XX-EUMETSAT-Darmstadt-FCI-1C-RRAD-FDHSI-FD-20241001005154-END.zip"
        )
        assert is_valid_fci_zip(path) is True

    def test_invalid_extension(self):
        from firecube_mtg_fci_l1c._data import is_valid_fci_zip

        path = Path("FCI-1C-RRAD-FDHSI-20241001005154.nc")
        assert is_valid_fci_zip(path) is False

    def test_missing_product_type(self):
        from firecube_mtg_fci_l1c._data import is_valid_fci_zip

        path = Path("some-other-product-20241001005154.zip")
        assert is_valid_fci_zip(path) is False

    def test_missing_timestamp(self):
        from firecube_mtg_fci_l1c._data import is_valid_fci_zip

        path = Path("FCI-1C-RRAD-FDHSI-notimestamp.zip")
        assert is_valid_fci_zip(path) is False


class TestTimestampExtraction:
    def test_extract_timestamp_real_filename(self):
        from datetime import datetime

        from firecube_mtg_fci_l1c._data import extract_timestamp_from_path

        path = Path(
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "x-x---x_C_EUMT_20241001120234_IDPFI_OPE_20241001120007_"
            "20241001120924_N__C_0073_0000.zip"
        )
        ts = extract_timestamp_from_path(path)
        assert ts is not None
        # Second-to-last timestamp is observation start time
        assert ts == datetime(2024, 10, 1, 12, 0, 7)

    def test_extract_timestamp_single(self):
        from firecube_mtg_fci_l1c._data import extract_timestamp_from_path

        path = Path("W_XX-EUMETSAT--20241001005154--END.zip")
        ts = extract_timestamp_from_path(path)
        assert ts is not None
        assert ts.year == 2024

    def test_no_timestamp(self):
        from firecube_mtg_fci_l1c._data import extract_timestamp_from_path

        path = Path("no-timestamp-here.zip")
        assert extract_timestamp_from_path(path) is None


class TestIndexSpecAndInspectItem:
    @pytest.mark.parametrize(
        ("product_type", "expected_groups"),
        [
            ("FDHSI", ["data_1km", "data_2km"]),
            ("HRFI", ["data_500m", "data_1km"]),
        ],
    )
    def test_index_spec_returns_resolution_groups(self, product_type, expected_groups):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig, MtgFciL1cIngestor

        ingestor = MtgFciL1cIngestor()
        cfg = MtgFciL1cConfig(product_type=product_type, time_slots=144)
        ingestor.plugin_config = cfg

        spec = ingestor.index_spec(SimpleNamespace(source="/tmp"))

        assert isinstance(spec, IndexSpec)
        assert spec is not None
        assert spec.name == ingestor.INDEX_MODEL
        assert list(spec.groups) == expected_groups
        for axis in spec.groups.values():
            assert isinstance(axis, RegularTimeAxis)
            assert axis.coordinate == "time"
            assert axis.epoch == f"{cfg.time_epoch}T00:00:00Z"
            assert axis.cadence_s == 600
            assert axis.mode == "floor"
            assert axis.slot_count == 144

    def test_inspect_item_returns_timestamp_coordinate(self):
        from datetime import datetime

        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cIngestor

        ingestor = MtgFciL1cIngestor()

        item = Path("W_XX-EUMETSAT--20241001005154--END.zip")
        info = ingestor.inspect_item(item, SimpleNamespace(source="/tmp"))

        assert info == ItemInfo(coordinate=datetime(2024, 10, 1, 0, 51, 54))

    def test_inspect_item_drops_invalid_items(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cIngestor

        ingestor = MtgFciL1cIngestor()

        info = ingestor.inspect_item(Path("no-timestamp-here.zip"), SimpleNamespace())

        assert info is None


class TestBuildWriteIntentsLogging:
    def test_logs_exception_when_nc_part_read_fails(self, monkeypatch):
        import datetime

        import firecube_mtg_fci_l1c.ingestor as ingestor_mod

        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig, MtgFciL1cIngestor

        ingestor = MtgFciL1cIngestor()
        ingestor.plugin_config = MtgFciL1cConfig(
            product_type="FDHSI",
            include_geolocation=False,
            include_pixel_time=False,
            include_calibration=False,
        )
        ingestor._log = MagicMock()

        class FakeScratch:
            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_zip(self, zip_path):
                return Path(f"/tmp/{Path(zip_path).stem}")

            def extract_zips_parallel(self, zip_paths, *, max_workers=4):
                del max_workers
                return {Path(p): self.extract_zip(p) for p in zip_paths}, {}

        class FakeReader:
            def __init__(self, part_path):
                self.part_path = part_path

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def has_time_map(self):
                return False

            def read_row_range(self, res):
                raise RuntimeError(f"failed for {res}")

        scratch_mod: Any = types.ModuleType("firecube_mtg_fci_l1c._scratch")
        scratch_mod.BatchScratch = FakeScratch
        monkeypatch.setitem(sys.modules, "firecube_mtg_fci_l1c._scratch", scratch_mod)
        monkeypatch.setattr(
            ingestor_mod, "list_fci_nc_parts", lambda _dir: [Path("/tmp/nc_part.nc")]
        )
        monkeypatch.setattr(ingestor_mod, "NCPartReader", FakeReader)
        monkeypatch.setattr(
            ingestor_mod,
            "extract_timestamp_from_path",
            lambda _zip_path: datetime.datetime(2024, 1, 1, 0, 0, 0),
        )

        batch: Any = SimpleNamespace(
            items=[Path("/tmp/input.zip")],
            metadata={},
            batch_id="batch-1",
        )

        ctx: Any = SimpleNamespace(
            run_id="run-1", source="/tmp", option=lambda *_args: None
        )

        intents = ingestor.build_write_intents(batch, ctx)  # pyright: ignore[reportArgumentType]

        assert all(intent.kind == "static" for intent in intents)
        assert any(intent.array == "channel_name" for intent in intents)
        assert batch.metadata["plugin_failure_counters"]["files_processed"] == 0
        assert batch.metadata["plugin_failure_counters"]["files_failed"] == 1
        ingestor._log.exception.assert_called_once()


class TestVariableDispatchRegressions:
    def test_spatial_phase_emits_callable_payloads(self):
        from firecube_mtg_fci_l1c._group_plan import GroupPlan
        from firecube_mtg_fci_l1c._decode import ChunkOwnedAssembler
        from firecube_mtg_fci_l1c.ingestor import (
            BatchResources,
            MtgFciL1cConfig,
            MtgFciL1cIngestor,
        )

        class CountingSharedReader:
            def __init__(self):
                self.decode_calls: list[tuple[Path, str]] = []

            def decode_spatial(
                self,
                part_path: Path,
                channel: str,
                index2time: dict[int, float] | None,
                pixel_time_dtype: np.dtype,
            ):
                del index2time, pixel_time_dtype
                self.decode_calls.append((part_path, channel))
                return SimpleNamespace(
                    counts=np.ones((2, 2), dtype=np.uint16),
                    pixel_quality=np.zeros((2, 2), dtype=np.uint8),
                    pixel_time=np.zeros((2, 2), dtype=np.float64),
                )

        config = MtgFciL1cConfig(
            product_type="FDHSI",
            include_pixel_quality=True,
            include_pixel_time=True,
            include_calibration=False,
            include_geolocation=False,
        )
        ingestor = MtgFciL1cIngestor()
        shared_reader = CountingSharedReader()
        part_path = Path("/tmp/part.nc")
        plan = GroupPlan(
            product_type="FDHSI",
            resolution="1km",
            group="data_1km",
            dimsize=2,
            logical_channels=("vis_04",),
            nc_channels=("vis_04",),
        )
        with ingestor._batch_resources_lock:
            ingestor._batch_resources["batch-1"] = BatchResources(
                chunk_owned_cache=ChunkOwnedAssembler(shared_reader)  # type: ignore[arg-type]
            )

        intents = ingestor._emit_spatial_intents(
            "batch-1",
            plan,
            0,
            [(part_path, (0, 2))],
            {0: 0.0},
            np.dtype(np.float64),
            config,
        )

        assert [intent.array for intent in intents] == [
            "counts",
            "pixel_quality",
            "pixel_time",
        ]
        assert all(callable(intent.data) for intent in intents)
        assert not any(isinstance(intent.data, np.ndarray) for intent in intents)
        assert shared_reader.decode_calls == []

        np.testing.assert_array_equal(
            intents[0].data(), np.ones((2, 2), dtype=np.uint16)
        )
        assert shared_reader.decode_calls == [(part_path, "vis_04")]

    def test_pixel_time_none_does_not_emit_intent(self):
        from firecube_mtg_fci_l1c._group_plan import GroupPlan
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig, MtgFciL1cIngestor

        class UnusedSharedReader:
            def __init__(self):
                self.decode_calls: list[tuple[Path, str]] = []

            def decode_spatial(self, *args, **kwargs):
                self.decode_calls.append((args, kwargs))
                raise AssertionError(
                    "decode_spatial must not be called when pixel_time is skipped"
                )

        config = MtgFciL1cConfig(
            product_type="FDHSI",
            include_pixel_quality=True,
            include_pixel_time=True,
            include_calibration=False,
            include_geolocation=False,
        )
        ingestor = MtgFciL1cIngestor()
        shared_reader = UnusedSharedReader()
        part_path = Path("/tmp/part.nc")
        plan = GroupPlan(
            product_type="FDHSI",
            resolution="1km",
            group="data_1km",
            dimsize=2,
            logical_channels=("vis_04",),
            nc_channels=("vis_04",),
        )

        intents = ingestor._emit_spatial_intents(
            "batch-1",
            plan,
            0,
            [(part_path, (0, 2))],
            None,
            np.dtype(np.float64),
            config,
        )

        arrays = [intent.array for intent in intents]
        assert "pixel_time" not in arrays
        assert arrays == ["counts", "pixel_quality"]
        assert shared_reader.decode_calls == []

    def test_time_channel_values_are_aggregated_once_per_timestamp(self, monkeypatch):
        import datetime

        import firecube_mtg_fci_l1c.ingestor as ingestor_mod

        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig, MtgFciL1cIngestor

        class FakeScratch:
            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_zip(self, zip_path):
                return Path(f"/tmp/{Path(zip_path).stem}")

            def extract_zips_parallel(self, zip_paths, *, max_workers=4):
                del max_workers
                return {Path(p): self.extract_zip(p) for p in zip_paths}, {}

        class FakeReader:
            def __init__(self, part_path):
                self.part_path = Path(part_path)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def has_time_map(self):
                return False

            def read_row_range(self, res):
                if res == "1km":
                    return (0, 2)
                raise KeyError(res)

            def read_calibration(self, channel):
                if self.part_path.name == "part-a.nc" and channel == "vis_04":
                    return (1.0, 10.0)
                if self.part_path.name == "part-b.nc" and channel == "vis_06":
                    return (2.0, 20.0)
                return None

            def read_channel_data(self, _channel):
                return (
                    np.ones((2, 2), dtype=np.uint16),
                    np.zeros((2, 2), dtype=np.uint8),
                    np.zeros((2, 2), dtype=np.int32),
                )

            def close(self):
                return None

        scratch_mod: Any = types.ModuleType("firecube_mtg_fci_l1c._scratch")
        scratch_mod.BatchScratch = FakeScratch
        monkeypatch.setitem(sys.modules, "firecube_mtg_fci_l1c._scratch", scratch_mod)
        monkeypatch.setattr(
            ingestor_mod,
            "list_fci_nc_parts",
            lambda _dir: [Path("/tmp/part-a.nc"), Path("/tmp/part-b.nc")],
        )
        monkeypatch.setattr(ingestor_mod, "NCPartReader", FakeReader)
        # SharedNcPartReader instantiates NCPartReader from _decode, so the
        # fake also has to replace that binding for calibration reads to hit it.
        import firecube_mtg_fci_l1c._decode as streaming_mod

        monkeypatch.setattr(streaming_mod, "NCPartReader", FakeReader)
        monkeypatch.setattr(
            ingestor_mod,
            "extract_timestamp_from_path",
            lambda _zip_path: datetime.datetime(2024, 1, 1, 0, 0, 0),
        )

        ingestor = MtgFciL1cIngestor()
        ingestor.plugin_config = MtgFciL1cConfig(
            product_type="FDHSI",
            time_epoch="2024-01-01",
            channels="vis_04,vis_06",
            include_geolocation=False,
            include_pixel_quality=False,
            include_pixel_time=False,
            include_calibration=True,
        )
        batch: Any = SimpleNamespace(
            items=[Path("/tmp/input.zip")],
            metadata={},
            batch_id="batch-1",
        )
        ctx: Any = SimpleNamespace(
            run_id="run-1", source="/tmp", option=lambda *_args: None
        )

        intents = ingestor.build_write_intents(batch, ctx)  # pyright: ignore[reportArgumentType]

        slope_intents = [intent for intent in intents if intent.array == "slope"]
        offset_intents = [intent for intent in intents if intent.array == "offset"]
        assert len(slope_intents) == 1
        assert len(offset_intents) == 1
        np.testing.assert_array_equal(slope_intents[0].data, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(offset_intents[0].data, np.array([10.0, 20.0]))


class TestGetBatchGroups:
    def test_get_batch_groups_filters_empty_channel_resolution(self):
        """Channel selection must drop resolution groups with no selected channels."""
        from firecube_mtg_fci_l1c import MtgFciL1cIngestor
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        config = MtgFciL1cConfig(channels="vis_06", product_type="FDHSI")
        ingestor = MtgFciL1cIngestor()
        ingestor.plugin_config = config
        # Canonical hook: get_batch_groups(items, ctx); product type comes from config.
        ctx: Any = SimpleNamespace(source="/tmp", option=lambda *_args: None)
        groups = ingestor.get_batch_groups([], ctx)  # pyright: ignore[reportArgumentType]
        assert groups == ["data_1km"]

    def test_get_batch_groups_no_channels_returns_all(self):
        from firecube_mtg_fci_l1c import MtgFciL1cIngestor
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        config = MtgFciL1cConfig(channels=None, product_type="FDHSI")
        ingestor = MtgFciL1cIngestor()
        ingestor.plugin_config = config
        ctx: Any = SimpleNamespace(source="/tmp", option=lambda *_args: None)
        groups = ingestor.get_batch_groups([], ctx)  # pyright: ignore[reportArgumentType]
        assert groups == ["data_1km", "data_2km"]
