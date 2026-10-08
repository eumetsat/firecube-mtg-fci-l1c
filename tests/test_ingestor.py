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

from collections.abc import Iterable
from typing import Any, cast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import re
import sys
import types

import numpy as np
import pytest

from firecube.core.api import IndexSpec, ItemInfo, RegularTimeAxis
from firecube.ingestor.api import ConfigurationError
from firecube_mtg_fci_l1c._decode import ChannelCalibration


class TestMtgFciL1cIngestorImport:
    def test_instantiate(self):
        from firecube_mtg_fci_l1c import MtgFciL1cIngestor

        ingestor = MtgFciL1cIngestor()
        assert ingestor is not None
        assert getattr(ingestor, "name", None) == "mtg_fci_l1c"


class TestMtgFciL1cConfig:
    def test_get_channels_parses_per_resolution(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig

        config = MtgFciL1cConfig(channels="vis_06,ir_105")
        assert config.get_channels("FDHSI") == {
            "1km": ["vis_06"],
            "2km": ["ir_105"],
        }

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
    def test_invalid_extension(self):
        from firecube_mtg_fci_l1c._data import is_valid_fci_zip

        path = Path("FCI-1C-RRAD-FDHSI-20241001005154.nc")
        assert is_valid_fci_zip(path) is False

    def test_missing_product_type(self):
        from firecube_mtg_fci_l1c._data import is_valid_fci_zip

        path = Path("some-other-product-20241001005154.zip")
        assert is_valid_fci_zip(path) is False


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

    @pytest.mark.parametrize(
        ("product_type", "channels", "expected_groups"),
        [
            ("FDHSI", "vis_06", {"data_1km"}),
            ("FDHSI", "ir_105,wv_63", {"data_2km"}),
            ("FDHSI", "vis_06,ir_105", {"data_1km", "data_2km"}),
            ("HRFI", "nir_22", {"data_500m"}),
        ],
    )
    def test_index_spec_groups_follow_channel_selection(
        self, product_type, channels, expected_groups
    ):
        """IndexSpec groups must be exactly the groups zarr_schema() declares."""
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig, MtgFciL1cIngestor

        ingestor = MtgFciL1cIngestor()
        ingestor.plugin_config = MtgFciL1cConfig(
            product_type=product_type, channels=channels, time_slots=144
        )
        ctx: Any = SimpleNamespace(source="unused", options={})

        spec = ingestor.index_spec(ctx)
        schema_groups = {group.group for group in ingestor.zarr_schema(ctx)}

        assert spec is not None
        assert set(spec.groups) == expected_groups
        assert schema_groups == expected_groups

    def test_inspect_item_returns_minute_floored_time_coordinate(self):
        from datetime import datetime

        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cIngestor

        ingestor = MtgFciL1cIngestor()

        item = Path("W_XX-EUMETSAT--20241001005154--END.zip")
        info = ingestor.inspect_item(item, SimpleNamespace(source="/tmp"))

        # Start 00:51:54 -> label 00:51:00.
        assert info == ItemInfo(coordinate=datetime(2024, 10, 1, 0, 51, 0))

    def test_inspect_item_drops_invalid_items(self):
        from firecube_mtg_fci_l1c.ingestor import MtgFciL1cIngestor

        ingestor = MtgFciL1cIngestor()

        info = ingestor.inspect_item(Path("no-timestamp-here.zip"), SimpleNamespace())

        assert info is None


class _FakeScratch:
    """Scratch stand-in: no real extraction, deterministic ``/tmp/<stem>`` dirs."""

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

    def close(self):
        return None


class TestBuildWriteIntentsLogging:
    def test_logs_exception_when_nc_part_read_fails(self, monkeypatch, tmp_path):
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

        def failing_row_range(_self, _item, resolution):
            raise RuntimeError(f"row range unreadable for {resolution}")

        # A real, readable nc_part: only the row-range read is made to fail.
        part = tmp_path / "nc_part.nc"
        _write_disk_rows_chunk(part, (0, 4))
        monkeypatch.setattr(
            ingestor_mod.SharedNcPartReader, "read_row_range", failing_row_range
        )
        scratch_mod: Any = types.ModuleType("firecube_mtg_fci_l1c._scratch")
        scratch_mod.BatchScratch = _FakeScratch
        monkeypatch.setitem(sys.modules, "firecube_mtg_fci_l1c._scratch", scratch_mod)
        monkeypatch.setattr(ingestor_mod, "list_fci_nc_parts", lambda _dir: [part])
        monkeypatch.setattr(
            ingestor_mod,
            "extract_slot_time_from_path",
            lambda _zip_path: datetime.datetime(2024, 1, 1, 0, 0, 0),
        )

        batch: Any = SimpleNamespace(
            items=[Path("/tmp/input.zip")],
            metadata={},
            batch_id="batch-1",
        )

        ctx: Any = SimpleNamespace(
            run_id="run-1",
            source="/tmp",
            option=lambda *_args: None,
            materialize=Path,
        )

        intents = ingestor.build_write_intents(batch, ctx)  # pyright: ignore[reportArgumentType]

        assert {intent.kind for intent in intents} == {"static"}
        assert any(intent.array == "channel" for intent in intents)
        assert batch.metadata["plugin_failure_counters"] == {
            "files_processed": 0,
            "files_failed": 1,
            "zip_errors": ["input.zip: row range unreadable for 1km"],
        }
        ingestor._log.exception.assert_called_once_with(
            "nc_part processing failed for %s", "input.zip"
        )
        ingestor._log.warning.assert_called_once()
        warning_args = ingestor._log.warning.call_args.args
        assert "row range unreadable for 1km" in str(warning_args[2])


class TestVariableDispatch:
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
                    # Two consecutive parts; overlapping rows would fail the scene.
                    return (0, 2) if self.part_path.name == "part-a.nc" else (2, 4)
                raise KeyError(res)

            def read_slot_geometry(self):
                return {}

            def read_calibration(self, channel):
                if self.part_path.name == "part-a.nc" and channel == "vis_04":
                    return ChannelCalibration(1.0, 10.0)
                if self.part_path.name == "part-b.nc" and channel == "vis_06":
                    return ChannelCalibration(2.0, 20.0)
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
        scratch_mod.BatchScratch = _FakeScratch
        monkeypatch.setitem(sys.modules, "firecube_mtg_fci_l1c._scratch", scratch_mod)
        monkeypatch.setattr(
            ingestor_mod,
            "list_fci_nc_parts",
            lambda _dir: [Path("/tmp/part-a.nc"), Path("/tmp/part-b.nc")],
        )
        # SharedNcPartReader instantiates NCPartReader from _decode, so the
        # fake also has to replace that binding for calibration reads to hit it.
        import firecube_mtg_fci_l1c._decode as streaming_mod

        monkeypatch.setattr(streaming_mod, "NCPartReader", FakeReader)
        monkeypatch.setattr(
            ingestor_mod,
            "extract_slot_time_from_path",
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
            run_id="run-1",
            source="/tmp",
            option=lambda *_args: None,
            materialize=Path,
        )

        intents = ingestor.build_write_intents(batch, ctx)  # pyright: ignore[reportArgumentType]

        assert batch.metadata["plugin_failure_counters"]["files_failed"] == 0
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


# --- Loose chunk discovery and the product_type contract -------------------


def _fci_chunk(product: str, kind: str, start: str, cycle: int, number: int) -> str:
    """Real-shaped chunk file name (pattern taken from a 2025 FDHSI product)."""
    return (
        f"W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-{product}-FD--"
        f"CHK-{kind}---NC4E_C_EUMT_{start}_IDPFI_OPE_{start}_{start}_N__O_"
        f"{cycle:04d}_{number:04d}.nc"
    )


def _fci_zip(product: str, start: str, cycle: int) -> str:
    return (
        f"W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-{product}-FD--x-x---x_C_EUMT_"
        f"{start}_IDPFI_OPE_{start}_{start}_N__O_{cycle:04d}_0000.zip"
    )


def _touch(directory: Path, names: list[str], size: int = 0) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = directory / name
        path.write_bytes(b"x" * size)
        paths.append(path)
    return paths


def _ingestor_for(product_type: str | None):
    from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig, MtgFciL1cIngestor

    ingestor = MtgFciL1cIngestor()
    ingestor.plugin_config = MtgFciL1cConfig(product_type=product_type)
    return ingestor


class TestLooseChunkDiscovery:
    def test_flat_directory_of_chunks_gives_one_bundle_per_cycle(self, tmp_path):
        from datetime import datetime

        from firecube_mtg_fci_l1c._data import SceneBundle

        cycle1 = _touch(
            tmp_path,
            [
                _fci_chunk("FDHSI", "TRAIL", "20240101000002", 1, 41),
                _fci_chunk("FDHSI", "BODY", "20240101000031", 1, 2),
                _fci_chunk("FDHSI", "BODY", "20240101000002", 1, 1),
            ],
        )
        cycle2 = _touch(
            tmp_path,
            [
                _fci_chunk("FDHSI", "BODY", "20240101001003", 2, 1),
                _fci_chunk("FDHSI", "BODY", "20240101001052", 2, 33),
            ],
        )
        _touch(tmp_path, ["notes.nc", "manifest.xml"])

        items = list(
            _ingestor_for("FDHSI").discover_source_files(
                SimpleNamespace(source=str(tmp_path))
            )
        )

        assert all(isinstance(item, SceneBundle) for item in items)
        assert [item.coordinate for item in items] == [
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 1, 0, 10),
        ]
        # Original item strings, BODY by chunk number, then TRAIL.
        assert items[0].members == tuple(str(cycle1[i]) for i in (2, 1, 0))
        assert items[1].members == tuple(str(p) for p in cycle2)

    def test_zip_only_discovery_returns_the_zip_items_in_name_order(self, tmp_path):
        zips = _touch(
            tmp_path,
            [
                _fci_zip("FDHSI", "20240101001003", 2),
                _fci_zip("FDHSI", "20240101000002", 1),
            ],
        )
        _touch(tmp_path, ["notes.nc"])

        items = list(
            _ingestor_for("FDHSI").discover_source_files(
                SimpleNamespace(source=str(tmp_path))
            )
        )

        assert items == [str(zips[1]), str(zips[0])]

    def test_zips_and_chunks_together_are_rejected_with_the_filter_hint(self, tmp_path):
        from firecube.ingestor.api import ConfigurationError

        _touch(
            tmp_path,
            [
                _fci_zip("FDHSI", "20240101000002", 1),
                _fci_chunk("FDHSI", "BODY", "20240101001003", 2, 1),
            ],
        )

        with pytest.raises(ConfigurationError) as excinfo:
            _ingestor_for("FDHSI").discover_source_files(
                SimpleNamespace(source=str(tmp_path))
            )

        message = str(excinfo.value)
        assert "--input-filters '[\"!*.zip\"]'" in message
        assert "--input-filters '[\"!*.nc\"]'" in message

    @pytest.mark.parametrize(
        ("configured", "names", "hint"),
        [
            (
                "FDHSI",
                [
                    _fci_chunk("FDHSI", "BODY", "20240101000002", 1, 1),
                    _fci_chunk("HRFI", "BODY", "20240101000002", 1, 1),
                ],
                "--input-filters '[\"!*HRFI*\"]'",
            ),
            (
                "HRFI",
                [_fci_zip("FDHSI", "20240101000002", 1)],
                "--input-filters '[\"!*FDHSI*\"]'",
            ),
        ],
    )
    def test_items_of_the_other_product_type_are_rejected_with_the_filter_hint(
        self, tmp_path, configured, names, hint
    ):
        from firecube.ingestor.api import ConfigurationError

        _touch(tmp_path, names)

        with pytest.raises(ConfigurationError) as excinfo:
            _ingestor_for(configured).discover_source_files(
                SimpleNamespace(source=str(tmp_path))
            )

        assert hint in str(excinfo.value)

    def test_bundle_hooks_label_size_and_keep_the_bundle(self, tmp_path):
        from datetime import datetime

        members = _touch(
            tmp_path,
            [
                _fci_chunk("FDHSI", "BODY", "20240101001003", 2, 1),
                _fci_chunk("FDHSI", "TRAIL", "20240101001003", 2, 41),
            ],
            size=7,
        )
        ingestor = _ingestor_for("FDHSI")
        ctx: Any = SimpleNamespace(source=str(tmp_path))
        (bundle,) = ingestor.discover_source_files(ctx)

        assert ingestor.filter_item(bundle, ctx) is True
        assert ingestor.inspect_item(bundle, ctx) == ItemInfo(
            coordinate=datetime(2024, 1, 1, 0, 10)
        )
        assert ingestor.item_size_bytes(bundle) == 14

        members[1].unlink()
        assert ingestor.item_size_bytes(bundle) is None

    def test_remote_zip_uri_passes_filter_item(self):
        ingestor = _ingestor_for("FDHSI")
        uri = "s3://bucket/fci/" + _fci_zip("FDHSI", "20240101000002", 1)

        assert ingestor.filter_item(uri, SimpleNamespace()) is True
        assert (
            ingestor.filter_item("s3://bucket/fci/notes.nc", SimpleNamespace()) is False
        )


class _SourceGuardCtx:
    """Context whose ``source`` must not be read."""

    run_id = "run-1"
    options: dict[str, Any] = {}

    @property
    def source(self) -> str:
        raise AssertionError("ctx.source was read")

    def option(self, _key: str, default: Any = None) -> Any:
        return default

    def materialize(self, _item: Any) -> Path:
        raise AssertionError("ctx.materialize was called")


class TestProductTypeRequired:
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda ing, ctx: ing.index_spec(ctx), id="index_spec"),
            pytest.param(lambda ing, ctx: ing.zarr_schema(ctx), id="zarr_schema"),
            pytest.param(
                lambda ing, ctx: ing.get_batch_groups([], ctx), id="get_batch_groups"
            ),
            pytest.param(
                lambda ing, ctx: ing.discover_source_files(ctx),
                id="discover_source_files",
            ),
            pytest.param(
                lambda ing, ctx: ing.build_write_intents(
                    SimpleNamespace(items=[], metadata={}, batch_id="batch_0000"), ctx
                ),
                id="build_write_intents",
            ),
        ],
    )
    def test_missing_product_type_raises_without_reading_the_source(self, call):
        from firecube.ingestor.api import ConfigurationError

        with pytest.raises(ConfigurationError, match="product_type") as excinfo:
            call(_ingestor_for(None), _SourceGuardCtx())

        assert "--option product_type=FDHSI" in str(excinfo.value)


class TestBundleMaterialisationFailure:
    def test_failed_member_counts_the_scene_as_failed_under_its_uri(self):
        from firecube_mtg_fci_l1c._data import group_chunks_into_bundles

        (bundle,) = group_chunks_into_bundles(
            [
                "s3://bucket/fci/"
                + _fci_chunk("FDHSI", "BODY", "20240101001003", 2, 1),
                "s3://bucket/fci/"
                + _fci_chunk("FDHSI", "BODY", "20240101001052", 2, 2),
            ]
        )
        ingestor = _ingestor_for("FDHSI")
        ingestor.plugin_config.include_geolocation = False
        calls: list[str] = []

        def materialize(item: str) -> Path:
            calls.append(item)
            if item == bundle.members[1]:
                raise OSError("download failed")
            return Path("/nonexistent") / Path(item).name

        batch: Any = SimpleNamespace(items=[bundle], metadata={}, batch_id="batch_0000")
        ctx: Any = SimpleNamespace(
            run_id="run-1", option=lambda *_args: None, materialize=materialize
        )
        try:
            intents = ingestor.build_write_intents(batch, ctx)
        finally:
            ingestor.cleanup_batch_data(batch, ctx)

        counters = batch.metadata["plugin_failure_counters"]
        assert counters["files_processed"] == 0
        assert counters["files_failed"] == 1
        assert counters["zip_errors"] == [
            "fci-scene://FDHSI/20240101001000: download failed"
        ]
        assert calls == list(bundle.members)
        assert {getattr(intent, "kind", None) for intent in intents} == {"static"}


# --- Partial coverage of output chunks ----------------------------------------


class _RowPayloadReader:
    """Fake shared reader: every pixel of disk row ``r`` holds ``r``.

    ``rows`` maps each part to its disk rows, as the scene's row metadata
    would; payloads are 2 columns wide.
    """

    def __init__(self, rows: dict[Path, tuple[int, int]]) -> None:
        self.rows = rows

    def decode_spatial(self, part, _channel, _index2time, pixel_time_dtype):
        from firecube_mtg_fci_l1c._decode import ChannelSlicePayload

        start, stop = self.rows[Path(part)]
        per_row = np.repeat(np.arange(start, stop)[:, None], 2, axis=1)
        return ChannelSlicePayload(
            counts=per_row.astype(np.uint16),
            pixel_quality=(per_row % 7).astype(np.uint8),
            pixel_time=(per_row * 10.0).astype(pixel_time_dtype),
        )


def _spatial_intents_for_rows(
    part_rows: dict[str, tuple[int, int]],
    *,
    dimsize: int,
    zarr_chunk_y: int | None = None,
) -> list[Any]:
    """Emit, without dispatching, FDHSI 1 km spatial intents for ``part_rows``."""
    from firecube_mtg_fci_l1c._decode import ChunkOwnedAssembler
    from firecube_mtg_fci_l1c._group_plan import GroupPlan
    from firecube_mtg_fci_l1c.ingestor import (
        BatchResources,
        MtgFciL1cConfig,
        MtgFciL1cIngestor,
    )

    rows = {Path(f"/tmp/{name}.nc"): span for name, span in part_rows.items()}
    config = MtgFciL1cConfig(
        product_type="FDHSI",
        zarr_chunk_y=zarr_chunk_y,
        include_geolocation=False,
    )
    plan = GroupPlan(
        product_type="FDHSI",
        resolution="1km",
        group="data_1km",
        dimsize=dimsize,
        logical_channels=("vis_04",),
        nc_channels=("vis_04",),
    )
    ingestor = MtgFciL1cIngestor()
    with ingestor._batch_resources_lock:
        ingestor._batch_resources["batch-1"] = BatchResources(
            chunk_owned_cache=ChunkOwnedAssembler(_RowPayloadReader(rows))  # type: ignore[arg-type]
        )
    intents = ingestor._emit_spatial_intents(
        "batch-1",
        plan,
        0,
        list(rows.items()),
        {0: 0.0},
        np.dtype(np.float64),
        config,
    )
    return intents


def _emit_with_rows(
    part_rows: dict[str, tuple[int, int]],
    *,
    dimsize: int,
    zarr_chunk_y: int | None = None,
) -> dict[str, list[tuple[tuple[int, int], list[int]]]]:
    """Emit and dispatch FDHSI 1 km spatial intents for ``part_rows``.

    Returns, per array, each intent's ``(y_slice, disk rows of its data)``
    in emission order. Data rows are read from the dispatched payload.
    """
    intents = _spatial_intents_for_rows(
        part_rows, dimsize=dimsize, zarr_chunk_y=zarr_chunk_y
    )
    emitted: dict[str, list[tuple[tuple[int, int], list[int]]]] = {}
    for intent in intents:
        data = np.asarray(intent.data())
        assert data.shape[0] == intent.y_slice.stop - intent.y_slice.start
        if intent.array == "pixel_time":
            data = data / 10.0
        elif intent.array == "pixel_quality":
            # r % 7 cannot be inverted; check it against the counts rows.
            counts_rows = np.arange(*emitted["counts"][-1][0])
            np.testing.assert_array_equal(data[:, 0], counts_rows % 7)
            data = np.repeat(counts_rows[:, None], 2, axis=1)
        emitted.setdefault(intent.array, []).append(
            (
                (intent.y_slice.start, intent.y_slice.stop),
                data[:, 0].astype(int).tolist(),
            )
        )
    return emitted


class TestPartialCoverage:
    @pytest.mark.parametrize(
        ("part_rows", "dimsize", "chunk_y", "expected_writes"),
        [
            pytest.param(
                # BODY chunk 32 of FDHSI 1 km holds disk rows [8649, 8908); with
                # the default 278-row chunks it meets output chunks [8618, 8896)
                # and [8896, 9174).
                {"body-32": (8649, 8908)},
                11136,
                None,
                [
                    ((8649, 8896), list(range(8649, 8896))),
                    ((8896, 8908), list(range(8896, 8908))),
                ],
                id="stripe-edge-chunk",
            ),
            pytest.param(
                # Parts 1 and 3 present, part 2 missing; one 12-row output chunk.
                {"body-1": (0, 4), "body-3": (8, 12)},
                12,
                12,
                [((0, 4), [0, 1, 2, 3]), ((8, 12), [8, 9, 10, 11])],
                id="gap-inside-one-output-chunk",
            ),
            pytest.param(
                # Three parts tile 12 rows; 5-row chunks straddle both part
                # boundaries. Same writes as before partial coverage existed.
                {"body-1": (0, 4), "body-2": (4, 8), "body-3": (8, 12)},
                12,
                5,
                [
                    ((0, 5), [0, 1, 2, 3, 4]),
                    ((5, 10), [5, 6, 7, 8, 9]),
                    ((10, 12), [10, 11]),
                ],
                id="full-scene-across-part-boundaries",
            ),
            pytest.param(
                {"body-2": (4, 8)},
                12,
                4,
                [((4, 8), [4, 5, 6, 7])],
                id="chunks-meeting-no-part",
            ),
        ],
    )
    def test_each_array_is_written_only_for_covered_rows(
        self, part_rows, dimsize, chunk_y, expected_writes
    ):
        emitted = _emit_with_rows(part_rows, dimsize=dimsize, zarr_chunk_y=chunk_y)

        assert sorted(emitted) == ["counts", "pixel_quality", "pixel_time"]
        for array, writes in emitted.items():
            assert len(writes) == len(expected_writes), array
            assert writes == expected_writes, array

    @pytest.mark.parametrize(
        "part_rows",
        [
            pytest.param({"body-1": (0, 5), "body-2": (4, 8)}, id="overlap"),
            pytest.param(
                {"body-1": (0, 2), "body-2": (2, 4), "body-3": (4, 8)},
                id="three-parts",
            ),
        ],
    )
    def test_overlapping_or_three_parts_in_one_chunk_are_rejected_at_emission(
        self, part_rows
    ):
        from firecube_mtg_fci_l1c._decode import AssemblyPreconditionError

        # No payload is dispatched: emission itself must refuse.
        with pytest.raises(AssemblyPreconditionError):
            _spatial_intents_for_rows(part_rows, dimsize=8, zarr_chunk_y=8)


def _write_disk_rows_chunk(
    path: Path,
    rows: tuple[int, int],
    *,
    index: int = 1,
    time: float = 60.0,
    latitude: float | None = None,
    scale_factor: float = 1.0,
    pixel_index: int | None = None,
) -> None:
    """Write a ``vis_04`` BODY chunk for disk rows ``[start, stop)``, 2 columns.

    Pixels of disk row ``r`` hold ``r``. The root table maps ``index`` to
    ``time`` seconds and, with ``latitude``, to a sub-satellite latitude;
    ``scale_factor`` is the calibration slope. Pixels point at ``index``
    unless ``pixel_index`` names another root-table row.
    """
    import h5netcdf

    start, stop = rows
    per_row = np.repeat(np.arange(start, stop)[:, None], 2, axis=1)
    with h5netcdf.File(path, "w") as ds:
        ds.dimensions["n_time"] = 1
        ds.create_variable("index", ("n_time",), data=np.array([index], np.uint16))
        ds.create_variable("time", ("n_time",), data=np.array([time]))
        if latitude is not None:
            platform = ds.create_group("state").create_group("platform")
            platform.create_variable(
                "subsatellite_latitude",
                ("n_time",),
                data=np.array([latitude], np.float32),
            )
        measured = (
            ds.create_group("data").create_group("vis_04").create_group("measured")
        )
        measured.dimensions["y"] = stop - start
        measured.dimensions["x"] = 2
        radiance = measured.create_variable(
            "effective_radiance", ("y", "x"), data=per_row.astype(np.uint16)
        )
        radiance.attrs["scale_factor"] = scale_factor
        radiance.attrs["add_offset"] = 0.0
        measured.create_variable("start_position_row", (), data=np.int32(start + 1))
        measured.create_variable("end_position_row", (), data=np.int32(stop))
        measured.create_variable(
            "pixel_quality", ("y", "x"), data=np.zeros(per_row.shape, np.uint8)
        )
        measured.create_variable(
            "index_map",
            ("y", "x"),
            data=np.full(per_row.shape, pixel_index or index, np.uint16),
        )


def _stripe_bundle(
    tmp_path: Path,
    rows_by_chunk: dict[int, tuple[int, int]],
    root_tables: dict[int, dict[str, Any]] | None = None,
):
    """Write BODY chunk files; ``root_tables`` gives per-chunk writer overrides."""
    from firecube_mtg_fci_l1c._data import group_chunks_into_bundles
    from tests._support import _chunk_name

    paths = []
    for number, rows in rows_by_chunk.items():
        path = tmp_path / _chunk_name("BODY", "20240101000002", 1, number)
        _write_disk_rows_chunk(path, rows, **(root_tables or {}).get(number, {}))
        paths.append(str(path))
    (bundle,) = group_chunks_into_bundles(paths)
    return bundle


def _scene_writes(
    item: Any, *, materialize: Any = Path, **options: Any
) -> tuple[list[Any], dict]:
    """Run build_write_intents on one item (FDHSI 1 km, vis_04 only).

    Payloads are resolved before batch cleanup; returns the slot and region
    intents as ``(kind, array, y_slice or None, data)`` and the failure
    counters. ``materialize`` stands in for ``ctx.materialize``.
    """
    from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig, MtgFciL1cIngestor

    ingestor = MtgFciL1cIngestor()
    ingestor.plugin_config = MtgFciL1cConfig(
        product_type="FDHSI",
        resolutions="1km",
        channels="vis_04",
        time_epoch="2024-01-01",
        include_geolocation=False,
        **options,
    )
    batch: Any = SimpleNamespace(items=[item], metadata={}, batch_id="batch_0000")
    ctx: Any = SimpleNamespace(
        run_id="run-1", option=lambda *_a: None, materialize=materialize
    )
    try:
        intents = ingestor.build_write_intents(batch, ctx)
        writes = [
            (
                i._kind,
                i.array,
                None if i.y_slice is None else (i.y_slice.start, i.y_slice.stop),
                np.asarray(i.data() if callable(i.data) else i.data),
            )
            for i in intents
            if getattr(i, "_kind", None) in ("slot", "region")
        ]
    finally:
        ingestor.cleanup_batch_data(batch, ctx)
    return writes, batch.metadata["plugin_failure_counters"]


def _build_scene_intents(
    bundle: Any, *, materialize: Any = Path, **options: Any
) -> tuple[list[Any], dict]:
    """Return a bundle's region intents as ``(array, y_slice, data)`` and counters."""
    writes, counters = _scene_writes(bundle, materialize=materialize, **options)
    regions = [
        (array, y_slice, data)
        for kind, array, y_slice, data in writes
        if kind == "region"
    ]
    return regions, counters


class TestPartialChunkOption:
    def test_fill_writes_the_covered_rows_of_a_real_chunk_file(self, tmp_path):
        bundle = _stripe_bundle(tmp_path, {32: (8649, 8908)})

        regions, counters = _build_scene_intents(bundle)

        assert counters == {"files_processed": 1, "files_failed": 0, "zip_errors": []}
        counts = [(y, data) for array, y, data in regions if array == "counts"]
        assert [y for y, _data in counts] == [(8649, 8896), (8896, 8908)]
        np.testing.assert_array_equal(counts[0][1][:, 0], np.arange(8649, 8896))
        np.testing.assert_array_equal(counts[1][1][:, 0], np.arange(8896, 8908))
        pixel_time = [data for array, _y, data in regions if array == "pixel_time"]
        assert [data.shape for data in pixel_time] == [(247, 2), (12, 2)]
        assert all(np.all(data == 60.0) for data in pixel_time)

    def test_fill_splits_a_chunk_around_a_missing_middle_chunk(self, tmp_path):
        # BODY 30 [8133, 8391) and 32 [8649, 8908), 31 missing. The 556-row
        # output chunk [8340, 8896) holds the end of 30, all of 31's rows and
        # the start of 32.
        bundle = _stripe_bundle(tmp_path, {30: (8133, 8391), 32: (8649, 8908)})

        regions, counters = _build_scene_intents(bundle, zarr_chunk_y=556)

        assert counters["files_failed"] == 0
        counts = [(y, data) for array, y, data in regions if array == "counts"]
        assert [y for y, _data in counts] == [
            (8133, 8340),
            (8340, 8391),
            (8649, 8896),
            (8896, 8908),
        ]
        for (y_start, y_stop), data in counts:
            np.testing.assert_array_equal(data[:, 0], np.arange(y_start, y_stop))

    def test_error_names_the_scene_group_and_missing_chunk(self, tmp_path):
        bundle = _stripe_bundle(tmp_path, {32: (8649, 8908)})

        with pytest.raises(ConfigurationError) as excinfo:
            _build_scene_intents(bundle, partial_chunk="error")

        message = str(excinfo.value)
        assert bundle.uri in message
        assert "'data_1km'" in message
        assert "rows [8618, 8649)" in message
        assert "missing BODY chunk(s) 31" in message
        assert "partial_chunk=fill" in message

    def test_error_names_every_chunk_missing_around_and_between_parts(self, tmp_path):
        # Gaps with 556-row chunks: [7784, 8133) -> BODY 28, 29;
        # [8391, 8649) -> 31; [8908, 9452) -> 33, 34.
        bundle = _stripe_bundle(tmp_path, {30: (8133, 8391), 32: (8649, 8908)})

        with pytest.raises(ConfigurationError) as excinfo:
            _build_scene_intents(bundle, partial_chunk="error", zarr_chunk_y=556)

        assert "missing BODY chunk(s) 28, 29, 31, 33, 34" in str(excinfo.value)

    @pytest.mark.parametrize("partial_chunk", ["fill", "error"])
    def test_full_disk_scene_writes_every_output_chunk_whole(
        self, tmp_path, partial_chunk
    ):
        from firecube_mtg_fci_l1c._constants import BODY_CHUNK_ROWS

        rows = dict(enumerate(BODY_CHUNK_ROWS["FDHSI"]["1km"], start=1))
        bundle = _stripe_bundle(tmp_path, rows)

        regions, counters = _build_scene_intents(bundle, partial_chunk=partial_chunk)

        assert counters["files_failed"] == 0
        counts = [(y, data) for array, y, data in regions if array == "counts"]
        expected_slices = [(start, start + 278) for start in range(0, 11120, 278)]
        assert [y for y, _data in counts] == [*expected_slices, (11120, 11136)]
        disk = np.concatenate([data for _y, data in counts])
        np.testing.assert_array_equal(disk[:, 0], np.arange(11136))


class TestRootTableFailureVsCoverage:
    """A failing root table must not mask a strict coverage error."""

    @staticmethod
    def _fail(monkeypatch, accumulator: str) -> None:
        from firecube_mtg_fci_l1c import ingestor as ingestor_mod

        def boom(self: Any, reader: Any) -> None:
            raise ValueError("malformed root table")

        monkeypatch.setattr(getattr(ingestor_mod, accumulator), "accumulate", boom)

    @pytest.mark.parametrize(
        "accumulator", ["TimeMapAccumulator", "SlotGeometryAccumulator"]
    )
    def test_error_mode_coverage_error_wins(self, tmp_path, monkeypatch, accumulator):
        bundle = _stripe_bundle(tmp_path, {32: (8649, 8908)})
        self._fail(monkeypatch, accumulator)

        with pytest.raises(ConfigurationError, match="missing BODY chunk"):
            _scene_writes(bundle, partial_chunk="error", include_pixel_time=True)

    @pytest.mark.parametrize(
        "accumulator", ["TimeMapAccumulator", "SlotGeometryAccumulator"]
    )
    def test_fill_mode_root_table_failure_is_a_scene_failure(
        self, tmp_path, monkeypatch, accumulator
    ):
        bundle = _stripe_bundle(tmp_path, {32: (8649, 8908)})
        self._fail(monkeypatch, accumulator)

        _writes, counters = _scene_writes(
            bundle, partial_chunk="fill", include_pixel_time=True
        )

        assert counters["files_failed"] == 1
        assert counters["files_processed"] == 0
        assert "malformed root table" in counters["zip_errors"][0]


def _write_body_2_only(directory: Path) -> None:
    """Write BODY 2 (rows 2-3 of the 4-row small grid) of cycle 1, values 7."""
    from tests._support import _chunk_name, _write_chunk_netcdf

    directory.mkdir()
    _write_chunk_netcdf(
        directory / _chunk_name("BODY", "20240101000002", 1, 2),
        rows=(2, 4),
        value=7,
        index=[1],
        times=[30.0],
    )


_LAYOUTS = [
    pytest.param({}, "group 'data_1km'", id="nested"),
    pytest.param(
        {"flat_store": True, "resolutions": "1km"}, "the root group", id="flat"
    ),
]


@pytest.mark.integration
@pytest.mark.plugin
@pytest.mark.parametrize(("layout", "where"), _LAYOUTS)
def test_partial_chunk_error_fails_the_run_before_the_store_is_written(
    tmp_path: Path, small_fci_layout: list[int], layout: dict, where: str
):
    from tests._store_files import store_files
    from tests._support import _run_ingest

    src = tmp_path / "loose"
    _write_body_2_only(src)
    workspace = tmp_path / "out"
    workspace.mkdir()

    # Core fails the batch on the plugin's ConfigurationError and ends the
    # run with PipelineFailedBatchesError (a RuntimeError) carrying it.
    with pytest.raises(
        RuntimeError,
        match=r"does not cover output chunks it would write: "
        + re.escape(where)
        + r" \(1km\) rows \[0, 2\)",
    ):
        _run_ingest(
            src,
            workspace,
            {"partial_chunk": "error", "include_geolocation": False, **layout},
        )

    # Core opens the store and records the failed run under .firecube/; no
    # group, array or chunk is written.
    target = workspace / "out.zarr"
    assert sorted(store_files(target)) == ["zarr.json"]


@pytest.mark.integration
@pytest.mark.plugin
def test_partial_chunk_error_batch_leaves_static_coordinates_to_later_batches(
    tmp_path: Path, small_fci_layout: list[int]
):
    import zarr

    from tests._support import _run_ingest, _write_scene_chunks

    # Batch 0 is the partial cycle 1 and fails; batch 1 is the complete
    # cycle 2 and must still write the static coordinates.
    src = tmp_path / "loose"
    _write_body_2_only(src)
    _write_scene_chunks(src, start="20240101001003", cycle=2, values=(7, 8))
    reference_src = tmp_path / "reference"
    _write_scene_chunks(reference_src, start="20240101001003", cycle=2, values=(7, 8))
    workspace = tmp_path / "out"
    workspace.mkdir()
    reference_workspace = tmp_path / "reference_out"
    reference_workspace.mkdir()
    options = {
        "partial_chunk": "error",
        "include_geolocation": False,
        "pipeline_batch_size": 1,
        "pipeline_workers": 1,
    }

    with pytest.raises(RuntimeError, match="does not cover output chunks"):
        _run_ingest(src, workspace, options)
    reference = _run_ingest(reference_src, reference_workspace, options)

    root = cast(Any, zarr.open_group(str(workspace / "out.zarr"), mode="r"))
    reference_root = cast(Any, zarr.open_group(str(reference), mode="r"))
    for group_name, channels in [
        ("data_1km", ["vis_04", "vis_06"]),
        ("data_2km", ["ir_38"]),
    ]:
        group = root[group_name]
        assert group["channel"][:].tolist() == channels
        for axis in ("x", "y"):
            expected = np.asarray(reference_root[group_name][axis][:])
            assert np.count_nonzero(expected) == expected.size
            np.testing.assert_array_equal(np.asarray(group[axis][:]), expected)
    counts = np.asarray(root["data_1km"]["counts"][:])
    np.testing.assert_array_equal(counts[1, :2, :, 0], np.full((2, 4), 7))


@pytest.mark.integration
@pytest.mark.plugin
@pytest.mark.parametrize(("layout", "where"), _LAYOUTS)
def test_partial_chunk_fill_leaves_uncovered_rows_at_the_fill_value(
    tmp_path: Path, small_fci_layout: list[int], layout: dict, where: str
):
    import zarr

    from tests._support import _run_ingest

    del where
    src = tmp_path / "loose"
    _write_body_2_only(src)

    out = _run_ingest(src, tmp_path, {"include_geolocation": False, **layout})

    root = cast(Any, zarr.open_group(str(out), mode="r"))
    groups = (
        [(root, 0, 7)]
        if layout
        else [
            (root["data_1km"], 0, 7),
            (root["data_2km"], 0, 9),
        ]
    )
    for group, channel, value in groups:
        counts = group["counts"]
        slot = np.asarray(counts[0, :, :, channel])
        assert counts.fill_value == np.iinfo(np.uint16).max
        np.testing.assert_array_equal(slot[:2], np.full((2, 4), 65535))
        np.testing.assert_array_equal(slot[2:], np.full((2, 4), value))


@pytest.mark.integration
@pytest.mark.plugin
def test_partial_chunk_change_does_not_bypass_the_resume_conflict(
    tmp_path: Path, small_fci_layout: list[int]
):
    import zarr
    from firecube.ingestor.api import ResumeConflictError

    from tests._support import _run_ingest, _write_scene_chunks

    first = tmp_path / "first"
    _write_scene_chunks(first, start="20240101000002", cycle=1, values=(5, 6))
    # The same scene again, with different pixel values so an overwrite shows.
    second = tmp_path / "second"
    _write_scene_chunks(second, start="20240101000002", cycle=1, values=(7, 8))
    options: dict[str, object] = {"include_geolocation": False}

    out = _run_ingest(first, tmp_path, {**options, "partial_chunk": "fill"})
    before = np.asarray(zarr.open_group(str(out), mode="r")["data_1km"]["counts"][:])
    np.testing.assert_array_equal(before[0, :2, :, 0], np.full((2, 4), 5))

    with pytest.raises(ResumeConflictError, match="Existing entries"):
        _run_ingest(
            second,
            tmp_path,
            {**options, "partial_chunk": "error", "force_reingest": False},
        )

    after = np.asarray(zarr.open_group(str(out), mode="r")["data_1km"]["counts"][:])
    np.testing.assert_array_equal(after, before)


# --- Stripe read path (body_chunks) --------------------------------------------

# Synthetic FDHSI 1 km grid: 24 rows, six BODY chunks of four rows. With
# body_chunks=[3, 4] (rows [8, 16)) and 3-row output chunks the window is
# [6, 18): BODY 2 and 5 reach into its edge chunks, BODY 1 and 6 lie outside.
_STRIPE_TABLE = ((0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24))
_STRIPE_OPTIONS: dict[str, Any] = {"zarr_chunk_y": 3, "body_chunks": [3, 4]}


@pytest.fixture
def stripe_grid(monkeypatch):
    """Patch FDHSI 1 km to the 24-row grid and its six-chunk row table."""
    from firecube_mtg_fci_l1c import _constants as const_mod

    monkeypatch.setitem(const_mod.CONSTANTS["FDHSI"]["1km"], "dimsize", 24)
    monkeypatch.setitem(const_mod.BODY_CHUNK_ROWS["FDHSI"], "1km", _STRIPE_TABLE)


def _stripe_parts(tmp_path: Path, numbers: Iterable[int]) -> Any:
    return _stripe_bundle(tmp_path, {n: _STRIPE_TABLE[n - 1] for n in numbers})


def _region_rows(regions: list[Any], array: str) -> list[tuple[tuple, list[int]]]:
    """Return ``(y_slice, disk rows of the data)`` of one array's region writes."""
    return [
        (y_slice, data[:, 0].astype(int).tolist())
        for name, y_slice, data in regions
        if name == array
    ]


@pytest.mark.usefixtures("stripe_grid")
class TestStripeReadPath:
    def test_parts_outside_the_window_feed_root_tables_but_their_pixels_are_never_read(
        self, tmp_path, monkeypatch
    ):
        from firecube_mtg_fci_l1c._data import parse_chunk_name
        from firecube_mtg_fci_l1c._decode import NCPartReader

        outside = {1, 6}
        read_channel_data = NCPartReader.read_channel_data

        def guarded(reader: Any, channel: str) -> Any:
            number = parse_chunk_name(reader.path).chunk_number
            if number in outside:
                raise AssertionError(f"pixels of out-of-window BODY {number} read")
            return read_channel_data(reader, channel)

        monkeypatch.setattr(NCPartReader, "read_channel_data", guarded)
        # BODY 1 (first) and 6 (last) lie outside the window. Each carries
        # distinctive root-table values: BODY 1 a calibration slope, a latitude
        # and time row 7; BODY 6 a latitude and a later time for row 1, which
        # wins over rows 2-5. BODY 4's pixels point at BODY 1's row 7.
        bundle = _stripe_bundle(
            tmp_path,
            {n: _STRIPE_TABLE[n - 1] for n in range(1, 7)},
            root_tables={
                1: {"index": 7, "time": 70.0, "latitude": 10.0, "scale_factor": 0.5},
                4: {"pixel_index": 7},
                6: {"index": 1, "time": 600.0, "latitude": 30.0},
            },
        )

        writes, counters = _scene_writes(bundle, **_STRIPE_OPTIONS)

        assert counters["files_failed"] == 0
        regions = [(a, y, d) for kind, a, y, d in writes if kind == "region"]
        slots = {a: d for kind, a, _y, d in writes if kind == "slot"}
        # Pixels still come from the in-window parts.
        assert [y for y, _rows in _region_rows(regions, "counts")] == [
            (0, 3),
            (3, 6),
            (6, 9),
            (9, 12),
        ]
        # Time map: row 1 from BODY 6, row 7 from BODY 1 (via BODY 4's pixels).
        pixel_time = [
            (y, d[:, 0].tolist()) for name, y, d in regions if name == "pixel_time"
        ]
        assert pixel_time == [
            ((0, 3), [600.0, 600.0, 600.0]),
            ((3, 6), [600.0, 600.0, 600.0]),
            ((6, 9), [70.0, 70.0, 70.0]),
            ((9, 12), [70.0, 600.0, 600.0]),
        ]
        # Slot geometry: mean of BODY 1's and BODY 6's latitudes.
        assert float(slots["subsatellite_latitude"]) == 20.0
        # Calibration comes from the first part in scene order, BODY 1.
        np.testing.assert_array_equal(slots["slope"], [0.5])

    def test_region_y_slices_are_relative_to_the_window_start(self, tmp_path):
        bundle = _stripe_parts(tmp_path, range(1, 7))

        regions, _counters = _build_scene_intents(bundle, **_STRIPE_OPTIONS)

        expected = [
            ((0, 3), [6, 7, 8]),
            ((3, 6), [9, 10, 11]),
            ((6, 9), [12, 13, 14]),
            ((9, 12), [15, 16, 17]),
        ]
        assert _region_rows(regions, "counts") == expected
        for array in ("pixel_quality", "pixel_time"):
            assert [y for name, y, _d in regions if name == array] == [
                y for y, _rows in expected
            ]

    def test_fill_writes_the_covered_window_rows_around_a_missing_chunk(self, tmp_path):
        bundle = _stripe_parts(tmp_path, [1, 2, 4, 5, 6])

        regions, counters = _build_scene_intents(bundle, **_STRIPE_OPTIONS)

        assert counters["files_failed"] == 0
        # Window chunk [6, 9) keeps BODY 2's rows 6-7; [9, 12) meets no part.
        assert _region_rows(regions, "counts") == [
            ((0, 2), [6, 7]),
            ((6, 9), [12, 13, 14]),
            ((9, 12), [15, 16, 17]),
        ]

    def test_error_names_only_the_missing_chunk_inside_the_window(self, tmp_path):
        # BODY 1, 3 and 6 are missing; only BODY 3's rows meet a window chunk
        # that a present part also meets.
        bundle = _stripe_parts(tmp_path, [2, 4, 5])

        with pytest.raises(ConfigurationError) as excinfo:
            _build_scene_intents(bundle, partial_chunk="error", **_STRIPE_OPTIONS)

        assert "group 'data_1km' (1km) rows [8, 9); missing BODY chunk(s) 3. " in str(
            excinfo.value
        )

    def test_error_mode_ignores_gaps_outside_the_window(self, tmp_path):
        # BODY 6 is missing: full-disk chunk [18, 21) would have a gap, but it
        # lies outside the window.
        bundle = _stripe_parts(tmp_path, range(1, 6))

        regions, counters = _build_scene_intents(
            bundle, partial_chunk="error", **_STRIPE_OPTIONS
        )

        assert counters["files_failed"] == 0
        assert [y for y, _rows in _region_rows(regions, "counts")] == [
            (0, 3),
            (3, 6),
            (6, 9),
            (9, 12),
        ]

    def test_scene_without_a_part_in_the_window_writes_nothing(self, tmp_path):
        bundle = _stripe_parts(tmp_path, [1, 6])

        writes, counters = _scene_writes(bundle, **_STRIPE_OPTIONS)

        assert counters["files_failed"] == 0
        assert writes == []

    def test_part_rows_differing_from_the_row_table_fail_the_scene(self, tmp_path):
        bundle = _stripe_bundle(tmp_path, {3: (8, 11), 4: (11, 16)})

        writes, counters = _scene_writes(bundle, **_STRIPE_OPTIONS)

        assert writes == []
        assert counters["files_failed"] == 1
        (error,) = counters["zip_errors"]
        assert error.startswith(bundle.uri)
        assert "BODY chunk 3" in error
        assert "rows [8, 11)" in error
        assert "row table gives [8, 12)" in error

    def test_full_disk_ingest_does_not_check_the_row_table(self, tmp_path):
        bundle = _stripe_bundle(tmp_path, {3: (8, 11), 4: (11, 16)})

        regions, counters = _build_scene_intents(bundle, zarr_chunk_y=3)

        assert counters["files_failed"] == 0
        # Full-disk y_slices are disk rows; the odd rows are placed as read.
        assert _region_rows(regions, "counts") == [
            ((8, 9), [8]),
            ((9, 12), [9, 10, 11]),
            ((12, 15), [12, 13, 14]),
            ((15, 16), [15]),
        ]

    def test_chunk_numbers_come_from_the_item_names_not_the_materialised_paths(
        self, tmp_path
    ):
        # Core's remote materialiser caches each URI as <sha256[:16]>.nc, so
        # the local file name carries no chunk number.
        import hashlib
        import shutil

        chunks = tmp_path / "chunks"
        cache = tmp_path / "_remote_cache"
        chunks.mkdir()
        cache.mkdir()
        bundle = _stripe_parts(chunks, range(1, 7))

        def hash_named_copy(item: str) -> Path:
            target = cache / f"{hashlib.sha256(item.encode()).hexdigest()[:16]}.nc"
            if not target.exists():
                shutil.copy(item, target)
            return target

        renamed, counters = _build_scene_intents(
            bundle, materialize=hash_named_copy, **_STRIPE_OPTIONS
        )
        original, _counters = _build_scene_intents(bundle, **_STRIPE_OPTIONS)

        assert counters == {"files_processed": 1, "files_failed": 0, "zip_errors": []}
        assert len(list(cache.glob("????????????????.nc"))) == 6
        assert [(a, y) for a, y, _d in renamed] == [(a, y) for a, y, _d in original]
        assert _region_rows(renamed, "counts") == [
            ((0, 3), [6, 7, 8]),
            ((3, 6), [9, 10, 11]),
            ((6, 9), [12, 13, 14]),
            ((9, 12), [15, 16, 17]),
        ]
        for (_a, _y, got), (_b, _z, want) in zip(renamed, original, strict=True):
            np.testing.assert_array_equal(got, want)

    def test_part_name_without_a_chunk_number_fails_the_scene(self, tmp_path):
        import zipfile

        from firecube_mtg_fci_l1c._constants import get_nc_part_prefix

        part = tmp_path / "part.nc"
        _write_disk_rows_chunk(part, _STRIPE_TABLE[2])
        zip_path = tmp_path / (
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--x-x---x_C_EUMT_"
            "20240101000318_IDPFI_OPE_20240101000002_20240101000934_N__O_0001_0000.zip"
        )
        member = f"{get_nc_part_prefix('FDHSI')}0003.nc"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(part, arcname=member)

        writes, counters = _scene_writes(
            str(zip_path), scratch_dir=str(tmp_path / "scratch"), **_STRIPE_OPTIONS
        )

        assert writes == []
        (error,) = counters["zip_errors"]
        assert f"Cannot read the BODY chunk number of {member}" in error
