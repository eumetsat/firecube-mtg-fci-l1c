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

# pyright: reportMissingImports=false

import copy
import zipfile
from pathlib import Path
from unittest.mock import patch

import h5netcdf
import numpy as np
import pytest

from firecube_mtg_fci_l1c import (  # pyright: ignore[reportMissingImports]
    _decode as streaming_mod,
)
from firecube_mtg_fci_l1c._decode import (
    AssemblyPreconditionError,
    ChannelSlicePayload,
    ChunkOwnedAssembler,
    NCPartReader,
    SharedNcPartReader,
    TimeMapAccumulator,
    _IdentityRef,
    expand_pixel_time,
    list_fci_nc_parts,
)  # pyright: ignore[reportMissingImports]
from firecube_mtg_fci_l1c._constants import (  # pyright: ignore[reportMissingImports]
    PRODUCT_TYPE_FDHSI,
    get_nc_part_prefix,
)


def _write_nc_part(
    path: Path,
    *,
    channel_specs: dict[str, tuple[int, int, int]],
    index_values: np.ndarray | None = None,
    time_values: np.ndarray | None = None,
    include_time_map: bool = True,
    scale_factor: float = 0.5,
    add_offset: float = 1.5,
) -> None:
    index_values_arr = (
        np.asarray(index_values, dtype=np.int64)
        if index_values is not None
        else np.asarray([0, 1, 2], dtype=np.int64)
    )
    time_values_arr = (
        np.asarray(time_values, dtype=np.float64)
        if time_values is not None
        else np.asarray([0.0, 60.0, 120.0], dtype=np.float64)
    )

    with h5netcdf.File(path, "w") as ds:
        if include_time_map:
            ds.dimensions["n_time"] = len(index_values_arr)
            ds.create_variable("index", ("n_time",), data=index_values_arr)
            ds.create_variable("time", ("n_time",), data=time_values_arr)

        data_group = ds.create_group("data")
        for channel, (start_row, end_row, base) in channel_specs.items():
            measured = data_group.create_group(channel).create_group("measured")
            measured.dimensions["y"] = 2
            measured.dimensions["x"] = 3

            radiance = measured.create_variable(
                "effective_radiance",
                ("y", "x"),
                data=np.full((2, 3), base, dtype=np.uint16),
            )
            radiance.attrs["start_position_row"] = start_row
            radiance.attrs["end_position_row"] = end_row
            radiance.attrs["scale_factor"] = scale_factor
            radiance.attrs["add_offset"] = add_offset

            measured.create_variable(
                "pixel_quality",
                ("y", "x"),
                data=np.full((2, 3), base + 1, dtype=np.uint8),
            )
            measured.create_variable(
                "index_map",
                ("y", "x"),
                data=np.asarray([[0, 1, 99], [2, -1, 1]], dtype=np.int32),
            )


@pytest.fixture
def _small_fci_constants(monkeypatch):
    from firecube_mtg_fci_l1c import _constants as const_mod  # pyright: ignore[reportMissingImports]

    backup = copy.deepcopy(const_mod.CONSTANTS)
    const_mod.CONSTANTS[PRODUCT_TYPE_FDHSI] = {
        "1km": {"channels": ["vis_04"], "dimsize": 2, "nc_channels": ["vis_04"]},
        "2km": {"channels": ["ir_38"], "dimsize": 2, "nc_channels": ["ir_38"]},
    }
    yield
    const_mod.CONSTANTS.clear()
    const_mod.CONSTANTS.update(backup)


@pytest.fixture
def fci_test_zip(tmp_path: Path, _small_fci_constants) -> Path:
    prefix = get_nc_part_prefix(PRODUCT_TYPE_FDHSI)
    trail_prefix = prefix.replace("BODY", "TRAIL")

    src = tmp_path / "src"
    src.mkdir()
    body_0001 = src / "body_0001.nc"
    body_0002 = src / "body_0002.nc"
    trail_0001 = src / "trail_0001.nc"
    other = src / "not_a_part.nc"

    _write_nc_part(
        body_0001,
        channel_specs={
            "vis_04": (1, 4, 11),
            "ir_38": (5, 8, 21),
        },
    )
    _write_nc_part(
        body_0002,
        channel_specs={
            "vis_04": (9, 12, 31),
            "ir_38": (13, 16, 41),
        },
    )
    _write_nc_part(
        trail_0001,
        channel_specs={
            "vis_04": (17, 20, 51),
            "ir_38": (21, 24, 61),
        },
    )
    _write_nc_part(other, channel_specs={"vis_04": (1, 2, 99)})

    zip_path = tmp_path / "W_XX-FCI-1C-RRAD-FDHSI-FD-20240101000000-END.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(body_0002, arcname=f"{prefix}0002.nc")
        zf.write(trail_0001, arcname=f"{trail_prefix}0001.nc")
        zf.write(body_0001, arcname=f"{prefix}0001.nc")
        zf.write(other, arcname="nested/not_a_part.nc")
    return zip_path


@pytest.mark.unit
def test_nc_part_reader_channel_data_and_row_ranges(fci_test_zip: Path, tmp_path: Path):
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    with zipfile.ZipFile(fci_test_zip) as zf:
        zf.extractall(extract_dir)

    first_body = list_fci_nc_parts(extract_dir)[0]

    with NCPartReader(first_body) as reader:
        assert reader.read_row_range("1km") == (0, 4)
        assert reader.read_row_range("2km") == (4, 8)
        assert reader.available_channels() == ["ir_38", "vis_04"]

        effective_radiance, pixel_quality, index_map = reader.read_channel_data(
            "vis_04"
        )
        assert effective_radiance.shape == (2, 3)
        assert pixel_quality.shape == (2, 3)
        assert index_map.shape == (2, 3)
        assert effective_radiance[0, 0] == 11
        assert pixel_quality[0, 0] == 12
        assert index_map[0, 2] == 99


@pytest.mark.unit
def test_time_map_accumulator_merges_and_overrides(tmp_path: Path):
    part1 = tmp_path / "W_XX-FCI-1C-RRAD-FDHSI-FD-body-0001.nc"
    part2 = tmp_path / "W_XX-FCI-1C-RRAD-FDHSI-FD-body-0002.nc"

    _write_nc_part(
        part1,
        channel_specs={"vis_04": (1, 2, 1)},
        index_values=np.asarray([0, 1], dtype=np.int64),
        time_values=np.asarray([0.0, 60.0], dtype=np.float64),
    )
    _write_nc_part(
        part2,
        channel_specs={"vis_04": (1, 2, 1)},
        index_values=np.asarray([1, 2], dtype=np.int64),
        time_values=np.asarray([61.0, 120.0], dtype=np.float64),
    )

    accumulator = TimeMapAccumulator()
    with NCPartReader(part1) as reader1:
        accumulator.accumulate(reader1)
    with NCPartReader(part2) as reader2:
        accumulator.accumulate(reader2)

    assert accumulator.build_index2time() == {0: 0.0, 1: 61.0, 2: 120.0}


@pytest.mark.unit
def test_time_map_accumulator_skips_parts_without_root_time_map(tmp_path: Path):
    body = tmp_path / "W_XX-FCI-1C-RRAD-FDHSI-FD-body-0001.nc"
    trail = tmp_path / "W_XX-FCI-1C-RRAD-FDHSI-FD-trail-0002.nc"

    _write_nc_part(
        body,
        channel_specs={"vis_04": (1, 2, 1)},
        index_values=np.asarray([5, 6], dtype=np.int64),
        time_values=np.asarray([50.0, 60.0], dtype=np.float64),
    )
    _write_nc_part(
        trail,
        channel_specs={"vis_04": (1, 2, 1)},
        include_time_map=False,
    )

    accumulator = TimeMapAccumulator()
    with NCPartReader(body) as body_reader:
        accumulator.accumulate(body_reader)
    with NCPartReader(trail) as trail_reader:
        accumulator.accumulate(trail_reader)

    assert accumulator.build_index2time() == {5: 50.0, 6: 60.0}


@pytest.mark.unit
def test_nc_part_reader_reads_calibration_from_radiance_attrs(tmp_path: Path):
    part = tmp_path / "W_XX-FCI-1C-RRAD-FDHSI-FD-body-0001.nc"
    _write_nc_part(
        part,
        channel_specs={"vis_04": (1, 2, 1)},
        scale_factor=0.25,
        add_offset=3.5,
    )

    with NCPartReader(part) as reader:
        assert reader.read_calibration("vis_04") == pytest.approx((0.25, 3.5))


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_expand_pixel_time_dtype_and_nan_for_unmapped(dtype):
    index_map = np.asarray([[0, 1, 3], [2, -1, 99]], dtype=np.int32)
    index2time = {0: 0.0, 1: 10.5, 2: 20.25}

    expanded = expand_pixel_time(
        index_map=index_map, index2time=index2time, dtype=dtype
    )

    assert expanded.dtype == np.dtype(dtype)
    assert expanded[0, 0] == pytest.approx(0.0)
    assert expanded[0, 1] == pytest.approx(10.5)
    assert expanded[1, 0] == pytest.approx(20.25)
    assert np.isnan(expanded[0, 2])
    assert np.isnan(expanded[1, 1])
    assert np.isnan(expanded[1, 2])


@pytest.mark.unit
def test_pixel_time_lookup_float32_matches_mask_based():
    index_map = np.asarray([[0, 1, 99], [2, 7, 1]], dtype=np.uint16)
    index2time = {0: 0.0, 1: 10.5, 2: 20.25}
    output_dtype = np.dtype(np.float32)
    fill_value = output_dtype.type(np.nan)

    direct = streaming_mod._expand_pixel_time_direct(
        index_map, index2time, output_dtype, fill_value
    )
    masked = streaming_mod._expand_pixel_time_masked(
        index_map, index2time, output_dtype, fill_value
    )

    assert np.array_equal(direct, masked, equal_nan=True)


@pytest.mark.unit
def test_pixel_time_lookup_float64_matches_mask_based():
    index_map = np.asarray([[0, 1, 99], [2, 7, 1]], dtype=np.uint16)
    index2time = {0: 0.0, 1: 10.5, 2: 20.25}
    output_dtype = np.dtype(np.float64)
    fill_value = output_dtype.type(np.nan)

    direct = streaming_mod._expand_pixel_time_direct(
        index_map, index2time, output_dtype, fill_value
    )
    masked = streaming_mod._expand_pixel_time_masked(
        index_map, index2time, output_dtype, fill_value
    )

    assert np.array_equal(direct, masked, equal_nan=True)


@pytest.mark.unit
def test_pixel_time_lookup_missing_key_is_nan():
    index_map = np.asarray([[0, 42], [1, 2]], dtype=np.uint16)
    index2time = {0: 0.0, 1: 10.5, 2: 20.25}

    expanded = expand_pixel_time(index_map, index2time, np.dtype(np.float32))

    assert np.isnan(expanded[0, 1])


@pytest.mark.unit
def test_pixel_time_lookup_unsigned_max_sentinel_is_nan():
    max_val = np.iinfo(np.uint16).max
    index_map = np.asarray([[0, max_val], [1, 2]], dtype=np.uint16)
    index2time = {0: 0.0, 1: 10.5, 2: 20.25, int(max_val): 999.0}

    expanded = expand_pixel_time(index_map, index2time, np.dtype(np.float32))

    assert np.isnan(expanded[0, 1])


@pytest.mark.unit
def test_pixel_time_lookup_falls_back_for_int64():
    index_map = np.asarray([[0, 1, 3], [2, -1, 99]], dtype=np.int64)
    index2time = {0: 0.0, 1: 10.5, 2: 20.25}

    with patch.object(
        streaming_mod,
        "_expand_pixel_time_masked",
        wraps=streaming_mod._expand_pixel_time_masked,
    ) as masked:
        expand_pixel_time(index_map, index2time, np.dtype(np.float32))

    masked.assert_called_once()


@pytest.mark.unit
def test_pixel_time_lookup_byte_parity_over_reference_fixture():
    rng = np.random.default_rng(20260824)
    index_map = rng.integers(0, 101, size=(100, 100), dtype=np.uint16)
    index_map[::10, ::10] = np.iinfo(np.uint16).max
    index2time = {idx: float(idx) + 0.5 for idx in range(100)}
    output_dtype = np.dtype(np.float32)
    fill_value = output_dtype.type(np.nan)

    expanded = expand_pixel_time(index_map, index2time, output_dtype)
    reference = streaming_mod._expand_pixel_time_masked(
        index_map, index2time, output_dtype, fill_value
    )

    assert np.array_equal(expanded, reference, equal_nan=True)


@pytest.mark.unit
def test_shared_nc_part_reader_opens_each_file_once_per_path(
    fci_test_zip: Path, tmp_path: Path, monkeypatch
):
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    with zipfile.ZipFile(fci_test_zip) as zf:
        zf.extractall(extract_dir)

    nc_parts = list_fci_nc_parts(extract_dir)
    assert len(nc_parts) >= 2, "fixture must produce at least two nc_parts"
    part_a = nc_parts[0]
    part_b = nc_parts[1]

    open_calls: list[Path] = []
    original_init = streaming_mod.h5netcdf.File.__init__

    def spy_init(self, path, *args, **kwargs):
        open_calls.append(Path(path))
        original_init(self, path, *args, **kwargs)

    monkeypatch.setattr(streaming_mod.h5netcdf.File, "__init__", spy_init)

    pixel_time_dtype = np.dtype(np.float32)
    with SharedNcPartReader() as shared:
        shared.decode_spatial(part_a, "vis_04", None, pixel_time_dtype)
        shared.decode_spatial(part_a, "ir_38", None, pixel_time_dtype)
        shared.decode_channel(part_a, "vis_04")
        shared.decode_channel(part_a, "ir_38")
        shared.decode_spatial(part_b, "vis_04", None, pixel_time_dtype)
        shared.decode_channel(part_b, "vis_04")

    opens_for_a = [p for p in open_calls if Path(p) == Path(part_a)]
    opens_for_b = [p for p in open_calls if Path(p) == Path(part_b)]
    assert len(opens_for_a) == 1, (
        f"expected 1 open for {part_a.name}, got {len(opens_for_a)}"
    )
    assert len(opens_for_b) == 1, (
        f"expected 1 open for {part_b.name}, got {len(opens_for_b)}"
    )


@pytest.mark.unit
def test_shared_nc_part_reader_caches_reader_instance_by_path(tmp_path: Path):
    part = tmp_path / "body.nc"
    _write_nc_part(part, channel_specs={"vis_04": (1, 2, 11)})

    with SharedNcPartReader() as shared:
        reader_first = shared._get_reader(part)
        reader_second = shared._get_reader(part)

        assert reader_first is reader_second
        assert list(shared._readers.keys()) == [Path(part)]


@pytest.mark.unit
def test_shared_nc_part_reader_decode_spatial_matches_eager_load(tmp_path: Path):
    part = tmp_path / "body.nc"
    _write_nc_part(part, channel_specs={"vis_04": (1, 2, 11)})
    pixel_time_dtype = np.dtype(np.float32)
    index2time = {0: 0.0, 1: 60.0, 2: 120.0}

    with NCPartReader(part) as eager_reader:
        eager_payload = streaming_mod.load_channel_slice(
            eager_reader, "vis_04", index2time, pixel_time_dtype
        )

    with SharedNcPartReader() as shared:
        shared_payload = shared.decode_spatial(
            part, "vis_04", index2time, pixel_time_dtype
        )

    np.testing.assert_array_equal(shared_payload.counts, eager_payload.counts)
    np.testing.assert_array_equal(
        shared_payload.pixel_quality, eager_payload.pixel_quality
    )
    assert shared_payload.pixel_time is not None
    assert eager_payload.pixel_time is not None
    np.testing.assert_array_equal(shared_payload.pixel_time, eager_payload.pixel_time)
    assert shared_payload.pixel_time.dtype == pixel_time_dtype


@pytest.mark.unit
def test_shared_nc_part_reader_decode_channel_matches_eager_calibration(
    tmp_path: Path,
):
    part = tmp_path / "body.nc"
    _write_nc_part(
        part,
        channel_specs={"vis_04": (1, 2, 11)},
        scale_factor=0.25,
        add_offset=3.5,
    )

    with NCPartReader(part) as eager_reader:
        eager_cal = eager_reader.read_calibration("vis_04")

    with SharedNcPartReader() as shared:
        shared_cal = shared.decode_channel(part, "vis_04")

    assert shared_cal == eager_cal
    assert shared_cal == pytest.approx((0.25, 3.5))


@pytest.mark.unit
def test_shared_nc_part_reader_close_releases_handles_and_clears_cache(
    tmp_path: Path,
):
    part = tmp_path / "body.nc"
    _write_nc_part(part, channel_specs={"vis_04": (1, 2, 11)})

    shared = SharedNcPartReader()
    shared.decode_channel(part, "vis_04")
    cached = shared._readers[Path(part)]
    assert cached._ds is not None

    shared.close()

    assert cached._ds is None
    assert shared._readers == {}


@pytest.mark.unit
def test_shared_nc_part_reader_context_manager_closes_on_exit(tmp_path: Path):
    part = tmp_path / "body.nc"
    _write_nc_part(part, channel_specs={"vis_04": (1, 2, 11)})

    with SharedNcPartReader() as shared:
        shared.decode_channel(part, "vis_04")
        cached = shared._readers[Path(part)]
        assert cached._ds is not None

    assert cached._ds is None
    assert shared._readers == {}


@pytest.mark.unit
def test_shared_nc_part_reader_close_is_idempotent(tmp_path: Path):
    part = tmp_path / "body.nc"
    _write_nc_part(part, channel_specs={"vis_04": (1, 2, 11)})

    shared = SharedNcPartReader()
    shared.decode_channel(part, "vis_04")

    shared.close()
    shared.close()


def _assembler_payload(base: int, *, rows: int = 2) -> ChannelSlicePayload:
    counts = np.full((rows, 3), base, dtype=np.uint16)
    quality = np.full((rows, 3), base + 1, dtype=np.uint8)
    pixel_time = np.full((rows, 3), float(base) + 0.5, dtype=np.float32)
    return ChannelSlicePayload(
        counts=counts,
        pixel_quality=quality,
        pixel_time=pixel_time,
    )


class _FakeSharedReader:
    def __init__(self) -> None:
        self.decode_calls: list[tuple[Path, str, int]] = []

    def decode_spatial(
        self,
        item: Path | str,
        nc_channel: str,
        index2time: dict[int, float] | None,
        pixel_time_dtype: np.dtype,
    ) -> ChannelSlicePayload:
        del pixel_time_dtype
        path = Path(item)
        self.decode_calls.append((path, nc_channel, id(index2time)))
        base = int("".join(ch for ch in path.stem if ch.isdigit()) or "0")
        return _assembler_payload(base)


@pytest.mark.unit
def test_assembler_bounded_cache_source_entries_max_two(tmp_path: Path):
    shared = _FakeSharedReader()
    assembler = ChunkOwnedAssembler(shared)  # type: ignore[arg-type]
    dtype = np.dtype(np.float32)

    for idx in range(3):
        part = tmp_path / f"part-{idx}.nc"
        assembler.assemble(
            [part],
            "vis_04",
            None,
            dtype,
            "data_1km",
            0,
            (idx * 2, idx * 2 + 2),
            frozenset({"counts"}),
        )

    assert len(assembler._source_cache) <= 2


@pytest.mark.unit
def test_assembler_bounded_cache_assembled_entries_max_one(tmp_path: Path):
    shared = _FakeSharedReader()
    assembler = ChunkOwnedAssembler(shared)  # type: ignore[arg-type]
    dtype = np.dtype(np.float32)
    part = tmp_path / "part-1.nc"

    assembler.assemble(
        [part],
        "vis_04",
        None,
        dtype,
        "data_1km",
        0,
        (0, 2),
        frozenset({"counts"}),
    )
    assembler.assemble(
        [part],
        "vis_04",
        None,
        dtype,
        "data_1km",
        0,
        (2, 4),
        frozenset({"counts"}),
    )

    assert len(assembler._assembled_cache) <= 1


@pytest.mark.unit
def test_assembler_rejects_non_contiguous_nc_parts(tmp_path: Path):
    from firecube_mtg_fci_l1c._group_plan import GroupPlan
    from firecube_mtg_fci_l1c.config import MtgFciL1cConfig
    from firecube_mtg_fci_l1c.ingestor import MtgFciL1cIngestor

    config = MtgFciL1cConfig(
        product_type="FDHSI",
        resolutions="1km",
        zarr_chunk_y=4,
        include_pixel_time=False,
    )
    plan = GroupPlan(
        product_type="FDHSI",
        resolution="1km",
        group="data_1km",
        dimsize=4,
        logical_channels=("vis_04",),
        nc_channels=("vis_04",),
    )
    nc_part_ranges = [
        (tmp_path / "part-1.nc", (0, 2)),
        (tmp_path / "part-2.nc", (3, 4)),
    ]

    with pytest.raises(AssemblyPreconditionError):
        MtgFciL1cIngestor()._emit_spatial_intents(
            "batch-1",
            plan,
            0,
            nc_part_ranges,
            None,
            np.dtype(np.float32),
            config,
        )


@pytest.mark.unit
def test_assembler_rejects_more_than_two_nc_parts_per_chunk(tmp_path: Path):
    assembler = ChunkOwnedAssembler(_FakeSharedReader())  # type: ignore[arg-type]

    with pytest.raises(AssemblyPreconditionError):
        assembler.assemble(
            [tmp_path / "part-1.nc", tmp_path / "part-2.nc", tmp_path / "part-3.nc"],
            "vis_04",
            None,
            np.dtype(np.float32),
            "data_1km",
            0,
            (0, 6),
            frozenset({"counts"}),
        )


@pytest.mark.unit
def test_assembler_byte_parity_optimized_vs_reference(tmp_path: Path):
    part_a = tmp_path / "part-1.nc"
    part_b = tmp_path / "part-2.nc"
    payload_a = _assembler_payload(10)
    payload_b = _assembler_payload(20)

    class Reader:
        def decode_spatial(self, item, *_args):
            return {part_a: payload_a, part_b: payload_b}[Path(item)]

    assembler = ChunkOwnedAssembler(Reader())  # type: ignore[arg-type]

    assembled = assembler.assemble(
        [part_a, part_b],
        "vis_04",
        {0: 1.0},
        np.dtype(np.float32),
        "data_1km",
        0,
        (0, 4),
        frozenset({"counts", "pixel_quality", "pixel_time"}),
    )

    np.testing.assert_array_equal(
        assembled.counts,
        np.concatenate([payload_a.counts, payload_b.counts], axis=0),
    )
    np.testing.assert_array_equal(
        assembled.pixel_quality,
        np.concatenate([payload_a.pixel_quality, payload_b.pixel_quality], axis=0),
    )
    assert assembled.pixel_time is not None
    assert np.array_equal(
        assembled.pixel_time,
        np.concatenate([payload_a.pixel_time, payload_b.pixel_time], axis=0),
        equal_nan=True,
    )


@pytest.mark.unit
def test_assembler_close_clears_cache(tmp_path: Path):
    assembler = ChunkOwnedAssembler(_FakeSharedReader())  # type: ignore[arg-type]
    dtype = np.dtype(np.float32)
    part = tmp_path / "part-1.nc"

    assembler.assemble(
        [part],
        "vis_04",
        None,
        dtype,
        "data_1km",
        0,
        (0, 2),
        frozenset({"counts"}),
    )
    assert assembler._source_cache
    assert assembler._assembled_cache

    assembler.close()

    assert assembler._source_cache == {}
    assert assembler._assembled_cache == {}


@pytest.mark.unit
def test_assembler_identity_via_is_not_equality_for_index2time(tmp_path: Path):
    import gc

    shared = _FakeSharedReader()
    assembler = ChunkOwnedAssembler(shared)  # type: ignore[arg-type]
    dtype = np.dtype(np.float32)
    part = tmp_path / "part-1.nc"
    dict1 = {0: 1.0}
    dict2 = {0: 1.0}
    assert dict1 == dict2
    assert dict1 is not dict2

    for index2time in (dict1, dict2):
        assembler.assemble(
            [part],
            "vis_04",
            index2time,
            dtype,
            "data_1km",
            0,
            (0, 2),
            frozenset({"counts"}),
        )

    assert shared.decode_calls == [
        (part, "vis_04", id(dict1)),
        (part, "vis_04", id(dict2)),
    ]

    d1 = {0: 1.0}
    ref1 = _IdentityRef(d1)
    addr1 = id(d1)
    del d1
    gc.collect()

    d2 = {0: 1.0}
    ref2 = _IdentityRef(d2)

    assert id(ref1.obj) == addr1
    assert id(d2) != addr1
    assert ref1 != ref2


@pytest.mark.unit
def test_decode_spatial_returns_read_only_payload(tmp_path: Path):
    part = tmp_path / "body.nc"
    _write_nc_part(part, channel_specs={"vis_04": (1, 2, 11)})
    pixel_time_dtype = np.dtype(np.float32)
    index2time = {0: 0.0, 1: 60.0, 2: 120.0}

    with SharedNcPartReader() as shared:
        payload = shared.decode_spatial(part, "vis_04", index2time, pixel_time_dtype)

    assert payload.counts.flags.writeable is False
    assert payload.pixel_quality.flags.writeable is False
    assert payload.pixel_time is not None
    assert payload.pixel_time.flags.writeable is False


@pytest.mark.unit
def test_variable_projections_do_not_mutate_cached_payload(tmp_path: Path):
    from firecube_mtg_fci_l1c.config import (  # pyright: ignore[reportMissingImports]
        MtgFciL1cConfig,
    )
    from firecube_mtg_fci_l1c._variables import (  # pyright: ignore[reportMissingImports]
        VariableContext,
        _counts_source,
        _pixel_quality_source,
        _pixel_time_source,
    )

    part = tmp_path / "body.nc"
    _write_nc_part(part, channel_specs={"vis_04": (1, 2, 11)})
    pixel_time_dtype = np.dtype(np.float32)
    index2time = {0: 0.0, 1: 60.0, 2: 120.0}

    with SharedNcPartReader() as shared:
        payload = shared.decode_spatial(part, "vis_04", index2time, pixel_time_dtype)

        counts_before = payload.counts.copy()
        pixel_quality_before = payload.pixel_quality.copy()
        assert payload.pixel_time is not None
        pixel_time_before = payload.pixel_time.copy()

        ctx = VariableContext(
            group="data_1km",
            product_type="FDHSI",
            config=MtgFciL1cConfig(),
            dimsize=2,
            n_channels=1,
            logical_channels=("vis_04",),
            channel_payload=payload,
        )

        projected_counts = _counts_source(ctx)
        projected_quality = _pixel_quality_source(ctx)
        projected_time = _pixel_time_source(ctx)

        assert projected_counts is payload.counts
        assert projected_quality is payload.pixel_quality
        assert projected_time is payload.pixel_time

        assert np.array_equal(payload.counts, counts_before)
        assert np.array_equal(payload.pixel_quality, pixel_quality_before)
        assert np.array_equal(payload.pixel_time, pixel_time_before, equal_nan=True)
