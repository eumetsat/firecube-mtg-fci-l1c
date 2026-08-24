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
import zipfile
from pathlib import Path
from typing import Any, cast

import h5netcdf
import numpy as np
import pytest
import xarray as xr
import zarr
from firecube_mtg_fci_l1c import MtgFciL1cIngestor
from firecube_mtg_fci_l1c._constants import (
    PRODUCT_TYPE_FDHSI,
    PRODUCT_TYPE_HRFI,
    get_nc_part_prefix,
)
from firecube_mtg_fci_l1c._data import validate_no_mixed_products
from firecube_mtg_fci_l1c.config import MtgFciL1cConfig
from firecube_mtg_fci_l1c._variables import build_specs

from firecube.core.cf.validator import validate_cf18
from firecube.core.product.identity import ProductIdentity
from firecube.core.storage.binding import StorageBinding
from firecube.core.storage.driver_config import StorageDriverConfig
from firecube.core.storage.session import StorageSession
from firecube.core.storage.uri import StorageUri
from firecube.ingestor.api import IngestContext, StorageContext


def _make_local_storage_session(target_path: Path) -> StorageSession:
    # Inlined from firecube.tests.helpers.storage.make_local_session
    # (core test helpers are not importable from plugin packages).
    uri = StorageUri.from_local_path(target_path)
    binding = StorageBinding(
        identity=ProductIdentity.from_uri(uri, "zarr", product_name=target_path.name),
        driver=StorageDriverConfig(driver="fsspec"),
    )
    return StorageSession(binding)


def _write_nc_part_netcdf(path: Path, nc_channels: list[str], dimsize: int) -> None:
    with h5netcdf.File(path, "w") as ds:
        ds.attrs["time_coverage_start"] = "20240101000000"

        ds.dimensions["n_time"] = 2
        ds.create_variable("index", ("n_time",), data=np.array([0, 1], dtype=np.uint16))
        time_var = ds.create_variable("time", ("n_time",), data=np.array([0.0, 60.0]))
        time_var.attrs["_FillValue"] = 0.0

        data_group = ds.create_group("data")
        for i, channel in enumerate(nc_channels):
            channel_group = data_group.create_group(channel)
            measured = channel_group.create_group("measured")
            measured.dimensions["y"] = dimsize
            measured.dimensions["x"] = dimsize

            radiance = measured.create_variable(
                "effective_radiance",
                ("y", "x"),
                data=np.full((dimsize, dimsize), i + 1, dtype=np.uint16),
            )
            radiance.attrs["scale_factor"] = float(i + 1)
            radiance.attrs["add_offset"] = float(i)

            measured.create_variable("start_position_row", (), data=np.int32(1))
            measured.create_variable("end_position_row", (), data=np.int32(dimsize))
            measured.create_variable(
                "pixel_quality",
                ("y", "x"),
                data=np.zeros((dimsize, dimsize), dtype=np.uint8),
            )
            measured.create_variable(
                "index_map",
                ("y", "x"),
                data=np.ones((dimsize, dimsize), dtype=np.uint16),
            )


def _make_zip_with_nc_part(
    zip_path: Path, product_type: str, nc_channels: list[str], dimsize: int
) -> Path:
    tmp_nc = zip_path.with_suffix(".nc")
    _write_nc_part_netcdf(tmp_nc, nc_channels=nc_channels, dimsize=dimsize)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(tmp_nc, arcname=f"{get_nc_part_prefix(product_type)}0001.nc")
    tmp_nc.unlink()
    return zip_path


def _run_ingest(
    source: Path, workspace: Path, options: dict[str, object] | None = None
) -> Path:
    output_name = "out.zarr"
    target_path = workspace / output_name
    ingestor = MtgFciL1cIngestor()
    ctx = IngestContext(
        source=str(source),
        target=str(target_path),
        output_format="zarr",
        storage=StorageContext(output=_make_local_storage_session(target_path)),
        options={
            "pipeline_parallel": False,
            "force_reingest": True,
            "write_mode": "direct",
            # Fixtures use 2024-01-01 timestamps (pre-dating real FCI data); anchor
            # the deterministic slot index there so they map to compact slots 0,1,...
            "time_epoch": "2024-01-01",
            **(options or {}),
        },
    )
    result = ingestor.run(ctx)
    result_path = Path(str(result.output_path))
    if result_path.exists():
        return result_path
    return target_path


@pytest.fixture
def small_fci_layout(monkeypatch):
    from firecube_mtg_fci_l1c import _constants as const_mod
    from firecube_mtg_fci_l1c.geolocation import provider as geolocation_mod

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

    compute_calls: list[int] = []

    def _fake_compute_latlon(_resolution_m: int):
        compute_calls.append(_resolution_m)
        lat = np.zeros((4, 4), dtype=np.float32)
        lon = np.zeros((4, 4), dtype=np.float32)
        lat[0, 0] = np.nan
        lon[0, 0] = np.nan
        return lat, lon

    monkeypatch.setattr(geolocation_mod, "compute_latlon", _fake_compute_latlon)
    yield compute_calls
    const_mod.CONSTANTS.clear()
    const_mod.CONSTANTS.update(constants_backup)


@pytest.fixture
def fdhsi_zip(tmp_path: Path, small_fci_layout) -> Path:
    src = tmp_path / "fdhsi"
    src.mkdir()
    zip_path = src / "W_XX-FCI-1C-RRAD-FDHSI-FD-20240101000000-END.zip"
    return _make_zip_with_nc_part(
        zip_path, PRODUCT_TYPE_FDHSI, ["vis_04", "vis_06", "ir_38"], dimsize=4
    )


@pytest.fixture
def hrfi_zip(tmp_path: Path, small_fci_layout) -> Path:
    src = tmp_path / "hrfi"
    src.mkdir()
    zip_path = src / "W_XX-FCI-1C-RRAD-HRFI-FD-20240101000000-END.zip"
    return _make_zip_with_nc_part(
        zip_path, PRODUCT_TYPE_HRFI, ["vis_06_hr", "ir_38_hr"], dimsize=4
    )


@pytest.mark.integration
@pytest.mark.plugin
def test_fdhsi_groups_created(tmp_path: Path, fdhsi_zip: Path):
    out = _run_ingest(fdhsi_zip.parent, tmp_path)
    root = zarr.open_group(str(out), mode="r")
    assert "data_1km" in root
    assert "data_2km" in root


@pytest.mark.integration
@pytest.mark.plugin
def test_hrfi_groups_created(tmp_path: Path, hrfi_zip: Path):
    out = _run_ingest(hrfi_zip.parent, tmp_path)
    root = zarr.open_group(str(out), mode="r")
    assert "data_500m" in root
    assert "data_1km" in root


@pytest.mark.integration
@pytest.mark.plugin
def test_mixed_rejection(tmp_path: Path, fdhsi_zip: Path, hrfi_zip: Path):
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / fdhsi_zip.name).write_bytes(fdhsi_zip.read_bytes())
    (mixed / hrfi_zip.name).write_bytes(hrfi_zip.read_bytes())
    with pytest.raises(ValueError, match=r"[Mm]ixed"):
        validate_no_mixed_products(list(mixed.glob("*.zip")))


@pytest.mark.integration
@pytest.mark.plugin
def test_geolocation_present(
    tmp_path: Path, fdhsi_zip: Path, small_fci_layout: list[int]
):
    out = _run_ingest(fdhsi_zip.parent, tmp_path)
    root = zarr.open_group(str(out), mode="r")
    grp = root["data_1km"]
    assert "counts" in grp
    assert any(res in small_fci_layout for res in (1000, 2000))


@pytest.mark.integration
@pytest.mark.plugin
def test_geolocation_static_values_written(
    tmp_path: Path, fdhsi_zip: Path, small_fci_layout: list[int]
):
    """Static lat/lon are written via the kind='static' intent path (direct mode).

    The fake grid is zeros with a NaN at [0, 0]; assert the on-disk array matches
    and carries core's write-once marker, proving the static-intent path persisted
    real data (not just an all-fill shell).
    """
    out = _run_ingest(fdhsi_zip.parent, tmp_path)
    grp = cast(Any, zarr.open_group(str(out), mode="r")["data_1km"])

    lat = np.asarray(grp["latitude"][:])
    assert lat.shape == (4, 4)
    assert lat.dtype == np.float32
    assert np.isnan(lat[0, 0])
    assert not np.isnan(lat[1, 1]) and lat[1, 1] == 0.0
    # Core stamps this marker once a static array commits (write-once).
    assert grp["latitude"].attrs.get("firecube_static_written") is True


@pytest.mark.integration
@pytest.mark.plugin
def test_backward_compatible_fdhsi(tmp_path: Path, fdhsi_zip: Path):
    out = _run_ingest(fdhsi_zip.parent, tmp_path)
    root = zarr.open_group(str(out), mode="r")
    grp = root["data_1km"]
    for var in [
        "counts",
        "pixel_quality",
        "pixel_time",
        "slope",
        "offset",
        "channel_name",
    ]:
        assert var in grp
    # lat/lon attrs come from declarative variable attrs and reach the store.
    grp_any: Any = grp
    latitude = grp_any["latitude"]
    longitude = grp_any["longitude"]
    assert latitude.attrs["standard_name"] == "latitude"
    assert longitude.attrs["units"] == "degrees_east"


@pytest.mark.integration
@pytest.mark.plugin
def test_channel_name_written(tmp_path: Path, fdhsi_zip: Path):
    """channel_name array is written to Zarr store with correct values."""
    out = _run_ingest(fdhsi_zip.parent, tmp_path)
    root = zarr.open_group(str(out), mode="r")
    data_1km = cast(Any, root["data_1km"])

    arr = data_1km["channel_name"]
    assert arr.shape == (2,)
    values = np.asarray(arr[:])
    assert values.dtype.kind == "S"
    assert len(values) == arr.shape[0]
    assert values.tolist() == [b"vis_04", b"vis_06"]


@pytest.mark.integration
@pytest.mark.plugin
def test_channel_name_static_replay_is_idempotent(tmp_path: Path, fdhsi_zip: Path):
    """Replaying a static bytes coordinate must not break resume/idempotency."""
    src = fdhsi_zip.parent

    out = _run_ingest(src, tmp_path, options={"include_geolocation": False})
    root = zarr.open_group(str(out), mode="r")
    data_1km = cast(Any, root["data_1km"])
    first = np.asarray(data_1km["channel_name"][:]).copy()

    _run_ingest(
        src,
        tmp_path,
        options={
            "include_geolocation": False,
            "force_reingest": False,
            "resume_existing": True,
        },
    )

    root = zarr.open_group(str(out), mode="r")
    data_1km = cast(Any, root["data_1km"])
    np.testing.assert_array_equal(np.asarray(data_1km["channel_name"][:]), first)


@pytest.mark.integration
@pytest.mark.plugin
def test_fdhsi_2km_content_written_correctly(tmp_path: Path, fdhsi_zip: Path):
    out = _run_ingest(
        fdhsi_zip.parent, tmp_path, options={"include_geolocation": False}
    )
    root = zarr.open_group(str(out), mode="r")
    data_2km = cast(Any, root["data_2km"])

    counts = np.asarray(data_2km["counts"][0, :, :, 0])
    pixel_quality = np.asarray(data_2km["pixel_quality"][0, :, :, 0])

    np.testing.assert_array_equal(counts, np.full((4, 4), 3, dtype=np.uint16))
    np.testing.assert_array_equal(pixel_quality, np.zeros((4, 4), dtype=np.uint8))
    assert data_2km["counts"].shape[-1] == 1
    assert data_2km["counts"].dtype == np.uint16

    # Schema-declared CF attributes reach the store (the schema.py worked example).
    assert data_2km["counts"].attrs["units"] == "1"
    assert data_2km["counts"].attrs["long_name"] == "FCI raw detector counts"


@pytest.mark.integration
@pytest.mark.plugin
def test_ingest_channels_option_filters_output(tmp_path: Path, fdhsi_zip: Path):
    out = _run_ingest(fdhsi_zip.parent, tmp_path, options={"channels": "vis_04"})
    root = zarr.open_group(str(out), mode="r")
    data_1km = cast(Any, root["data_1km"])

    assert data_1km["counts"].shape[-1] == 1
    if "data_2km" in root:
        data_2km = cast(Any, root["data_2km"])
        assert "counts" not in list(data_2km.array_keys())


def _make_fdhsi_zip_at(src_dir: Path, ts_str: str) -> Path:
    src_dir.mkdir(parents=True, exist_ok=True)
    zip_path = src_dir / f"W_XX-FCI-1C-RRAD-FDHSI-FD-{ts_str}-END.zip"
    return _make_zip_with_nc_part(
        zip_path, PRODUCT_TYPE_FDHSI, ["vis_04", "vis_06", "ir_38"], dimsize=4
    )


@pytest.mark.integration
@pytest.mark.plugin
def test_append_and_duplicate_timestamp_idempotent(
    tmp_path: Path, small_fci_layout: list[int]
):
    store_path = tmp_path / "out.zarr"

    # Given: a source directory with two ZIPs at distinct timestamps A and B
    src = tmp_path / "src"
    _make_fdhsi_zip_at(src, "20240101000000")
    _make_fdhsi_zip_at(src, "20240101001000")

    # When: the batch is ingested
    _run_ingest(src, tmp_path)
    root = zarr.open_group(str(store_path), mode="r")
    data_1km = cast(Any, root["data_1km"])
    timestamps_first = np.asarray(data_1km["time"][:]).copy()
    assert data_1km["counts"].shape[0] == 2, (
        f"expected 2 timestamps after first ingest, got {data_1km['counts'].shape[0]}"
    )
    assert data_1km["time"].shape == (2,)
    assert len(set(timestamps_first.tolist())) == 2, (
        f"expected 2 distinct timestamps, got {timestamps_first.tolist()}"
    )

    # Then: re-ingesting the same source is idempotent
    # (canonical re-ingest: resume_existing + force_reingest=False)
    _run_ingest(
        src, tmp_path, options={"force_reingest": False, "resume_existing": True}
    )
    root = zarr.open_group(str(store_path), mode="r")
    data_1km = cast(Any, root["data_1km"])
    timestamps_second = np.asarray(data_1km["time"][:])
    assert data_1km["counts"].shape[0] == 2, (
        f"re-ingest must not grow timestamp axis; got {data_1km['counts'].shape[0]}"
    )
    assert data_1km["time"].shape == (2,)
    np.testing.assert_array_equal(
        timestamps_first,
        timestamps_second,
        err_msg="re-ingest changed the timestamp coordinate values (idempotency broken)",
    )


@pytest.mark.integration
@pytest.mark.plugin
def test_cross_batch_append_preserves_existing_timestamps(
    tmp_path: Path, small_fci_layout: list[int]
):
    store_path = tmp_path / "out.zarr"

    # Separate runs against the same store must append, not overwrite.
    src_a = tmp_path / "src_a"
    src_b = tmp_path / "src_b"
    _make_fdhsi_zip_at(src_a, "20240101000000")
    _make_fdhsi_zip_at(src_b, "20240101001000")

    _run_ingest(src_a, tmp_path)
    root = zarr.open_group(str(store_path), mode="r")
    data_1km = cast(Any, root["data_1km"])
    assert data_1km["time"].shape == (1,), (
        f"after first ingest, expected 1 timestamp, got shape {data_1km['time'].shape}"
    )
    ts_after_a = np.asarray(data_1km["time"][:]).copy()

    _run_ingest(
        src_b, tmp_path, options={"force_reingest": False, "resume_existing": True}
    )

    root = zarr.open_group(str(store_path), mode="r")
    data_1km = cast(Any, root["data_1km"])
    ts_after_b = np.asarray(data_1km["time"][:])
    assert data_1km["time"].shape == (2,), (
        f"after cross-batch append, expected 2 timestamps, "
        f"got shape {data_1km['time'].shape}"
    )
    assert ts_after_b[0] == ts_after_a[0], (
        f"second ingest overwrote slot 0 ({ts_after_a[0]} -> {ts_after_b[0]})"
    )
    assert len(set(ts_after_b.tolist())) == 2, (
        f"expected 2 distinct timestamps after cross-batch append, got {ts_after_b.tolist()}"
    )

    # Re-ingesting A is an idempotent no-op.
    _run_ingest(
        src_a, tmp_path, options={"force_reingest": False, "resume_existing": True}
    )
    root = zarr.open_group(str(store_path), mode="r")
    data_1km = cast(Any, root["data_1km"])
    ts_after_reingest = np.asarray(data_1km["time"][:])
    assert data_1km["time"].shape == (2,), (
        f"re-ingest of A grew the timestamp axis to {data_1km['time'].shape}"
    )
    np.testing.assert_array_equal(
        ts_after_reingest,
        ts_after_b,
        err_msg="re-ingest of A changed existing timestamp slots (idempotency broken)",
    )


@pytest.mark.integration
@pytest.mark.plugin
def test_staged_mode_append_and_reingest_idempotent(
    tmp_path: Path, small_fci_layout: list[int]
):
    """Staged append preserves existing slots and re-ingest is idempotent."""
    store_path = tmp_path / "out.zarr"
    # Geolocation has its own staged-resume guard test below.
    staged = {"write_mode": "staged", "include_geolocation": False}
    staged_resume = {
        "write_mode": "staged",
        "resume_existing": True,
        "force_reingest": False,
        "include_geolocation": False,
    }

    src_a = tmp_path / "src_a"
    src_b = tmp_path / "src_b"
    _make_fdhsi_zip_at(src_a, "20240101000000")
    _make_fdhsi_zip_at(src_b, "20240101001000")

    # Run A (staged): single timestamp T_A at slot 0.
    _run_ingest(src_a, tmp_path, options=staged)
    data_1km = cast(Any, zarr.open_group(str(store_path), mode="r")["data_1km"])
    assert data_1km["time"].shape == (1,), (
        f"after staged run A, expected 1 timestamp, got {data_1km['time'].shape}"
    )
    ts_after_a = np.asarray(data_1km["time"][:]).copy()

    # Run B (staged, resume): T_B must APPEND at slot 1, not overwrite slot 0.
    _run_ingest(src_b, tmp_path, options=staged_resume)
    data_1km = cast(Any, zarr.open_group(str(store_path), mode="r")["data_1km"])
    ts_after_b = np.asarray(data_1km["time"][:])
    assert data_1km["time"].shape == (2,), (
        f"staged append must grow the time axis to 2, got {data_1km['time'].shape}"
    )
    assert ts_after_b[0] == ts_after_a[0], (
        f"staged append overwrote slot 0 ({ts_after_a[0]} -> {ts_after_b[0]})"
    )
    assert len(set(ts_after_b.tolist())) == 2, (
        f"expected 2 distinct timestamps after staged append, got {ts_after_b.tolist()}"
    )

    # Re-ingest A (staged, resume): T_A already present -> idempotent no-op.
    _run_ingest(src_a, tmp_path, options=staged_resume)
    data_1km = cast(Any, zarr.open_group(str(store_path), mode="r")["data_1km"])
    ts_after_reingest = np.asarray(data_1km["time"][:])
    assert data_1km["time"].shape == (2,), (
        f"staged re-ingest of A duplicated an existing timestamp; "
        f"time axis grew to {data_1km['time'].shape}"
    )
    np.testing.assert_array_equal(
        ts_after_reingest,
        ts_after_b,
        err_msg="staged re-ingest of A changed existing timestamp slots (idempotency broken)",
    )


@pytest.mark.integration
@pytest.mark.plugin
def test_deterministic_slot_placement_across_days(
    tmp_path: Path, small_fci_layout: list[int]
):
    """A timestamp lands at its EUMETSAT repeat-cycle slot, not by arrival order.

    With epoch 2024-01-01, an acquisition on 2024-01-02 at 00:00 is day 1, cycle
    0 -> slot 144. The single ingested timestamp must sit at index 144 (array
    grown to 145), with the leading slots left as NaT: the deterministic,
    appendable layout, not the old compact append-order layout.
    """
    store_path = tmp_path / "out.zarr"
    src = tmp_path / "src"
    _make_fdhsi_zip_at(src, "20240102000000")  # day 1 after epoch, cycle 0

    _run_ingest(src, tmp_path, options={"include_geolocation": False})

    data_1km = cast(Any, zarr.open_group(str(store_path), mode="r")["data_1km"])
    assert data_1km["time"].shape == (145,), (
        f"expected time axis grown to 145 (slot 144), got {data_1km['time'].shape}"
    )
    ts = np.asarray(data_1km["time"][:])
    assert not np.isnat(ts[144]), "slot 144 should hold the acquisition"
    assert np.isnat(ts[0]), "leading slots should be NaT (gap), not back-filled"
    assert ts[144] == np.datetime64("2024-01-02T00:00:00")


@pytest.mark.integration
@pytest.mark.plugin
def test_index_model_attrs_recorded_and_epoch_mismatch_rejected(
    tmp_path: Path, small_fci_layout: list[int]
):
    """Slot-range ingest stamps the slot-index model and rejects epoch drift."""
    store_path = tmp_path / "out.zarr"
    src = tmp_path / "src"
    _make_fdhsi_zip_at(src, "20240101000000")

    base = {
        "write_mode": "direct",
        "resolutions": "1km",
        "time_epoch": "2024-01-01",
        "time_slots": 1,
        "include_geolocation": False,
    }
    _run_ingest(src, tmp_path, options={**base, "slot_start": 0, "slot_end": 1})

    root = zarr.open_group(str(store_path), mode="r")
    # firecube v0.1.6+ stores resolved index under firecube_resolved_index.
    model_attr = root.attrs.get("firecube_resolved_index") or root.attrs.get(
        "firecube_slot_index_model"
    )
    assert model_attr is not None, "Expected firecube index attr not found in store"
    assert "eumetsat_repeat_cycle_v1" in str(model_attr)
    assert "2024-01-01" in str(model_attr)

    # force_reingest gets past span overlap so epoch validation is reached.
    with pytest.raises(
        Exception, match="mismatch|misalign|epoch|conflict|incompatible|drift"
    ):
        _run_ingest(
            src,
            tmp_path,
            options={
                **base,
                "slot_start": 0,
                "slot_end": 1,
                "time_epoch": "2024-09-24",
                "force_reingest": True,
            },
        )


@pytest.mark.integration
@pytest.mark.plugin
def test_staged_mode_geolocation_reingest_resumes_existing_store(
    tmp_path: Path, small_fci_layout: list[int]
):
    """Staged re-ingest into an existing store works with geolocation enabled."""
    src = tmp_path / "src"
    _make_fdhsi_zip_at(src, "20240101000000")

    staged = {"write_mode": "staged", "include_geolocation": True}
    _run_ingest(src, tmp_path, options=staged)  # fresh target: OK
    _run_ingest(
        src,
        tmp_path,
        options={**staged, "resume_existing": True, "force_reingest": False},
    )


@pytest.mark.integration
@pytest.mark.plugin
def test_reused_ingestor_emits_static_for_each_target(
    tmp_path: Path, small_fci_layout: list[int]
):
    """A reused ingestor instance must re-emit static lat/lon for a new target.

    The once-per-group guard is run-local; both stores must get coordinates.
    """
    src = tmp_path / "src"
    _make_fdhsi_zip_at(src, "20240101000000")

    ingestor = MtgFciL1cIngestor()  # one instance, two runs

    def _run(target_path: Path) -> None:
        ctx = IngestContext(
            source=str(src),
            target=str(target_path),
            output_format="zarr",
            storage=StorageContext(output=_make_local_storage_session(target_path)),
            options={
                "pipeline_parallel": False,
                "force_reingest": True,
                "write_mode": "direct",
                "time_epoch": "2024-01-01",
                "resolutions": "1km",
            },
        )
        ingestor.run(ctx)

    for name in ("first.zarr", "second.zarr"):
        target = tmp_path / name
        _run(target)
        grp = cast(Any, zarr.open_group(str(target), mode="r")["data_1km"])
        assert "latitude" in grp, f"{name} missing static latitude (guard not reset)"
        assert grp["latitude"].attrs.get("firecube_static_written") is True


@pytest.mark.integration
@pytest.mark.plugin
def test_slot_range_disjoint_writes(tmp_path: Path, small_fci_layout: list[int]):
    """Two slot-range 'pods' write disjoint slots into one store without clobber.

    Exercises the full slot-range path for this plugin (capability gate,
    engine-owned item filtering via inspect_item, resolved-index schema
    sizing, and the post-intent range assertion). The two runs are sequential
    here; true concurrent claim coordination is covered by core's own
    parallel fixtures.
    """
    store_path = tmp_path / "out.zarr"
    src = tmp_path / "src"
    # Four cycles on the epoch day -> slots 0,1,2,3.
    for ts in (
        "20240101000000",
        "20240101001000",
        "20240101002000",
        "20240101003000",
    ):
        _make_fdhsi_zip_at(src, ts)

    base = {
        "write_mode": "direct",
        "resolutions": "1km",  # single group keeps the slot-range path simple
        "time_epoch": "2024-01-01",
        "time_slots": 4,
        "include_geolocation": False,
    }
    # Pod A owns [0,2); pod B owns [2,4). Both see all four ZIPs and filter.
    _run_ingest(src, tmp_path, options={**base, "slot_start": 0, "slot_end": 2})
    _run_ingest(
        src,
        tmp_path,
        options={
            **base,
            "slot_start": 2,
            "slot_end": 4,
            "force_reingest": False,
            "resume_existing": True,
        },
    )

    data_1km = cast(Any, zarr.open_group(str(store_path), mode="r")["data_1km"])
    assert data_1km["time"].shape == (4,), (
        f"expected 4 slots after both pods, got {data_1km['time'].shape}"
    )
    ts = np.asarray(data_1km["time"][:])
    assert not np.any(np.isnat(ts)), f"all four slots must be filled, got {ts.tolist()}"
    expected = [np.datetime64(f"2024-01-01T00:{m:02d}:00") for m in (0, 10, 20, 30)]
    np.testing.assert_array_equal(ts, np.array(expected, dtype=ts.dtype))


@pytest.mark.integration
@pytest.mark.plugin
def test_batches_use_isolated_scratch_roots(
    tmp_path: Path, small_fci_layout: list[int], monkeypatch
):
    """Each batch extracts into a distinct scratch root tagged by batch_id."""
    import firecube_mtg_fci_l1c._scratch as scratch_mod

    # Three distinct timestamps -> three batches at batch_size=1.
    src = tmp_path / "src"
    _make_fdhsi_zip_at(src, "20240101000000")
    _make_fdhsi_zip_at(src, "20240101001000")
    _make_fdhsi_zip_at(src, "20240101002000")

    seen_roots: list[str] = []
    original_init = scratch_mod.BatchScratch.__init__

    def _recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        seen_roots.append(str(self._scratch_root))

    monkeypatch.setattr(scratch_mod.BatchScratch, "__init__", _recording_init)

    _run_ingest(src, tmp_path, options={"pipeline_batch_size": 1})

    # One isolated root per batch.
    assert len(seen_roots) == 3, (
        f"expected one scratch root per batch, got {seen_roots}"
    )
    assert len(set(seen_roots)) == 3, (
        f"batches shared a scratch root (cross-batch deletion hazard): {seen_roots}"
    )
    # Root names carry batch ids, which keeps concurrent cleanup disjoint.
    for batch_idx in range(3):
        assert any(f"batch_{batch_idx:04d}" in root for root in seen_roots), (
            f"no scratch root tagged for batch_{batch_idx:04d}: {seen_roots}"
        )

    # And the run still wrote all three timestamps.
    store_path = tmp_path / "out.zarr"
    data_1km = cast(Any, zarr.open_group(str(store_path), mode="r")["data_1km"])
    assert data_1km["time"].shape == (3,), (
        f"expected 3 timestamps after ingest, got {data_1km['time'].shape}"
    )
    assert len(set(np.asarray(data_1km["time"][:]).tolist())) == 3


@pytest.mark.integration
@pytest.mark.plugin
def test_geolocation_idempotent_across_batches(
    tmp_path: Path, small_fci_layout: list[int]
):
    store_path = tmp_path / "out.zarr"

    # Given: a first batch produces latitude/longitude
    src_a = tmp_path / "src_a"
    _make_fdhsi_zip_at(src_a, "20240101000000")
    _run_ingest(src_a, tmp_path)
    root = zarr.open_group(str(store_path), mode="r")
    data_1km_first = cast(Any, root["data_1km"])
    lat_first = np.asarray(data_1km_first["latitude"][:]).copy()
    lon_first = np.asarray(data_1km_first["longitude"][:]).copy()

    # When: a second batch is appended with a different timestamp
    # (canonical append: resume_existing + force_reingest=False)
    src_b = tmp_path / "src_b"
    _make_fdhsi_zip_at(src_b, "20240101001000")
    _run_ingest(
        src_b, tmp_path, options={"force_reingest": False, "resume_existing": True}
    )
    root = zarr.open_group(str(store_path), mode="r")
    data_1km_second = cast(Any, root["data_1km"])
    lat_second = np.asarray(data_1km_second["latitude"][:])
    lon_second = np.asarray(data_1km_second["longitude"][:])

    # Then: lat/lon are unchanged bytewise
    np.testing.assert_array_equal(
        lat_first,
        lat_second,
        err_msg="latitude changed across batches — geolocation idempotency broken",
    )
    np.testing.assert_array_equal(
        lon_first,
        lon_second,
        err_msg="longitude changed across batches — geolocation idempotency broken",
    )


@pytest.mark.integration
@pytest.mark.plugin
def test_zarr_metadata_invariants(tmp_path: Path, fdhsi_zip: Path, hrfi_zip: Path):
    # Given: an FDHSI ingest
    fdhsi_ws = tmp_path / "fdhsi_ws"
    fdhsi_ws.mkdir()
    _run_ingest(fdhsi_zip.parent, fdhsi_ws)
    root_fdhsi = zarr.open_group(str(fdhsi_ws / "out.zarr"), mode="r")

    # Then: only FDHSI groups exist
    fdhsi_groups = set(root_fdhsi.group_keys())
    assert "data_1km" in fdhsi_groups, (
        f"data_1km missing in FDHSI; groups: {fdhsi_groups}"
    )
    assert "data_2km" in fdhsi_groups, (
        f"data_2km missing in FDHSI; groups: {fdhsi_groups}"
    )
    assert "data_500m" not in fdhsi_groups, "data_500m must not exist in FDHSI output"

    data_1km = cast(Any, root_fdhsi["data_1km"])
    array_keys_1km = set(data_1km.array_keys())
    expected_arrays = {
        "counts",
        "pixel_quality",
        "pixel_time",
        "slope",
        "offset",
        "time",
        "latitude",
        "longitude",
    }
    missing = expected_arrays - array_keys_1km
    assert not missing, f"data_1km missing arrays: {missing}; have: {array_keys_1km}"

    assert data_1km["counts"].dtype == np.uint16
    assert data_1km["pixel_quality"].dtype == np.uint8
    assert data_1km["slope"].dtype == np.float64
    assert data_1km["offset"].dtype == np.float64
    assert data_1km["latitude"].dtype == np.float32
    assert data_1km["longitude"].dtype == np.float32

    assert data_1km["counts"].ndim == 4, "counts must be (timestamp, y, x, channel)"
    assert data_1km["latitude"].ndim == 2
    assert data_1km["latitude"].shape == data_1km["longitude"].shape
    assert data_1km["latitude"].shape == data_1km["counts"].shape[1:3]

    # Chunking invariant: even on tiny fixtures, the schema's nc_part-aligned
    # chunk_shape must be applied (zarr v3 may collapse the sharding codec for
    # arrays smaller than a single shard, so `.shards` is unreliable here; the
    # written chunk_shape is the stable invariant exposed by the schema).
    counts_arr = data_1km["counts"]
    assert counts_arr.chunks is not None, "data_1km/counts must have explicit chunks"
    assert counts_arr.chunks != counts_arr.shape, (
        f"counts chunks {counts_arr.chunks} collapsed to shape {counts_arr.shape}; "
        "schema chunk_shape was not applied"
    )

    data_2km = cast(Any, root_fdhsi["data_2km"])
    array_keys_2km = set(data_2km.array_keys())
    assert {"counts", "time", "latitude", "longitude"} <= array_keys_2km, (
        f"data_2km missing core arrays; have: {array_keys_2km}"
    )

    # Given: an HRFI ingest
    hrfi_ws = tmp_path / "hrfi_ws"
    hrfi_ws.mkdir()
    _run_ingest(hrfi_zip.parent, hrfi_ws)
    root_hrfi = zarr.open_group(str(hrfi_ws / "out.zarr"), mode="r")

    # Then: only HRFI groups exist
    hrfi_groups = set(root_hrfi.group_keys())
    assert "data_500m" in hrfi_groups, (
        f"data_500m missing in HRFI; groups: {hrfi_groups}"
    )
    assert "data_1km" in hrfi_groups, f"data_1km missing in HRFI; groups: {hrfi_groups}"
    assert "data_2km" not in hrfi_groups, "data_2km must not exist in HRFI output"

    data_500m = cast(Any, root_hrfi["data_500m"])
    array_keys_500m = set(data_500m.array_keys())
    assert {"counts", "time", "latitude", "longitude"} <= array_keys_500m, (
        f"data_500m missing core arrays; have: {array_keys_500m}"
    )
    assert data_500m["counts"].dtype == np.uint16
    assert data_500m["latitude"].dtype == np.float32


@pytest.mark.integration
@pytest.mark.plugin
def test_partial_failure_metrics_preserved(tmp_path: Path, small_fci_layout: list[int]):
    src = tmp_path / "partial"
    src.mkdir()

    # Given: one valid FDHSI ZIP
    _make_fdhsi_zip_at(src, "20240101000000")

    # Given: one structurally invalid ZIP - valid filename pattern (so the
    # ZIP discovery picks it up and timestamp parsing succeeds) but the bytes
    # are NOT a real ZIP archive, forcing extract_zip to raise.
    corrupt = src / "W_XX-FCI-1C-RRAD-FDHSI-FD-20240101001000-END.zip"
    corrupt.write_bytes(b"not-a-real-zip")

    target_path = tmp_path / "out.zarr"
    ingestor = MtgFciL1cIngestor()
    ctx = IngestContext(
        source=str(src),
        target=str(target_path),
        output_format="zarr",
        storage=StorageContext(output=_make_local_storage_session(target_path)),
        options={
            "pipeline_parallel": False,
            "force_reingest": True,
            "write_mode": "direct",
            "time_epoch": "2024-01-01",
        },
    )
    result = ingestor.run(ctx)

    # Then: the batch was processed without aborting (continue-on-error)
    assert result.outputs.primary is not None, "ingest did not produce an output"

    # Then: per-batch counters surface on the run-level result.metrics via the
    # ``_aggregate_metrics`` override (see ingestor.py). Consumers read these
    # four keys to detect partial failures.
    files_processed = result.metrics.get("files_processed")
    files_failed = result.metrics.get("files_failed")
    zip_errors = result.metrics.get("zip_errors") or []

    assert files_processed == 1, (
        f"expected files_processed=1 on result.metrics, got {files_processed!r}"
    )
    assert files_failed == 1, (
        f"expected files_failed=1 on result.metrics, got {files_failed!r}"
    )
    assert len(zip_errors) == 1, (
        f"expected exactly one zip_errors entry, got {zip_errors!r}"
    )
    assert (
        "corrupt" in zip_errors[0].lower()
        or "zip" in zip_errors[0].lower()
        or "file is not a zip" in zip_errors[0].lower()
    ), f"zip_errors entry should describe the failure; got {zip_errors[0]!r}"

    # Then: the valid ZIP still produced data in the store
    root = zarr.open_group(str(target_path), mode="r")
    assert "data_1km" in set(root.group_keys())
    data_1km = cast(Any, root["data_1km"])
    assert data_1km["counts"].shape[0] >= 1, (
        "valid ZIP did not produce any timestamps in the store"
    )


_CF_DIM_SIZES = {"time": 1, "y": 4, "x": 4, "channel": 2}


def _build_cf_dataset_from_specs(specs: list[Any], group: str) -> xr.Dataset:
    """Construct a tiny xr.Dataset mirroring ``build_specs(...)`` output for ``group``.

    The dataset preserves variable names, dims, dtypes, and attrs declared by the
    schema so that ``validate_cf18`` exercises the same metadata the plugin would
    write to a real Zarr store.
    """
    spec = next(s for s in specs if s.group == group)

    data_vars: dict[str, xr.DataArray] = {}
    coords: dict[str, xr.DataArray] = {}

    for arr in spec.arrays:
        dim_names = tuple(arr.dimension_names or ())
        shape = tuple(_CF_DIM_SIZES.get(d, 4) for d in dim_names)

        dtype = np.dtype(arr.dtype)
        if dtype.kind == "M":
            data = np.array(
                [np.datetime64("2024-01-01T00:00:00")] * (shape[0] if shape else 1),
                dtype=dtype,
            )
        elif dtype.kind == "S":
            length = shape[0] if shape else 1
            data = np.array([f"ch{i:02d}".encode() for i in range(length)], dtype=dtype)
        else:
            data = np.zeros(shape, dtype=dtype)

        if arr.name == "x":
            data = np.linspace(0.156, -0.156, shape[0], dtype=dtype)
        elif arr.name == "y":
            data = np.linspace(-0.156, 0.156, shape[0], dtype=dtype)

        attrs = {str(k): v for k, v in (arr.attrs or {}).items()}
        da = xr.DataArray(data, dims=dim_names, attrs=attrs)

        if arr.name in spec.coord_names or arr.name in dim_names:
            coords[arr.name] = da
        else:
            data_vars[arr.name] = da

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={str(k): v for k, v in (spec.attrs or {}).items()},
    )


def _format_errors(report: Any) -> str:
    return "\n".join(
        f"  {f.id} ({f.path}): {f.message}"
        for f in report.findings
        if f.severity.value == "error"
    )


@pytest.mark.integration
@pytest.mark.plugin
@pytest.mark.xfail(
    strict=False,
    reason=(
        "CFFinding.path attribute missing; tracked as plans/TODO.md §3. "
        "Remove xfail once the CF advisor call site is updated for the current "
        "CFFinding API or the upstream advisor bug is patched."
    ),
)
@pytest.mark.parametrize("include_geolocation", [True, False])
@pytest.mark.parametrize(
    "product_type,groups",
    [
        ("FDHSI", ("data_1km", "data_2km")),
        ("HRFI", ("data_500m", "data_1km")),
    ],
)
def test_cf_advisor_zero_errors_per_group(
    product_type: str, groups: tuple[str, ...], include_geolocation: bool
) -> None:
    """firecube advise compliance --profile cf-18 must find zero CF errors per group."""
    config = MtgFciL1cConfig(include_geolocation=include_geolocation)
    specs = build_specs(config, product_type)

    for group in groups:
        ds = _build_cf_dataset_from_specs(specs, group)
        report = validate_cf18(
            ds, product=f"file:///tmp/fci-cf/{product_type}.zarr", group=group
        )
        assert report.summary.errors == 0, (
            f"CF advisor reported {report.summary.errors} errors for {product_type}/{group} "
            f"(include_geolocation={include_geolocation}):\n{_format_errors(report)}"
        )
