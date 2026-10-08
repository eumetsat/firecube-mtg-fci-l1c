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

"""Shared builders and the ingest run harness for the plugin tests.

Fixtures built on these live in ``conftest.py``. Import this module as
``tests._support`` so every test module sees one module object.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import h5netcdf
import numpy as np
from firecube_mtg_fci_l1c import MtgFciL1cIngestor
from firecube_mtg_fci_l1c._constants import PRODUCT_TYPE_FDHSI, get_nc_part_prefix

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


_NC_FLOAT32_FILL = 9.96921e36


def _write_nc_part_netcdf(
    path: Path,
    nc_channels: list[str],
    dimsize: int,
    radiance_attrs: dict[str, dict[str, float]] | None = None,
    counts: dict[str, np.ndarray] | None = None,
    measured_scalars: dict[str, dict[str, float]] | None = None,
    state_tables: dict[str, list[float]] | None = None,
) -> None:
    with h5netcdf.File(path, "w") as ds:
        ds.attrs["time_coverage_start"] = "20240101000000"

        ds.dimensions["n_time"] = 2
        ds.create_variable("index", ("n_time",), data=np.array([0, 1], dtype=np.uint16))
        time_var = ds.create_variable("time", ("n_time",), data=np.array([0.0, 60.0]))
        time_var.attrs["_FillValue"] = 0.0
        # Index-dimensioned tables such as state/platform/platform_altitude.
        for table_path, values in (state_tables or {}).items():
            group_path, _, name = table_path.rpartition("/")
            group = ds
            for part in group_path.split("/"):
                group = (
                    group[part] if part in group.groups else group.create_group(part)
                )
            group.create_variable(
                name, ("n_time",), data=np.asarray(values, dtype=np.float32)
            )

        data_group = ds.create_group("data")
        for i, channel in enumerate(nc_channels):
            channel_group = data_group.create_group(channel)
            measured = channel_group.create_group("measured")
            measured.dimensions["y"] = dimsize
            measured.dimensions["x"] = dimsize

            radiance = measured.create_variable(
                "effective_radiance",
                ("y", "x"),
                data=(counts or {}).get(
                    channel, np.full((dimsize, dimsize), i + 1, dtype=np.uint16)
                ),
            )
            radiance.attrs["scale_factor"] = float(i + 1)
            radiance.attrs["add_offset"] = float(i)
            for name, value in (radiance_attrs or {}).get(channel, {}).items():
                radiance.attrs[name] = np.float32(value)

            for name, value in (measured_scalars or {}).get(channel, {}).items():
                # Mirrors the L1C float32 measured scalars and their default fill.
                measured.create_variable(
                    name,
                    (),
                    data=np.float32(value),
                    fillvalue=np.float32(_NC_FLOAT32_FILL),
                )

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
    zip_path: Path,
    product_type: str,
    nc_channels: list[str],
    dimsize: int,
    radiance_attrs: dict[str, dict[str, float]] | None = None,
    counts: dict[str, np.ndarray] | None = None,
    measured_scalars: dict[str, dict[str, float]] | None = None,
    state_tables: dict[str, list[float]] | None = None,
) -> Path:
    tmp_nc = zip_path.with_suffix(".nc")
    _write_nc_part_netcdf(
        tmp_nc,
        nc_channels=nc_channels,
        dimsize=dimsize,
        radiance_attrs=radiance_attrs,
        counts=counts,
        measured_scalars=measured_scalars,
        state_tables=state_tables,
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(tmp_nc, arcname=f"{get_nc_part_prefix(product_type)}0001.nc")
    tmp_nc.unlink()
    return zip_path


def _run_ingest(
    source: Path,
    workspace: Path,
    options: dict[str, object] | None = None,
    *,
    product_type: str = PRODUCT_TYPE_FDHSI,
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
            "force_reingest": True,
            "write_mode": "direct",
            # Fixtures use 2024-01-01 timestamps (pre-dating real FCI data); anchor
            # the deterministic slot index there so they map to compact slots 0,1,...
            "time_epoch": "2024-01-01",
            "product_type": product_type,
            **(options or {}),
        },
    )
    result = ingestor.run(ctx)
    result_path = Path(str(result.output_path))
    if result_path.exists():
        return result_path
    return target_path


def _make_fdhsi_zip_at(src_dir: Path, ts_str: str) -> Path:
    src_dir.mkdir(parents=True, exist_ok=True)
    zip_path = src_dir / f"W_XX-FCI-1C-RRAD-FDHSI-FD-{ts_str}-END.zip"
    return _make_zip_with_nc_part(
        zip_path, PRODUCT_TYPE_FDHSI, ["vis_04", "vis_06", "ir_38"], dimsize=4
    )


def _chunk_name(kind: str, start: str, cycle: int, number: int) -> str:
    """Real-shaped FDHSI chunk file name (pattern taken from a 2025 product)."""
    return (
        "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
        f"CHK-{kind}---NC4E_C_EUMT_{start}_IDPFI_OPE_{start}_{start}_N__O_"
        f"{cycle:04d}_{number:04d}.nc"
    )


def _write_chunk_netcdf(
    path: Path,
    *,
    rows: tuple[int, int] | None,
    value: int,
    index: list[int],
    times: list[float],
) -> None:
    """Write one chunk: BODY rows ``[start, stop)`` of every channel, or a TRAIL.

    A TRAIL (``rows=None``) carries only the root ``index``/``time`` table.
    Every pixel's ``index_map`` is 1, so ``pixel_time`` reads ``time`` at
    index 1 from whichever part wins that row of the root table.
    """
    with h5netcdf.File(path, "w") as ds:
        ds.dimensions["n_time"] = len(index)
        ds.create_variable("index", ("n_time",), data=np.asarray(index, np.uint16))
        ds.create_variable("time", ("n_time",), data=np.asarray(times, np.float64))
        if rows is None:
            return
        n_rows = rows[1] - rows[0]
        data_group = ds.create_group("data")
        for i, channel in enumerate(["vis_04", "vis_06", "ir_38"]):
            measured = data_group.create_group(channel).create_group("measured")
            measured.dimensions["y"] = n_rows
            measured.dimensions["x"] = 4
            radiance = measured.create_variable(
                "effective_radiance",
                ("y", "x"),
                data=np.full((n_rows, 4), value + i, dtype=np.uint16),
            )
            radiance.attrs["scale_factor"] = float(i + 1)
            radiance.attrs["add_offset"] = float(i)
            measured.create_variable(
                "start_position_row", (), data=np.int32(rows[0] + 1)
            )
            measured.create_variable("end_position_row", (), data=np.int32(rows[1]))
            measured.create_variable(
                "pixel_quality", ("y", "x"), data=np.zeros((n_rows, 4), np.uint8)
            )
            measured.create_variable(
                "index_map", ("y", "x"), data=np.ones((n_rows, 4), np.uint16)
            )


def _write_scene_chunks(
    directory: Path, *, start: str, cycle: int, values: tuple[int, int]
) -> list[Path]:
    """Write BODY 1 (rows 0-1), BODY 2 (rows 2-3) and TRAIL 41 of one cycle.

    BODY 1 and BODY 2 both carry root index 1 with different times, so the
    stored ``pixel_time`` shows which part was read last.
    """
    directory.mkdir(parents=True, exist_ok=True)
    specs = [
        ("BODY", 1, (0, 2), values[0], [0, 1], [10.0, 20.0]),
        ("BODY", 2, (2, 4), values[1], [1, 2], [30.0, 40.0]),
        ("TRAIL", 41, None, 0, [2], [50.0]),
    ]
    paths = []
    for kind, number, rows, value, index, times in specs:
        path = directory / _chunk_name(kind, start, cycle, number)
        _write_chunk_netcdf(path, rows=rows, value=value, index=index, times=times)
        paths.append(path)
    return paths
