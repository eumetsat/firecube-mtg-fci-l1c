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

"""Partial scenes and chunk-stripe stores, ingested end to end.

A synthetic product of three BODY chunks per resolution (12 rows at 2 km, 24
at 1 km, 48 at 500 m, so 4, 8 and 16 rows per chunk) is written as loose chunk
files with real-shaped names and ingested through the core harness. The output
chunk height is 3 rows, so the part edges (rows 4/8, 8/16, 16/32) fall inside
output chunks: partial coverage and stripe windows snapped to the chunk grid
are both exercised.

Pixel values encode their disk row and column, so a row written to the wrong
place is visible. Each chunk's root ``index``/``time`` table overlaps its
neighbours' by two indices, with a time that names the chunk that carries it
(``100 * index + chunk``), and the last pixel row of every chunk but the last
refers to an overlapping index. That row's ``pixel_time`` therefore shows
whether the next chunk was present: the later chunk wins on shared indices.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import h5netcdf
import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).parent))
from _store_compare import assert_rows_equal, assert_stores_bitwise_equal  # noqa: E402
from _store_files import store_files  # noqa: E402
from tests._support import _run_ingest  # noqa: E402
from test_unzipped_input import _chunk_file_name  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.plugin]

# product -> resolution -> (nc channels, rows and columns of the full disk)
_LAYOUT: dict[str, dict[str, tuple[tuple[str, ...], int]]] = {
    "FDHSI": {"1km": (("vis_04", "vis_06"), 24), "2km": (("ir_38",), 12)},
    "HRFI": {"500m": (("vis_06_hr",), 48), "1km": (("ir_38_hr",), 24)},
}
_PARTS = 3
_CHUNK_Y = 3
_PIXEL_ARRAYS = ("counts", "pixel_quality", "pixel_time")
_PLATFORM_ARRAYS = (
    "subsatellite_latitude",
    "subsatellite_longitude",
    "platform_altitude",
)
_RESUME = {"force_reingest": False, "resume_existing": True}
_PRODUCTS = pytest.mark.parametrize("product_type", ["FDHSI", "HRFI"])


def _rows_per_part(dim: int) -> int:
    return dim // _PARTS


def _part_rows(dim: int, number: int) -> tuple[int, int]:
    """Disk rows ``[start, stop)`` of BODY chunk ``number`` (1-based)."""
    size = _rows_per_part(dim)
    return size * (number - 1), size * number


def _part_indices(number: int) -> range:
    """Root ``index`` values chunk ``number`` carries: its four and two shared."""
    first = 4 * (number - 1)
    return range(first, min(first + 6, 4 * _PARTS))


def _window_start(dim: int, first_part: int) -> int:
    """First row of a stripe starting at ``first_part``, on the 3-row chunk grid."""
    return _part_rows(dim, first_part)[0] // _CHUNK_Y * _CHUNK_Y


def _window_stop(dim: int, last_part: int) -> int:
    return min(-(-_part_rows(dim, last_part)[1] // _CHUNK_Y) * _CHUNK_Y, dim)


@pytest.fixture
def stripe_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink both products to the three-chunk layout above."""
    from firecube_mtg_fci_l1c import _constants as const_mod
    from firecube_mtg_fci_l1c.geolocation import provider as geolocation_mod

    for product_type, resolutions in _LAYOUT.items():
        monkeypatch.setitem(
            const_mod.CONSTANTS,
            product_type,
            {
                res: {
                    "channels": list(channels),
                    "dimsize": dim,
                    "nc_channels": list(channels),
                }
                for res, (channels, dim) in resolutions.items()
            },
        )
        monkeypatch.setitem(const_mod.BODY_CHUNK_COUNT, product_type, _PARTS)
        for res, (_channels, dim) in resolutions.items():
            monkeypatch.setitem(
                const_mod.BODY_CHUNK_ROWS[product_type],
                res,
                tuple(_part_rows(dim, number) for number in range(1, _PARTS + 1)),
            )

    def fake_compute_latlon(resolution_m: int) -> tuple[np.ndarray, np.ndarray]:
        """Latitude is the row, longitude the column: a wrong row shows."""
        dim = {500: 48, 1000: 24, 2000: 12}[resolution_m]
        rows = np.repeat(np.arange(dim, dtype=np.float32)[:, None], dim, axis=1)
        return rows, rows.T.copy()

    monkeypatch.setattr(geolocation_mod, "compute_latlon", fake_compute_latlon)


# --- Synthetic scene ---------------------------------------------------------


def _expected_counts(channel_index: int, rows: range, dim: int) -> np.ndarray:
    """The counts the synthetic chunks hold for disk ``rows`` of one channel."""
    row_values = np.asarray(rows, dtype=np.int64)[:, None]
    return (10000 * (channel_index + 1) + 100 * row_values + np.arange(dim)).astype(
        np.uint16
    )


def _write_part(path: Path, product_type: str, number: int) -> None:
    indices = list(_part_indices(number))
    with h5netcdf.File(path, "w") as ds:
        ds.dimensions["n_time"] = len(indices)
        ds.create_variable("index", ("n_time",), data=np.asarray(indices, np.uint16))
        ds.create_variable(
            "time",
            ("n_time",),
            data=np.asarray([100.0 * i + number for i in indices], np.float64),
        )
        platform = ds.create_group("state").create_group("platform")
        for name, values in (
            ("subsatellite_latitude", [10.0 + i for i in indices]),
            ("subsatellite_longitude", [20.0 + 2 * i for i in indices]),
            ("platform_altitude", [35_786_000.0 + 1000 * i for i in indices]),
        ):
            platform.create_variable(
                name, ("n_time",), data=np.asarray(values, np.float32)
            )
        data = ds.create_group("data")
        for _res, (channels, dim) in _LAYOUT[product_type].items():
            start, stop = _part_rows(dim, number)
            rows = range(start, stop)
            for channel_index, channel in enumerate(channels):
                measured = data.create_group(channel).create_group("measured")
                measured.dimensions["y"] = stop - start
                measured.dimensions["x"] = dim
                radiance = measured.create_variable(
                    "effective_radiance",
                    ("y", "x"),
                    data=_expected_counts(channel_index, rows, dim),
                )
                radiance.attrs["scale_factor"] = float(channel_index + 1)
                radiance.attrs["add_offset"] = float(channel_index)
                measured.create_variable(
                    "start_position_row", (), data=np.int32(start + 1)
                )
                measured.create_variable("end_position_row", (), data=np.int32(stop))
                row_col = np.asarray(rows)[:, None] + np.arange(dim)
                measured.create_variable(
                    "pixel_quality", ("y", "x"), data=(row_col % 4).astype(np.uint8)
                )
                # Four index bins per chunk; the last row refers to the first
                # index the next chunk shares.
                local = np.arange(stop - start)
                index_map = np.repeat(
                    (indices[0] + 4 * local // (stop - start))[:, None], dim, axis=1
                )
                if number < _PARTS:
                    index_map[-1, :] = indices[0] + 4
                measured.create_variable(
                    "index_map", ("y", "x"), data=index_map.astype(np.uint16)
                )


def _write_scene(directory: Path, product_type: str, parts: tuple[int, ...]) -> Path:
    """Write the given BODY chunks of cycle 0001 of 2024-01-01 as loose files."""
    directory.mkdir(parents=True)
    for number in parts:
        _write_part(
            directory / _chunk_file_name(product_type, number), product_type, number
        )
    return directory


def _zip_scene(directory: Path, product_type: str, parts: tuple[int, ...]) -> Path:
    """A directory holding one ZIP of the given chunks, like a product download."""
    staging = _write_scene(
        directory.parent / f"{directory.name}-staging", product_type, parts
    )
    directory.mkdir()
    zip_path = directory / f"W_XX-FCI-1C-RRAD-{product_type}-FD-20240101000000-END.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for part in sorted(staging.iterdir()):
            archive.write(part, arcname=part.name)
    shutil.rmtree(staging)
    return directory


def _ingest(
    source: Path,
    workspace: Path,
    product_type: str,
    **options: object,
) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    return _run_ingest(
        source,
        workspace,
        {"zarr_chunk_y": _CHUNK_Y, **options},
        product_type=product_type,
    )


def _group_names(product_type: str) -> list[tuple[str, str, tuple[str, ...], int]]:
    return [
        (f"data_{res}", res, channels, dim)
        for res, (channels, dim) in _LAYOUT[product_type].items()
    ]


def _open(store: Path) -> Any:
    return zarr.open_group(str(store), mode="r")


def _is_fill(values: np.ndarray, fill: Any) -> bool:
    if np.issubdtype(values.dtype, np.floating) and np.isnan(fill):
        return bool(np.isnan(values).all())
    return bool((values == fill).all())


# --- What a partial input may and may not change -----------------------------


def _platform_means(parts: tuple[int, ...]) -> dict[str, float]:
    """Mean of each platform table over the indices the given chunks carry."""
    indices = np.asarray(sorted({i for n in parts for i in _part_indices(n)}))
    return {
        "subsatellite_latitude": float(np.mean(10.0 + indices)),
        "subsatellite_longitude": float(np.mean(20.0 + 2 * indices)),
        "platform_altitude": float(np.mean(35_786_000.0 + 1000 * indices)),
    }


def _assert_documented_differences(
    full_store: Path,
    partial_store: Path,
    product_type: str,
    present: tuple[int, ...],
    *,
    y_offsets: dict[str, int],
) -> None:
    """The partial store equals the full-input store except where documented.

    Both stores hold the same rows; array row ``i`` of a group is disk row
    ``y_offsets[resolution] + i``. Allowed differences, and nothing else:

    * the platform variables: means over the indices of the chunks given;
    * ``pixel_time`` of the last row of the last present chunk, when a later
      chunk exists: the later chunk wins the shared index, and it was not given;
    * rows of an output chunk that the given chunks do not cover: fill.
    """
    full, partial = _open(full_store), _open(partial_store)
    first, last = min(present), max(present)
    expected_means = _platform_means(present)
    for group, res, channels, dim in _group_names(product_type):
        y0 = y_offsets[res]
        covered_start = _part_rows(dim, first)[0]
        covered_stop = _part_rows(dim, last)[1]
        assert sorted(full[group].array_keys()) == sorted(partial[group].array_keys())
        for name, full_arr in full[group].arrays():
            partial_arr = partial[group][name]
            if name in _PLATFORM_ARRAYS:
                np.testing.assert_allclose(
                    np.asarray(partial_arr[:], np.float64),
                    expected_means[name],
                    rtol=1e-6,
                    err_msg=f"{group}/{name}",
                )
                assert not np.array_equal(full_arr[:], partial_arr[:]), (
                    f"{group}/{name}: means over other chunks cannot be equal"
                )
            elif name not in _PIXEL_ARRAYS:
                assert (
                    np.asarray(full_arr[...]).tobytes()
                    == np.asarray(partial_arr[...]).tobytes()
                ), f"{group}/{name} differs"

        for name in _PIXEL_ARRAYS:
            full_px = np.asarray(full[group][name][...])
            partial_px = np.asarray(partial[group][name][...])
            fill = full[group][name].fill_value
            assert partial_px.shape == full_px.shape
            for disk_row in range(y0, y0 + full_px.shape[1]):
                row = disk_row - y0
                if covered_start <= disk_row < covered_stop:
                    want = full_px[:, row]
                    if (
                        name == "pixel_time"
                        and disk_row == covered_stop - 1
                        and last < _PARTS
                    ):
                        # The chunk after the last given one wins this row's
                        # index in the full input: chunk ``last + 1``'s time.
                        index = 4 * last
                        np.testing.assert_array_equal(want, 100.0 * index + last + 1)
                        np.testing.assert_array_equal(
                            partial_px[:, row], 100.0 * index + last
                        )
                        continue
                    assert not _is_fill(want, fill), f"{group}/{name} row {disk_row}"
                    assert want.tobytes() == partial_px[:, row].tobytes(), (
                        f"{group}/{name} disk row {disk_row} differs"
                    )
                else:
                    assert not _is_fill(full_px[:, row], fill), (
                        f"{group}/{name} disk row {disk_row} is empty in the full input"
                    )
                    assert _is_fill(partial_px[:, row], fill), (
                        f"{group}/{name} disk row {disk_row} was written"
                    )
        # The rows themselves are the synthetic ones, not merely equal.
        counts = np.asarray(partial[group]["counts"][...])
        for channel_index in range(len(channels)):
            rows = range(covered_start, covered_stop)
            np.testing.assert_array_equal(
                counts[0, covered_start - y0 : covered_stop - y0, :, channel_index],
                _expected_counts(channel_index, rows, dim),
            )


def _full_disk_offsets(product_type: str) -> dict[str, int]:
    return {res: 0 for res in _LAYOUT[product_type]}


def _stripe_offsets(product_type: str, first_part: int) -> dict[str, int]:
    return {
        res: _window_start(dim, first_part)
        for res, (_channels, dim) in _LAYOUT[product_type].items()
    }


# --- (a) A partial bundle in the full-disk shape -------------------------------


@_PRODUCTS
@pytest.mark.parametrize("sharding", [True, False], ids=["sharded", "unsharded"])
@pytest.mark.parametrize("present", [(2, 3), (1, 2)], ids=["parts-2-3", "parts-1-2"])
def test_partial_bundle_equals_the_full_ingest_on_its_rows(
    tmp_path: Path,
    stripe_layout: None,
    product_type: str,
    sharding: bool,
    present: tuple[int, ...],
):
    full_source = _write_scene(tmp_path / "full", product_type, (1, 2, 3))
    partial_source = _write_scene(tmp_path / "partial", product_type, present)

    full = _ingest(
        full_source, tmp_path / "ws_full", product_type, zarr_sharding=sharding
    )
    partial = _ingest(
        partial_source, tmp_path / "ws_partial", product_type, zarr_sharding=sharding
    )

    _assert_documented_differences(
        full,
        partial,
        product_type,
        present,
        y_offsets=_full_disk_offsets(product_type),
    )


@_PRODUCTS
def test_output_chunks_that_meet_no_part_are_not_written(
    tmp_path: Path, stripe_layout: None, product_type: str
):
    source = _write_scene(tmp_path / "src", product_type, (2, 3))

    store = _ingest(source, tmp_path / "ws", product_type, zarr_sharding=False)

    files = store_files(store)
    for group, _res, _channels, dim in _group_names(product_type):
        first_chunk = _part_rows(dim, 2)[0] // _CHUNK_Y
        for name in _PIXEL_ARRAYS:
            written = {
                int(rel.split("/")[-3])
                for rel in files
                if rel.startswith(f"{group}/{name}/c/")
            }
            # y chunk indices: the edge chunk that holds rows of part 1 and
            # part 2 is written (covered rows only), the ones before are not.
            assert written == set(range(first_chunk, dim // _CHUNK_Y)), (
                f"{group}/{name}: chunk files {sorted(written)}"
            )


# --- (b) partial_chunk=error ----------------------------------------------------


@_PRODUCTS
@pytest.mark.parametrize(
    "options", [{}, {"fci_chunks": [2, 3]}], ids=["full-disk", "stripe"]
)
def test_partial_chunk_error_fails_the_run_before_anything_is_written(
    tmp_path: Path, stripe_layout: None, product_type: str, options: dict[str, object]
):
    source = _write_scene(tmp_path / "src", product_type, (2, 3))
    workspace = tmp_path / "ws"
    workspace.mkdir()

    # Core fails the batch on the plugin's ConfigurationError and ends the run
    # with PipelineFailedBatchesError, a RuntimeError carrying the message.
    with pytest.raises(RuntimeError) as excinfo:
        _ingest(source, workspace, product_type, partial_chunk="error", **options)

    message = str(excinfo.value)
    assert "does not cover output chunks it would write" in message
    for group, res, _channels, dim in _group_names(product_type):
        chunk_start = _window_start(dim, 2)
        gap = f"[{chunk_start}, {_part_rows(dim, 2)[0]})"
        assert f"group '{group}' ({res}) rows {gap}; missing BODY chunk(s) 1" in message
    # Core opened the store and recorded the failed run under .firecube/;
    # no group, array or chunk was written.
    assert sorted(store_files(workspace / "out.zarr")) == ["zarr.json"]


@_PRODUCTS
def test_partial_chunk_error_accepts_a_bundle_that_fills_every_chunk_it_touches(
    tmp_path: Path, stripe_layout: None, product_type: str
):
    # Chunks of 4 rows start on every part edge (4, 8 and 16 rows per part), so
    # the chunks parts 2-3 touch are covered whole. Chunks meeting no part
    # raise nothing: they are never written.
    options = {"zarr_chunk_y": 4, "partial_chunk": "error"}
    source = _write_scene(tmp_path / "src", product_type, (2, 3))
    full_source = _write_scene(tmp_path / "full", product_type, (1, 2, 3))

    strict = _ingest(source, tmp_path / "ws", product_type, **options)
    reference = _ingest(full_source, tmp_path / "ws_full", product_type, zarr_chunk_y=4)

    _assert_documented_differences(
        reference,
        strict,
        product_type,
        (2, 3),
        y_offsets=_full_disk_offsets(product_type),
    )


# --- (c) stripe shape against the full-disk shape -------------------------------


@_PRODUCTS
@pytest.mark.parametrize(
    ("parts", "window"),
    [((2, 3), (2, 3)), ((1, 2), (1, 2)), ((1, 2, 3), (1, 2))],
    ids=["window-2-3", "window-1-2", "window-1-2-with-part-3-given"],
)
def test_stripe_store_holds_the_window_rows_of_the_full_disk_store(
    tmp_path: Path,
    stripe_layout: None,
    product_type: str,
    parts: tuple[int, ...],
    window: tuple[int, int],
):
    source = _write_scene(tmp_path / "src", product_type, parts)

    full_shape = _ingest(source, tmp_path / "ws_full", product_type)
    stripe = _ingest(
        source, tmp_path / "ws_stripe", product_type, fci_chunks=list(window)
    )

    full_root, stripe_root = _open(full_shape), _open(stripe)
    for group, _res, channels, dim in _group_names(product_type):
        y0 = _window_start(dim, window[0])
        y1 = _window_stop(dim, window[1])
        assert sorted(stripe_root[group].array_keys()) == sorted(
            full_root[group].array_keys()
        )
        for name, stripe_arr in stripe_root[group].arrays():
            assert_rows_equal(
                full_root[group][name], stripe_arr, y0, name=f"{group}/{name}"
            )
        # The stripe is the window, not the disk, and it holds real rows.
        assert stripe_root[group]["counts"].shape == (1, y1 - y0, dim, len(channels))
        assert list(stripe_root[group]["y"].shape) == [y1 - y0]
        covered = [
            row
            for number in parts
            for row in range(*_part_rows(dim, number))
            if y0 <= row < y1
        ]
        first_row = min(covered)
        counts = np.asarray(stripe_root[group]["counts"][0, first_row - y0 :, :, 0])
        np.testing.assert_array_equal(
            counts[: len(covered)],
            _expected_counts(0, range(first_row, first_row + len(covered)), dim),
        )
        latitude = np.asarray(stripe_root[group]["latitude"][:, 0])
        np.testing.assert_array_equal(latitude, np.arange(y0, y1, dtype=np.float32))


# --- (d) ZIP and loose stripes, full and partial input --------------------------


@_PRODUCTS
def test_zip_and_loose_bundle_of_the_same_chunks_give_identical_stripe_stores(
    tmp_path: Path, stripe_layout: None, product_type: str
):
    loose = _write_scene(tmp_path / "loose", product_type, (2, 3))
    zipped = _zip_scene(tmp_path / "zipped", product_type, (2, 3))
    options: dict[str, Any] = {"fci_chunks": [2, 3]}

    from_loose = _ingest(loose, tmp_path / "ws_loose", product_type, **options)
    from_zip = _ingest(zipped, tmp_path / "ws_zip", product_type, **options)

    group = _open(from_loose)[_group_names(product_type)[0][0]]
    assert (
        group["counts"].shape[1]
        < _LAYOUT[product_type][_group_names(product_type)[0][1]][1]
    )
    assert np.any(np.asarray(group["counts"][0, -1]) != np.iinfo(np.uint16).max)
    assert_stores_bitwise_equal(from_loose, from_zip)


@_PRODUCTS
@pytest.mark.parametrize(
    ("window", "present"),
    [((2, 3), (2, 3)), ((1, 2), (1, 2))],
    ids=["window-2-3-without-part-1", "window-1-2-without-part-3"],
)
def test_partial_input_stripe_differs_from_full_input_stripe_only_as_documented(
    tmp_path: Path,
    stripe_layout: None,
    product_type: str,
    window: tuple[int, int],
    present: tuple[int, ...],
):
    full_source = _write_scene(tmp_path / "full", product_type, (1, 2, 3))
    partial_source = _write_scene(tmp_path / "partial", product_type, present)
    options: dict[str, Any] = {"fci_chunks": list(window)}

    full = _ingest(full_source, tmp_path / "ws_full", product_type, **options)
    partial = _ingest(partial_source, tmp_path / "ws_partial", product_type, **options)

    _assert_documented_differences(
        full,
        partial,
        product_type,
        present,
        y_offsets=_stripe_offsets(product_type, window[0]),
    )


# --- (e) A stripe store keeps its window -----------------------------------------


@_PRODUCTS
@pytest.mark.parametrize(
    ("second", "incoming_name"),
    [
        ({"fci_chunks": [1, 2]}, "eumetsat_repeat_cycle_v1_stripe_c1_2"),
        ({}, "eumetsat_repeat_cycle_v1"),
    ],
    ids=["other-window", "full-disk-run"],
)
def test_run_with_another_window_is_rejected_and_the_stripe_store_is_untouched(
    tmp_path: Path,
    stripe_layout: None,
    product_type: str,
    second: dict[str, object],
    incoming_name: str,
):
    first_source = _write_scene(tmp_path / "first", product_type, (2, 3))
    second_source = _write_scene(tmp_path / "second", product_type, (1, 2))
    workspace = tmp_path / "ws"
    store = _ingest(first_source, workspace, product_type, fci_chunks=[2, 3])
    before = store_files(store)
    group = _group_names(product_type)[0][0]
    counts_before = np.asarray(_open(store)[group]["counts"][...])
    assert np.any(counts_before != np.iinfo(np.uint16).max)

    with pytest.raises(Exception) as excinfo:
        _ingest(second_source, workspace, product_type, **second)

    assert type(excinfo.value).__name__ == "ResolvedIndexConflictError"
    assert (
        f'name: stored="eumetsat_repeat_cycle_v1_stripe_c2_3" incoming="{incoming_name}"'
        in str(excinfo.value)
    )
    assert store_files(store) == before
    np.testing.assert_array_equal(_open(store)[group]["counts"][...], counts_before)

    # The store still takes its own window.
    _ingest(first_source, workspace, product_type, fci_chunks=[2, 3])
    assert store_files(store) == before


# --- (f) Resuming a stripe -------------------------------------------------------


@_PRODUCTS
def test_same_stripe_twice_resumes_without_error_and_leaves_the_data_equal(
    tmp_path: Path, stripe_layout: None, product_type: str
):
    source = _write_scene(tmp_path / "src", product_type, (2, 3))
    workspace = tmp_path / "ws"
    store = _ingest(source, workspace, product_type, fci_chunks=[2, 3])
    before = store_files(store)
    assert any("/counts/c/" in rel for rel in before)

    resumed = _ingest(source, workspace, product_type, fci_chunks=[2, 3], **_RESUME)

    assert resumed == store
    assert store_files(store) == before


# --- (g) A later ingest completes a slot -----------------------------------------


@_PRODUCTS
@pytest.mark.parametrize("sharding", [True, False], ids=["sharded", "unsharded"])
@pytest.mark.parametrize(
    "second_run",
    [{"force_reingest": True}, _RESUME],
    ids=["force_reingest", "resume_existing"],
)
def test_part_ingested_later_completes_the_slot_and_keeps_the_rows_already_written(
    tmp_path: Path,
    stripe_layout: None,
    product_type: str,
    sharding: bool,
    second_run: dict[str, object],
):
    full_source = _write_scene(tmp_path / "full", product_type, (1, 2, 3))
    later_source = _write_scene(tmp_path / "part1", product_type, (1,))
    first_source = _write_scene(tmp_path / "parts23", product_type, (2, 3))
    reference = _ingest(
        full_source, tmp_path / "ws_ref", product_type, zarr_sharding=sharding
    )

    # Same store, same slot. Core's resume guard refuses a second run into a
    # store with entries unless it is told to overwrite (force_reingest) or to
    # continue (resume_existing); both are accepted here, see the next test for
    # the refusal.
    workspace = tmp_path / "ws"
    store = _ingest(first_source, workspace, product_type, zarr_sharding=sharding)
    root = _open(store)
    rows_before = {
        (group, name): np.asarray(root[group][name][...])
        for group, _res, _channels, _dim in _group_names(product_type)
        for name in _PIXEL_ARRAYS
    }
    files_before = store_files(store)

    _ingest(later_source, workspace, product_type, zarr_sharding=sharding, **second_run)

    root, reference_root = _open(store), _open(reference)
    for group, _res, channels, dim in _group_names(product_type):
        start, stop = _part_rows(dim, 2)[0], dim
        later = _part_rows(dim, 1)
        for name in _PIXEL_ARRAYS:
            now = np.asarray(root[group][name][...])
            before = rows_before[(group, name)]
            want = np.asarray(reference_root[group][name][...])
            fill = root[group][name].fill_value
            # Rows of parts 2-3 are byte-identical to what the first run wrote.
            assert now[:, start:stop].tobytes() == before[:, start:stop].tobytes(), (
                f"{group}/{name}: rows written by parts 2-3 changed"
            )
            # Part 1's rows were empty and now hold data.
            assert _is_fill(before[:, later[0] : later[1]], fill)
            assert not _is_fill(now[:, later[0] : later[1]], fill)
            if name == "pixel_time":
                # The last row of part 1 refers to an index chunk 2 shares and
                # wins in the full input; this run read part 1 alone.
                edge = later[1] - 1
                np.testing.assert_array_equal(now[:, edge], 100.0 * 4 + 1)
                np.testing.assert_array_equal(want[:, edge], 100.0 * 4 + 2)
                keep = np.ones(dim, dtype=bool)
                keep[edge] = False
                assert now[:, keep].tobytes() == want[:, keep].tobytes()
            else:
                assert now.tobytes() == want.tobytes(), f"{group}/{name}"
        for channel_index in range(len(channels)):
            np.testing.assert_array_equal(
                np.asarray(root[group]["counts"][0, :, :, channel_index]),
                _expected_counts(channel_index, range(dim), dim),
            )
        # Slot-level variables come from the latest ingest only.
        means = _platform_means((1,))
        for name in _PLATFORM_ARRAYS:
            np.testing.assert_allclose(
                np.asarray(root[group][name][:], np.float64), means[name], rtol=1e-6
            )

    if not sharding:
        # Chunk files wholly inside parts 2-3 were not rewritten.
        files_after = store_files(store)
        for group, _res, _channels, dim in _group_names(product_type):
            first_whole = -(-_part_rows(dim, 2)[0] // _CHUNK_Y)
            for name in _PIXEL_ARRAYS:
                whole = [
                    rel
                    for rel in files_before
                    if rel.startswith(f"{group}/{name}/c/")
                    and int(rel.split("/")[-3]) >= first_whole
                ]
                assert whole
                for rel in whole:
                    assert files_after[rel] == files_before[rel], rel


@_PRODUCTS
def test_second_ingest_of_a_slot_without_force_or_resume_is_refused_and_writes_nothing(
    tmp_path: Path, stripe_layout: None, product_type: str
):
    from firecube.ingestor.api import ResumeConflictError

    first_source = _write_scene(tmp_path / "parts23", product_type, (2, 3))
    later_source = _write_scene(tmp_path / "part1", product_type, (1,))
    workspace = tmp_path / "ws"
    store = _ingest(first_source, workspace, product_type)
    before = store_files(store)

    with pytest.raises(ResumeConflictError, match="Existing entries"):
        _ingest(later_source, workspace, product_type, force_reingest=False)

    assert store_files(store) == before


# --- (h) Staged write mode takes complete scenes only ----------------------------
#
# A staged run replaces every output chunk (every shard, when sharded) it writes
# in the target, so a piece of a scene would reset the rows other pieces wrote
# there. Completing a slot in pieces is direct mode's job, see (g).

_STAGED = {"write_mode": "staged"}


@_PRODUCTS
@pytest.mark.parametrize("sharding", [True, False], ids=["sharded", "unsharded"])
@pytest.mark.parametrize(
    "second_run",
    [{"force_reingest": True}, _RESUME],
    ids=["force_reingest", "resume_existing"],
)
def test_staged_ingest_of_a_missing_piece_is_refused_and_the_written_rows_survive(
    tmp_path: Path,
    stripe_layout: None,
    product_type: str,
    sharding: bool,
    second_run: dict[str, object],
):
    first_source = _write_scene(tmp_path / "parts23", product_type, (2, 3))
    later_source = _write_scene(tmp_path / "part1", product_type, (1,))
    workspace = tmp_path / "ws"
    store = _ingest(first_source, workspace, product_type, zarr_sharding=sharding)
    before = store_files(store)

    with pytest.raises(RuntimeError) as excinfo:
        _ingest(
            later_source,
            workspace,
            product_type,
            zarr_sharding=sharding,
            **_STAGED,
            **second_run,
        )

    message = str(excinfo.value)
    assert "--write-mode direct" in message
    for group, res, _channels, dim in _group_names(product_type):
        gap = f"[{_part_rows(dim, 1)[1]}, {dim})"
        assert f"group '{group}' ({res}) rows {gap}; missing BODY chunk(s) 2, 3" in (
            message
        )
    assert store_files(store) == before
    root = _open(store)
    for group, _res, _channels, dim in _group_names(product_type):
        counts = np.asarray(root[group]["counts"][0, :, :, 0])
        rows = range(_part_rows(dim, 2)[0], dim)
        np.testing.assert_array_equal(
            counts[rows.start :], _expected_counts(0, rows, dim)
        )


@_PRODUCTS
@pytest.mark.parametrize("partial_chunk", ["fill", "error"])
@pytest.mark.parametrize(
    ("options", "stripe"),
    [({}, False), ({"fci_chunks": [2, 3]}, True), ({"zarr_chunk_y": 4}, False)],
    # chunk-aligned: no output chunk has a gap, but the shard (or a later
    # piece's chunk) still spans rows the scene does not hold.
    ids=["full-disk", "stripe", "chunk-aligned"],
)
def test_staged_mode_rejects_a_partial_scene_before_anything_is_written(
    tmp_path: Path,
    stripe_layout: None,
    product_type: str,
    partial_chunk: str,
    options: dict[str, object],
    stripe: bool,
):
    source = _write_scene(tmp_path / "src", product_type, (2, 3))
    workspace = tmp_path / "ws"
    workspace.mkdir()

    with pytest.raises(RuntimeError) as excinfo:
        _ingest(
            source,
            workspace,
            product_type,
            partial_chunk=partial_chunk,
            **_STAGED,
            **options,
        )

    message = str(excinfo.value)
    assert "Use --write-mode direct to ingest a scene in pieces" in message
    assert "partial_chunk=fill" not in message
    for group, res, _channels, dim in _group_names(product_type):
        start = _window_start(dim, 2) if stripe else 0
        gap = f"[{start}, {_part_rows(dim, 2)[0]})"
        assert f"group '{group}' ({res}) rows {gap}; missing BODY chunk(s) 1" in message
    assert sorted(store_files(workspace / "out.zarr")) == ["zarr.json"]


@_PRODUCTS
@pytest.mark.parametrize("sharding", [True, False], ids=["sharded", "unsharded"])
@pytest.mark.parametrize(
    ("parts", "options"),
    [
        ((1, 2, 3), {}),
        # Window [2,3] widened to the 3-row grid reaches into part 1.
        ((1, 2, 3), {"fci_chunks": [2, 3]}),
        # On a 4-row grid the window is exactly parts 2-3.
        ((2, 3), {"fci_chunks": [2, 3], "zarr_chunk_y": 4}),
    ],
    ids=["full-disk", "stripe-widened", "stripe-exact"],
)
def test_staged_mode_ingests_a_scene_that_covers_its_window_as_direct_mode_does(
    tmp_path: Path,
    stripe_layout: None,
    product_type: str,
    sharding: bool,
    parts: tuple[int, ...],
    options: dict[str, object],
):
    source = _write_scene(tmp_path / "src", product_type, parts)
    staged_ws = tmp_path / "ws_staged"
    _ingest(
        source,
        staged_ws,
        product_type,
        zarr_sharding=sharding,
        partial_chunk="error",
        **_STAGED,
        **options,
    )
    direct = _ingest(
        source, tmp_path / "ws_direct", product_type, zarr_sharding=sharding, **options
    )

    staged = staged_ws / "out.zarr"
    assert_stores_bitwise_equal(staged, direct)
    root = _open(staged)
    for group, _res, _channels, dim in _group_names(product_type):
        counts = np.asarray(root[group]["counts"][0, :, :, 0])
        first = dim - counts.shape[0]
        np.testing.assert_array_equal(
            counts, _expected_counts(0, range(first, dim), dim)
        )
