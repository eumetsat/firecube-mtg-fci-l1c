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

"""A store keeps the layout and grid it was first written with.

Switching between the flat and the grouped layout on an existing store, or
writing a flat store of one resolution with another resolution, must be
rejected by Firecube as a resolved-index conflict before any Zarr data is
touched. The store then still accepts runs with its original layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from _store_files import store_files
from test_integration import _make_fdhsi_zip_at, _run_ingest

pytestmark = [pytest.mark.integration, pytest.mark.plugin]

FIRST_SLOT = "20240101000000"
SECOND_SLOT = "20240101001000"
RESUME = {"force_reingest": False, "resume_existing": True}
GROUPED_1KM = {"resolutions": "1km"}
FLAT_1KM = {"resolutions": "1km", "flat_store": True}
FLAT_2KM = {"resolutions": "2km", "flat_store": True}
FIRST_SLOT_TIMES = [np.datetime64("2024-01-01T00:00:00", "s")]


def _layout(store: Path) -> tuple[list[str], list[str]]:
    root = zarr.open_group(str(store), mode="r")
    return sorted(root.group_keys()), sorted(root.array_keys())


def _times(array: Any) -> list[np.datetime64]:
    return np.asarray(array[:]).astype("datetime64[s]").tolist()


def _assert_resolved_index_conflict(
    excinfo: pytest.ExceptionInfo[Exception], stored_name: str, incoming_name: str
) -> None:
    assert type(excinfo.value).__name__ == "ResolvedIndexConflictError"
    assert f'name: stored="{stored_name}" incoming="{incoming_name}"' in str(
        excinfo.value
    )


@pytest.mark.parametrize(
    ("stored", "requested", "group", "stored_name", "incoming_name"),
    [
        pytest.param(
            GROUPED_1KM,
            FLAT_1KM,
            "data_1km",
            "eumetsat_repeat_cycle_v1",
            "eumetsat_repeat_cycle_v1_fdhsi_1km",
            id="grouped-store-flat-run",
        ),
        pytest.param(
            FLAT_1KM,
            GROUPED_1KM,
            "",
            "eumetsat_repeat_cycle_v1_fdhsi_1km",
            "eumetsat_repeat_cycle_v1",
            id="flat-store-grouped-run",
        ),
    ],
)
def test_run_with_the_other_layout_is_rejected_and_store_is_untouched(
    tmp_path: Path,
    small_fci_layout: list[int],
    stored: dict[str, object],
    requested: dict[str, object],
    group: str,
    stored_name: str,
    incoming_name: str,
) -> None:
    source = _make_fdhsi_zip_at(tmp_path / "src_first", FIRST_SLOT).parent
    store = _run_ingest(source, tmp_path, options=stored)
    counts_path = f"{group}/counts".lstrip("/")
    layout = _layout(store)
    before = store_files(store)
    assert f"{counts_path}/c/0/0/0/0" in before
    counts_before = np.asarray(zarr.open_group(str(store))[counts_path][:])

    with pytest.raises(Exception) as excinfo:
        _run_ingest(source, tmp_path, options=requested)

    _assert_resolved_index_conflict(excinfo, stored_name, incoming_name)
    assert _layout(store) == layout
    assert store_files(store) == before
    np.testing.assert_array_equal(
        zarr.open_group(str(store))[counts_path][:], counts_before
    )

    # The same run with the store's own layout goes through.
    _run_ingest(source, tmp_path, options=stored)
    written: Any = zarr.open_group(str(store), mode="r")
    if group:
        written = written[group]
    assert _times(written["time"]) == FIRST_SLOT_TIMES
    assert _layout(store) == layout


@pytest.mark.parametrize(
    (
        "stored",
        "requested",
        "extra",
        "append",
        "counts_shape",
        "stored_name",
        "incoming_name",
    ),
    [
        pytest.param(
            FLAT_1KM,
            FLAT_2KM,
            {},
            False,
            (1, 4, 4, 2),
            "eumetsat_repeat_cycle_v1_fdhsi_1km",
            "eumetsat_repeat_cycle_v1_fdhsi_2km",
            id="1km-to-2km",
        ),
        # The 1 km grid has two channels, the 2 km grid one: this direction would
        # grow the channel axis of the stored arrays if the run went through.
        pytest.param(
            FLAT_2KM,
            FLAT_1KM,
            RESUME,
            True,
            (1, 4, 4, 1),
            "eumetsat_repeat_cycle_v1_fdhsi_2km",
            "eumetsat_repeat_cycle_v1_fdhsi_1km",
            id="2km-to-1km-resume-append",
        ),
    ],
)
def test_flat_run_with_another_resolution_is_rejected_and_arrays_keep_their_shape(
    tmp_path: Path,
    small_fci_layout: list[int],
    stored: dict[str, object],
    requested: dict[str, object],
    extra: dict[str, object],
    append: bool,
    counts_shape: tuple[int, ...],
    stored_name: str,
    incoming_name: str,
) -> None:
    first_source = _make_fdhsi_zip_at(tmp_path / "src_first", FIRST_SLOT).parent
    store = _run_ingest(first_source, tmp_path, options=stored)
    root: Any = zarr.open_group(str(store), mode="r")
    assert root["counts"].shape == counts_shape
    before = store_files(store)
    assert "counts/c/0/0/0/0" in before
    second_source = first_source
    if append:
        second_source = _make_fdhsi_zip_at(tmp_path / "src_second", SECOND_SLOT).parent

    with pytest.raises(Exception) as excinfo:
        _run_ingest(second_source, tmp_path, options={**requested, **extra})

    _assert_resolved_index_conflict(excinfo, stored_name, incoming_name)
    root = zarr.open_group(str(store), mode="r")
    assert root["counts"].shape == counts_shape
    assert store_files(store) == before

    # The same run with the store's own resolution reaches the write path.
    _run_ingest(second_source, tmp_path, options={**stored, **extra})
    root = zarr.open_group(str(store), mode="r")
    expected_time_len = 2 if append else 1
    assert root["counts"].shape == (expected_time_len, *counts_shape[1:])
    assert len(_times(root["time"])) == expected_time_len
