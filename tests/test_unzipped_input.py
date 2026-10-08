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

"""A scene given as loose BODY chunk files ingests exactly like its ZIP.

The synthetic ZIP fixtures are unpacked into a directory of chunk files with
real-shaped names (2024-01-01, repeat cycle 0001, so the nominal time is
00:00:00 like the ZIP's label). Both forms are ingested with the same options
into separate stores; the stores must be bitwise equal. The comparators used
for that are tested here too, because every parity claim rests on them.
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import zarr
from firecube.ingestor.api import ConfigurationError

sys.path.insert(0, str(Path(__file__).parent))
from _store_compare import assert_rows_equal, assert_stores_bitwise_equal  # noqa: E402
from _store_files import store_files  # noqa: E402
from test_golden_output import (  # noqa: E402
    GOLDEN_FILE,
    _capture_zarr_structure,
    _hash,
)
from test_integration import _run_ingest  # noqa: E402

_SENSING_START = "20240101000000"
_SENSING_END = "20240101000934"
_DISSEMINATED = "20240101000318"


def _chunk_file_name(product_type: str, number: int) -> str:
    """Real-shaped BODY chunk name of repeat cycle 0001 of 2024-01-01."""
    return (
        "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-"
        f"{product_type}-FD--CHK-BODY---NC4E_C_EUMT_{_DISSEMINATED}_IDPFI_OPE_"
        f"{_SENSING_START}_{_SENSING_END}_N__O_0001_{number:04d}.nc"
    )


def _unpack_zip_to_chunks(zip_path: Path, dest: Path, product_type: str) -> Path:
    """Unpack the ZIP's parts as loose chunk files, numbered by their part suffix."""
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            number = int(Path(member).stem[-4:])
            (dest / _chunk_file_name(product_type, number)).write_bytes(
                archive.read(member)
            )
    return dest


@pytest.fixture
def fdhsi_chunks(tmp_path: Path, fdhsi_zip: Path) -> Path:
    return _unpack_zip_to_chunks(fdhsi_zip, tmp_path / "fdhsi_chunks", "FDHSI")


@pytest.fixture
def hrfi_chunks(tmp_path: Path, hrfi_zip: Path) -> Path:
    return _unpack_zip_to_chunks(hrfi_zip, tmp_path / "hrfi_chunks", "HRFI")


# --- The comparators -------------------------------------------------------


def _first_data_file(store: Path) -> str:
    return next(rel for rel in store_files(store) if not rel.endswith("zarr.json"))


@pytest.mark.integration
@pytest.mark.plugin
def test_store_comparator_accepts_two_ingests_of_the_same_zip(
    tmp_path: Path, fdhsi_zip: Path
):
    first = _run_ingest(fdhsi_zip.parent, tmp_path / "a", product_type="FDHSI")
    second = _run_ingest(fdhsi_zip.parent, tmp_path / "b", product_type="FDHSI")

    assert store_files(first)  # a store with content, not two empty ones
    assert_stores_bitwise_equal(first, second)


@pytest.mark.integration
@pytest.mark.plugin
def test_store_comparator_names_the_path_of_a_one_byte_difference(
    tmp_path: Path, fdhsi_zip: Path
):
    original = _run_ingest(fdhsi_zip.parent, tmp_path / "a", product_type="FDHSI")
    copy = tmp_path / "copy.zarr"
    shutil.copytree(original, copy)
    target = _first_data_file(copy)
    path = copy / target
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0x01
    path.write_bytes(bytes(raw))

    with pytest.raises(AssertionError, match=target.replace("+", r"\+")):
        assert_stores_bitwise_equal(original, copy)


@pytest.mark.integration
@pytest.mark.plugin
def test_store_comparator_names_the_json_key_that_differs_and_ignores_layout(
    tmp_path: Path, fdhsi_zip: Path
):
    original = _run_ingest(fdhsi_zip.parent, tmp_path / "a", product_type="FDHSI")
    copy = tmp_path / "copy.zarr"
    shutil.copytree(original, copy)
    meta = copy / "data_1km" / "counts" / "zarr.json"
    document = json.loads(meta.read_text())

    # Same document, other formatting: not a difference.
    meta.write_text(json.dumps(document, indent=7, sort_keys=True))
    assert_stores_bitwise_equal(original, copy)

    document["attributes"]["long_name"] = "changed"
    meta.write_text(json.dumps(document))
    with pytest.raises(AssertionError, match=r"data_1km/counts/zarr.json.*long_name"):
        assert_stores_bitwise_equal(original, copy)


@pytest.mark.integration
@pytest.mark.plugin
def test_store_comparator_reports_a_file_present_in_only_one_store(
    tmp_path: Path, fdhsi_zip: Path
):
    original = _run_ingest(fdhsi_zip.parent, tmp_path / "a", product_type="FDHSI")
    copy = tmp_path / "copy.zarr"
    shutil.copytree(original, copy)
    (copy / "data_1km" / "stray").write_bytes(b"x")

    with pytest.raises(AssertionError, match=r"data_1km/stray: only in the second"):
        assert_stores_bitwise_equal(original, copy)


def _make_arrays(
    dims: tuple[str, ...], full_shape: tuple[int, ...], y0: int, ny: int
) -> tuple[Any, Any, np.ndarray]:
    """A full array and the stripe array holding its rows ``[y0, y0 + ny)``."""
    rng = np.random.default_rng(7)
    data = rng.random(full_shape).astype(np.float32)
    data.flat[1] = np.nan  # NaN must compare equal to NaN
    y_axis = dims.index("y")
    chunks = tuple(
        2 if axis == y_axis else (1 if axis == 0 else size)
        for axis, size in enumerate(full_shape)
    )
    full = zarr.create_array(
        zarr.storage.MemoryStore(),
        shape=full_shape,
        chunks=chunks,
        dtype="float32",
        dimension_names=dims,
        fill_value=float("nan"),
        attributes={"units": "K"},
    )
    full[...] = data
    index = tuple(
        slice(y0, y0 + ny) if a == y_axis else slice(None) for a in range(len(dims))
    )
    stripe = zarr.create_array(
        zarr.storage.MemoryStore(),
        shape=data[index].shape,
        chunks=tuple(min(c, s) for c, s in zip(chunks, data[index].shape)),
        dtype="float32",
        dimension_names=dims,
        fill_value=float("nan"),
        attributes={"units": "K"},
    )
    stripe[...] = data[index]
    return full, stripe, data


@pytest.mark.parametrize(
    ("dims", "shape"),
    [
        (("time", "y", "x", "channel"), (3, 8, 3, 2)),
        (("time", "y", "x"), (3, 8, 3)),
        (("y", "x"), (8, 3)),
        (("y",), (8,)),
    ],
    ids=["t-y-x-c", "t-y-x", "y-x", "y"],
)
def test_rows_comparator_finds_the_stripe_by_dimension_name(
    dims: tuple[str, ...], shape: tuple[int, ...]
):
    full, stripe, _ = _make_arrays(dims, shape, y0=2, ny=4)

    assert_rows_equal(full, stripe, 2)

    # The same stripe is not the rows from another offset.
    with pytest.raises(AssertionError, match="values differ"):
        assert_rows_equal(full, stripe, 4)


@pytest.mark.parametrize("dims", [("time", "y", "x", "channel"), ("y",)])
def test_rows_comparator_detects_one_changed_value_in_the_last_slab(
    dims: tuple[str, ...],
):
    shape = (3, 8, 3, 2) if len(dims) == 4 else (8,)
    full, stripe, data = _make_arrays(dims, shape, y0=2, ny=4)
    last = tuple(s - 1 for s in stripe.shape)
    stripe[last] = np.float32(data[tuple(s - 1 for s in stripe.shape)] + 1)

    with pytest.raises(AssertionError, match="values differ in stripe slab"):
        assert_rows_equal(full, stripe, 2)


def test_rows_comparator_rejects_other_dtype_fill_dims_and_attrs():
    full, stripe, _ = _make_arrays(("time", "y", "x"), (2, 8, 3), y0=0, ny=4)

    other_dtype = zarr.create_array(
        zarr.storage.MemoryStore(),
        shape=stripe.shape,
        dtype="float64",
        dimension_names=("time", "y", "x"),
        fill_value=float("nan"),
        attributes={"units": "K"},
    )
    other_dtype[...] = stripe[...]
    with pytest.raises(AssertionError, match=r"dtype float32 vs float64"):
        assert_rows_equal(full, other_dtype, 0)

    other_fill = zarr.create_array(
        zarr.storage.MemoryStore(),
        shape=stripe.shape,
        dtype="float32",
        dimension_names=("time", "y", "x"),
        fill_value=0.0,
        attributes={"units": "K"},
    )
    other_fill[...] = stripe[...]
    with pytest.raises(AssertionError, match="fill_value"):
        assert_rows_equal(full, other_fill, 0)

    other_dims = zarr.create_array(
        zarr.storage.MemoryStore(),
        shape=stripe.shape,
        dtype="float32",
        dimension_names=("time", "y", "z"),
        fill_value=float("nan"),
        attributes={"units": "K"},
    )
    with pytest.raises(AssertionError, match="dimension_names"):
        assert_rows_equal(full, other_dims, 0)

    stripe.attrs["units"] = "m"
    with pytest.raises(AssertionError, match="attrs differ"):
        assert_rows_equal(full, stripe, 0)


def test_rows_comparator_compares_arrays_without_y_whole():
    def table(values: list[float]) -> Any:
        arr = zarr.create_array(
            zarr.storage.MemoryStore(),
            shape=(len(values),),
            dtype="float32",
            dimension_names=("time",),
        )
        arr[...] = np.asarray(values, np.float32)
        return arr

    assert_rows_equal(table([1.0, 2.0]), table([1.0, 2.0]), 5)
    with pytest.raises(AssertionError, match="values differ"):
        assert_rows_equal(table([1.0, 2.0]), table([1.0, 3.0]), 5)


# --- ZIP and unpacked directory give the same store -------------------------

_GOLDEN_KEYS = {"FDHSI": "fdhsi_defaults", "HRFI": "hrfi_defaults"}


def _read_group(store: Path, group: str) -> Any:
    root = cast(Any, zarr.open_group(str(store), mode="r"))
    return root[group] if group else root


@pytest.mark.integration
@pytest.mark.plugin
@pytest.mark.parametrize("product_type", ["FDHSI", "HRFI"])
def test_unpacked_directory_gives_the_same_store_as_its_zip(
    request: pytest.FixtureRequest, tmp_path: Path, product_type: str
):
    zip_path: Path = request.getfixturevalue(f"{product_type.lower()}_zip")
    chunks: Path = request.getfixturevalue(f"{product_type.lower()}_chunks")

    from_zip = _run_ingest(zip_path.parent, tmp_path / "zip", product_type=product_type)
    unpacked = _run_ingest(chunks, tmp_path / "chunks", product_type=product_type)

    # The scene was read: slot 0 is 2024-01-01T00:00 and holds the fixture counts.
    group = _read_group(unpacked, "data_1km")
    assert group["time"][:].tolist() == [np.datetime64("2024-01-01T00:00:00")]
    assert np.all(np.asarray(group["counts"][0, :, :, 0]) > 0)

    assert_stores_bitwise_equal(from_zip, unpacked)

    golden = json.loads(GOLDEN_FILE.read_text())[_GOLDEN_KEYS[product_type]]
    structure = _capture_zarr_structure(unpacked)
    assert structure == golden["structure"]
    assert _hash(structure) == golden["hash"]


@pytest.mark.integration
@pytest.mark.plugin
def test_unpacked_directory_gives_the_same_flat_store_as_its_zip(
    tmp_path: Path, fdhsi_zip: Path, fdhsi_chunks: Path
):
    options: dict[str, object] = {"resolutions": "1km", "flat_store": True}

    from_zip = _run_ingest(
        fdhsi_zip.parent, tmp_path / "zip", options, product_type="FDHSI"
    )
    unpacked = _run_ingest(
        fdhsi_chunks, tmp_path / "chunks", options, product_type="FDHSI"
    )

    root = _read_group(unpacked, "")
    assert sorted(root.group_keys()) == []
    assert root["time"][:].tolist() == [np.datetime64("2024-01-01T00:00:00")]
    assert np.all(np.asarray(root["counts"][0, :, :, 0]) > 0)
    assert_stores_bitwise_equal(from_zip, unpacked)


@pytest.mark.integration
@pytest.mark.plugin
def test_quicklook_and_xml_next_to_the_chunks_are_ignored(
    tmp_path: Path, fdhsi_zip: Path, fdhsi_chunks: Path
):
    clean = _run_ingest(fdhsi_chunks, tmp_path / "clean", product_type="FDHSI")

    chunk_name = _chunk_file_name("FDHSI", 1)
    (fdhsi_chunks / "MANIFEST.xml").write_text("<manifest/>")
    (fdhsi_chunks / (chunk_name[:-3] + ".xml")).write_text("<metadata/>")
    (fdhsi_chunks / (chunk_name[:-3] + "-quicklook.jpg")).write_bytes(b"\xff\xd8")
    (fdhsi_chunks / "quicklook.png").write_bytes(b"\x89PNG")
    noisy = _run_ingest(fdhsi_chunks, tmp_path / "noisy", product_type="FDHSI")

    assert_stores_bitwise_equal(clean, noisy)
    from_zip = _run_ingest(fdhsi_zip.parent, tmp_path / "zip", product_type="FDHSI")
    assert_stores_bitwise_equal(from_zip, noisy)


@pytest.mark.integration
@pytest.mark.plugin
def test_directory_with_a_zip_and_chunks_fails_before_the_store_is_written(
    tmp_path: Path, fdhsi_zip: Path, fdhsi_chunks: Path
):
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    shutil.copy(fdhsi_zip, mixed / fdhsi_zip.name)
    for chunk in fdhsi_chunks.iterdir():
        shutil.copy(chunk, mixed / chunk.name)
    workspace = tmp_path / "mixed_out"
    workspace.mkdir()

    with pytest.raises(ConfigurationError, match=r"Mixed ZIP and unpacked chunk input"):
        _run_ingest(mixed, workspace, product_type="FDHSI")

    target = workspace / "out.zarr"
    assert not target.exists() or not store_files(target, skip_control_plane=False)
