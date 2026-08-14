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

"""Tests for GitHub issue #3.

The time variable currently declares native ``datetime64[s]`` dtype AND writes
``units``/``calendar`` to attrs, causing ``xr.open_zarr(store).to_zarr(new_store)``
to raise ``"Key 'units' already exists in attrs"``. The fix is to drop
``units``/``calendar`` from the time variable's attrs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.mark.unit
def test_time_variable_attrs_declared_without_units_or_calendar() -> None:
    """The time Variable in the schema registry must not carry units/calendar.

    The zarr array is stored with native ``datetime64[s]`` dtype, so declaring
    ``units``/``calendar`` in attrs is both redundant and breaks the
    ``open_zarr -> to_zarr`` roundtrip (xarray tries to re-write encoding-derived
    ``units`` into attrs that already contain them).
    """
    from firecube_mtg_fci_l1c.schema import VARIABLES

    time_var = next(v for v in VARIABLES if v.name == "time")
    attrs = time_var.attrs or {}
    assert "units" not in attrs, (
        "time variable must not declare 'units' in attrs; native datetime64 "
        "storage does not need it, and it breaks the open->write roundtrip."
    )
    assert "calendar" not in attrs, (
        "time variable must not declare 'calendar' in attrs; native datetime64 "
        "storage does not need it, and it breaks the open->write roundtrip."
    )
    assert attrs.get("standard_name") == "time"
    assert attrs.get("axis") == "T"


@pytest.mark.integration
@pytest.mark.plugin
def test_time_attrs_do_not_contain_units_or_calendar_after_ingest(
    tmp_path: Path, fdhsi_zip: Path
) -> None:
    """After an end-to-end ingest, the on-disk time coordinate must not carry
    ``units``/``calendar`` in its attrs.
    """
    import xarray as xr

    from test_integration import _run_ingest

    store_path = _run_ingest(fdhsi_zip.parent, tmp_path, options={})
    ds = xr.open_zarr(str(store_path), group="data_1km", consolidated=False)
    try:
        assert "units" not in ds.time.attrs, (
            f"time.attrs still contains 'units': {dict(ds.time.attrs)!r}"
        )
        assert "calendar" not in ds.time.attrs, (
            f"time.attrs still contains 'calendar': {dict(ds.time.attrs)!r}"
        )
        assert str(ds.time.dtype).startswith("datetime64"), (
            f"time dtype should be datetime64-like, got {ds.time.dtype!r}"
        )
    finally:
        ds.close()


@pytest.mark.integration
@pytest.mark.plugin
def test_open_then_write_roundtrip_does_not_raise(
    tmp_path: Path, fdhsi_zip: Path
) -> None:
    """Reading a store with xarray and writing it back to a new zarr must not
    raise ``"Key 'units' already exists in attrs"``.

    This is the concrete failure mode reported in GitHub issue #3.
    """
    import xarray as xr

    from test_integration import _run_ingest

    store_path = _run_ingest(fdhsi_zip.parent, tmp_path, options={})
    ds = xr.open_zarr(str(store_path), group="data_1km", consolidated=False)
    try:
        roundtrip_path = tmp_path / "roundtrip.zarr"
        try:
            ds.to_zarr(str(roundtrip_path), mode="w")
        except ValueError as exc:
            if "already exists in attrs" in str(exc):
                pytest.fail(
                    "open_zarr -> to_zarr roundtrip failed with the issue #3 "
                    f"symptom: {exc!r}"
                )
            raise
    finally:
        ds.close()

    ds2 = xr.open_zarr(str(roundtrip_path), consolidated=False)
    try:
        assert str(ds2.time.dtype).startswith("datetime64"), (
            f"roundtripped time dtype should be datetime64-like, "
            f"got {ds2.time.dtype!r}"
        )
    finally:
        ds2.close()
