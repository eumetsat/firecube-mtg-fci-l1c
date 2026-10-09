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

"""Shared pytest fixtures for the MTG FCI L1C plugin tests."""

import copy
import sys
from pathlib import Path

import pytest
from firecube_mtg_fci_l1c._constants import PRODUCT_TYPE_FDHSI, PRODUCT_TYPE_HRFI

sys.path.insert(0, str(Path(__file__).parent))

from tests._small_grid import SMALL_CONSTANTS, fake_compute_latlon  # noqa: E402
from tests._support import _make_zip_with_nc_part  # noqa: E402


@pytest.fixture
def small_fci_layout(monkeypatch):
    from firecube_mtg_fci_l1c import _constants as const_mod
    from firecube_mtg_fci_l1c.geolocation import provider as geolocation_mod

    constants_backup = copy.deepcopy(const_mod.CONSTANTS)
    const_mod.CONSTANTS[PRODUCT_TYPE_FDHSI] = copy.deepcopy(
        SMALL_CONSTANTS[PRODUCT_TYPE_FDHSI]
    )
    const_mod.CONSTANTS[PRODUCT_TYPE_HRFI] = copy.deepcopy(
        SMALL_CONSTANTS[PRODUCT_TYPE_HRFI]
    )

    compute_calls: list[int] = []

    def _recording_compute_latlon(resolution_m: int):
        compute_calls.append(resolution_m)
        return fake_compute_latlon(resolution_m)

    monkeypatch.setattr(geolocation_mod, "compute_latlon", _recording_compute_latlon)
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
