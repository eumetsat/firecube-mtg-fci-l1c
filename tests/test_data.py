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

from datetime import datetime
from pathlib import Path

import pytest

from firecube_mtg_fci_l1c._constants import PRODUCT_TYPE_FDHSI, PRODUCT_TYPE_HRFI
from firecube_mtg_fci_l1c._data import (
    detect_product_type,
    extract_timestamp_from_path,
    is_valid_fci_zip,
    validate_no_mixed_products,
)


@pytest.mark.unit
def test_detect_product_type_fdhsi():
    assert (
        detect_product_type("W_XX-FCI-1C-RRAD-FDHSI-FD-20241001005154.zip")
        == PRODUCT_TYPE_FDHSI
    )


@pytest.mark.unit
def test_detect_product_type_hrfi():
    assert (
        detect_product_type("W_XX-FCI-1C-RRAD-HRFI-FD-20241001005154.zip")
        == PRODUCT_TYPE_HRFI
    )


@pytest.mark.unit
def test_detect_product_type_invalid():
    with pytest.raises(ValueError):
        detect_product_type("W_XX-FCI-1C-RRAD-UNKNOWN-20241001005154.zip")


@pytest.mark.unit
def test_validate_no_mixed_products_single():
    assert (
        validate_no_mixed_products([Path("A-FCI-1C-RRAD-FDHSI-20241001005154.zip")])
        == PRODUCT_TYPE_FDHSI
    )


@pytest.mark.unit
def test_validate_no_mixed_products_mixed_rejected():
    with pytest.raises(ValueError, match=r"[Mm]ixed"):
        validate_no_mixed_products(
            [
                Path("A-FCI-1C-RRAD-FDHSI-20241001005154.zip"),
                Path("A-FCI-1C-RRAD-HRFI-20241001015154.zip"),
            ]
        )


@pytest.mark.unit
def test_valid_fci_filename_fdhsi():
    path = Path("W_XX-EUMETSAT-FCI-1C-RRAD-FDHSI-FD-20241001005154-END.zip")
    assert is_valid_fci_zip(path) is True


@pytest.mark.unit
def test_valid_fci_filename_hrfi():
    path = Path("W_XX-EUMETSAT-FCI-1C-RRAD-HRFI-FD-20241001005154-END.zip")
    assert is_valid_fci_zip(path) is True


@pytest.mark.unit
def test_missing_timestamp_is_invalid():
    assert is_valid_fci_zip(Path("FCI-1C-RRAD-HRFI-notimestamp.zip")) is False


@pytest.mark.unit
def test_extract_timestamp_real_filename():
    path = Path(
        "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
        "x-x---x_C_EUMT_20241001120234_IDPFI_OPE_20241001120007_"
        "20241001120924_N__C_0073_0000.zip"
    )
    ts = extract_timestamp_from_path(path)
    assert ts == datetime(2024, 10, 1, 12, 0, 7)


@pytest.mark.unit
def test_extract_timestamp_hrfi_filename():
    path = Path(
        "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-HRFI-FD--"
        "x-x---x_C_EUMT_20241001120234_IDPFI_OPE_20241001120007_"
        "20241001120924_N__C_0073_0000.zip"
    )
    ts = extract_timestamp_from_path(path)
    assert ts == datetime(2024, 10, 1, 12, 0, 7)
