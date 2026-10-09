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

import pytest
from firecube_mtg_fci_l1c._constants import (
    BODY_CHUNK_COUNT,
    BODY_CHUNK_ROWS,
    CHUNK_DEFAULTS_BY_RESOLUTION,
    CONSTANTS,
    FCI_COLLECTION_IDS,
    FCI_PROJ_SCALE_RAD_PER_INDEX,
    FCI_PROJ_SWEEP_AXIS,
    PRODUCT_TYPE_FDHSI,
    PRODUCT_TYPE_HRFI,
    VALID_RESOLUTIONS,
    get_nc_part_prefix,
    logical_channel_resolution_map,
    nc_channel_resolution_map,
    stripe_rows,
)


@pytest.mark.unit
def test_fdhsi_constants_present():
    assert len(CONSTANTS[PRODUCT_TYPE_FDHSI]["1km"]["channels"]) == 8
    assert len(CONSTANTS[PRODUCT_TYPE_FDHSI]["2km"]["channels"]) == 8


@pytest.mark.unit
def test_hrfi_constants_present():
    assert CONSTANTS[PRODUCT_TYPE_HRFI]["500m"]["channels"] == ["vis_06", "nir_22"]
    assert CONSTANTS[PRODUCT_TYPE_HRFI]["500m"]["nc_channels"] == [
        "vis_06_hr",
        "nir_22_hr",
    ]
    assert CONSTANTS[PRODUCT_TYPE_HRFI]["1km"]["channels"] == ["ir_38", "ir_105"]


@pytest.mark.unit
def test_get_nc_part_prefix_per_product():
    fd = get_nc_part_prefix(PRODUCT_TYPE_FDHSI)
    hr = get_nc_part_prefix(PRODUCT_TYPE_HRFI)
    assert "FDHSI" in fd
    assert "HRFI" in hr


@pytest.mark.unit
def test_valid_resolutions_map():
    assert VALID_RESOLUTIONS[PRODUCT_TYPE_FDHSI] == ["1km", "2km"]
    assert VALID_RESOLUTIONS[PRODUCT_TYPE_HRFI] == ["500m", "1km"]


@pytest.mark.unit
def test_collection_ids_present():
    assert FCI_COLLECTION_IDS[PRODUCT_TYPE_FDHSI] == "EO:EUM:DAT:0662"
    assert FCI_COLLECTION_IDS[PRODUCT_TYPE_HRFI] == "EO:EUM:DAT:0665"


@pytest.mark.unit
def test_chunk_defaults_match_legacy():
    """CHUNK_DEFAULTS_BY_RESOLUTION values must match legacy hardcoded values."""
    assert CHUNK_DEFAULTS_BY_RESOLUTION["500m"] == 556
    assert CHUNK_DEFAULTS_BY_RESOLUTION["1km"] == 278
    assert CHUNK_DEFAULTS_BY_RESOLUTION["2km"] == 139


@pytest.mark.unit
def test_projection_sampling_constants_match_source_netcdf():
    assert FCI_PROJ_SCALE_RAD_PER_INDEX["1km"] == 2.79435763233999e-05
    assert FCI_PROJ_SCALE_RAD_PER_INDEX["500m"] == 1.39717881617e-05
    assert FCI_PROJ_SCALE_RAD_PER_INDEX["2km"] == 5.58871526468e-05
    assert FCI_PROJ_SWEEP_AXIS == "y"
    assert (
        round(
            FCI_PROJ_SCALE_RAD_PER_INDEX["2km"] / FCI_PROJ_SCALE_RAD_PER_INDEX["1km"],
            6,
        )
        == 2.0
    )


@pytest.mark.unit
def test_channel_resolution_map_fdhsi():
    m = logical_channel_resolution_map(PRODUCT_TYPE_FDHSI)
    assert m["vis_06"] == "1km"
    assert m["ir_105"] == "2km"


@pytest.mark.unit
def test_channel_resolution_map_hrfi():
    m = logical_channel_resolution_map(PRODUCT_TYPE_HRFI)
    assert m["vis_06"] == "500m"
    assert m["ir_105"] == "1km"


@pytest.mark.unit
def test_nc_channel_resolution_map_hrfi():
    m = nc_channel_resolution_map(PRODUCT_TYPE_HRFI)
    assert m["vis_06_hr"] == "500m"
    assert m["ir_105_hr"] == "1km"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("product_type", "resolution", "height"),
    [
        ("FDHSI", "1km", 11136),
        ("FDHSI", "2km", 5568),
        ("HRFI", "500m", 22272),
        ("HRFI", "1km", 11136),
    ],
)
def test_body_chunk_rows_tile_full_disk_without_gap_or_overlap(
    product_type: str, resolution: str, height: int
) -> None:
    rows = BODY_CHUNK_ROWS[product_type][resolution]
    assert len(rows) == BODY_CHUNK_COUNT[product_type] == 40
    assert rows[0][0] == 0
    assert rows[-1][1] == height
    for (_, prev_end), (start, end) in zip(rows, rows[1:], strict=False):
        assert start == prev_end
        assert end > start


@pytest.mark.unit
def test_body_chunk_rows_cover_exactly_the_declared_resolutions() -> None:
    for product_type, resolutions in VALID_RESOLUTIONS.items():
        assert sorted(BODY_CHUNK_ROWS[product_type]) == sorted(resolutions)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("product_type", "resolution", "first", "last", "expected"),
    [
        ("FDHSI", "1km", 32, 40, (8649, 11136)),
        ("FDHSI", "1km", 1, 1, (0, 278)),
        ("FDHSI", "2km", 1, 40, (0, 5568)),
        ("HRFI", "500m", 2, 2, (557, 1113)),
        ("HRFI", "1km", 2, 2, (278, 556)),
    ],
)
def test_stripe_rows_returns_span_of_first_to_last_chunk(
    product_type: str,
    resolution: str,
    first: int,
    last: int,
    expected: tuple[int, int],
) -> None:
    assert stripe_rows(product_type, resolution, first, last) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("product_type", "resolution", "first", "last"),
    [
        ("FDHSI", "1km", 40, 32),  # reversed
        ("FDHSI", "1km", 0, 5),  # below range
        ("FDHSI", "1km", 1, 41),  # just past the last chunk
        ("NOPE", "1km", 1, 2),  # unknown product
        ("FDHSI", "500m", 1, 2),  # resolution not in this product
        ("HRFI", "2km", 1, 2),
    ],
)
def test_stripe_rows_rejects_bad_input(
    product_type: str, resolution: str, first: int, last: int
) -> None:
    with pytest.raises(ValueError):
        stripe_rows(product_type, resolution, first, last)
