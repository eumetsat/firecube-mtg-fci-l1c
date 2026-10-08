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
    classify_items,
    detect_product_type,
    extract_slot_time_from_path,
    extract_timestamp_from_path,
    group_chunks_into_bundles,
    is_valid_fci_chunk,
    is_valid_fci_zip,
    nominal_cycle_start,
    parse_chunk_name,
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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("start", "expected"),
    [
        ("20241001120007", datetime(2024, 10, 1, 12, 0, 0)),  # typical start
        ("20241001120000", datetime(2024, 10, 1, 12, 0, 0)),  # already round
        ("20241001120959", datetime(2024, 10, 1, 12, 9, 0)),  # last second
        ("20241231235959", datetime(2024, 12, 31, 23, 59, 0)),  # no carry-over
    ],
)
def test_slot_time_is_observation_start_floored_to_the_minute(
    start: str, expected: datetime
) -> None:
    path = Path(
        "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
        f"x-x---x_C_EUMT_20241001120234_IDPFI_OPE_{start}_"
        "20241001120924_N__C_0073_0000.zip"
    )

    assert extract_slot_time_from_path(path) == expected


@pytest.mark.unit
def test_slot_time_is_none_without_timestamp_in_filename() -> None:
    assert extract_slot_time_from_path(Path("no-timestamp-here.zip")) is None


# --- Unpacked chunk input -------------------------------------------------


def _chunk(
    kind: str = "BODY",
    *,
    product: str = "FDHSI",
    start: str = "20250701000352",
    cycle: int = 1,
    number: int = 18,
) -> str:
    pad = "---" if kind == "BODY" else "--"
    return (
        f"W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-{product}-FD--"
        f"CHK-{kind}{pad}NC4E_C_EUMT_20250701000756_IDPFI_OPE_{start}_"
        f"20250701000421_N__O_{cycle:04d}_{number:04d}.nc"
    )


@pytest.mark.unit
def test_parse_chunk_name_reads_real_body_name() -> None:
    chunk = parse_chunk_name(Path(_chunk()))

    assert chunk.product_type == PRODUCT_TYPE_FDHSI
    assert chunk.sensing_start == datetime(2025, 7, 1, 0, 3, 52)
    assert chunk.repeat_cycle == 1
    assert chunk.chunk_number == 18
    assert chunk.is_trail is False


@pytest.mark.unit
def test_parse_chunk_name_marks_trail_and_hrfi() -> None:
    chunk = parse_chunk_name(_chunk("TRAIL", product="HRFI", number=41))

    assert chunk.product_type == PRODUCT_TYPE_HRFI
    assert chunk.chunk_number == 41
    assert chunk.is_trail is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        "preview.jpg",
        "quicklook.png",
        "MTI1-FCI-1C-RRAD-FDHSI-FD--CHK-BODY---NC4E_manifest.xml",
        "W_XX-FCI-1C-RRAD-FDHSI-FD-20241001005154.zip",
        _chunk().replace("CHK-BODY", "CHK-HEAD"),
        _chunk().replace("FCI-1C-RRAD", "FCI-1C-XXXX"),
    ],
)
def test_non_chunks_are_invalid_and_do_not_parse(name: str) -> None:
    assert is_valid_fci_chunk(name) is False
    with pytest.raises(ValueError):
        parse_chunk_name(name)


@pytest.mark.unit
def test_chunk_with_unparseable_tail_is_valid_by_filter_but_fails_to_parse() -> None:
    name = "FCI-1C-RRAD-FDHSI-FD--CHK-BODY---NC4E_garbage.nc"

    assert is_valid_fci_chunk(name) is True
    with pytest.raises(ValueError, match="parse"):
        parse_chunk_name(name)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("sensing", "cycle", "expected"),
    [
        (datetime(2025, 7, 1, 0, 0, 2), 1, datetime(2025, 7, 1, 0, 0, 0)),
        (datetime(2025, 7, 1, 12, 0, 7), 73, datetime(2025, 7, 1, 12, 0, 0)),
        (datetime(2025, 7, 1, 12, 9, 40), 73, datetime(2025, 7, 1, 12, 0, 0)),
        (datetime(2025, 7, 1, 23, 50, 3), 144, datetime(2025, 7, 1, 23, 50, 0)),
        # last cycle of the day, chunk sensed after midnight -> previous day
        (datetime(2025, 7, 2, 0, 3, 52), 144, datetime(2025, 7, 1, 23, 50, 0)),
    ],
)
def test_nominal_cycle_start(sensing: datetime, cycle: int, expected: datetime) -> None:
    assert nominal_cycle_start(sensing, cycle) == expected


@pytest.mark.unit
@pytest.mark.parametrize("cycle", [0, 145, -1])
def test_nominal_cycle_start_rejects_invalid_cycle(cycle: int) -> None:
    with pytest.raises(ValueError, match="1..144"):
        nominal_cycle_start(datetime(2025, 7, 1, 12, 0, 0), cycle)


@pytest.mark.unit
def test_bundle_has_deterministic_uri_and_trail_sorts_last() -> None:
    names = [
        _chunk("TRAIL", number=41, start="20250701000318"),
        _chunk(number=18),
        _chunk(number=2, start="20250701000010"),
    ]

    (bundle,) = group_chunks_into_bundles(names)

    assert bundle.uri == "fci-scene://FDHSI/20250701000000"
    assert str(bundle) == bundle.uri
    assert [m[-12:] for m in bundle.members] == [
        "0001_0002.nc",
        "0001_0018.nc",
        "0001_0041.nc",
    ]
    assert "CHK-TRAIL" in bundle.members[-1]


@pytest.mark.unit
def test_trail_sorts_after_body_even_with_lower_chunk_number() -> None:
    names = [_chunk("TRAIL", number=3), _chunk(number=9)]

    (bundle,) = group_chunks_into_bundles(names)

    assert "CHK-TRAIL" in bundle.members[-1]


@pytest.mark.unit
def test_partial_bundle_shares_coordinate_and_uri_with_full_bundle() -> None:
    partial = group_chunks_into_bundles(
        [_chunk(number=n, start="20250701000352") for n in (32, 40)]
    )
    full = group_chunks_into_bundles(
        [_chunk(number=n, start="20250701000002") for n in range(1, 41)]
    )

    assert len(partial) == len(full) == 1
    assert len(partial[0].members) == 2
    assert len(full[0].members) == 40
    assert partial[0].coordinate == full[0].coordinate == datetime(2025, 7, 1)
    assert partial[0].uri == full[0].uri


@pytest.mark.unit
def test_duplicate_chunk_number_names_the_chunk() -> None:
    first = _chunk(number=18)
    second = first.replace("20250701000756", "20250701000800")

    with pytest.raises(ValueError, match="0018"):
        group_chunks_into_bundles([first, second])


@pytest.mark.unit
def test_bundles_split_by_cycle_and_product_in_deterministic_order() -> None:
    names = [
        _chunk(product="HRFI", cycle=2, start="20250701001002"),
        _chunk(cycle=2, start="20250701001002"),
        _chunk(cycle=1, start="20250701000002"),
    ]

    forward = [b.uri for b in group_chunks_into_bundles(names)]
    backward = [b.uri for b in group_chunks_into_bundles(reversed(names))]

    assert (
        forward
        == backward
        == [
            "fci-scene://FDHSI/20250701000000",
            "fci-scene://FDHSI/20250701001000",
            "fci-scene://HRFI/20250701001000",
        ]
    )


@pytest.mark.unit
def test_bundling_rejects_invalid_cycle() -> None:
    with pytest.raises(ValueError, match="1..144"):
        group_chunks_into_bundles([_chunk(cycle=0)])


@pytest.mark.unit
def test_classify_items_ignores_quicklooks_and_xml() -> None:
    chunk = _chunk()
    items = [chunk, "quicklook.jpg", Path("quicklook.png"), "manifest.xml"]

    zips, chunks, ignored = classify_items(items)

    assert zips == []
    assert chunks == [chunk]
    assert ignored == ["quicklook.jpg", Path("quicklook.png"), "manifest.xml"]


@pytest.mark.unit
def test_classify_items_separates_zips() -> None:
    zip_path = Path("W_XX-EUMETSAT-FCI-1C-RRAD-FDHSI-FD-20241001005154-END.zip")

    zips, chunks, ignored = classify_items([zip_path, "readme.txt"])

    assert zips == [zip_path]
    assert chunks == []
    assert ignored == ["readme.txt"]


@pytest.mark.unit
def test_classify_items_rejects_mixed_zip_and_chunk_input() -> None:
    zip_name = "W_XX-EUMETSAT-FCI-1C-RRAD-FDHSI-FD-20241001005154-END.zip"

    with pytest.raises(ValueError, match="[Mm]ixed ZIP and unpacked chunk"):
        classify_items([zip_name, _chunk()])


@pytest.mark.unit
def test_classify_items_keeps_remote_uris_byte_for_byte() -> None:
    chunk_uri = "s3://bucket/prefix/" + _chunk()
    zip_uri = (
        "s3://bucket/prefix/W_XX-EUMETSAT-FCI-1C-RRAD-FDHSI-FD-20241001005154-END.zip"
    )
    junk_uri = "s3://bucket/prefix/quicklook.jpg"

    zips, chunks, ignored = classify_items([chunk_uri, junk_uri])
    assert (zips, chunks, ignored) == ([], [chunk_uri], [junk_uri])

    zips, chunks, ignored = classify_items([zip_uri])
    assert zips == [zip_uri]


@pytest.mark.unit
def test_bundle_from_remote_uris_keeps_uris_and_matches_local_scene_uri() -> None:
    names = [_chunk("TRAIL", number=41), _chunk(number=18)]
    uris = ["s3://bucket/prefix/" + n for n in names]

    (remote,) = group_chunks_into_bundles(uris)
    (local,) = group_chunks_into_bundles([Path("/data") / n for n in names])

    assert remote.members == (uris[1], uris[0])
    assert remote.uri == local.uri == "fci-scene://FDHSI/20250701000000"
    assert all(m.startswith("s3://bucket/prefix/") for m in remote.members)


@pytest.mark.unit
def test_remote_uri_chunk_name_parses_from_basename() -> None:
    uri = "s3://bucket/prefix/" + _chunk(product="HRFI", number=7)

    assert is_valid_fci_chunk(uri) is True
    chunk = parse_chunk_name(uri)
    assert (chunk.product_type, chunk.chunk_number) == (PRODUCT_TYPE_HRFI, 7)
