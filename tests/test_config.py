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

"""Unit tests for the extracted ``MtgFciL1cConfig`` surface.

These exercise the operator-facing ``--option`` parsing in isolation from the
ingestion pipeline. They double as a worked example of how the config layer maps
plugin options onto resolutions/channels.
"""

from __future__ import annotations

import pytest

from firecube_mtg_fci_l1c.config import MtgFciL1cConfig

pytestmark = pytest.mark.unit


def test_defaults():
    cfg = MtgFciL1cConfig()
    # Epoch defaults to the dataset's first available date (see _constants).
    assert cfg.time_epoch == "2024-09-24"
    assert cfg.include_geolocation is True
    assert cfg.pixel_time_dtype == "float64"


def test_pixel_time_dtype_accepts_all_valid_values():
    for dtype in ("float64", "float32", "int32", "int64"):
        cfg = MtgFciL1cConfig(pixel_time_dtype=dtype)
        assert cfg.pixel_time_dtype == dtype


def test_pixel_time_dtype_default_unchanged():
    cfg = MtgFciL1cConfig()
    assert cfg.pixel_time_dtype == "float64"


def test_pixel_time_dtype_rejects_bogus():
    with pytest.raises(ValueError, match="pixel_time_dtype"):
        MtgFciL1cConfig(pixel_time_dtype="bogus")


def test_get_resolutions_filters_by_product_type():
    cfg = MtgFciL1cConfig(resolutions="1km,2km,500m")
    # 500m is not valid for FDHSI, so it is dropped.
    assert cfg.get_resolutions("FDHSI") == ["1km", "2km"]
    # 500m is valid for HRFI; 2km is not.
    assert cfg.get_resolutions("HRFI") == ["1km", "500m"]


@pytest.mark.parametrize(
    ("product_type", "config_kwargs", "expected"),
    [
        ("FDHSI", {"resolutions": "1km"}, ("1km",)),
        ("FDHSI", {"resolutions": "2km,1km"}, ("1km", "2km")),
        ("FDHSI", {"channels": "vis_06,nir_16"}, ("1km",)),
        ("FDHSI", {"resolutions": "500m"}, ()),
        ("HRFI", {"resolutions": "500m"}, ("500m",)),
        ("HRFI", {"channels": "ir_38"}, ("1km",)),
    ],
)
def test_effective_resolutions_applies_resolution_and_channel_filters(
    product_type: str, config_kwargs: dict[str, str], expected: tuple[str, ...]
) -> None:
    cfg = MtgFciL1cConfig(**config_kwargs)  # type: ignore[arg-type]
    assert cfg.effective_resolutions(product_type) == expected


def test_get_resolutions_defaults_when_unset():
    assert MtgFciL1cConfig().get_resolutions("FDHSI") == ["1km", "2km"]


def test_get_channels_none_when_unset():
    assert MtgFciL1cConfig().get_channels("FDHSI") is None


def test_get_channels_unknown_channel_raises():
    cfg = MtgFciL1cConfig(channels="not_a_channel")
    with pytest.raises(ValueError, match="Unknown channel"):
        cfg.get_channels("FDHSI")


def test_get_group_chunk_shape_unknown_resolution_raises():
    with pytest.raises(ValueError, match="Unknown resolution"):
        MtgFciL1cConfig().get_group_chunk_shape("999m")


@pytest.mark.parametrize("bad", [0, -1, -1024])
def test_zarr_shard_target_bytes_must_be_positive(bad):
    with pytest.raises(ValueError, match="zarr_shard_target_bytes must be a positive"):
        MtgFciL1cConfig(zarr_shard_target_bytes=bad)


def test_zarr_shard_overrides_unknown_group_rejected():
    with pytest.raises(ValueError, match="Invalid zarr_shard_overrides group"):
        MtgFciL1cConfig(zarr_shard_overrides={"data_999m": (1, 139, 5568, 1)})  # pyright: ignore[reportArgumentType]


def test_zarr_shard_overrides_wrong_rank_rejected():
    with pytest.raises(ValueError, match="must be rank-4"):
        MtgFciL1cConfig(zarr_shard_overrides={"data_2km": (1, 5568, 5568)})  # pyright: ignore[reportArgumentType]


def test_zarr_shard_overrides_non_positive_rejected():
    with pytest.raises(ValueError, match="must contain positive ints"):
        MtgFciL1cConfig(zarr_shard_overrides={"data_2km": (1, 0, 5568, 1)})


def test_zarr_shard_defaults():
    cfg = MtgFciL1cConfig()
    assert cfg.template_config.zarr_sharding is True
    assert cfg.zarr_shard_target_bytes == 128 * 1024 * 1024
    assert cfg.zarr_shard_overrides is None


def test_zarr_chunk_overrides_default_is_none() -> None:
    assert MtgFciL1cConfig().zarr_chunk_overrides is None


def test_config_rejects_zarr_chunk_y_exceeding_2x_nc_part_rows_1km() -> None:
    with pytest.raises(
        ValueError, match="exceeds max supported for chunk-owned assembly"
    ):
        MtgFciL1cConfig(zarr_chunk_y=1200)


def test_config_accepts_zarr_chunk_y_at_2x_nc_part_rows() -> None:
    MtgFciL1cConfig(zarr_chunk_y=556)


def test_config_accepts_zarr_chunk_y_below_nc_part_rows() -> None:
    MtgFciL1cConfig(zarr_chunk_y=100)


def test_config_accepts_none_zarr_chunk_y_default() -> None:
    MtgFciL1cConfig()


def test_zarr_chunk_overrides_accepts_valid_rank4() -> None:
    cfg = MtgFciL1cConfig(zarr_chunk_overrides={"data_1km": (1, 556, 11136, 1)})
    assert cfg.zarr_chunk_overrides == {"data_1km": (1, 556, 11136, 1)}


def test_config_rejects_zarr_chunk_overrides_exceeding_2x_nc_part_rows() -> None:
    with pytest.raises(
        ValueError, match="exceeds max supported for chunk-owned assembly"
    ):
        MtgFciL1cConfig(zarr_chunk_overrides={"data_1km": (1, 557, 11136, 1)})


def test_zarr_chunk_overrides_rejects_bogus_group() -> None:
    with pytest.raises(ValueError, match="Invalid zarr_chunk_overrides group"):
        MtgFciL1cConfig(zarr_chunk_overrides={"bogus": (1, 100, 100, 1)})


def test_zarr_chunk_overrides_rejects_rank3() -> None:
    with pytest.raises(ValueError, match="rank-4"):
        MtgFciL1cConfig(zarr_chunk_overrides={"data_1km": (1, 100, 100)})  # type: ignore[dict-item]


def test_zarr_chunk_overrides_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        MtgFciL1cConfig(zarr_chunk_overrides={"data_1km": (1, -1, 100, 1)})


def test_zarr_chunk_overrides_rejects_time_not_one() -> None:
    with pytest.raises(ValueError, match="time dim must be 1"):
        MtgFciL1cConfig(zarr_chunk_overrides={"data_1km": (2, 100, 100, 1)})


def test_zarr_chunk_overrides_rejects_channel_not_one() -> None:
    with pytest.raises(ValueError, match="channel dim must be 1"):
        MtgFciL1cConfig(zarr_chunk_overrides={"data_1km": (1, 100, 100, 2)})


def test_partial_chunk_and_body_chunks_defaults() -> None:
    cfg = MtgFciL1cConfig()
    assert cfg.partial_chunk == "fill"
    assert cfg.body_chunks is None


@pytest.mark.parametrize("value", ["fill", "error"])
def test_partial_chunk_accepts_allowed_values(value: str) -> None:
    assert MtgFciL1cConfig(partial_chunk=value).partial_chunk == value


@pytest.mark.parametrize("value", ["skip", "FILL", ""])
def test_partial_chunk_rejects_other_values(value: str) -> None:
    with pytest.raises(ValueError, match=r"partial_chunk.*\['error', 'fill'\]"):
        MtgFciL1cConfig(partial_chunk=value)


def test_partial_chunk_rejected_via_option_parsing() -> None:
    with pytest.raises(ValueError, match="partial_chunk"):
        MtgFciL1cConfig.from_options({"partial_chunk": "skip"})


@pytest.mark.parametrize("product_type", [None, "FDHSI", "HRFI"])
def test_body_chunks_accepts_valid_range_up_to_last_chunk(
    product_type: str | None,
) -> None:
    cfg = MtgFciL1cConfig(product_type=product_type, body_chunks=[32, 40])
    assert cfg.body_chunks == [32, 40]


@pytest.mark.parametrize("pair", [[1, 1], [40, 40], [1, 40]])
def test_body_chunks_accepts_boundary_ranges(pair: list[int]) -> None:
    assert MtgFciL1cConfig(body_chunks=pair).body_chunks == pair


@pytest.mark.parametrize(
    "bad",
    [
        [40, 32],  # reversed
        [0, 5],  # below range
        [-1, 5],
        [1, 41],  # just past the last chunk
        [41, 41],
        [32],  # wrong length
        [],
        [1, 2, 3],
        [1.5, 3],  # non-int
        ["1", "3"],
        [True, 3],  # bool is not a chunk number
        [None, 3],
    ],
)
@pytest.mark.parametrize("product_type", [None, "FDHSI", "HRFI"])
def test_body_chunks_rejects_invalid(bad: list, product_type: str | None) -> None:
    with pytest.raises(ValueError, match="body_chunks"):
        MtgFciL1cConfig(product_type=product_type, body_chunks=bad)


def test_body_chunks_json_string_arrives_as_int_list() -> None:
    cfg = MtgFciL1cConfig.from_options({"body_chunks": "[32, 40]"})
    assert cfg.body_chunks == [32, 40]
    assert all(type(n) is int for n in cfg.body_chunks)


def test_body_chunks_none_string_arrives_as_none() -> None:
    assert MtgFciL1cConfig.from_options({"body_chunks": "none"}).body_chunks is None


@pytest.mark.parametrize("raw", ["[40, 32]", "[1, 41]", "[32]", '["a", "b"]'])
def test_body_chunks_invalid_json_rejected_via_option_parsing(raw: str) -> None:
    with pytest.raises(ValueError, match="body_chunks"):
        MtgFciL1cConfig.from_options({"body_chunks": raw})
