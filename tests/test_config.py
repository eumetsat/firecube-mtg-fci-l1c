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


def test_batch_workers_rejected():
    with pytest.raises(TypeError, match="batch_workers"):
        MtgFciL1cConfig(batch_workers=1)  # pyright: ignore[reportCallIssue]


def test_get_resolutions_filters_by_product_type():
    cfg = MtgFciL1cConfig(resolutions="1km,2km,500m")
    # 500m is not valid for FDHSI, so it is dropped.
    assert cfg.get_resolutions("FDHSI") == ["1km", "2km"]
    # 500m is valid for HRFI; 2km is not.
    assert cfg.get_resolutions("HRFI") == ["1km", "500m"]


def test_get_resolutions_defaults_when_unset():
    assert MtgFciL1cConfig().get_resolutions("FDHSI") == ["1km", "2km"]


def test_get_channels_none_when_unset():
    assert MtgFciL1cConfig().get_channels("FDHSI") is None


def test_get_channels_unknown_channel_raises():
    cfg = MtgFciL1cConfig(channels="not_a_channel")
    with pytest.raises(ValueError, match="Unknown channel"):
        cfg.get_channels("FDHSI")


def test_get_streaming_chunk_shape_unknown_group_raises():
    with pytest.raises(ValueError, match="Unknown resolution group"):
        MtgFciL1cConfig().get_streaming_chunk_shape("data_999m")


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
    assert cfg.zarr_sharding is True
    assert cfg.zarr_shard_target_bytes == 128 * 1024 * 1024
    assert cfg.zarr_shard_overrides is None


def test_zarr_chunk_overrides_default_is_none() -> None:
    assert MtgFciL1cConfig().zarr_chunk_overrides is None


def test_zarr_chunk_overrides_accepts_valid_rank4() -> None:
    cfg = MtgFciL1cConfig(zarr_chunk_overrides={"data_1km": (1, 2784, 11136, 1)})
    assert cfg.zarr_chunk_overrides == {"data_1km": (1, 2784, 11136, 1)}


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
