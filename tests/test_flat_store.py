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

"""Schema-level contract of the opt-in ``flat_store`` layout.

With ``flat_store=true`` and one effective resolution, every array is declared
in the store root (group ``""``) so ``xr.open_zarr(store)`` needs no ``group=``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from firecube.ingestor.api import ConfigurationError

from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig, MtgFciL1cIngestor

pytestmark = pytest.mark.unit

EXPECTED_ARRAY_NAMES = [
    "counts",
    "pixel_quality",
    "pixel_time",
    "slope",
    "offset",
    "latitude",
    "longitude",
    "x",
    "y",
    "time",
    "channel_name",
    "radiance_unit_conversion_coefficient",
    "radiance_to_bt_conversion_coefficient_wavenumber",
    "radiance_to_bt_conversion_coefficient_a",
    "radiance_to_bt_conversion_coefficient_b",
    "radiance_to_bt_conversion_constant_c1",
    "radiance_to_bt_conversion_constant_c2",
    "channel_effective_solar_irradiance",
    "spatial_ref",
]

# Groups holding ir_38 also declare the warm-gain calibration arrays.
EXPECTED_ARRAY_NAMES_WITH_IR38 = [
    "counts",
    "pixel_quality",
    "pixel_time",
    "slope",
    "offset",
    "warm_slope",
    "warm_offset",
    "latitude",
    "longitude",
    "x",
    "y",
    "time",
    "channel_name",
    "radiance_unit_conversion_coefficient",
    "radiance_to_bt_conversion_coefficient_wavenumber",
    "radiance_to_bt_conversion_coefficient_a",
    "radiance_to_bt_conversion_coefficient_b",
    "radiance_to_bt_conversion_constant_c1",
    "radiance_to_bt_conversion_constant_c2",
    "channel_effective_solar_irradiance",
    "spatial_ref",
]


def _flat_ingestor(**config_kwargs: Any) -> MtgFciL1cIngestor:
    ingestor = MtgFciL1cIngestor()
    ingestor.plugin_config = MtgFciL1cConfig(
        flat_store=True, time_slots=144, **config_kwargs
    )
    return ingestor


def _ctx() -> Any:
    return SimpleNamespace(source="unused", options={})


def _array(ingestor: MtgFciL1cIngestor, name: str) -> Any:
    (group_spec,) = ingestor.zarr_schema(_ctx())
    return next(array for array in group_spec.arrays if array.name == name)


@pytest.mark.parametrize("hook", ["zarr_schema", "index_spec"])
@pytest.mark.parametrize(
    ("config_kwargs", "expected_in_message"),
    [
        ({"product_type": "FDHSI"}, "['1km', '2km']"),
        ({"product_type": "FDHSI", "channels": "vis_06,ir_105"}, "['1km', '2km']"),
    ],
)
def test_flat_store_with_several_resolutions_is_rejected(
    hook: str, config_kwargs: dict[str, str], expected_in_message: str
) -> None:
    ingestor = _flat_ingestor(**config_kwargs)

    with pytest.raises(ConfigurationError) as excinfo:
        getattr(ingestor, hook)(_ctx())

    message = str(excinfo.value)
    assert expected_in_message in message
    assert "flat_store" in message


@pytest.mark.parametrize(
    ("config_kwargs", "expected_counts_shape", "expected_index_name", "expected_names"),
    [
        (
            {"product_type": "FDHSI", "resolutions": "1km"},
            (1, 11136, 11136, 8),
            "eumetsat_repeat_cycle_v1_fdhsi_1km",
            EXPECTED_ARRAY_NAMES,
        ),
        (
            {"product_type": "HRFI", "resolutions": "500m"},
            (1, 22272, 22272, 2),
            "eumetsat_repeat_cycle_v1_hrfi_500m",
            EXPECTED_ARRAY_NAMES,
        ),
        (
            {"product_type": "HRFI", "resolutions": "1km"},
            (1, 11136, 11136, 2),
            "eumetsat_repeat_cycle_v1_hrfi_1km",
            EXPECTED_ARRAY_NAMES_WITH_IR38,
        ),
        (
            {"product_type": "FDHSI", "channels": "ir_105,wv_63"},
            (1, 5568, 5568, 2),
            "eumetsat_repeat_cycle_v1_fdhsi_2km",
            EXPECTED_ARRAY_NAMES,
        ),
    ],
)
def test_flat_store_single_resolution_declares_everything_at_root(
    config_kwargs: dict[str, str],
    expected_counts_shape: tuple[int, ...],
    expected_index_name: str,
    expected_names: list[str],
) -> None:
    ingestor = _flat_ingestor(**config_kwargs)

    specs = ingestor.zarr_schema(_ctx())
    index_spec = ingestor.index_spec(_ctx())

    assert [spec.group for spec in specs] == [""]
    assert [array.name for array in specs[0].arrays] == expected_names
    assert _array(ingestor, "counts").shape == expected_counts_shape
    assert index_spec is not None
    assert index_spec.name == expected_index_name
    assert list(index_spec.groups) == [""]
    assert ingestor.get_batch_groups([], _ctx()) == [""]


def test_flat_store_applies_data_res_keyed_chunk_and_shard_overrides() -> None:
    ingestor = _flat_ingestor(
        product_type="FDHSI",
        resolutions="1km",
        # Arbitrary non-default shapes, so the overrides are visible in the spec.
        zarr_chunk_overrides={"data_1km": (1, 464, 11136, 1)},
        zarr_shard_overrides={"data_1km": (1, 2784, 11136, 1)},
    )

    for name in ("counts", "pixel_quality", "pixel_time"):
        array = _array(ingestor, name)
        assert array.chunks == (1, 464, 11136, 1), name
        assert array.shards == (1, 2784, 11136, 1), name


@pytest.mark.parametrize("hook", ["zarr_schema", "index_spec"])
@pytest.mark.parametrize("flat_store", [True, False])
def test_selection_that_leaves_no_resolution_is_rejected(
    hook: str, flat_store: bool
) -> None:
    ingestor = MtgFciL1cIngestor()
    ingestor.plugin_config = MtgFciL1cConfig(
        flat_store=flat_store,
        product_type="FDHSI",
        resolutions="1km",
        channels="ir_105",
        time_slots=144,
    )

    with pytest.raises(ConfigurationError) as excinfo:
        getattr(ingestor, hook)(_ctx())

    message = str(excinfo.value)
    assert "No FDHSI resolution is left" in message
    assert "resolutions='1km'" in message
    assert "channels='ir_105'" in message
