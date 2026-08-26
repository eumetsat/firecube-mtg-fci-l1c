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

"""Route --option zarr_sharding through zarr_schema() into template_config.

W18: regression tests that ``ctx.options["zarr_sharding"]`` reaches
``config.template_config.zarr_sharding`` before the schema is projected. The
plugin default is sharding-on; the CLI/operator override must win.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from firecube.ingestor.api import ZarrTemplateConfig

from firecube_mtg_fci_l1c.ingestor import MtgFciL1cConfig, MtgFciL1cIngestor

DYNAMIC_ARRAYS_4D = ("counts", "pixel_quality", "pixel_time")
STATIC_ARRAYS_NEVER_SHARDED = ("latitude", "longitude", "x", "y")


def _make_ingestor(
    *, zarr_sharding: bool | None = None, **config_kwargs: Any
) -> MtgFciL1cIngestor:
    ingestor = MtgFciL1cIngestor()
    ingestor.plugin_config = MtgFciL1cConfig(
        product_type="FDHSI",
        time_slots=144,
        **config_kwargs,
    )
    if zarr_sharding is not None:
        ingestor.template_config = ZarrTemplateConfig(zarr_sharding=zarr_sharding)
    return ingestor


def _dynamic_specs(specs: list[Any]) -> list[Any]:
    return [
        array_spec
        for group_spec in specs
        for array_spec in group_spec.arrays
        if array_spec.name in DYNAMIC_ARRAYS_4D
    ]


def _static_specs(specs: list[Any]) -> list[Any]:
    return [
        array_spec
        for group_spec in specs
        for array_spec in group_spec.arrays
        if array_spec.name in STATIC_ARRAYS_NEVER_SHARDED
    ]


@pytest.mark.parametrize("false_value", [False, "false", "False", "0", "no", "off"])
def test_zarr_sharding_explicit_false_propagates_to_all_dynamic_arrays(
    false_value: Any,
) -> None:
    """--option zarr_sharding=false must disable shards on every 4-D data array.

    Static coord arrays (lat/lon/x/y) are never sharded regardless of the flag;
    assert that invariant holds so a future refactor cannot silently attach
    shards to them.
    """
    ingestor = _make_ingestor(zarr_sharding=False)
    ctx: Any = SimpleNamespace(
        source="/tmp",
        options={"zarr_sharding": false_value},
    )

    specs = ingestor.zarr_schema(ctx)

    dynamic = _dynamic_specs(specs)
    assert dynamic, "expected at least one 4-D dynamic array spec"
    for spec in dynamic:
        assert spec.shards is None, (
            f"{spec.name}: expected shards=None with zarr_sharding=False, got {spec.shards!r}"
        )

    for spec in _static_specs(specs):
        assert spec.shards is None, (
            f"{spec.name}: static arrays must never be sharded, got {spec.shards!r}"
        )


def test_zarr_sharding_explicit_true_keeps_plugin_shard_derivation() -> None:
    """--option zarr_sharding=true keeps the plugin's byte-budgeted 4-D shard derivation."""
    ingestor = _make_ingestor(zarr_sharding=True)
    ctx: Any = SimpleNamespace(
        source="/tmp",
        options={"zarr_sharding": True},
    )

    specs = ingestor.zarr_schema(ctx)

    dynamic = _dynamic_specs(specs)
    assert dynamic, "expected at least one 4-D dynamic array spec"
    assert any(spec.shards is not None for spec in dynamic), (
        "with zarr_sharding=True, plugin's byte-budgeted derivation should produce "
        "non-None shards on at least one 4-D array"
    )


def test_zarr_sharding_routing_reads_from_template_config_not_options_string() -> None:
    """Schema routing must use Firecube's typed template config, not option text."""
    ingestor = _make_ingestor(zarr_sharding=False)
    ctx: Any = SimpleNamespace(
        source="/tmp",
        options={"zarr_sharding": "true"},
    )

    specs = ingestor.zarr_schema(ctx)

    dynamic = _dynamic_specs(specs)
    assert dynamic and all(spec.shards is None for spec in dynamic)


def test_zarr_sharding_omitted_keeps_current_default() -> None:
    """No zarr_sharding key in ctx.options means: fall back to plugin default (on).

    The plugin default is ``ZarrTemplateConfig(zarr_sharding=True)``; omitting
    the option must produce an identical schema to explicit ``True``.
    """
    ingestor_omitted = _make_ingestor()
    ctx_omitted: Any = SimpleNamespace(source="/tmp", options={})
    specs_omitted = ingestor_omitted.zarr_schema(ctx_omitted)

    ingestor_true = _make_ingestor(zarr_sharding=True)
    ctx_true: Any = SimpleNamespace(source="/tmp", options={"zarr_sharding": True})
    specs_true = ingestor_true.zarr_schema(ctx_true)

    shards_omitted = [
        (group.group, array.name, array.shards)
        for group in specs_omitted
        for array in group.arrays
    ]
    shards_true = [
        (group.group, array.name, array.shards)
        for group in specs_true
        for array in group.arrays
    ]
    assert shards_omitted == shards_true


def test_zarr_sharding_toggle_is_self_consistent_within_one_process() -> None:
    """Toggling zarr_sharding across two calls yields consistent per-call schemas.

    Regression guard: the plugin_config's ``template_config`` is a mutable
    dataclass. Two consecutive ``zarr_schema`` calls with different options
    must each produce the schema matching *their* option, not leak the earlier
    call's mutation into the later call's default.
    """
    ingestor = _make_ingestor()

    ingestor.template_config = ZarrTemplateConfig(zarr_sharding=False)
    ctx_false: Any = SimpleNamespace(source="/tmp", options={"zarr_sharding": False})
    specs_false = ingestor.zarr_schema(ctx_false)
    dynamic_false = _dynamic_specs(specs_false)
    assert dynamic_false and all(spec.shards is None for spec in dynamic_false)

    ingestor.template_config = ZarrTemplateConfig(zarr_sharding=True)
    ctx_true: Any = SimpleNamespace(source="/tmp", options={"zarr_sharding": True})
    specs_true = ingestor.zarr_schema(ctx_true)
    dynamic_true = _dynamic_specs(specs_true)
    assert dynamic_true and any(spec.shards is not None for spec in dynamic_true)

    ingestor.template_config = ZarrTemplateConfig(zarr_sharding=False)
    specs_false_again = ingestor.zarr_schema(ctx_false)
    dynamic_false_again = _dynamic_specs(specs_false_again)
    assert dynamic_false_again and all(
        spec.shards is None for spec in dynamic_false_again
    )
