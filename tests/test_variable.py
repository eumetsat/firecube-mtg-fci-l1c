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

"""Comprehensive unit tests for the flat Variable primitive and VARIABLES registry."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from firecube_mtg_fci_l1c.config import MtgFciL1cConfig
from firecube_mtg_fci_l1c.schema import (
    VARIABLES,
    Variable,
    VariableContext,
    build_specs,
    variable_enabled,
)


# ─────────────────────────────────────────────────────────────
# Group 1: Variable dataclass
# ─────────────────────────────────────────────────────────────


def test_variable_pickle_roundtrip() -> None:
    v = Variable(name="x", dims=("x",), dtype="f8", fill_value=None, attrs={"units": "m"})
    assert pickle.loads(pickle.dumps(v)) == v


def test_variable_equality() -> None:
    v1 = Variable(name="x", dims=("x",), dtype="f8", fill_value=None)
    v2 = Variable(name="x", dims=("x",), dtype="f8", fill_value=None)
    assert v1 == v2


def test_variable_immutable() -> None:
    v = Variable(name="x", dims=("x",), dtype="f8", fill_value=None)
    with pytest.raises((AttributeError, TypeError)):
        v.name = "y"  # type: ignore[misc]


def test_variable_with_source_func_pickles() -> None:
    # Source is a module-level function on VARIABLES; the registry must pickle.
    v = VARIABLES[0]
    assert v.source is not None
    restored = pickle.loads(pickle.dumps(v))
    assert restored == v
    assert restored.source is v.source


# ─────────────────────────────────────────────────────────────
# Group 2: VariableContext
# ─────────────────────────────────────────────────────────────


def test_variable_context_optional_fields() -> None:
    ctx = VariableContext(
        group="data_1km",
        product_type="FDHSI",
        config=MtgFciL1cConfig(),
        dimsize=11136,
        n_channels=8,
        logical_channels=("vis_06",),
    )
    assert ctx.y_slice is None
    assert ctx.timestamp is None


def test_variable_context_with_runtime_fields() -> None:
    ctx = VariableContext(
        group="data_1km",
        product_type="FDHSI",
        config=MtgFciL1cConfig(),
        dimsize=11136,
        n_channels=8,
        logical_channels=("vis_06",),
        y_slice=slice(0, 278),
    )
    assert ctx.y_slice == slice(0, 278)


# ─────────────────────────────────────────────────────────────
# Group 3: variable_enabled
# ─────────────────────────────────────────────────────────────


def test_variable_enabled_default_true() -> None:
    v = Variable(name="x", dims=("x",), dtype="f8", fill_value=None)
    assert variable_enabled(v, MtgFciL1cConfig()) is True


def test_variable_enabled_by_config_flag() -> None:
    v = Variable(
        name="x",
        dims=("x",),
        dtype="f8",
        fill_value=None,
        enabled_by="include_pixel_quality",
    )
    assert variable_enabled(v, MtgFciL1cConfig(include_pixel_quality=True)) is True
    assert variable_enabled(v, MtgFciL1cConfig(include_pixel_quality=False)) is False


def test_variable_enabled_missing_attr_defaults_true() -> None:
    v = Variable(
        name="x",
        dims=("x",),
        dtype="f8",
        fill_value=None,
        enabled_by="nonexistent_flag",
    )
    assert variable_enabled(v, MtgFciL1cConfig()) is True


# ─────────────────────────────────────────────────────────────
# Group 4: VARIABLES registry
# ─────────────────────────────────────────────────────────────


def test_variables_count() -> None:
    assert len(VARIABLES) == 12, f"Expected 12, got {len(VARIABLES)}: {[v.name for v in VARIABLES]}"


def test_channel_name_in_variables() -> None:
    names = [v.name for v in VARIABLES]
    assert "channel_name" in names


def test_variables_name_uniqueness() -> None:
    names = [v.name for v in VARIABLES]
    assert len(names) == len(set(names)), f"Duplicate names: {names}"


def test_variables_all_instances_of_variable() -> None:
    assert all(isinstance(v, Variable) for v in VARIABLES)


def test_variables_pickle_roundtrip() -> None:
    restored = pickle.loads(pickle.dumps(VARIABLES))
    assert len(restored) == len(VARIABLES)
    for r, v in zip(restored, VARIABLES, strict=True):
        assert r.name == v.name
        assert r.dims == v.dims
        assert r.dtype == v.dtype
        assert r.enabled_by == v.enabled_by
        assert r.source is v.source
        if r.attrs is None:
            assert v.attrs is None
        else:
            assert v.attrs is not None
            assert set(r.attrs.keys()) == set(v.attrs.keys())
            for k in r.attrs:
                rv, vv = r.attrs[k], v.attrs[k]
                if isinstance(rv, np.ndarray) or isinstance(vv, np.ndarray):
                    assert np.array_equal(rv, vv), f"attrs[{k!r}] mismatch"
                else:
                    assert rv == vv, f"attrs[{k!r}] mismatch"


def test_variables_no_lambda_sources() -> None:
    lambdas = [
        v.name for v in VARIABLES if v.source is not None and v.source.__name__ == "<lambda>"
    ]
    assert not lambdas, f"Lambda sources detected (not picklable): {lambdas}"


# ─────────────────────────────────────────────────────────────
# Group 5: build_specs
# ─────────────────────────────────────────────────────────────


def test_build_specs_fdhsi_returns_2_groups() -> None:
    specs = build_specs(MtgFciL1cConfig(), "FDHSI")
    groups = [s.group for s in specs]
    assert "data_1km" in groups
    assert "data_2km" in groups


def test_build_specs_hrfi_returns_groups() -> None:
    specs = build_specs(MtgFciL1cConfig(), "HRFI")
    assert len(specs) >= 1


def test_build_specs_geolocation_disabled_excludes_lat_lon() -> None:
    specs = build_specs(MtgFciL1cConfig(include_geolocation=False), "FDHSI")
    for group_spec in specs:
        array_names = [a.name for a in group_spec.arrays]
        assert "latitude" not in array_names, f"latitude found in {group_spec.group}"
        assert "longitude" not in array_names, f"longitude found in {group_spec.group}"


# ─────────────────────────────────────────────────────────────
# Group 6: CF-compliance projection coordinates (x / y)
# ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_x_y_variables_in_variables_list() -> None:
    names = [v.name for v in VARIABLES]
    assert "x" in names
    assert "y" in names


@pytest.mark.unit
def test_x_y_attrs_meter_mode() -> None:
    # Check Variable-level invariants (dims, dtype, fill_value) via VARIABLES registry.
    x_var_reg = next(v for v in VARIABLES if v.name == "x")
    y_var_reg = next(v for v in VARIABLES if v.name == "y")
    assert x_var_reg.dims == ("x",)
    assert y_var_reg.dims == ("y",)
    assert x_var_reg.dtype == np.float64
    assert y_var_reg.dtype == np.float64
    assert x_var_reg.fill_value is None
    assert y_var_reg.fill_value is None

    # Check CF attrs via build_specs (attrs_resolver merges at spec-build time).
    specs = build_specs(MtgFciL1cConfig(), "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    x_spec = next(a for a in g.arrays if a.name == "x")
    y_spec = next(a for a in g.arrays if a.name == "y")
    assert x_spec.attrs is not None
    assert y_spec.attrs is not None
    assert x_spec.attrs["standard_name"] == "projection_x_coordinate"
    assert y_spec.attrs["standard_name"] == "projection_y_coordinate"
    assert x_spec.attrs["units"] == "m"
    assert y_spec.attrs["units"] == "m"
    assert x_spec.attrs["axis"] == "X"
    assert y_spec.attrs["axis"] == "Y"


@pytest.mark.unit
def test_x_y_attrs_radian_mode() -> None:
    specs = build_specs(MtgFciL1cConfig(projection_units="radian"), "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    x_var = next(a for a in g.arrays if a.name == "x")
    y_var = next(a for a in g.arrays if a.name == "y")
    assert x_var.attrs is not None
    assert y_var.attrs is not None
    assert x_var.attrs["standard_name"] == "projection_x_angular_coordinate"
    assert y_var.attrs["standard_name"] == "projection_y_angular_coordinate"
    assert x_var.attrs["units"] == "radian"
    assert y_var.attrs["units"] == "radian"
    assert x_var.attrs["axis"] == "X"
    assert y_var.attrs["axis"] == "Y"


@pytest.mark.unit
def test_projection_units_default_is_meter() -> None:
    assert MtgFciL1cConfig().projection_units == "meter"


@pytest.mark.unit
def test_projection_units_accepts_meter_metre_radian() -> None:
    assert MtgFciL1cConfig(projection_units="meter").projection_units == "meter"
    assert MtgFciL1cConfig(projection_units="metre").projection_units == "metre"
    assert MtgFciL1cConfig(projection_units="radian").projection_units == "radian"


@pytest.mark.unit
def test_projection_units_metre_is_alias_for_meter() -> None:
    from firecube_mtg_fci_l1c._variable import VariableContext as _VC
    from firecube_mtg_fci_l1c.schema import _projection_x_source

    ctx_meter = _VC(
        group="data_1km",
        product_type="FDHSI",
        config=MtgFciL1cConfig(projection_units="meter"),
        dimsize=11136,
        n_channels=8,
        logical_channels=(),
    )
    ctx_metre = _VC(
        group="data_1km",
        product_type="FDHSI",
        config=MtgFciL1cConfig(projection_units="metre"),
        dimsize=11136,
        n_channels=8,
        logical_channels=(),
    )
    x_meter = _projection_x_source(ctx_meter)
    x_metre = _projection_x_source(ctx_metre)
    assert x_meter is not None
    assert x_metre is not None
    assert np.array_equal(x_metre, x_meter)


@pytest.mark.unit
def test_projection_units_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="projection_units"):
        MtgFciL1cConfig(projection_units="foot")


@pytest.mark.unit
def test_x_y_source_values_1km_radian_mode() -> None:
    from firecube_mtg_fci_l1c._variable import VariableContext as _VC
    from firecube_mtg_fci_l1c.schema import _projection_x_source, _projection_y_source

    ctx = _VC(
        group="data_1km",
        product_type="FDHSI",
        config=MtgFciL1cConfig(projection_units="radian"),
        dimsize=11136,
        n_channels=8,
        logical_channels=(),
    )
    x_arr = _projection_x_source(ctx)
    y_arr = _projection_y_source(ctx)
    assert x_arr is not None
    assert y_arr is not None
    assert x_arr.shape == (11136,)
    assert y_arr.shape == (11136,)
    assert x_arr.dtype == np.float64
    # x is monotonically increasing (east-positive)
    assert np.all(np.diff(x_arr) > 0)
    # y is monotonically increasing
    assert np.all(np.diff(y_arr) > 0)
    # Check first values roughly match the FCI offset constant
    assert abs(x_arr[0] - (-0.1556038047568524)) < 1e-10
    assert abs(y_arr[0] - (-0.1556038047568524)) < 1e-10


@pytest.mark.unit
def test_x_y_source_values_meter_mode_1km() -> None:
    from firecube_mtg_fci_l1c._variable import VariableContext as _VC
    from firecube_mtg_fci_l1c.schema import _projection_x_source, _projection_y_source

    ctx = _VC(
        group="data_1km",
        product_type="FDHSI",
        config=MtgFciL1cConfig(),
        dimsize=11136,
        n_channels=8,
        logical_channels=(),
    )
    x_arr = _projection_x_source(ctx)
    y_arr = _projection_y_source(ctx)
    assert x_arr is not None
    assert y_arr is not None
    assert x_arr.shape == (11136,)
    assert y_arr.shape == (11136,)
    assert x_arr.dtype == np.float64
    assert np.all(np.diff(x_arr) > 0)
    assert np.all(np.diff(y_arr) > 0)
    assert abs(x_arr[0] - (-5568500.0)) < 1.0


@pytest.mark.unit
def test_x_y_source_values_meter_mode_500m_and_2km() -> None:
    from firecube_mtg_fci_l1c._variable import VariableContext as _VC
    from firecube_mtg_fci_l1c.schema import _projection_x_source, _projection_y_source

    for group, dimsize in [("data_500m", 22272), ("data_2km", 5568)]:
        ctx = _VC(
            group=group,
            product_type="FDHSI",
            config=MtgFciL1cConfig(),
            dimsize=dimsize,
            n_channels=8,
            logical_channels=(),
        )
        x_arr = _projection_x_source(ctx)
        y_arr = _projection_y_source(ctx)
        assert x_arr is not None, f"x is None for {group}"
        assert y_arr is not None, f"y is None for {group}"
        assert x_arr.shape == (dimsize,)
        assert np.all(np.diff(x_arr) > 0), f"x not increasing for {group}"
        assert np.all(np.diff(y_arr) > 0), f"y not increasing for {group}"


@pytest.mark.unit
def test_x_y_position_in_variables() -> None:
    names = [v.name for v in VARIABLES]
    lon_idx = names.index("longitude")
    x_idx = names.index("x")
    y_idx = names.index("y")
    time_idx = names.index("time")
    assert lon_idx < x_idx < y_idx < time_idx, "x/y must be between longitude and time"


# ─────────────────────────────────────────────────────────────
# Group 7: CF-compliance attrs on data variables
# ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_counts_cf_attrs() -> None:
    c = next(v for v in VARIABLES if v.name == "counts")
    assert c.attrs is not None
    assert c.attrs["grid_mapping"] == "spatial_ref"
    assert c.attrs["ancillary_variables"] == "pixel_quality pixel_time"
    assert c.attrs["units"] == "1"
    assert "standard_name" not in c.attrs


@pytest.mark.unit
def test_pixel_quality_flag_attrs() -> None:
    pq = next(v for v in VARIABLES if v.name == "pixel_quality")
    assert pq.attrs is not None
    assert pq.attrs["grid_mapping"] == "spatial_ref"
    fm = pq.attrs["flag_masks"]
    assert fm == [1, 2, 4, 8, 16, 32, 64, 128]
    assert all(0 <= v <= 255 for v in fm)
    meanings = pq.attrs["flag_meanings"]
    tokens = meanings.split()
    assert len(tokens) == 8
    assert "," not in meanings


@pytest.mark.unit
def test_pixel_time_cf_attrs() -> None:
    pt = next(v for v in VARIABLES if v.name == "pixel_time")
    assert pt.attrs is not None
    assert pt.attrs["grid_mapping"] == "spatial_ref"
    assert pt.attrs["standard_name"] == "time"
    assert pt.attrs["calendar"] == "standard"
    assert pt.attrs["units"] == "seconds since 2000-01-01"


@pytest.mark.unit
def test_spatial_ref_no_cf_violations() -> None:
    sr = next(v for v in VARIABLES if v.name == "spatial_ref")
    assert sr.attrs is not None
    assert "units" not in sr.attrs
    assert "coordinates" not in sr.attrs
    assert "grid_mapping_name" in sr.attrs
    assert "crs_wkt" in sr.attrs
    assert "spatial_ref" in sr.attrs
    assert sr.attrs["crs_wkt"] == sr.attrs["spatial_ref"]
    assert sr.attrs["crs_wkt"].startswith('PROJCRS["MTG Geostationary"')


# ─────────────────────────────────────────────────────────────
# Group 8: coordinates attribute resolution at build_specs time
# ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_coordinates_attr_conditional_geolocation_on() -> None:
    specs = build_specs(MtgFciL1cConfig(include_geolocation=True), "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    a = {arr.name: dict(arr.attrs or {}) for arr in g.arrays}
    assert a["counts"]["coordinates"] == "latitude longitude"
    assert a["pixel_quality"]["coordinates"] == "latitude longitude"
    assert a["pixel_time"]["coordinates"] == "latitude longitude"
    assert "coordinates" not in a.get("slope", {})
    assert "coordinates" not in a.get("x", {})


@pytest.mark.unit
def test_coordinates_attr_absent_when_geolocation_off() -> None:
    specs = build_specs(MtgFciL1cConfig(include_geolocation=False), "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    a = {arr.name: dict(arr.attrs or {}) for arr in g.arrays}
    assert "coordinates" not in a.get("counts", {})
    assert "coordinates" not in a.get("pixel_quality", {})
    assert "coordinates" not in a.get("pixel_time", {})


# ─────────────────────────────────────────────────────────────
# Group 9: Regression guards (rename / removal protection)
# ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_slope_offset_names_and_units_unchanged() -> None:
    """Regression guard: slope/offset must NOT be renamed to scale_factor/add_offset."""
    names = [v.name for v in VARIABLES]
    assert "slope" in names, "slope was removed or renamed!"
    assert "offset" in names, "offset was removed or renamed!"
    assert "scale_factor" not in names, "slope was wrongly renamed to scale_factor!"
    assert "add_offset" not in names, "offset was wrongly renamed to add_offset!"
    slope_var = next(v for v in VARIABLES if v.name == "slope")
    offset_var = next(v for v in VARIABLES if v.name == "offset")
    assert slope_var.attrs is not None
    assert offset_var.attrs is not None
    assert slope_var.attrs["units"] == "mW m-2 sr-1 (cm-1)-1"
    assert offset_var.attrs["units"] == "mW m-2 sr-1 (cm-1)-1"


@pytest.mark.unit
def test_no_calibration_coefficients_variable() -> None:
    """calibration_coefficients was explicitly rejected."""
    names = [v.name for v in VARIABLES]
    assert "calibration_coefficients" not in names


# ─────────────────────────────────────────────────────────────
# Group 10: zarr_chunk_overrides — precedence, cross-validation, integration
# ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_chunk_override_takes_precedence_over_chunk_y() -> None:
    cfg = MtgFciL1cConfig(
        zarr_chunk_y=999,
        zarr_chunk_overrides={"data_1km": (1, 2784, 11136, 1)},
    )
    specs = build_specs(cfg, "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    counts = next(a for a in g.arrays if a.name == "counts")
    assert counts.chunks == (1, 2784, 11136, 1)


@pytest.mark.unit
def test_chunk_override_combined_with_shard_override_full_disk() -> None:
    """Power-user recipe: chunks divide dimsize, shards = full dimsize."""
    cfg = MtgFciL1cConfig(
        zarr_chunk_overrides={"data_1km": (1, 2784, 11136, 1)},
        zarr_shard_overrides={"data_1km": (1, 11136, 11136, 1)},
    )
    specs = build_specs(cfg, "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    counts = next(a for a in g.arrays if a.name == "counts")
    assert counts.chunks == (1, 2784, 11136, 1)
    assert counts.shards == (1, 11136, 11136, 1)
    assert counts.shards[1] // counts.chunks[1] == 4
    assert counts.shards[2] // counts.chunks[2] == 1


@pytest.mark.unit
def test_chunk_override_without_shard_override() -> None:
    """User overrides chunks only; shards stay byte-budgeted using new chunks."""
    cfg = MtgFciL1cConfig(zarr_chunk_overrides={"data_1km": (1, 2784, 11136, 1)})
    specs = build_specs(cfg, "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    counts = next(a for a in g.arrays if a.name == "counts")
    assert counts.chunks == (1, 2784, 11136, 1)
    # byte-budget math: chunk_bytes = 1*2784*11136*2 = 62,029,824; budget = 134,217,728
    # multiples = 134217728 // 62029824 = 2; cap = 11136 // 2784 = 4
    # shard_y = max(2784, min(2,4) * 2784) = 5568
    assert counts.shards == (1, 5568, 11136, 1)


@pytest.mark.unit
def test_chunk_override_non_divisible_with_shard_override_raises() -> None:
    cfg = MtgFciL1cConfig(
        zarr_chunk_overrides={"data_1km": (1, 100, 11136, 1)},
        zarr_shard_overrides={"data_1km": (1, 11136, 11136, 1)},
    )
    with pytest.raises(ValueError, match="not a whole multiple"):
        build_specs(cfg, "FDHSI")


@pytest.mark.unit
def test_chunk_override_y_exceeds_dimsize_raises() -> None:
    cfg = MtgFciL1cConfig(zarr_chunk_overrides={"data_1km": (1, 99999, 11136, 1)})
    with pytest.raises(ValueError, match="exceeds dimsize"):
        build_specs(cfg, "FDHSI")


@pytest.mark.unit
def test_default_chunk_shape_unchanged_when_no_override() -> None:
    """Regression: defaults are identical to v0.3.0."""
    cfg = MtgFciL1cConfig()
    specs = build_specs(cfg, "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    counts = next(a for a in g.arrays if a.name == "counts")
    assert counts.chunks == (1, 278, 11136, 1)  # v0.3.0 default


@pytest.mark.unit
def test_mixed_per_resolution_override() -> None:
    """Only overriding data_1km; data_2km uses defaults."""
    cfg = MtgFciL1cConfig(zarr_chunk_overrides={"data_1km": (1, 2784, 11136, 1)})
    specs = build_specs(cfg, "FDHSI")
    g1km = next(g for g in specs if g.group == "data_1km")
    g2km = next(g for g in specs if g.group == "data_2km")
    counts_1km = next(a for a in g1km.arrays if a.name == "counts")
    counts_2km = next(a for a in g2km.arrays if a.name == "counts")
    assert counts_1km.chunks == (1, 2784, 11136, 1)  # overridden
    assert counts_2km.chunks == (1, 139, 5568, 1)  # default 2km


@pytest.mark.unit
def test_zarr_sharding_false_ignores_shard_overrides_but_keeps_chunk_overrides() -> None:
    """zarr_sharding=False produces shards=None regardless of shard overrides,
    but chunk overrides still apply to chunk shape."""
    cfg = MtgFciL1cConfig(
        zarr_sharding=False,
        zarr_chunk_overrides={"data_1km": (1, 2784, 11136, 1)},
        zarr_shard_overrides={"data_1km": (1, 11136, 11136, 1)},
    )
    specs = build_specs(cfg, "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    counts = next(a for a in g.arrays if a.name == "counts")
    assert counts.chunks == (1, 2784, 11136, 1)  # chunk override still applied
    assert counts.shards is None  # sharding fully disabled


@pytest.mark.unit
def test_chunk_overrides_cli_string_parser_path() -> None:
    """from_options() correctly parses a JSON-string zarr_chunk_overrides and
    the resulting config (with list values, not tuples) still passes validation
    and produces the correct schema."""
    # NOTE: from_options parses JSON strings, yielding lists not tuples
    options = {"zarr_chunk_overrides": '{"data_1km":[1,2784,11136,1]}'}
    cfg = MtgFciL1cConfig.from_options(options)
    assert cfg.zarr_chunk_overrides is not None
    assert "data_1km" in cfg.zarr_chunk_overrides
    assert tuple(cfg.zarr_chunk_overrides["data_1km"]) == (1, 2784, 11136, 1)
    # End-to-end: schema build accepts list values
    specs = build_specs(cfg, "FDHSI")
    counts = next(
        a for g in specs if g.group == "data_1km" for a in g.arrays if a.name == "counts"
    )
    assert tuple(counts.chunks) == (1, 2784, 11136, 1)


@pytest.mark.unit
def test_chunk_overrides_dict_of_lists_accepted() -> None:
    """Validation in __post_init__ must handle both tuples and lists without error."""
    cfg = MtgFciL1cConfig(
        zarr_chunk_overrides={"data_1km": [1, 2784, 11136, 1]}  # type: ignore[dict-item]
    )
    specs = build_specs(cfg, "FDHSI")
    counts = next(
        a for g in specs if g.group == "data_1km" for a in g.arrays if a.name == "counts"
    )
    assert tuple(counts.chunks) == (1, 2784, 11136, 1)
