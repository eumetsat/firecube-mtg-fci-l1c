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
import zarr

from firecube_mtg_fci_l1c.config import MtgFciL1cConfig
from firecube_mtg_fci_l1c._variables import (
    TIME_COORD_NAME,
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
    v = Variable(
        name="x", dims=("x",), dtype="f8", fill_value=None, attrs={"units": "m"}
    )
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
        resolution="1km",
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
        resolution="1km",
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
    assert variable_enabled(v, MtgFciL1cConfig(), ("vis_06",)) is True


def test_variable_enabled_by_config_flag() -> None:
    v = Variable(
        name="x",
        dims=("x",),
        dtype="f8",
        fill_value=None,
        enabled_by="include_pixel_quality",
    )
    assert (
        variable_enabled(v, MtgFciL1cConfig(include_pixel_quality=True), ("vis_06",))
        is True
    )
    assert (
        variable_enabled(v, MtgFciL1cConfig(include_pixel_quality=False), ("vis_06",))
        is False
    )


def test_variable_enabled_missing_attr_defaults_true() -> None:
    v = Variable(
        name="x",
        dims=("x",),
        dtype="f8",
        fill_value=None,
        enabled_by="nonexistent_flag",
    )
    assert variable_enabled(v, MtgFciL1cConfig(), ("vis_06",)) is True


# ─────────────────────────────────────────────────────────────
# Group 4: VARIABLES registry
# ─────────────────────────────────────────────────────────────


def test_variables_count() -> None:
    assert len(VARIABLES) == 25, (
        f"Expected 25, got {len(VARIABLES)}: {[v.name for v in VARIABLES]}"
    )


def test_channel_coordinate_declared_as_text_without_fill_value() -> None:
    specs = build_specs(MtgFciL1cConfig(channels="ir_105,ir_38"), "FDHSI")
    group = next(g for g in specs if g.group == "data_2km")
    channel = next(a for a in group.arrays if a.name == "channel")

    assert channel.dimension_names == ("channel",)
    assert channel.shape == (2,)
    assert channel.time_indexed is False
    assert isinstance(channel.dtype, np.dtypes.StringDType)
    assert channel.fill_value is None
    assert "channel" in group.coord_names

    variable = next(v for v in VARIABLES if v.name == "channel")
    assert variable.source is not None
    ctx = VariableContext(
        group="data_2km",
        resolution="2km",
        product_type="FDHSI",
        config=MtgFciL1cConfig(),
        dimsize=5568,
        n_channels=2,
        logical_channels=("ir_105", "ir_38"),
    )
    assert variable.source(ctx).tolist() == ["ir_105", "ir_38"]


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
        v.name
        for v in VARIABLES
        if v.source is not None and v.source.__name__ == "<lambda>"
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
    from firecube_mtg_fci_l1c._schema import VariableContext as _VC
    from firecube_mtg_fci_l1c._variables import _projection_x_source

    ctx_meter = _VC(
        group="data_1km",
        resolution="1km",
        product_type="FDHSI",
        config=MtgFciL1cConfig(projection_units="meter"),
        dimsize=11136,
        n_channels=8,
        logical_channels=(),
    )
    ctx_metre = _VC(
        group="data_1km",
        resolution="1km",
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
    from firecube_mtg_fci_l1c._schema import VariableContext as _VC
    from firecube_mtg_fci_l1c._variables import (
        _projection_x_source,
        _projection_y_source,
    )

    ctx = _VC(
        group="data_1km",
        resolution="1km",
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
    # Index 0 is pixel centre -(dimsize/2 - 0.5) sampling steps from nadir:
    # 5567.5 * 2.79435763233999e-05 rad. Symmetric around 0.
    assert abs(x_arr[0] - (-0.15557586)) < 1e-8
    assert abs(y_arr[0] - (-0.15557586)) < 1e-8
    assert abs(x_arr[0] + x_arr[-1]) < 1e-15
    assert abs(x_arr[5567] + x_arr[5568]) < 1e-15


@pytest.mark.unit
def test_x_y_source_values_meter_mode_1km() -> None:
    from firecube_mtg_fci_l1c._schema import VariableContext as _VC
    from firecube_mtg_fci_l1c._variables import (
        _projection_x_source,
        _projection_y_source,
    )

    ctx = _VC(
        group="data_1km",
        resolution="1km",
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
    # -5567.5 px * 2.79435763233999e-05 rad/px * 35786400 m
    assert abs(x_arr[0] - (-5567500.0)) < 1.0


@pytest.mark.unit
def test_x_y_source_values_meter_mode_500m_and_2km() -> None:
    from firecube_mtg_fci_l1c._schema import VariableContext as _VC
    from firecube_mtg_fci_l1c._variables import (
        _projection_x_source,
        _projection_y_source,
    )

    for resolution, dimsize in [("500m", 22272), ("2km", 5568)]:
        group = f"data_{resolution}"
        ctx = _VC(
            group=group,
            resolution=resolution,
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


@pytest.mark.unit
def test_time_coord_spec_matches_cf_attrs() -> None:
    specs = build_specs(MtgFciL1cConfig(), "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    time_spec = next(a for a in g.arrays if a.name == TIME_COORD_NAME)
    assert time_spec.chunks is None
    assert time_spec.dimension_names == (TIME_COORD_NAME,)
    assert time_spec.time_indexed is True
    assert time_spec.attrs is not None
    assert time_spec.attrs == {
        "standard_name": "time",
        "long_name": "observation time",
        "axis": "T",
    }


@pytest.mark.unit
def test_time_coord_spec_does_not_declare_units_or_calendar() -> None:
    specs = build_specs(MtgFciL1cConfig(), "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    time_spec = next(a for a in g.arrays if a.name == TIME_COORD_NAME)
    attrs = time_spec.attrs or {}
    assert "units" not in attrs
    assert "calendar" not in attrs


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
# Group 9: Persisted variable names and units
# ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_slope_offset_names_and_units_unchanged() -> None:
    """slope/offset keep their names; they are not CF scale_factor/add_offset."""
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
        zarr_chunk_overrides={"data_1km": (1, 556, 11136, 1)},
    )
    specs = build_specs(cfg, "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    counts = next(a for a in g.arrays if a.name == "counts")
    assert counts.chunks == (1, 556, 11136, 1)


@pytest.mark.unit
def test_chunk_override_combined_with_shard_override_full_disk() -> None:
    """Power-user recipe: shards are a whole multiple of explicit chunks."""
    cfg = MtgFciL1cConfig(
        zarr_chunk_overrides={"data_1km": (1, 556, 11136, 1)},
        zarr_shard_overrides={"data_1km": (1, 11120, 11136, 1)},
    )
    specs = build_specs(cfg, "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    counts = next(a for a in g.arrays if a.name == "counts")
    assert counts.chunks == (1, 556, 11136, 1)
    assert counts.shards == (1, 11120, 11136, 1)
    assert counts.shards[1] // counts.chunks[1] == 20
    assert counts.shards[2] // counts.chunks[2] == 1


@pytest.mark.unit
def test_chunk_override_without_shard_override() -> None:
    """User overrides chunks only; shards stay byte-budgeted using new chunks."""
    cfg = MtgFciL1cConfig(zarr_chunk_overrides={"data_1km": (1, 556, 11136, 1)})
    specs = build_specs(cfg, "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    counts = next(a for a in g.arrays if a.name == "counts")
    assert counts.chunks == (1, 556, 11136, 1)
    # byte-budget math: chunk_bytes = 1*556*11136*2 = 12,383,232; budget = 134,217,728
    # multiples = 134217728 // chunk_bytes = 10; cap = 11136 // 556 = 20
    # shard_y = max(556, min(10,20) * 556) = 5560
    assert counts.shards == (1, 5560, 11136, 1)


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
    with pytest.raises(ValueError, match="exceeds max supported"):
        MtgFciL1cConfig(zarr_chunk_overrides={"data_1km": (1, 99999, 11136, 1)})


@pytest.mark.unit
def test_default_chunk_shape_unchanged_when_no_override() -> None:
    """Without overrides, 1 km counts chunk one netCDF part of rows, full width, one channel."""
    cfg = MtgFciL1cConfig()
    specs = build_specs(cfg, "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    counts = next(a for a in g.arrays if a.name == "counts")
    assert counts.chunks == (1, 278, 11136, 1)


@pytest.mark.unit
def test_mixed_per_resolution_override() -> None:
    """Only overriding data_1km; data_2km uses defaults."""
    cfg = MtgFciL1cConfig(zarr_chunk_overrides={"data_1km": (1, 556, 11136, 1)})
    specs = build_specs(cfg, "FDHSI")
    g1km = next(g for g in specs if g.group == "data_1km")
    g2km = next(g for g in specs if g.group == "data_2km")
    counts_1km = next(a for a in g1km.arrays if a.name == "counts")
    counts_2km = next(a for a in g2km.arrays if a.name == "counts")
    assert counts_1km.chunks == (1, 556, 11136, 1)  # overridden
    assert counts_2km.chunks == (1, 139, 5568, 1)  # default 2km


@pytest.mark.unit
def test_zarr_sharding_false_ignores_shard_overrides_but_keeps_chunk_overrides() -> (
    None
):
    """zarr_sharding=False produces shards=None regardless of shard overrides,
    but chunk overrides still apply to chunk shape."""
    cfg = MtgFciL1cConfig(
        zarr_chunk_overrides={"data_1km": (1, 556, 11136, 1)},
        zarr_shard_overrides={"data_1km": (1, 11120, 11136, 1)},
    )
    cfg.template_config.zarr_sharding = False
    specs = build_specs(cfg, "FDHSI")
    g = next(g for g in specs if g.group == "data_1km")
    counts = next(a for a in g.arrays if a.name == "counts")
    assert counts.chunks == (1, 556, 11136, 1)  # chunk override still applied
    assert counts.shards is None  # sharding fully disabled


@pytest.mark.unit
def test_chunk_overrides_cli_string_parser_path() -> None:
    """from_options() correctly parses a JSON-string zarr_chunk_overrides and
    the resulting config (with list values, not tuples) still passes validation
    and produces the correct schema."""
    # NOTE: from_options parses JSON strings, yielding lists not tuples
    options = {"zarr_chunk_overrides": '{"data_1km":[1,556,11136,1]}'}
    cfg = MtgFciL1cConfig.from_options(options)
    assert cfg.zarr_chunk_overrides is not None
    assert "data_1km" in cfg.zarr_chunk_overrides
    assert tuple(cfg.zarr_chunk_overrides["data_1km"]) == (1, 556, 11136, 1)
    # End-to-end: schema build accepts list values
    specs = build_specs(cfg, "FDHSI")
    counts = next(
        a
        for g in specs
        if g.group == "data_1km"
        for a in g.arrays
        if a.name == "counts"
    )
    assert tuple(counts.chunks) == (1, 556, 11136, 1)


@pytest.mark.unit
def test_chunk_overrides_dict_of_lists_accepted() -> None:
    """Validation in __post_init__ must handle both tuples and lists without error."""
    cfg = MtgFciL1cConfig(
        zarr_chunk_overrides={"data_1km": [1, 556, 11136, 1]}  # type: ignore[dict-item]
    )
    specs = build_specs(cfg, "FDHSI")
    counts = next(
        a
        for g in specs
        if g.group == "data_1km"
        for a in g.arrays
        if a.name == "counts"
    )
    assert tuple(counts.chunks) == (1, 556, 11136, 1)


# ─────────────────────────────────────────────────────────────
# Radiance conversion constants: per-product (time, channel) variables
# ─────────────────────────────────────────────────────────────

_CONVERSION_VARIABLES = (
    "radiance_unit_conversion_coefficient",
    "radiance_to_bt_conversion_coefficient_wavenumber",
    "radiance_to_bt_conversion_coefficient_a",
    "radiance_to_bt_conversion_coefficient_b",
    "radiance_to_bt_conversion_constant_c1",
    "radiance_to_bt_conversion_constant_c2",
    "channel_effective_solar_irradiance",
)


def _conversion_values(name: str) -> np.ndarray:
    """Project one conversion variable from a mixed IR/VNIR calibration table."""
    from firecube_mtg_fci_l1c._decode import ChannelCalibration

    nan = float("nan")
    variable = next(v for v in VARIABLES if v.name == name)
    assert variable.source is not None
    # HRFI 1 km group reads logical ir_105/ir_38 from the *_hr groups; the
    # third channel has no calibration entry in this slot.
    ctx = VariableContext(
        group="data_1km",
        resolution="1km",
        product_type="HRFI",
        config=MtgFciL1cConfig(),
        dimsize=11136,
        n_channels=3,
        logical_channels=("ir_105", "ir_38", "vis_06"),
        nc_channels=("ir_105_hr", "ir_38_hr", "vis_06_hr"),
        calibration_table={
            "ir_105_hr": ChannelCalibration(
                0.0528,
                -10.77,
                radiance_unit_conversion_coefficient=0.09,
                radiance_to_bt_conversion_coefficient_wavenumber=950.5,
                radiance_to_bt_conversion_coefficient_a=0.998,
                radiance_to_bt_conversion_coefficient_b=0.36,
                radiance_to_bt_conversion_constant_c1=1.2e-05,
                radiance_to_bt_conversion_constant_c2=1.44,
                channel_effective_solar_irradiance=nan,
            ),
            "ir_38_hr": ChannelCalibration(
                0.00122,
                -0.2498,
                radiance_unit_conversion_coefficient=0.7,
                radiance_to_bt_conversion_coefficient_wavenumber=2645.0,
                radiance_to_bt_conversion_coefficient_a=0.9967,
                radiance_to_bt_conversion_coefficient_b=2.25,
                radiance_to_bt_conversion_constant_c1=1.2e-05,
                radiance_to_bt_conversion_constant_c2=1.44,
                channel_effective_solar_irradiance=15.4,
            ),
        },
    )
    data = variable.source(ctx)
    assert data is not None
    return data


@pytest.mark.unit
def test_conversion_constants_are_time_channel_arrays() -> None:
    specs = build_specs(MtgFciL1cConfig(), "FDHSI")
    for group in ("data_1km", "data_2km"):
        arrays = {a.name: a for a in next(g for g in specs if g.group == group).arrays}
        for name in _CONVERSION_VARIABLES:
            spec = arrays[name]
            assert spec.dimension_names == (TIME_COORD_NAME, "channel"), name
            assert spec.shape[1] == 8, name
            assert spec.time_indexed is True, name
            assert np.dtype(spec.dtype) == np.float32, name
            assert spec.attrs is not None and spec.attrs["units"], name


@pytest.mark.unit
def test_conversion_constants_take_each_channels_product_value() -> None:
    np.testing.assert_array_equal(
        _conversion_values("radiance_to_bt_conversion_coefficient_wavenumber"),
        np.array([950.5, 2645.0, np.nan], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _conversion_values("radiance_unit_conversion_coefficient"),
        np.array([0.09, 0.7, np.nan], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _conversion_values("radiance_to_bt_conversion_constant_c2"),
        np.array([1.44, 1.44, np.nan], dtype=np.float32),
    )


@pytest.mark.unit
def test_conversion_constants_nan_where_product_marks_not_applicable() -> None:
    # ir_105 carries no solar irradiance; ir_38 does, although it is IR.
    np.testing.assert_array_equal(
        _conversion_values("channel_effective_solar_irradiance"),
        np.array([np.nan, 15.4, np.nan], dtype=np.float32),
    )


@pytest.mark.unit
def test_conversion_constants_omitted_without_calibration() -> None:
    specs = build_specs(MtgFciL1cConfig(include_calibration=False), "FDHSI")
    for group_spec in specs:
        names = {a.name for a in group_spec.arrays}
        assert names.isdisjoint(_CONVERSION_VARIABLES), group_spec.group


@pytest.mark.unit
def test_warm_calibration_only_for_ir38_nan_elsewhere() -> None:
    from firecube_mtg_fci_l1c._decode import ChannelCalibration

    # HRFI 1 km group: logical ir_38/ir_105 read from ir_38_hr/ir_105_hr.
    # The files set warm_* to 0.0 on ir_105; the store must show NaN.
    ctx = VariableContext(
        group="data_1km",
        resolution="1km",
        product_type="HRFI",
        config=MtgFciL1cConfig(),
        dimsize=11136,
        n_channels=2,
        logical_channels=("ir_105", "ir_38"),
        nc_channels=("ir_105_hr", "ir_38_hr"),
        calibration_table={
            "ir_105_hr": ChannelCalibration(0.0528, -10.77, 0.0, 0.0),
            "ir_38_hr": ChannelCalibration(0.00122, -0.2498, 0.0242, -94.42),
        },
    )
    by_name = {v.name: v for v in VARIABLES}
    warm_slope = by_name["warm_slope"].source(ctx)  # type: ignore[misc]
    warm_offset = by_name["warm_offset"].source(ctx)  # type: ignore[misc]

    np.testing.assert_array_equal(warm_slope, np.array([np.nan, 0.0242]))
    np.testing.assert_array_equal(warm_offset, np.array([np.nan, -94.42]))
    assert by_name["warm_slope"].dims == (TIME_COORD_NAME, "channel")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("product_type", "channels", "expected_groups"),
    [
        ("FDHSI", None, {"data_2km"}),
        ("HRFI", None, {"data_1km"}),
        ("FDHSI", "vis_06,ir_105", set()),
        ("FDHSI", "vis_06,ir_38", {"data_2km"}),
    ],
)
def test_warm_calibration_declared_only_in_groups_with_ir38(
    product_type: str, channels: str | None, expected_groups: set[str]
) -> None:
    specs = build_specs(MtgFciL1cConfig(channels=channels), product_type)

    for name in ("warm_slope", "warm_offset"):
        groups = {g.group for g in specs if name in {a.name for a in g.arrays}}
        assert groups == expected_groups, name
    # slope/offset stay in every group.
    for group_spec in specs:
        names = {a.name for a in group_spec.arrays}
        assert {"slope", "offset"} <= names, group_spec.group


# ─────────────────────────────────────────────────────────────
# Group 11: body_chunks stripe stores (schema, static coordinates, identity)
# ─────────────────────────────────────────────────────────────

# Expected windows, written out from the BODY chunk row table: chunks 32-40
# start at full-disk row 8649 (1 km), 4324 (2 km) and 17299 (HRFI 500 m), and
# chunk 40 ends at the disk edge. The window starts at the output chunk that
# holds that row: 31 * 278 = 8618, 31 * 139 = 4309, 31 * 556 = 17236.
_STRIPE_32_40 = [
    # product, resolution, dimsize, chunk_y, n_channels, window start, window stop
    ("FDHSI", "1km", 11136, 278, 8, 8618, 11136),
    ("FDHSI", "2km", 5568, 139, 8, 4309, 5568),
    ("HRFI", "500m", 22272, 556, 2, 17236, 22272),
    ("HRFI", "1km", 11136, 278, 2, 8618, 11136),
]


def _group_arrays(specs: list, group: str) -> tuple[dict, dict]:
    group_spec = next(g for g in specs if g.group == group)
    return dict(group_spec.attrs or {}), {a.name: a for a in group_spec.arrays}


def _static_payloads(config: MtgFciL1cConfig, product_type: str) -> dict:
    """Resolve every static write intent the ingestor emits for *config*."""
    from firecube_mtg_fci_l1c._group_plan import resolve_group_plans
    from firecube_mtg_fci_l1c.ingestor import MtgFciL1cIngestor

    ingestor = MtgFciL1cIngestor()
    ingestor.plugin_config = config
    plans = resolve_group_plans(config, product_type)
    intents = ingestor._emit_static_intents(config, product_type, plans)
    return {(intent.group, intent.array): intent.data() for intent in intents}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("product_type", "resolution", "dimsize", "chunk_y", "n_ch", "start", "stop"),
    _STRIPE_32_40,
)
def test_stripe_arrays_span_the_window_snapped_to_the_groups_chunk_grid(
    product_type: str,
    resolution: str,
    dimsize: int,
    chunk_y: int,
    n_ch: int,
    start: int,
    stop: int,
) -> None:
    group = f"data_{resolution}"
    full_attrs, full = _group_arrays(
        build_specs(MtgFciL1cConfig(product_type=product_type), product_type), group
    )
    attrs, arrays = _group_arrays(
        build_specs(
            MtgFciL1cConfig(product_type=product_type, body_chunks=[32, 40]),
            product_type,
        ),
        group,
    )
    ny = stop - start

    assert start % chunk_y == 0
    for name in ("counts", "pixel_quality", "pixel_time"):
        array = arrays[name]
        assert array.shape == (1, ny, dimsize, n_ch), name
        # Same chunk grid as the full-disk store: chunk boundaries coincide.
        assert array.chunks == full[name].chunks == (1, chunk_y, dimsize, 1)
        # Shards: whole chunks, no taller than the window in whole chunks.
        assert array.shards is not None
        assert array.shards[1] % chunk_y == 0, name
        assert array.shards[1] <= -(-ny // chunk_y) * chunk_y, name
        assert array.shards[2:] == array.chunks[2:], name
        # Zarr itself rejects a shard that is not a whole number of chunks.
        created = zarr.create_array(
            store=zarr.storage.MemoryStore(),
            shape=array.shape,
            chunks=array.chunks,
            shards=array.shards,
            dtype=array.dtype,
            fill_value=array.fill_value,
        )
        assert created.shape == array.shape
    for name in ("latitude", "longitude"):
        assert arrays[name].shape == (ny, dimsize), name
        assert arrays[name].chunks[1] == dimsize
        assert arrays[name].chunks[0] <= ny
    assert arrays["y"].shape == arrays["y"].chunks == (ny,)
    assert arrays["x"].shape == full["x"].shape == (dimsize,)
    assert attrs["body_chunks"] == [32, 40]
    assert attrs["disk_row_start"] == start
    assert attrs["disk_row_stop"] == stop
    assert {"body_chunks", "disk_row_start", "disk_row_stop"}.isdisjoint(full_attrs)
    assert {k: v for k, v in attrs.items() if k in full_attrs} == full_attrs


@pytest.mark.unit
def test_stripe_window_start_is_floor_of_chunk_32_first_row() -> None:
    """Chunk 32 starts at 1 km row 8649; with zarr_chunk_y=200 the window starts at 8600."""
    attrs, arrays = _group_arrays(
        build_specs(
            MtgFciL1cConfig(
                product_type="FDHSI",
                resolutions="1km",
                zarr_chunk_y=200,
                body_chunks=[32, 40],
            ),
            "FDHSI",
        ),
        "data_1km",
    )
    assert attrs["disk_row_start"] == 8600
    assert attrs["disk_row_stop"] == 11136
    assert arrays["counts"].shape[1] == 2536
    assert arrays["counts"].chunks[1] == 200


@pytest.mark.unit
@pytest.mark.parametrize(
    ("product_type", "body_chunks", "expected"),
    [
        # Chunk 40 ends at the disk edge: the window stops at dimsize, not at
        # the next multiple of the chunk height (11120 + 278 > 11136).
        ("FDHSI", [40, 40], {"data_1km": (10842, 11136), "data_2km": (5421, 5568)}),
        ("HRFI", [40, 40], {"data_500m": (21684, 22272), "data_1km": (10842, 11136)}),
        # Chunk 1 starts at row 0; the 2 km chunks 1-3 end exactly on a chunk edge.
        ("FDHSI", [1, 3], {"data_1km": (0, 1112), "data_2km": (0, 417)}),
    ],
)
def test_stripe_window_at_the_disk_edges(
    product_type: str, body_chunks: list[int], expected: dict[str, tuple[int, int]]
) -> None:
    specs = build_specs(
        MtgFciL1cConfig(product_type=product_type, body_chunks=body_chunks),
        product_type,
    )
    for group, (start, stop) in expected.items():
        attrs, arrays = _group_arrays(specs, group)
        assert (attrs["disk_row_start"], attrs["disk_row_stop"]) == (start, stop)
        assert arrays["counts"].shape[1] == stop - start
        assert arrays["latitude"].shape[0] == stop - start
        assert arrays["y"].shape == (stop - start,)


@pytest.mark.unit
def test_stripe_shard_budget_is_capped_by_window_height() -> None:
    """1 km counts: the byte budget allows 21 chunks; the 2518-row window holds 9."""
    _full_attrs, full = _group_arrays(
        build_specs(MtgFciL1cConfig(product_type="FDHSI"), "FDHSI"), "data_1km"
    )
    _attrs, stripe = _group_arrays(
        build_specs(
            MtgFciL1cConfig(product_type="FDHSI", body_chunks=[32, 40]), "FDHSI"
        ),
        "data_1km",
    )
    assert full["counts"].shards == (1, 21 * 278, 11136, 1)
    assert stripe["counts"].shards == (1, 9 * 278, 11136, 1)


@pytest.mark.unit
def test_stripe_inside_the_last_partial_chunk_keeps_the_chunk_height() -> None:
    """zarr_chunk_y=400: the last 1 km output chunk is [10800, 11136), 336 rows.

    Chunk 40 lies inside it, so the window is one partial chunk. The chunk
    height stays 400 (the full-disk grid) and the shard is that one chunk.
    """
    attrs, arrays = _group_arrays(
        build_specs(
            MtgFciL1cConfig(
                product_type="FDHSI",
                resolutions="1km",
                zarr_chunk_y=400,
                body_chunks=[40, 40],
            ),
            "FDHSI",
        ),
        "data_1km",
    )
    counts = arrays["counts"]
    assert (attrs["disk_row_start"], attrs["disk_row_stop"]) == (10800, 11136)
    assert counts.shape == (1, 336, 11136, 8)
    assert counts.chunks == (1, 400, 11136, 1)
    assert counts.shards == (1, 400, 11136, 1)
    created = zarr.create_array(
        store=zarr.storage.MemoryStore(),
        shape=counts.shape,
        chunks=counts.chunks,
        shards=counts.shards,
        dtype=counts.dtype,
        fill_value=counts.fill_value,
    )
    created[0, 330:336, :, 0] = 7
    assert int(created[0, 335, 0, 0]) == 7


@pytest.mark.unit
def test_stripe_rejects_a_shard_override_taller_than_the_window() -> None:
    # chunk 556: window [8340, 11136) is 2796 rows, 6 whole chunks = 3336 rows.
    def config(shard_y: int, body_chunks: list[int] | None) -> MtgFciL1cConfig:
        return MtgFciL1cConfig(
            product_type="FDHSI",
            resolutions="1km",
            zarr_chunk_overrides={"data_1km": (1, 556, 11136, 1)},
            zarr_shard_overrides={"data_1km": (1, shard_y, 11136, 1)},
            body_chunks=body_chunks,
        )

    _attrs, arrays = _group_arrays(
        build_specs(config(3336, [32, 40]), "FDHSI"), "data_1km"
    )
    assert arrays["counts"].shape[1] == 2796
    assert arrays["counts"].shards == (1, 3336, 11136, 1)

    with pytest.raises(ValueError, match="exceeds the body_chunks stripe of 2796 rows"):
        build_specs(config(3892, [32, 40]), "FDHSI")
    # The full-disk recipe keeps working without body_chunks.
    _attrs, full = _group_arrays(build_specs(config(11120, None), "FDHSI"), "data_1km")
    assert full["counts"].shards == (1, 11120, 11136, 1)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("product_type", "resolutions", "starts"),
    [
        ("FDHSI", None, {"data_1km": 8618, "data_2km": 4309}),
        ("HRFI", None, {"data_500m": 17236, "data_1km": 8618}),
    ],
)
def test_stripe_projection_axes_are_the_full_disk_values_of_the_window(
    product_type: str, resolutions: str | None, starts: dict[str, int]
) -> None:
    common = {
        "product_type": product_type,
        "resolutions": resolutions,
        "include_geolocation": False,
    }
    full = _static_payloads(MtgFciL1cConfig(**common), product_type)
    stripe_config = MtgFciL1cConfig(**common, body_chunks=[32, 40])
    stripe = _static_payloads(stripe_config, product_type)
    specs = build_specs(stripe_config, product_type)

    for group, start in starts.items():
        _attrs, arrays = _group_arrays(specs, group)
        ny = arrays["y"].shape[0]
        y = stripe[(group, "y")]
        assert y.shape == (ny,)
        np.testing.assert_array_equal(y, full[(group, "y")][start : start + ny])
        np.testing.assert_array_equal(stripe[(group, "x")], full[(group, "x")])


@pytest.mark.unit
def test_stripe_latitude_longitude_are_the_full_disk_rows_of_the_window() -> None:
    common = {"product_type": "FDHSI", "resolutions": "2km"}
    full = _static_payloads(MtgFciL1cConfig(**common), "FDHSI")
    stripe = _static_payloads(MtgFciL1cConfig(**common, body_chunks=[32, 40]), "FDHSI")

    for name in ("latitude", "longitude"):
        window = stripe[("data_2km", name)]
        assert window.shape == (1259, 5568), name
        assert window.dtype == np.float32
        np.testing.assert_array_equal(window, full[("data_2km", name)][4309:5568])
    # The window reaches the southern limb: off-disk pixels stay NaN.
    assert np.isnan(stripe[("data_2km", "latitude")][-1, 0])


@pytest.mark.unit
def test_stripe_token_separates_store_identities() -> None:
    from types import SimpleNamespace

    from firecube_mtg_fci_l1c.ingestor import MtgFciL1cIngestor

    def index_name(**kwargs: object) -> str:
        ingestor = MtgFciL1cIngestor()
        ingestor.plugin_config = MtgFciL1cConfig(time_slots=144, **kwargs)  # type: ignore[arg-type]
        spec = ingestor.index_spec(SimpleNamespace(source="/tmp"))  # type: ignore[arg-type]
        assert spec is not None
        return spec.name

    full = index_name(product_type="FDHSI")
    stripe = index_name(product_type="FDHSI", body_chunks=[32, 40])
    other = index_name(product_type="FDHSI", body_chunks=[30, 40])
    flat = index_name(
        product_type="FDHSI", resolutions="1km", flat_store=True, body_chunks=[32, 40]
    )

    assert full == "eumetsat_repeat_cycle_v1"
    assert stripe == "eumetsat_repeat_cycle_v1_stripe_c32_40"
    assert other == "eumetsat_repeat_cycle_v1_stripe_c30_40"
    assert flat == "eumetsat_repeat_cycle_v1_fdhsi_1km_stripe_c32_40"
    assert index_name(product_type="HRFI", body_chunks=[32, 40]) == stripe


@pytest.mark.unit
def test_slice_meta_tells_stripes_apart_but_not_partial_modes() -> None:
    from types import SimpleNamespace

    from firecube_mtg_fci_l1c.ingestor import MtgFciL1cIngestor

    def meta(**kwargs: object) -> dict:
        ingestor = MtgFciL1cIngestor()
        ingestor.plugin_config = MtgFciL1cConfig(product_type="FDHSI", **kwargs)  # type: ignore[arg-type]
        values = ingestor.slice_meta(SimpleNamespace(options={}))  # type: ignore[arg-type]
        return {key: values[key] for key in ingestor.slice_meta_keys()}

    full = meta()
    stripe = meta(body_chunks=[32, 40])
    error_mode = meta(partial_chunk="error")

    assert full["body_chunks"] is None
    assert stripe["body_chunks"] == [32, 40]
    assert stripe != full
    # partial_chunk picks which rows of a partial scene are written; the slice
    # is the same, so a mode change must still meet the resume guard.
    assert error_mode == full
