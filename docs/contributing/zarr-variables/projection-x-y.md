# Projection Coordinates (x, y)

Add 1D `x` and `y` coordinate arrays that map dimension indices to geostationary projection angles or distances: the OPERA-SEVIRI cube convention. These are static per group (constant across all timestamps).

## Background

The FCI L1C product stores per-nc_part packed `int16` arrays at `data/<channel>/measured/x` and `data/<channel>/measured/y` with `scale_factor` and `add_offset` attributes that decode to radians under the standard CF projection convention:

- `x` (shape `(dimsize,)`): same across all nc_parts and all channels at one resolution; decodes to `projection_x_angular_coordinate` (azimuth angle from sub-satellite point)
- `y` (shape `(rows_per_nc_part,)`): different rows per nc_part; assemble across all nc_parts to get the full-disk `(dimsize,)` coord; decodes to `projection_y_angular_coordinate` (elevation angle)

Nominal 1km decoding constants:

- `scale_factor`: `±2.79435763233999e-05` rad/index (negative for x, positive for y)
- `add_offset`: `±0.1556038047568524` rad

Multiply by satellite altitude (35,786,400 m for MTG) to convert radians to ground-projected metres.

## How (two cases)

These coordinates are static per group, so the source function lives in the static phase and runs once per group.

### Case A: Computed analytically from constants

> **Status: IMPLEMENTED in FCI L1C plugin**: see `_projection_x_source` / `_projection_y_source` in `schema.py`
> and `FCI_PROJ_SCALE_RAD_PER_INDEX` / `FCI_PROJ_OFFSET_RAD` in `_constants.py`.

If you know the projection geometry (FCI uses fixed sampling), you can compute the coords without touching the NetCDF:

```python
# In schema.py: radians version
import numpy as np

_FCI_PROJ_SCALE: dict[str, float] = {
    "500m": 1.39717881617e-05,
    "1km":  2.79435763233999e-05,
    "2km":  5.58871526468e-05,
}
_FCI_PROJ_OFFSET: dict[str, float] = {
    "500m": 0.1556038047568524,
    "1km":  0.1556038047568524,
    "2km":  0.1556038047568524,
}


def _projection_x_radians_source(ctx: VariableContext) -> np.ndarray | None:
    res = ctx.group.removeprefix("data_")
    if res not in _FCI_PROJ_SCALE:
        return None
    scale = -_FCI_PROJ_SCALE[res]   # x scale is negative (column 0 is east-of-nadir)
    offset = _FCI_PROJ_OFFSET[res]
    return np.arange(ctx.dimsize, dtype=np.float64) * scale + offset


Variable(
    name="x",
    dims=("x",),
    dtype=np.float64,
    fill_value=np.nan,
    attrs={
        "units": "radian",
        "standard_name": "projection_x_angular_coordinate",
        "long_name": "MTG geostationary projection x angle",
        "axis": "X",
    },
    source=_projection_x_radians_source,
),
```

Mirror for `y` with the positive scale and inverted offset sign:

```python
def _projection_y_radians_source(ctx: VariableContext) -> np.ndarray | None:
    res = ctx.group.removeprefix("data_")
    if res not in _FCI_PROJ_SCALE:
        return None
    scale = _FCI_PROJ_SCALE[res]    # y scale is positive
    offset = -_FCI_PROJ_OFFSET[res]
    return np.arange(ctx.dimsize, dtype=np.float64) * scale + offset


Variable(
    name="y",
    dims=("y",),
    dtype=np.float64,
    fill_value=np.nan,
    attrs={
        "units": "radian",
        "standard_name": "projection_y_angular_coordinate",
        "long_name": "MTG geostationary projection y angle",
        "axis": "Y",
    },
    source=_projection_y_radians_source,
),
```

### Case B: Decoded from NetCDF (data-driven)

If you need the actual packed values from the NetCDF (in case calibration drifts or you want exact byte-level fidelity), extend the static phase to read x/y from the first nc_part of each batch. This requires architectural changes since the static phase currently has no nc_part access:

1. **`src/firecube_mtg_fci_l1c/_streaming.py`**: add `NCPartReader.read_projection_x()` and `read_projection_y()` that return `(int16_array, scale_factor, add_offset)` from `data/<first_channel>/measured/x` and `y`.
2. **`src/firecube_mtg_fci_l1c/ingestor.py`**: during the first nc_part read in `build_write_intents`, capture x/y for each plan and stash on a new `VariableContext` field (e.g., `projection_x_decoded: np.ndarray | None`). For `y`, accumulate across nc_parts using `read_row_range()` to know each part's row offsets.
3. **`src/firecube_mtg_fci_l1c/_variable.py`**: add the new ctx fields.
4. **`src/firecube_mtg_fci_l1c/schema.py`**: add pure-projection source functions.

```python
# In schema.py: Case B source (after the extensions above)
def _projection_x_radians_source(ctx: VariableContext) -> np.ndarray | None:
    return ctx.projection_x_decoded if ctx.projection_x_decoded is not None else None
```

## Metres variant

Multiply by satellite altitude for ground-projected metres (useful for xarray + cartopy plots):

```python
_FCI_SATELLITE_ALTITUDE_M = 35_786_400.0


def _projection_x_metres_source(ctx: VariableContext) -> np.ndarray | None:
    res = ctx.group.removeprefix("data_")
    if res not in _FCI_PROJ_SCALE:
        return None
    scale = -_FCI_PROJ_SCALE[res]
    offset = _FCI_PROJ_OFFSET[res]
    radians = np.arange(ctx.dimsize, dtype=np.float64) * scale + offset
    return radians * _FCI_SATELLITE_ALTITUDE_M


Variable(
    name="x_meters",
    dims=("x",),
    dtype=np.float64,
    fill_value=np.nan,
    attrs={
        "units": "m",
        "standard_name": "projection_x_coordinate",
        "long_name": "MTG geostationary projection x distance",
        "axis": "X",
    },
    source=_projection_x_metres_source,
),
```

Use the angular version (`radian`, `projection_x_angular_coordinate`) when you want CF compliance for the canonical FCI projection. Use the metres version when downstream tools (cartopy, xESMF) expect projected linear coordinates.

## Files touched

| Case | Files |
|---|---|
| A (analytical) | `src/firecube_mtg_fci_l1c/schema.py` |
| B (NetCDF-decoded) | `_streaming.py` + `_variable.py` + `ingestor.py` + `schema.py` |

## Common mistakes

- Using `dims=("y", "x")`. Those are 2D fields (lat/lon). Projection coords are 1D: `("x",)` or `("y",)`.
- Forgetting the `axis` attribute. CF requires `axis="X"` or `axis="Y"` for projection coords so tools like xarray and cartopy recognise them.
- Mixing radian and metre units in the same array. Pick one per Variable and label it correctly in `attrs["units"]`.
- Sign-flipping x scale. FCI's x scale factor is negative (column 0 is east-of-nadir, increasing column index moves west). Match the source NetCDF or your column-major decoding will be mirrored.

## See also

- [How to add a Zarr variable](../add-zarr-variable.md)
- [Static 2D Grid (y, x)](xy-coordinates.md): for `latitude`/`longitude`
- [CF Conventions §5.6: Horizontal Coordinate Reference Systems, Grids, and Projections](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#grid-mappings-and-projections)
