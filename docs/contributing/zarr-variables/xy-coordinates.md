# Static 2D Grid (y, x)

Add a new static 2D array covering the group's grid (the full detector grid, or a stripe's rows), such as `latitude` or `longitude`. The ingestor writes these once per group during the static phase.

## How

The existing `latitude` and `longitude` variables show the full pattern. Both live in `src/firecube_mtg_fci_l1c/_variables.py`.

### Source function

```python
def _lat_lon_pair(ctx: VariableContext) -> tuple[np.ndarray, np.ndarray] | None:
    """The group's (lat, lon) rows: the stripe's window, or the full disk."""
    if ctx.geo_provider is None:
        return None
    res_m = ctx.geo_provider.resolution_m(ctx.resolution)
    if res_m is None:
        return None
    if ctx.y_window is None:
        return ctx.geo_provider.get_lat_lon(ctx.config.fci_grids_file, res_m)
    return ctx.geo_provider.get_lat_lon(
        ctx.config.fci_grids_file, res_m, rows=ctx.y_window
    )


def _latitude_source(ctx: VariableContext) -> np.ndarray | None:
    pair = _lat_lon_pair(ctx)
    return None if pair is None else pair[0]
```

`ctx.geo_provider` is populated during the static phase. Call `resolution_m(ctx.resolution)` to map a bare resolution string (`"1km"`, `"2km"`, `"500m"`) to metres, then `get_lat_lon(grids_file, res_m)` to get the full-disk grid. A group of a stripe store (`body_chunks`) holds only the rows of `ctx.y_window`, so pass `rows=ctx.y_window` to get just those rows: the provider then caches only that slice. The provider caches the result on `(grids_file, resolution_m, rows)` so repeated calls within a run are free. Return `None` to skip writing.

### Variable entry

```python
Variable(
    name="latitude",
    dims=("y", "x"),
    dtype=np.float32,
    fill_value=np.float32(np.nan),
    attrs={
        "units": "degrees_north",
        "standard_name": "latitude",
        "long_name": "latitude",
    },
    source=_latitude_source,
    enabled_by="include_geolocation",
),
```

`enabled_by="include_geolocation"` gates the variable on the `MtgFciL1cConfig.include_geolocation` flag. Omit `enabled_by` if the variable should always be written.

## Adding your own static 2D grid

Follow the same two-step pattern. The example below adds `solar_zenith_angle`: a per-pixel solar geometry field.

FCI L1C does not include per-pixel solar zenith angles in the input; they would come from external solar geometry computation (e.g., `pyorbital.astronomy.sun_zenith_angle`). The extension pattern is: add the computation to the geolocation subpackage (mirroring `compute_latlon`), expose it via a `LatLonProvider` method, then add the projection source in `_variables.py`. The code below shows the schema-side shape. The provider method `get_solar_zenith_angle` does not exist yet: the recipe describes what it would look like once you add it.

```python
def _solar_zenith_angle_source(ctx: VariableContext) -> np.ndarray | None:
    if ctx.geo_provider is None:
        return None
    res_m = ctx.geo_provider.resolution_m(ctx.resolution)
    if res_m is None:
        return None
    # get_solar_zenith_angle is a user-added provider method.
    # Add it to geolocation/provider.py mirroring the get_lat_lon caching pattern.
    return ctx.geo_provider.get_solar_zenith_angle(
        ctx.config.fci_grids_file, res_m, ctx.timestamp
    )


Variable(
    name="solar_zenith_angle",
    dims=("y", "x"),
    dtype=np.float32,
    fill_value=np.float32(np.nan),
    attrs={
        "units": "degree",
        "standard_name": "solar_zenith_angle",
        "long_name": "solar zenith angle",
    },
    source=_solar_zenith_angle_source,
),
```

If `get_solar_zenith_angle` is a new provider method, add it to `src/firecube_mtg_fci_l1c/geolocation/provider.py` first, mirroring the `get_lat_lon` caching pattern.

## Files touched

| File | Change |
|------|--------|
| `src/firecube_mtg_fci_l1c/_variables.py` | Add source function + append `Variable(...)` to `VARIABLES` |
| `src/firecube_mtg_fci_l1c/geolocation/provider.py` | Only if the source needs a new provider method (e.g. `get_solar_zenith_angle`) |

## Common mistakes

- Calling `get_lat_lon(res_m)` with a single argument. The signature is `get_lat_lon(grids_file, resolution_m, rows=None)`: pass `ctx.config.fci_grids_file` as the first argument, and `rows=ctx.y_window` when the group may be a stripe.
- Defining a local `_resolution_m` helper in `_variables.py`. Use the method on the provider: `ctx.geo_provider.resolution_m(ctx.resolution)`.
- Returning pixel indices instead of projected or geographic coordinates. The source must return real-world values.
- Defining the source as a lambda. It must be module-level for pickle-safety.
- Computing static `(y, x)` arrays at every timestamp. Static arrays are written ONCE per group during the static phase. If your data varies per timestamp, you need a 4D spatial field: see [spatial-field.md](spatial-field.md).

## See also

[How to add a Zarr variable](../add-zarr-variable.md)
