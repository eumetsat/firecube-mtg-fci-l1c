# Channel Name (channel)

The `channel_name` variable maps channel index `0..n-1` back to logical channel names like `vis_04` or `ir_105`. This is the worked example for adding a static 1D coordinate.

## Implementation

Both pieces live in `src/firecube_mtg_fci_l1c/schema.py`.

### Source function

```python
def _channel_name_source(ctx: VariableContext) -> np.ndarray:
    return np.asarray(ctx.logical_channels, dtype="S16")
```

`ctx.logical_channels` is a tuple of strings populated during the static phase. The source converts them to a fixed-width byte string array.

### Variable entry

```python
Variable(
    name="channel_name",
    dims=("channel",),
    dtype="S16",
    fill_value=b"",
    attrs={"long_name": "FCI logical channel name"},
    source=_channel_name_source,
),
```

This is already in `VARIABLES`. The recipe below shows how to add a similar static 1D coordinate.

## Adding your own static 1D coordinate

Follow the same two-step pattern. The example below adds `channel_central_wavelength`: a real per-channel constant from the FCI instrument spec, stored as a lookup table. This is the canonical way to add static `(channel,)` data when no per-nc_part read is needed (the static phase currently has no nc_part access).

```python
# Approximate central wavelengths for FCI channels (micrometres),
# per the MTG FCI instrument spec.  Replace with values from your
# preferred reference if you need higher precision.
_FCI_CENTRAL_WAVELENGTH_UM: dict[str, float] = {
    "vis_04": 0.444, "vis_05": 0.510, "vis_06": 0.640, "vis_08": 0.865,
    "vis_09": 0.914, "nir_13": 1.380, "nir_16": 1.610, "nir_22": 2.250,
    "ir_38":  3.800, "wv_63":  6.300, "wv_73":  7.350, "ir_87":  8.700,
    "ir_97":  9.660, "ir_105": 10.500, "ir_123": 12.300, "ir_133": 13.300,
}


def _channel_central_wavelength_source(ctx: VariableContext) -> np.ndarray | None:
    return np.array(
        [_FCI_CENTRAL_WAVELENGTH_UM[ch] for ch in ctx.logical_channels],
        dtype=np.float64,
    )


Variable(
    name="channel_central_wavelength",
    dims=("channel",),
    dtype=np.float64,
    fill_value=np.nan,
    attrs={
        "units": "um",
        "standard_name": "sensor_band_central_radiation_wavelength",
        "long_name": "FCI channel central wavelength",
    },
    source=_channel_central_wavelength_source,
),
```

## Files touched

| File | Change |
|------|--------|
| `src/firecube_mtg_fci_l1c/schema.py` | Add source function + append `Variable(...)` to `VARIABLES` |

## Common mistakes

- Writing NetCDF internal channel names like `vis_06_hr`. Use logical names (`vis_06`) from `ctx.logical_channels`.
- Defining the source as a lambda. It must be module-level for pickle-safety.
- Using `dims=("time", "channel")`. Channel names and static per-channel constants are static; use `dims=("channel",)`.
- Putting `(channel,)` data that varies per acquisition. If the value changes from one scan to the next (like calibration slope), use `dims=("time", "channel")` instead: see [time-channel-variable.md](time-channel-variable.md).

## See also

[How to add a Zarr variable](../add-zarr-variable.md)
