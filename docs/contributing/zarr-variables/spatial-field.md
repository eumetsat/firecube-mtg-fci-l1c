# Spatial Field (time, y, x, channel)

Add a new 4D per-acquisition array, such as a quality mask or derived per-pixel field. The ingestor dispatches writes for any `Variable` with `dims=("time", "y", "x", "channel")` automatically.

## How (two cases)

Spatial source functions are **pure projections** from `ctx.channel_payload`. They never call `ctx.reader.*` or perform I/O: the ingestor pre-loads the channel slice once per `(nc_part, channel)` and hands the payload to the source.

### Case A: Derived from existing payload fields

If your new field is computed from the already-loaded `counts`, `pixel_quality`, or `pixel_time` arrays, the only edit is to `src/firecube_mtg_fci_l1c/schema.py`.

The `pixel_quality` field is an 8-bit flag array. Each bit has a defined meaning per the FCI L1C pixel quality specification (see the [`pixel_quality` bit table](../../fci-data-in-zarr.md#pixel_quality-bits)):

| Bit | Flag |
|-----|------|
| 0 | missing_warning |
| 1 | radiometric_warning |
| 2 | noise_warning |
| 3 | geolocation_warning |
| 4 | saturation_warning |
| 5 | straylight_correction_warning |
| 6 | extended_dynamic_range_warning |
| 7 | encoding_saturation_warning |

**Saturation mask (bit 4):**

```python
def _saturation_mask_source(ctx: VariableContext) -> np.ndarray | None:
    if ctx.channel_payload is None:
        return None
    # Bit 4 of pixel_quality is the saturation flag (FCI L1C pixel quality flags)
    return (ctx.channel_payload.pixel_quality & 0b00010000).astype(np.uint8)


Variable(
    name="saturation_mask",
    dims=("time", "y", "x", "channel"),
    dtype=np.uint8,
    fill_value=0,
    attrs={"units": "1", "long_name": "FCI saturation mask",
           "grid_mapping": "spatial_ref"},
    source=_saturation_mask_source,
),
```

The same pattern generalises to any bit. **Extended dynamic range mask (bit 6):**

```python
def _extended_dynamic_range_mask_source(ctx: VariableContext) -> np.ndarray | None:
    if ctx.channel_payload is None:
        return None
    # Bit 6 of pixel_quality is the extended_dynamic_range_warning flag
    return (ctx.channel_payload.pixel_quality & 0b01000000).astype(np.uint8)


Variable(
    name="extended_dynamic_range_mask",
    dims=("time", "y", "x", "channel"),
    dtype=np.uint8,
    fill_value=0,
    attrs={"units": "1", "long_name": "FCI extended dynamic range mask",
           "grid_mapping": "spatial_ref"},
    source=_extended_dynamic_range_mask_source,
),
```

Both functions return a 2D `(y_rows, x_cols)` array for the current `(nc_part, channel)` slice.

### Case B: Requires new NetCDF data

The FCI L1C product does not currently include cloud masks or geometry arrays (solar/satellite zenith angles) per pixel: those would come from L2 products. If you have an external per-pixel field at the same `(y, x)` resolution, the extension pattern is identical: add a reader method on `NCPartReader` (or your own loader), add the field to `ChannelSlicePayload`, populate it in `load_channel_slice()`, then add the projection source in `schema.py`.

The code below shows the schema-side shape. If you had a `read_external_field(channel)` method on `NCPartReader`, this is what the full extension looks like:

```python
# In _streaming.py
@dataclasses.dataclass(frozen=True)
class ChannelSlicePayload:
    counts: np.ndarray
    pixel_quality: np.ndarray
    pixel_time: np.ndarray | None
    external_field: np.ndarray | None   # NEW: user-added

# In schema.py
def _external_field_source(ctx: VariableContext) -> np.ndarray | None:
    return ctx.channel_payload.external_field if ctx.channel_payload else None
```

The source stays a one-liner. All I/O lives in `_streaming.py`.

## Files touched

| Case | Files |
|---|---|
| A (derived) | `src/firecube_mtg_fci_l1c/schema.py` |
| B (new data) | `src/firecube_mtg_fci_l1c/_streaming.py` + `src/firecube_mtg_fci_l1c/schema.py` |

## Common mistakes

- Calling `ctx.reader.read_*` inside a source function. `ctx.reader` does not exist: source functions are pure projections, not I/O.
- Returning a full 4D array from the source. Return a 2D `(y_rows, x_cols)` slice; the ingestor handles the time and channel dimensions.
- Defining the source as a lambda or nested function. It must be module-level so `VARIABLES` stays picklable for `ProcessPoolExecutor` workers.
- Checking `ctx.geo_provider` in a spatial source. That field is only populated during the static phase; use `ctx.channel_payload` here.

## See also

[How to add a Zarr variable](../add-zarr-variable.md)
