# Time-Channel Variable (time, channel)

Add a new 2D variable with one scalar value per acquisition and channel. The existing `slope` and `offset` calibration arrays follow this pattern.

## How (two cases)

Time-channel source functions are **pure projections** from per-channel aggregated data on `ctx`. The current canonical example is `_slope_source` reading from `ctx.calibration_table`.

### Case A: Derived from existing aggregated data

If your value is derived from `ctx.calibration_table` (the slope/offset dict), the only edit is to `src/firecube_mtg_fci_l1c/_variables.py`.

The example below computes a per-channel SNR estimate as `slope / noise_floor`, where `noise_floor` is a hardcoded per-channel constant from the FCI instrument spec. This is illustrative: production would source noise floor values from external calibration files.

```python
# Approximate noise floor (mW m-2 sr-1 (cm-1)-1) per FCI channel.
# Replace with values from your preferred calibration reference.
_FCI_NOISE_FLOOR: dict[str, float] = {
    "vis_04": 0.05, "vis_05": 0.05, "vis_06": 0.04, "vis_08": 0.04,
    "vis_09": 0.04, "nir_13": 0.03, "nir_16": 0.03, "nir_22": 0.03,
    "ir_38":  0.10, "wv_63":  0.08, "wv_73":  0.08, "ir_87":  0.07,
    "ir_97":  0.07, "ir_105": 0.07, "ir_123": 0.07, "ir_133": 0.07,
}


def _channel_snr_estimate_source(ctx: VariableContext) -> np.ndarray | None:
    if ctx.calibration_table is None:
        return None
    vec = np.full(ctx.n_channels, np.nan, dtype=np.float64)
    found = False
    for i, ch in enumerate(ctx.nc_channels):
        cal = ctx.calibration_table.get(ch)
        noise = _FCI_NOISE_FLOOR.get(ch)
        if cal is not None and noise and noise > 0:
            vec[i] = cal[0] / noise
            found = True
    return vec if found else None


Variable(
    name="channel_snr_estimate",
    dims=("time", "channel"),
    dtype=np.float64,
    fill_value=np.nan,
    attrs={"units": "1", "long_name": "FCI channel SNR estimate (slope / noise floor)"},
    source=_channel_snr_estimate_source,
),
```

The function returns a 1D array of length `ctx.n_channels`, or `None` to skip.

### Case B: Requires new per-channel data aggregation

If you need a value that isn't already aggregated, the schema source remains pure but you must also extend the ingestor. The example below adds `noise_warning_count`: the total number of noise-warning pixels per channel per acquisition, read from `data/<channel>/quality_channel/number_of_noise_warning_pixels` in the NetCDF.

**1. `src/firecube_mtg_fci_l1c/_decode.py`**: add a reader method on `NCPartReader`:

```python
def read_noise_warning_count(self, channel: str) -> int | None:
    """Return the noise-warning pixel count for one channel, or None if absent.

    Reads data/<channel>/quality_channel/number_of_noise_warning_pixels
    (uint32 scalar per nc_part).
    """
    ds = self._open()
    if "data" not in ds.groups or channel not in ds["data"].groups:
        return None
    ch_group = ds[f"data/{channel}"]
    if "quality_channel" not in ch_group.groups:
        return None
    quality = ch_group["quality_channel"]
    if "number_of_noise_warning_pixels" not in quality.variables:
        return None
    return int(np.asarray(quality["number_of_noise_warning_pixels"][...]).item())
```

**2. `src/firecube_mtg_fci_l1c/_schema.py`**: add a new field on `VariableContext`:

```python
@dataclasses.dataclass(frozen=True)
class VariableContext:
    ...
    noise_warning_table: dict[str, int] | None = None   # NEW
```

**3. `src/firecube_mtg_fci_l1c/ingestor.py`**: in `build_write_intents`, add an aggregation block that sums across nc_parts per channel (mirror the existing `calibration_table` block, but SUM not first-wins), then pass it to `_emit_time_channel_intents`:

```python
# After the calibration_table block, inside `_intents_for_plan`:
noise_warning_table: dict[str, int] = {}
for part_idx, part_path in enumerate(nc_parts):
    if part_idx not in row_ranges:
        continue
    for ch in plan.nc_channels:
        count = shared_reader.reader_for(part_path).read_noise_warning_count(ch)
        if count is not None:
            noise_warning_table[ch] = noise_warning_table.get(ch, 0) + count

# Pass to the emitter:
intents.extend(
    self._emit_time_channel_intents(
        config, product_type, res, logical_channels, timestamp,
        calibration_table,
        nc_channels=nc_channels,
        noise_warning_table=noise_warning_table or None,
    )
)
```

Two things to copy from the surrounding code rather than invent:

- Read through `shared_reader`, never `NCPartReader(part_path)` directly. The
  shared reader keeps one open handle per nc_part for the whole batch; opening
  your own reintroduces a file open per phase. Use `reader_for()` for a raw
  reader, or add a passthrough on `SharedNcPartReader` if your access pattern
  deserves a named method (as `read_row_range` and `decode_channel` do).
- Pass `timestamp`, not a slot index. Core resolves the coordinate to a slot
  when it compiles the `IndexedWrite`; the plugin no longer computes one.

Update `_emit_time_channel_intents` to accept and forward the new field when constructing `VariableContext`.

**4. `src/firecube_mtg_fci_l1c/_variables.py`**: add the pure-projection source:

```python
def _noise_warning_count_source(ctx: VariableContext) -> np.ndarray | None:
    if ctx.noise_warning_table is None:
        return None
    return np.array(
        [ctx.noise_warning_table.get(ch, 0) for ch in ctx.nc_channels],
        dtype=np.uint32,
    )


Variable(
    name="noise_warning_count",
    dims=("time", "channel"),
    dtype=np.uint32,
    fill_value=np.iinfo(np.uint32).max,
    attrs={
        "units": "pixel",
        "long_name": "Number of noise-warning pixels per channel",
    },
    source=_noise_warning_count_source,
),
```

The NetCDF path is `data/<channel>/quality_channel/number_of_noise_warning_pixels`. The same six `number_of_*_pixels` variants in `quality_channel/` follow the identical pattern.

**Static scalars use first-wins, not sum.** The same Case B template works for per-channel static scalars like `channel_effective_solar_irradiance` (path: `data/<channel>/measured/channel_effective_solar_irradiance`). The only difference is the aggregation semantics: static scalars are identical across all nc_parts, so use first-wins (skip if already in the table) rather than summing. The `calibration_table` block in `ingestor.py` already shows this pattern: copy it directly.

## Files touched

| Case | Files |
|---|---|
| A (derived) | `src/firecube_mtg_fci_l1c/_variables.py` |
| B (new data) | `_decode.py` + `_schema.py` + `ingestor.py` + `_variables.py` |

## Shape rule

Always declare `("time", "channel")`, not `("channel", "time")`. If the source product stores data in `(channel, time)` order, transpose before returning.

## Common mistakes

- Calling `ctx.reader.read_*` inside the source. `ctx.reader` does not exist: pre-aggregate in `build_write_intents` and project here.
- Using `ctx.nc_channels or ctx.logical_channels`. The fallback was removed; `ctx.nc_channels` is always populated for time-channel emits. Iterate over `ctx.nc_channels` directly.
- Declaring `dims=("channel", "time")`. The ingestor expects time first.
- Defining the source as a lambda or nested function. It must be module-level for pickle-safety.
- Returning a 2D array from the source. Return a 1D vector of length `ctx.n_channels`.
- Summing quality counts when you want static scalars. Use first-wins for values that are constant across nc_parts (calibration coefficients, solar irradiance). Use sum for counts that accumulate across nc_parts (noise warnings, saturation warnings).

## See also

[How to add a Zarr variable](../add-zarr-variable.md)
