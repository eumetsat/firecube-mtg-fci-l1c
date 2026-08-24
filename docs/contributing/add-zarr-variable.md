# How to Add a Zarr Variable

All variables are declared in `src/firecube_mtg_fci_l1c/_variables.py`. No other file needs editing.

## The Recipe

1. Define a module-level `source` function.
2. Append a `Variable(...)` entry to `VARIABLES`.

That's it. The ingestor's generic phase emitters dispatch by `dims` shape, so the right write path is chosen automatically.

## Decision Table

| What you want | `dims` | Recipe |
|---|---|---|
| Edit CF attrs on an existing variable | n/a | [cf-attributes.md](zarr-variables/cf-attributes.md) |
| New 4D spatial field | `("time", "y", "x", "channel")` | [spatial-field.md](zarr-variables/spatial-field.md) |
| New 2D time-channel scalar | `("time", "channel")` | [time-channel-variable.md](zarr-variables/time-channel-variable.md) |
| New 1D static channel coord | `("channel",)` | [channel-name.md](zarr-variables/channel-name.md) |
| New 2D static grid | `("y", "x")` | [xy-coordinates.md](zarr-variables/xy-coordinates.md) |
| New 1D projection coord | `("x",)` or `("y",)` | [projection-x-y.md](zarr-variables/projection-x-y.md) |
| Scalar attrs-only container | `()` | Append `Variable(name=..., dims=(), source=None, attrs={...})` |

## Key Rules

**One file.** `_variables.py` is the single edit target. Don't touch `ingestor.py` for schema or attrs changes.

**No lambdas.** Source functions must be module-level so `VARIABLES` stays picklable for `ProcessPoolExecutor` workers.

**Use `enabled_by`.** Set `enabled_by="include_pixel_quality"` (or any `MtgFciL1cConfig` flag name) to gate a variable on a config option. Omit it for variables that are always written.

**`source=None`.** For attrs-only variables like `spatial_ref`, set `source=None`. The ingestor writes the array with fill values and attaches the attrs.

## VariableContext Fields

The source function receives a `VariableContext`. Different fields are populated for different phases: a source function only reads the fields relevant to its phase:

| Phase | Populated fields |
|---|---|
| Static | `group`, `product_type`, `config`, `dimsize`, `n_channels`, `logical_channels`, `geo_provider` |
| Timestamp | `group`, `product_type`, `config`, `dimsize`, `n_channels`, `logical_channels`, `timestamp` |
| Time-channel | `group`, `product_type`, `config`, `dimsize`, `n_channels`, `logical_channels`, `nc_channels`, `calibration_table` |
| Spatial | `group`, `product_type`, `config`, `dimsize`, `n_channels`, `logical_channels`, `nc_channels`, `y_slice`, `channel_payload` |

**Source functions are pure projections**: they never perform I/O. All NetCDF reads happen in the ingestor's `build_write_intents` (which pre-loads `ChannelSlicePayload`s and aggregates `calibration_table`s). Source functions only project from the pre-loaded data carried on the context.

Return `None` from a source to skip writing for that call.

## Validation

Run the narrowest tests first:

```bash
uv run ruff check src/
uv run pytest tests/test_schema.py -q
```

For changes that write data:

```bash
uv run pytest tests/test_integration.py -q
```

Before review:

```bash
uv run pytest -q
```
