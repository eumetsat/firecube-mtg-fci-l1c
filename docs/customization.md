# Customizing `mtg_fci_l1c`

Operator-facing plugin and production-script settings.

- [Plugin `--option` flags](#plugin---option-flags)
- [Projection units](#projection-units)
- [Time-axis options (`time_epoch`, `time_slots`)](#time-axis-options)
- [`scripts/fci-ingest.sh` environment variables](#scriptsfci-ingestsh-environment-variables)
- [Chunk and shard layout](#chunk-and-shard-layout)
- [Compression options](#compression-options)
- [Geolocation grids workflow](#geolocation-grids-workflow)
- [Fix FillValue (legacy stores)](#fix-fillvalue-legacy-stores)

---

## Firecube template options

These options come from Firecube's template config, not the plugin layer.

| Option | Default | Description |
|---|---|---|
| `zarr_sharding` | `true` | Enable Zarr v3 sharding of the 4-D data arrays |
| `zarr_compression` | `true` | Compress arrays with `ZstdCodec(level=0)`. Set `false` for uncompressed output. |
| `zarr_codecs` | `null` | Custom codec pipeline as a JSON list. Overrides `zarr_compression` when set. |

## Plugin `--option` flags

Pass with `--option key=value` to `firecube ingest` and `firecube zarr preallocate`.
All are optional.

| Option | Default | Description |
|---|---|---|
| `resolutions` | auto per product | Comma-separated list, e.g. `1km` or `500m,1km` |
| `channels` | all channels for selected resolutions | Comma-separated logical channel names, e.g. `vis_06,ir_105` |
| `product_type` | auto-detect | Override: `FDHSI` or `HRFI` |
| `include_pixel_quality` | `true` | Include the 8-bit warning flag array |
| `include_pixel_time` | `true` | Include per-pixel observation timestamps |
| `include_calibration` | `true` | Include `slope` and `offset` arrays |
| `include_geolocation` | `true` | Include static `latitude` / `longitude` arrays |
| `fci_grids_file` | `null` | Path to pre-generated `.npz` grids (see [Geolocation grids workflow](#geolocation-grids-workflow)) |
| `pixel_time_dtype` | `float64` | `float64`, `float32`, `int32`, or `int64`; `float32` halves storage but loses sub-minute absolute-epoch precision |
| `scratch_dir` | `null` | Base directory for temporary ZIP extraction (uses system temp when unset) |
| `zarr_chunk_y` | `null` | Y-dimension chunk size for Zarr arrays (defaults to nc_part-aligned) |
| `zarr_shard_target_bytes` | `134217728` (128 MiB) | Target bytes per shard for the default policy |
| `zarr_shard_overrides` | `null` | Explicit per-group `(time, y, x, channel)` shard shapes, e.g. `{"data_1km": [1, 2784, 11136, 1]}` |
| `zarr_chunk_overrides` | `null` | Explicit per-group chunk shapes; takes precedence over `zarr_chunk_y` |
| `projection_units` | `meter` | Units for the `x` and `y` projection coordinate arrays. Valid values: `meter` (default), `metre` (alias for `meter`), `radian`. See [Projection units](#projection-units). |

```bash
# Process only 1 km data, drop pixel_time
firecube ingest mtg_fci_l1c \
    --input-data /path/to/fci-zips \
    --target file:///path/to/output.zarr \
    --output-format zarr --write-mode staged \
    --option resolutions=1km \
    --option include_pixel_time=false
```

```bash
# Keep only selected channels across default resolutions
firecube ingest mtg_fci_l1c \
    --input-data /path/to/fci-zips \
    --target file:///path/to/output.zarr \
    --output-format zarr --write-mode staged \
    --option channels=vis_06,ir_105
```

---

## Projection units

The `projection_units` option controls the units written to the `x` and `y`
coordinate arrays.

| Value | `standard_name` | `units` | When to use |
|---|---|---|---|
| `meter` (default) | `projection_x_coordinate` | `m` | Works out of the box with rioxarray, cartopy, GDAL, and satpy |
| `metre` | `projection_x_coordinate` | `m` | Alias for `meter`; identical output |
| `radian` | `projection_x_angular_coordinate` | `radian` | Native unit from the source netCDF; use when working directly with satellite geometry |

> **Warning:** changing `projection_units` between ingests to the same store
> raises `SchemaDriftError`. Choose a value once per store, before
> preallocation, and keep it consistent across all pods.

```bash
# Change to radian coordinates 
firecube ingest mtg_fci_l1c \
    --input-data /path/to/fci-zips \
    --target file:///path/to/output.zarr \
    --output-format zarr --write-mode staged \
    --option projection_units=radian
```

---

## Time-axis options

`firecube zarr preallocate` and every `firecube ingest` writing to the same
store share two options that fix the Zarr time axis:

| Option | Meaning |
|---|---|
| `time_epoch=YYYY-MM-DD` | UTC-midnight date that maps to slot 0. For a full MTG FCI L1C cube, use `2024-09-24`, the first FCI L1C availability date in the EUMETSAT Data Store. It cannot be shifted later without re-preallocating from scratch. |
| `time_slots=N` | Total length of the time axis, in slots. Choose `N = 144 × <days>` to cover the full intended range up-front (`144` = 1 day, `1008` = 1 week, `4320` = 30 days). For a full cube, calculate `N` from the desired horizon before preallocation; a long sparse axis, such as several years, is valid if needed. The axis shape is fixed at preallocation time; ingest into subsets of it over time, but do not expect to grow it after data has been written. |

All pods writing to the same store **must** use identical `time_epoch` and
`time_slots` values.

`scripts/fci-ingest.sh` exposes these as `TIME_EPOCH` and `TIME_SLOTS`.

---

## `scripts/fci-ingest.sh` environment variables

### Data source and target

| Variable | Default | Description |
|---|---|---|
| `INPUT` | `/data/fci-zips` | Directory or URI of FCI L1C `.zip` files (same for all pods) |
| `TARGET` | `s3://mtg-fci-l1c.zarr/` | Zarr store URI (`file:///abs/path` or `s3://bucket/key/`) |
| `PRODUCT_NAME` | derived from `TARGET` basename | Logical store name |
| `PRODUCT_TYPE` | `FDHSI` | `FDHSI` or `HRFI` |
| `RESOLUTIONS` | all for `PRODUCT_TYPE` | Optional subset, e.g. `1km` or `500m,1km` |
| `PLUGIN` | `mtg_fci_l1c` | Firecube plugin name passed to `firecube ingest` and `firecube zarr preallocate` |
| `FIRECUBE` | `firecube` | Firecube executable path or wrapper command |

### Time axis

Same semantics as [`--option time_epoch` / `--option time_slots`](#time-axis-options).

| Variable | Default | Description |
|---|---|---|
| `TIME_EPOCH` | *required* | Slot-0 anchor date (`YYYY-MM-DD`, UTC midnight) |
| `TIME_SLOTS` | one required | Axis length in slots; takes precedence over `TIME_END` |
| `TIME_END` | one required | Axis end date; length = `(TIME_END − TIME_EPOCH).days × 144` |

### Window (which slots this run ingests)

| Variable | Default | Description |
|---|---|---|
| `SLOT_START` | `0` | First slot index (inclusive) |
| `SLOT_END` | `TIME_SLOTS` | Last slot index (exclusive) |
| `FROM` | unset | Alternative to `SLOT_START` as ISO datetime (on/after `TIME_EPOCH`), e.g. `2024-09-24T06:00` |
| `TO` | unset | Alternative to `SLOT_END` as ISO datetime |

### Fan-out shape

| Variable | Default | Description |
|---|---|---|
| `SLOTS_PER_POD` | `6` | Slots per pod (`6` = 1 hour). Smaller: finer-grained retry. Larger: less startup overhead |
| `PARALLELISM` | `8` | Concurrent pods (`xargs -P`). Cap by host RAM; see [Sizing](guides/production-ingestion.md#sizing-the-fan-out) |

### Storage

| Variable | Default | Description |
|---|---|---|
| `WRITE_MODE` | `direct` | `direct` writes chunks straight to the store; `staged` writes locally first, then uploads |
| `STORAGE_TYPE` | inferred from `TARGET` scheme | `local` (from `file://`) or `s3` (from `s3://`); override when inference is wrong |
| `STORAGE_DRIVER` | `fsspec` | `fsspec` for local. For S3, use **`obstore`** because parallel writes are much faster than `fsspec` (requires the `obstore` extra: `uv sync --extra obstore`) |

### Behavior

| Variable | Default | Description |
|---|---|---|
| `FORCE_REINGEST` | `1` | `1` overwrites written slots (idempotent re-runs); `0` errors on existing slots |
| `ASSUME_YES` | `0` | `1` skips the interactive confirmation prompt (use in CI) |
| `DO_PREALLOCATE` | `1` | `0` skips phase 0 (use when preallocation runs separately) |

### Shared geolocation grids

| Variable | Default | Description |
|---|---|---|
| `GRIDS_FILE` | unset | Filesystem path to a shared `.npz` file. Passed as `--option fci_grids_file=...` to every pod |
| `GEN_GRIDS` | `0` | `1` generates `GRIDS_FILE` in phase 0 only when `GRIDS_FILE` is set and the file does not exist |

### Logs

| Variable | Default | Description |
|---|---|---|
| `LOG_ROOT` | `/root/logs` | Parent directory for per-run log folders |
| `LOGDIR` | `<LOG_ROOT>/fci-ingest-<timestamp>-<pid>` | Specific log folder. Contains `run.log`, `pod_<start>_<end>.log`, and `results.txt` (ok/FAIL per pod) |

---

## Chunk and shard layout

Defaults are nc_part-aligned so each nc_part write fills exactly one chunk
(no read-modify-write). Default shards are byte-budgeted at 128 MiB.

Override when you need a specific layout, for example **one full disk per shard**:

```bash
firecube ingest mtg_fci_l1c \
  --input-data /path/to/zips \
  --target file:///path/to/output.zarr \
  --output-format zarr --write-mode staged \
  --option zarr_chunk_overrides='{"data_1km":[1,2784,11136,1]}' \
  --option zarr_shard_overrides='{"data_1km":[1,11136,11136,1]}'
```

This produces `data_1km/counts` shards of shape `(1, 11136, 11136, 1)`: one
full disk per `(time, channel)` pair, with 4 inner chunks along Y.

Full recipes for 500 m and 2 km, the resulting shard sizes (990 MB / 248 MB / 62 MB
for uint16), and the tradeoffs are in
[Performance Tuning → Chunk and Shard Tuning](performance-tuning.md#chunk-and-shard-tuning).

**Other Firecube-core options**: `pipeline_batch_size`, `pipeline_workers` (must stay `1`
for FCI; see [Performance Tuning: Concurrency](performance-tuning.md#concurrency-pipeline_workers1)),
and `extract_workers` (parallel ZIP extraction inside a batch, default `4`; independent
of `pipeline_workers` and safe to raise on fast local disks).

---

## Compression options

The plugin does not set explicit compression on individual arrays. Compression is
an operator concern passed through the Firecube template config.

### Default: Zstd level 0

`zarr_compression=true` (the default) preserves the `ZstdCodec(level=0)` behavior.
No action needed to keep the default.

### Uncompressed output

```bash
firecube ingest mtg_fci_l1c \
    --input-data /path/to/fci-zips \
    --target file:///path/to/output.zarr \
    --output-format zarr --write-mode staged \
    --option zarr_compression=false
```

Uncompressed output is larger on disk but avoids codec overhead on read. Useful
for analysis cubes where read speed matters more than storage cost.

### Custom codec pipeline

```bash
firecube ingest mtg_fci_l1c \
    --input-data /path/to/fci-zips \
    --target file:///path/to/output.zarr \
    --output-format zarr --write-mode staged \
    --option zarr_codecs='[{"name": "blosc", "configuration": {"cname": "lz4", "clevel": 5}}]'
```

`zarr_codecs` accepts a JSON list of codec objects in Zarr v3 format. When set,
it overrides `zarr_compression`. See [Performance Tuning: Codec choice](performance-tuning.md#codec-choice)
for trade-offs between archive and analysis workloads.

> **Warning:** changing `zarr_compression` or `zarr_codecs` between ingests to
> the same store raises `SchemaDriftError`. Choose a codec configuration once,
> before preallocation, and keep it consistent across all pods and re-ingest runs.
> To switch codecs, re-preallocate from scratch.

---

## Geolocation grids workflow

Each resolution group has static `latitude[y, x]` and `longitude[y, x]` arrays.
Computing them on the fly takes 1–20 s and 200 MB–4 GB per pod. In parallel
runs, every pod would recompute the same grids. Pre-generate them once and
share them through a filesystem path visible inside every pod.

### Generate

```bash
firecube plugins mtg_fci_l1c geo generate \
  --resolutions 1km,2km \
  --sub-satellite-lon 0.0 \
  --output /shared/fci_grids.npz
```

Use `--overwrite` to replace an existing `.npz` file.

`--sub-satellite-lon` sets the sub-satellite longitude in degrees. Keep
`0.0` for MTG-I1. A nonzero longitude needs deliberate validation because
`scripts/fci-ingest.sh` generates grids with the default `0.0`, and the current
Zarr projection metadata is centered on `0.0`.

When `scripts/fci-ingest.sh` runs with `GEN_GRIDS=1`, it only generates a file
when `GRIDS_FILE` is set and that path does not exist. Script-generated grids
use the default `--sub-satellite-lon 0.0`. For any other longitude, generate
the file manually and pass it through `GRIDS_FILE`.

### Pass to every ingest run

```bash
firecube ingest mtg_fci_l1c \
  --input-data /data/fci-zips \
  --target file:///data/fci_l1c.zarr \
  --output-format zarr --write-mode staged \
  --option fci_grids_file=/shared/fci_grids.npz
```

Writes to the Zarr store are idempotent: if `latitude`/`longitude` already
exist in a group, they are not re-written. Safe for parallel pods.

When `fci_grids_file` is not set, `latitude`/`longitude` are computed on the fly
for each process. This is fine for local development and wasteful in production.
If the file is missing or lacks a requested resolution, ingestion logs a warning
and recomputes that grid on the fly.

### Inspect

```bash
firecube plugins mtg_fci_l1c geo info --grids-file /shared/fci_grids.npz
```

### Disable geolocation entirely

Saves storage and skips the compute:

```bash
firecube ingest mtg_fci_l1c \
  --input-data /path/to/fci-zips \
  --target file:///path/to/output.zarr \
  --output-format zarr --write-mode staged \
  --option include_geolocation=false
```

Array specs (dtype, per-resolution sizes, NaN-at-limb semantics) are in
[FCI Data in Zarr](fci-data-in-zarr.md).

---

## Fix FillValue (legacy stores)

Firecube now stamps the CF `_FillValue` attribute on numeric arrays at ingest
time, so new stores need no extra step. Stores written before that behavior
lack the attribute, and xarray then shows fill pixels as raw integer values
instead of masking them to `NaN` on read.

The `fix-fillvalue` command stamps `_FillValue` on numeric arrays in such an
existing store without re-running ingestion.

**Run only after ingestion has completed.** The store must be offline (no
active ingest pods writing to it).

### Dry run (default)

Preview what would be stamped without making any changes:

```bash
firecube plugins mtg_fci_l1c fix-fillvalue --store <path>
```

### Apply

```bash
firecube plugins mtg_fci_l1c fix-fillvalue --store <path> --yes-i-really-mean-it
```

After running, xarray will mask fill pixels to `NaN` on read for all stamped
arrays. The command is idempotent: arrays already stamped with the expected
value are skipped, and a conflicting existing value stops the command without
writing anything.

---
