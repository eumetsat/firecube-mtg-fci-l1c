# Customizing `mtg_fci_l1c`

Operator-facing plugin and production-script settings.

- [Plugin `--option` flags](#plugin---option-flags)
- [FCI chunks (`fci_chunks`)](#fci-chunks)
- [Flat store layout](#flat-store-layout)
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
Only `product_type` is required.

| Option | Default | Description |
|---|---|---|
| `product_type` | *required* | `FDHSI` or `HRFI`. The plugin does not infer it from the input. Without it, `firecube ingest` stops with `product_type is required`. |
| `resolutions` | all for `product_type` | Comma-separated list, e.g. `1km` or `500m,1km` |
| `flat_store` | `false` | Write the variables at the store root instead of `data_<res>/`. Needs exactly one resolution; see [Flat store layout](#flat-store-layout) |
| `channels` | all channels for selected resolutions | Comma-separated logical channel names, e.g. `vis_06,ir_105` |
| `partial_chunk` | `fill` | `fill` or `error`: what to do with rows that no input file covers. |
| `fci_chunks` | `null` (full disk) | Inclusive `[first, last]` BODY chunk numbers (1 to 40): store only the rows of those chunks. See [FCI chunks](#fci-chunks) |
| `include_pixel_quality` | `true` | Include the 8-bit warning flag array |
| `include_pixel_time` | `true` | Include per-pixel observation timestamps |
| `include_calibration` | `true` | Include `slope` and `offset` arrays |
| `include_geolocation` | `true` | Include static `latitude` / `longitude` arrays |
| `fci_grids_file` | `null` | Path to pre-generated `.npz` grids (see [Geolocation grids workflow](#geolocation-grids-workflow)) |
| `pixel_time_dtype` | `float64` | `float64`, `float32`, `int32`, or `int64`; `float32` halves storage but loses sub-minute absolute-epoch precision |
| `scratch_dir` | `null` | Base directory for temporary ZIP extraction (uses system temp when unset) |
| `zarr_chunk_y` | `null` | Y-dimension chunk size for Zarr arrays (defaults to the nominal BODY chunk height of each resolution: 556 / 278 / 139 rows at 500 m / 1 km / 2 km; see [Chunk and shard layout](#chunk-and-shard-layout)) |
| `zarr_shard_target_bytes` | `134217728` (128 MiB) | Target bytes per shard for the default policy |
| `zarr_shard_overrides` | `null` | Explicit per-group `(time, y, x, channel)` shard shapes, e.g. `{"data_1km": [1, 5560, 11136, 1]}` (a whole multiple of the chunk height) |
| `zarr_chunk_overrides` | `null` | Explicit per-group chunk shapes; takes precedence over `zarr_chunk_y` |
| `projection_units` | `meter` | Units for the `x` and `y` projection coordinate arrays. Valid values: `meter` (default), `metre` (alias for `meter`), `radian`. See [Projection units](#projection-units). |

```bash
# Process only 1 km data, drop pixel_time
firecube ingest mtg_fci_l1c \
    --input-data /path/to/fci-zips \
    --target file:///path/to/output.zarr \
    --output-format zarr --write-mode direct \
    --option product_type=FDHSI \
    --option resolutions=1km \
    --option include_pixel_time=false \
    --option cleanup_workspace=true
```

```bash
# Keep only selected channels across default resolutions
firecube ingest mtg_fci_l1c \
    --input-data /path/to/fci-zips \
    --target file:///path/to/output.zarr \
    --output-format zarr --write-mode direct \
    --option product_type=FDHSI \
    --option channels=vis_06,ir_105 \
    --option cleanup_workspace=true
```

---

## FCI chunks

`fci_chunks=[first,last]` stores only a band of the disk. The numbers are
inclusive BODY chunk numbers from 1 to 40, for FDHSI and HRFI alike. Chunk 1
is at the southern edge of the disk and chunk 40 at the northern edge, so
`[32,40]` is the northern part: at 1 km it is rows 8649 to 11136 of 11136.

```bash
--option 'fci_chunks=[32,40]'
```

Pass the same value to every ingest of the store. Use no spaces when you set it
through `scripts/fci-ingest.sh`.

What the store holds:

- **Rows.** In each group the `y` extent is the rows of the chosen chunks, widened to that
  group's output-chunk grid, so chunk boundaries are those of a full-disk
  store. For `[32,40]` of FDHSI, `data_1km` has 2518 rows (disk rows 8618 to
  11136) and `data_2km` has 1259 rows (disk rows 4309 to 5568). Measured on
  one real cycle. `x` keeps the full width.
- **Coordinates.** `y`, `latitude` and `longitude` cover those rows only.
- **Group attributes.** A store created by `firecube ingest` carries
  `fci_chunks`, `disk_row_start` and `disk_row_stop` on each group (disk rows
  counted from the southern edge). A store created by `firecube zarr
  preallocate` carries no group attributes at all, as for the CF attributes of
  a full-disk store; read its window from the shape of `y`.
- **Identity.** The window is part of the store's identity. Ingesting another
  window, or a full-disk run, into a store written with `fci_chunks` (or an `fci_chunks` run into a full-disk
  store) is refused with `plugin declares incompatible resolved index`.
- **Checks.** Every input file's rows are checked against the table of BODY
  chunk rows; a file that does not match stops its scene with an error naming
  the chunk.
- **Reads.** Pixel data is read only from files inside the window. Ingesting
  chunks 32 to 40 reads 9 files, not 40.

Limits and caveats:

- A store written with `fci_chunks` cannot be grown yet. To cover other rows, ingest into a new
  store.
- Platform variables (`subsatellite_*`, `platform_altitude`) are means over the
  files you give it.
- `pixel_time` at the northern edge of the chosen chunks can differ from a full-disk
  ingest unless the next chunk to the north is present.
- `fci_chunks` is part of the resume identity of a run. Stores ingested before
  the option existed carry no value for it, so a single-pod run into such a
  store (no `--slot-start`/`--slot-end`) can fail with `ResumeConflictError`. Add
  `--option resume_existing=true` to continue, or `--option force_reingest=true`
  to overwrite, as for any resume conflict. See
  [Performance Tuning → Failure Recovery](performance-tuning.md#failure-recovery).

### Example: the northern chunks of one repeat cycle

Download chunks 32 to 40 of one cycle with [eumdac](https://user.eumetsat.int/resources/user-guides/eumetsat-data-access-client-eumdac-guide),
then ingest them. Set the sensing window so that it matches one product, and
check that with `eumdac search` first.

```bash
eumdac download -c EO:EUM:DAT:0662 \
    -s 2025-07-01T00:00:00 -e 2025-07-01T00:09:59 \
    --entry '*CHK-BODY*_003[2-9].nc' '*CHK-BODY*_0040.nc' \
    --onedir -o /path/to/fci-chunks

firecube ingest mtg_fci_l1c \
    --input-data /path/to/fci-chunks \
    --target file:///path/to/fci-chunks.zarr \
    --output-format zarr --write-mode direct \
    --option product_type=FDHSI \
    --option 'fci_chunks=[32,40]' \
    --option cleanup_workspace=true
```

The `--entry` patterns match the chunk numbers 0032 to 0040 at the end of the
file name. The window starts at disk row 8618, inside chunk 31, so chunks 32
to 40 are a partial scene for it and need `--write-mode direct`. Open the result with `xr.open_zarr("/path/to/fci-chunks.zarr", group="data_1km")`.

---

## Flat store layout

By default each resolution goes into its own `data_<res>/` group. When a store
holds one resolution only, `flat_store=true` writes the variables at the store
root instead, so xarray opens the store without a `group=` argument. Supported
grids: FDHSI 1 km, FDHSI 2 km, HRFI 500 m, and HRFI 1 km.

```bash
firecube ingest mtg_fci_l1c \
    --input-data /path/to/fci-zips \
    --target file:///path/to/fci-1km.zarr \
    --output-format zarr --write-mode direct \
    --option product_type=FDHSI \
    --option resolutions=1km \
    --option flat_store=true \
    --option cleanup_workspace=true
```

```
fci-1km.zarr/
├── counts/  pixel_quality/  pixel_time/  slope/  offset/
├── latitude/  longitude/  x/  y/  time/  channel/  spatial_ref/
└── zarr.json
```

```python
import xarray as xr

ds = xr.open_zarr("/path/to/fci-1km.zarr")  # no group= needed
print(ds["counts"].sizes)
```

Rules:

- **Exactly one effective resolution.** Select it with `resolutions`, or with a
  `channels` list whose channels all share one resolution. Otherwise
  `firecube ingest` and `firecube zarr preallocate` stop with a configuration
  error before writing anything, for example
  `flat_store=true requires exactly one effective resolution for FDHSI, got ['1km', '2km']`.
- **One layout and one grid per store.** Pass the same `flat_store`,
  `product_type`, and `resolutions` to preallocation and every ingest of a
  store. Firecube refuses to switch an existing store between grouped and flat,
  or to write another resolution or product type into a flat store, before any
  array data is written; the error reads
  `plugin declares incompatible resolved index` and lists the differing groups.
  Stores are not converted in place: ingest into a new store.
- **Override keys keep the group name.** `zarr_chunk_overrides` and
  `zarr_shard_overrides` stay keyed by `data_<res>` (for example `data_1km`) in
  both layouts.

To keep both FDHSI resolutions flat, write one store per resolution, for example
`fci-1km.zarr` with `resolutions=1km` and `fci-2km.zarr` with `resolutions=2km`.
`scripts/fci-ingest.sh` sets the option with `FLAT_STORE=1`.

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
    --output-format zarr --write-mode direct \
    --option product_type=FDHSI \
    --option projection_units=radian \
    --option cleanup_workspace=true
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
| `INPUT` | `/data/fci-zips` | Directory or URI of FCI L1C `.zip` files or unpacked chunk `.nc` files, not both (same for all pods). |
| `TARGET` | `s3://mtg-fci-l1c.zarr/` | Zarr store URI (`file:///abs/path` or `s3://bucket/key/`) |
| `PRODUCT_NAME` | derived from `TARGET` basename | Logical store name |
| `PRODUCT_TYPE` | `FDHSI` | `FDHSI` or `HRFI`. Always passed as `--option product_type=...`, so set it to `HRFI` for HRFI input |
| `RESOLUTIONS` | all for `PRODUCT_TYPE` | Optional subset, e.g. `1km` or `500m,1km` |
| `FLAT_STORE` | unset | `1`, `true`, `yes`, or `on` (any case) adds `--option flat_store=true`; needs a single-resolution `RESOLUTIONS`. See [Flat store layout](#flat-store-layout) |
| `FCI_CHUNKS` | unset | Adds `--option fci_chunks=...` to preallocation and every pod, e.g. `[32,40]`. Write it without spaces. See [FCI chunks](#fci-chunks) |
| `PARTIAL_CHUNK` | unset | Adds `--option partial_chunk=...` (`fill` or `error`) to preallocation and every pod. |
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
| `WRITE_MODE` | `direct` | `direct` writes chunks straight to the store; `staged` writes locally first, then uploads, and refuses partial scenes |
| `STORAGE_TYPE` | inferred from `TARGET` scheme | `local` (from `file://`) or `s3` (from `s3://`); override when inference is wrong |
| `STORAGE_DRIVER` | `fsspec` | `fsspec` for local. For S3, use **`obstore`** because parallel writes are much faster than `fsspec` (requires the `obstore` extra: `uv sync --extra obstore`) |

### Behavior

| Variable | Default | Description |
|---|---|---|
| `FORCE_REINGEST` | `1` | `1` overwrites written slots (idempotent re-runs); `0` errors on existing slots |
| `CLEANUP_WORKSPACE` | `true` | `1`, `true`, `yes`, or `on` (any case) adds `--option cleanup_workspace=true` to preallocation and every pod; `0`, `false`, `no`, or `off` passes nothing and keeps the workspaces. |
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

The default chunk height is the nominal BODY chunk height of the resolution:
556 rows at 500 m, 278 at 1 km and 139 at 2 km, by the full grid width. Real
BODY chunks vary a little around that height (258 to 300 rows at 1 km), so an
output chunk is assembled from one or two chunk files and written once.
Default shards are byte-budgeted at 128 MiB.

Override when you need a specific layout, for example **one full disk per shard**:

```bash
firecube ingest mtg_fci_l1c \
  --input-data /path/to/zips \
  --target file:///path/to/output.zarr \
  --output-format zarr --write-mode direct \
  --option product_type=FDHSI \
  --option zarr_shard_overrides='{"data_1km":[1,11398,11136,1]}' \
  --option cleanup_workspace=true
```

This produces `data_1km/counts` chunks of `(1, 278, 11136, 1)` (the default) and
shards of shape `(1, 11398, 11136, 1)`: one full disk per `(time, channel)` pair,
with 41 inner chunks along Y. The shard height must be a whole multiple of the
chunk height, so it is rounded up from 11136 to 41 chunks. Keep the default chunk
height: larger heights pass config validation but can make an output chunk meet
three BODY chunk files, which the plugin rejects.

Full recipes for 500 m and 2 km, the resulting data sizes (968 / 242 / 61 MiB
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
    --output-format zarr --write-mode direct \
    --option product_type=FDHSI \
    --option zarr_compression=false \
    --option cleanup_workspace=true
```

Uncompressed output is larger on disk but avoids codec overhead on read. Useful
for analysis cubes where read speed matters more than storage cost.

### Custom codec pipeline

```bash
firecube ingest mtg_fci_l1c \
    --input-data /path/to/fci-zips \
    --target file:///path/to/output.zarr \
    --output-format zarr --write-mode direct \
    --option product_type=FDHSI \
    --option zarr_codecs='[{"name": "blosc", "configuration": {"cname": "lz4", "clevel": 5}}]' \
    --option cleanup_workspace=true
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
  --output-format zarr --write-mode direct \
  --option product_type=FDHSI \
  --option fci_grids_file=/shared/fci_grids.npz \
  --option cleanup_workspace=true
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
  --output-format zarr --write-mode direct \
  --option product_type=FDHSI \
  --option include_geolocation=false \
  --option cleanup_workspace=true
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
existing store without re-running ingestion. It works on grouped and
[flat](#flat-store-layout) stores, detects the layout from the store, and
refuses a store that is empty, is not an FCI store, or mixes both layouts.

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
