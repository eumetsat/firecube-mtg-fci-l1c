# Performance Tuning

## Purpose

This page covers memory sizing, storage layout choices, and parallel ingestion
for the `mtg_fci_l1c` plugin. Use it when planning resource requirements,
tuning Zarr storage layout, or deploying multi-pod parallel ingestion.

## Prerequisites

- Firecube ≥ 0.1.7 with `mtg_fci_l1c` installed.
- A Zarr store target: `file:///` for local storage or `s3://` for object storage.
- For parallel ingestion: all pods must have read access to the same input ZIP
  or chunk files, and the Zarr store must be preallocated before the first pod starts.

---

## Memory Considerations

Peak resident memory (RSS) of one `firecube ingest` process ingesting one
slot. Measured on an aarch64 host (NVIDIA GB10) on 2026-10-08 with
`/usr/bin/time -v`, for FDHSI, 1 km + 2 km, all 16 channels, default options,
`pipeline_workers=1`, `--write-mode direct` to a local store preallocated for
144 slots, with a precomputed `fci_grids_file`:

| Input | Peak RSS per process | Label |
|---|---|---|
| Full-disk ZIP | about 2.0 GiB (mean 2.017 GiB over 4 runs) | measured |
| Full-disk unpacked chunks | about 2.0 GiB (mean 2.019 GiB over 2 runs) | measured |
| Stripe store, `fci_chunks=[32, 40]` | about 1.25 GiB (3 runs, 1.252 to 1.253 GiB) | measured |

A single run now and then peaks 50 to 130 MiB higher than the others (highest
seen: 2.15 GiB). Plan for 2.5 GiB per pod to leave room for that (calculated,
not a measured limit).

N workers need about N times the figure above (calculated). Without a shared
`fci_grids_file` each process computes the grids itself and peaks higher; that
case was not re-measured.

The options that shrink the stored output (`include_pixel_time=false`,
`pixel_time_dtype=float32`, `include_pixel_quality=false`,
`include_geolocation=false`) have not been re-measured against these figures;
do not assume they lower peak RSS by the size of the arrays they drop.

## Geolocation Grid Compute

`latitude` and `longitude` are static arrays in the Zarr output. For production
runs, use a shared `GRIDS_FILE` so each pod loads precomputed grids instead of
recomputing them.

This reduces repeated startup work and transient memory pressure. It does not
remove the final `latitude` and `longitude` arrays from the Zarr store. To omit
those arrays, set `--option include_geolocation=false`.

Use the [Geolocation grids workflow](customization.md#geolocation-grids-workflow)
to generate and inspect the `.npz` file.

## Chunk and Shard Tuning

### Default chunk heights

An FCI L1C repeat cycle holds 40 BODY chunks. The default chunk Y-dim is the
nominal height of one BODY chunk at that resolution, and a chunk spans the full
grid width. BODY chunk heights are not uniform (at 1 km they range from 258 to
300 rows), so an output chunk is assembled from one or two chunk files and
written once. Config validation accepts a chunk height up to twice the nominal
one, but heights above the defaults can make an output chunk meet three BODY
chunk files, which the plugin rejects when it assembles the scene (`Output
chunk y=(...) intersects 3 nc_parts; max supported is 2.`). Keep the defaults:

| Resolution | Array size (y=x) | Default chunk Y |
|---|---|---|
| 500m | 22272 | 556 |
| 1km | 11136 | 278 |
| 2km | 5568 | 139 |

Default shards are byte-budgeted (128 MiB target). For 1 km uint16 this
groups ~21 chunks along Y, giving a shard shape of approximately
`(1, 5838, 11136, 1)`.

### When to override chunks

- **Larger Y chunks** than the defaults are not recommended. They pass config
  validation (up to twice the default), but on the real BODY chunk layout the
  maximum heights (556 at 1 km, 278 at 2 km, 1112 at 500 m) make an output chunk
  meet three BODY chunk files and the scene fails.
- **Smaller Y chunks** enable finer spatial subsetting but multiply object count.

### When to override shards

- **Full-disk shards** (`zarr_shard_overrides`): minimise S3 object count
  (~1 object per `(time, channel)` per array).
- **Smaller shards**: bound shard size for object-storage size limits.

### Full-disk-per-shard recipe

| Resolution | Chunk (default, no override) | Shard override |
|---|---|---|
| 500m | `(1, 556, 22272, 1)` | `(1, 22796, 22272, 1)` |
| 1km | `(1, 278, 11136, 1)` | `(1, 11398, 11136, 1)` |
| 2km | `(1, 139, 5568, 1)` | `(1, 5699, 5568, 1)` |

Override only the shard and keep the default chunk height. The shard height
must be a whole multiple of the chunk height, so it is rounded up to cover the
grid: 41 chunks along Y (for example `41 * 278 = 11398` rows at 1 km, for 11136
rows). The 41st chunk holds only the last 16 rows at 1 km (32 at 500 m, 8 at
2 km); the rest of that chunk is padding.

```bash
firecube ingest mtg_fci_l1c \
  --input-data /path/to/zips \
  --target file:///path/to/output.zarr \
  --output-format zarr --write-mode direct \
  --option product_type=FDHSI \
  --option zarr_shard_overrides='{"data_1km":[1,11398,11136,1]}' \
  --option cleanup_workspace=true
```

Override keys are always `data_<res>`, also for a store written with
`flat_store=true` where the arrays sit at the root; see
[Flat store layout](customization.md#flat-store-layout).

### Resulting shard byte size (uint16 data arrays)

| Resolution | Full-disk shard | Uncompressed disk data (uint16) |
|---|---|---|
| 500m | `(1, 22796, 22272, 1)` | ~968 MiB |
| 1km | `(1, 11398, 11136, 1)` | ~242 MiB |
| 2km | `(1, 5699, 5568, 1)` | ~61 MiB |

Sizes are for the 22272, 11136 and 5568 grid rows; the padding in the last
chunk adds up to one chunk of rows. For float64 `pixel_time` at 500m the data
would be ~3.8 GiB. Disable
`pixel_time` via `--option include_pixel_time=false` if this is too large.

### Known caveats

- **Mid-store strategy change**: switching chunks or shards on an existing store
  raises `SchemaDriftError`. Re-ingest from source to apply a new layout.
- **`zarr_sharding=false` overrides everything**: `zarr_shard_overrides` shapes
  are ignored when sharding is disabled globally. Chunk overrides still apply.
- **Chunk height and write mode**: keep the default chunk heights; larger ones
  can make an output chunk meet three BODY chunk files, which the plugin
  rejects (see [Default chunk heights](#default-chunk-heights)). Do not switch
  to `--write-mode staged` to avoid rewriting chunks in a populated store: a
  `staged` run replaces each output chunk or shard it writes, so rows an
  earlier run stored there are lost. See
  [Staged write mode](customization.md#staged-write-mode-takes-complete-scenes-only).

## Codec choice

The default codec (`ZstdCodec(level=0)`) is a reasonable starting point for most
workloads. It compresses FCI uint16 counts data well and adds minimal CPU overhead
at level 0. Before committing to a non-default codec in production, measure on a
representative FCI slot with your actual read and write patterns.

### Archive vs analysis trade-offs

| Goal | Recommended approach | Notes |
|---|---|---|
| Minimize storage cost | Default `ZstdCodec(level=0)` or higher level | Higher Zstd levels (e.g. level 9) compress better but slow writes significantly. Measure before using in production. |
| Maximize read throughput | `zarr_compression=false` (uncompressed) | Removes decompression overhead on read. Storage cost roughly doubles for uint16 counts. |
| Balanced archive | Default `ZstdCodec(level=0)` | Good compression ratio with fast decompression. Suitable for long-term storage with occasional access. |
| Custom pipeline | `zarr_codecs='[{"name": "blosc", ...}]'` | Blosc with LZ4 can be faster than Zstd for read-heavy workloads. Verify codec availability in your environment. |

### Codec lock-in warning

Changing `zarr_compression` or `zarr_codecs` on an existing store raises
`SchemaDriftError`. The codec configuration is fixed at preallocation time.
Choose once, before preallocation, and keep it consistent across all pods and
re-ingest runs. To switch codecs, re-preallocate from scratch.

See [Compression options](customization.md#compression-options) for the full
`--option` syntax and examples.

## Concurrency: pipeline_workers=1

**Always set `pipeline_workers=1` for FCI workloads.**

Same-slot conflicts raise `ClaimConflictError` immediately with no retry and no
queue. The batch is dropped and the ingest run aborts. This is deterministic:
any configuration where two workers target the same slot will fail on every run.

**RAM scales linearly**: N workers × about 2.0 GiB per worker (measured; see [Memory Considerations](#memory-considerations)). Scale horizontally
with separate pods over disjoint slot ranges; do not raise `pipeline_workers`
inside a pod.

## Parallel Ingestion: Slot-Range Partitioning

Each pod must ingest a disjoint subset of the time axis. No two pods should
touch the same slot. The Zarr store must be preallocated before any pod starts.

This plugin uses Firecube's direct-region Zarr path. The plugin declares the
schema and exact write intents; Firecube owns schema setup, chunk claims, run
records, and coordinated writes. For the core mechanics, see
[Direct Region Zarr](https://eumetsat.github.io/firecube/concepts/output-formats/zarr/direct-region/),
[Parallel Zarr Writes](https://eumetsat.github.io/firecube/concepts/output-formats/zarr/parallel-writes/),
and [Direct Zarr Plugins](https://eumetsat.github.io/firecube/concepts/plugins/direct-zarr/).

### Slot index

FCI repeats every 10 minutes; each repeat maps to one integer slot:

```
slot = (timestamp_utc - time_epoch_utc_midnight) / cadence
```

There are 144 slots per day. For aligned 10-minute acquisitions, this is:
`(date - epoch).days * 144 + hour * 6 + minute // 10`

Preallocation is **idempotent** when re-run with identical arguments (safe to
retry after a failure). The axis shape is fixed at preallocation time. Choose
`time_epoch=2024-09-24` for a full MTG FCI L1C cube and choose `time_slots` to
cover the full intended range up-front. A long sparse axis, such as several
years, is valid if needed. Ingest into subsets of the preallocated axis over
time; do not expect to grow it after data has been written.

Use the [Production Ingestion Guide](guides/production-ingestion.md) for the
actual `scripts/fci-ingest.sh` workflow, S3 setup, host sizing table, and
multi-host command examples.

Use [Performance Benchmarks](reference/performance-benchmarks.md) for published
scaling plots and benchmark workload notes.

## Failure Recovery

| Symptom | Cause | Recovery |
|---|---|---|
| `ClaimConflictError` on startup | Two pods have overlapping slot ranges | Ensure `--slot-start`/`--slot-end` ranges do not overlap. The failed pod can be re-submitted with a corrected range. |
| `ResumeConflictError` on a store ingested before `fci_chunks` existed | Spans written by plugin 0.2.0 or earlier carry no `fci_chunks` value, so a single-pod run (no `--slot-start`/`--slot-end`) cannot prove it matches them | Add `--option resume_existing=true` to continue, or `--option force_reingest=true` to overwrite |
| `ResumeConflictError` on restart | A previous run was interrupted (SIGKILL, OOM) and left a `started` record | 1. `firecube chunks runs list --product-name <name> --status started` to find the stale run ID. 2. `firecube chunks runs abandon --product-name <name> --run-id <id> --reason "crash recovery" --yes-i-really-mean-it` to clear the record. 3. Re-run the same slot range. Data written before the kill is intact; re-ingest overwrites the partial slot cleanly. |
| Some slots missing after a run | A pod exited with an error | Re-submit the missing slot range. Writes are idempotent with `force_reingest=true` (the default in `scripts/fci-ingest.sh`). |

## Operational Notes

- Use `pipeline_workers=1` per pod. Scale throughput via pod count, not thread count.
- Assign slot ranges by integer index. Convert dates to slots using
  `slot = (timestamp_utc - time_epoch_utc_midnight) / cadence` before setting
  `--slot-start`/`--slot-end`.
- Re-ingesting a completed slot overwrites it cleanly. Data integrity is
  preserved after abandon + re-ingest.
