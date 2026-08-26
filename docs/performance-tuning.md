# Performance Tuning

## Purpose

This page covers memory sizing, storage layout choices, and parallel ingestion
for the `mtg_fci_l1c` plugin. Use it when planning resource requirements,
tuning Zarr storage layout, or deploying multi-pod parallel ingestion.

## Prerequisites

- Firecube ≥ 0.1.4 with `mtg_fci_l1c` installed.
- A Zarr store target: `file:///` for local storage or `s3://` for object storage.
- For parallel ingestion: all pods must have read access to the same input ZIP
  files, and the Zarr store must be preallocated before the first pod starts.

---

## Memory Considerations

All figures below apply to this workload unless noted: **FDHSI, 1 km + 2 km,
all 16 channels, `pipeline_workers=1`, `pixel_time_dtype=float64`, one slot**.
Measured baseline: **14.625 GiB** peak RSS per worker.

### Per-component breakdown

| Component | Theoretical footprint | Notes |
|---|---|---|
| Pixel time (`float64`) | ~9.24 GiB | Largest single component. `include_pixel_time=false` removes it entirely. |
| Counts + calibration | ~2.31 GiB | Irreducible baseline; always included. |
| Pixel quality | ~1.15 GiB | `include_pixel_quality=false` removes it. |
| Geolocation (lat/lon) | ~1.15 GiB | `include_geolocation=false` skips it. |
| Runtime / writer overhead | ~0.78 GiB | Zarr write buffers, HDF5 read-side, Python runtime. |
| **Total observed** | **~14.6 GiB** | Measured baseline with all features enabled. |

### Feature-flag knobs

The following options reduce per-worker peak RSS. Savings are expected values
based on payload-size analysis; they are not independently re-measured for every
configuration.

| Option | Default | Expected saving per slot | Trade-off |
|---|---|---|---|
| `pixel_time_dtype=float32` | `float64` | ~4.7 GiB | Pixel-time precision drops to ~64–128 s absolute-epoch resolution. Safe only when sub-minute precision is not required downstream. |
| `include_pixel_time=false` | `true` | ~9.5 GiB | Drops per-pixel observation timestamps entirely. Safe when downstream consumers do not use pixel_time. |
| `include_geolocation=false` | `true` | ~1.2 GiB | Skips static lat/lon arrays. Safe when coordinates are available from another source or not needed. See [Geolocation Grid Compute](#geolocation-grid-compute). |
| `include_pixel_quality=false` | `true` | ~1.2 GiB | Skips pixel quality mask. Safe when quality filtering is done at source. |

Example: disable pixel time to reduce per-worker peak from ~14.6 GiB to ~5.1 GiB:

```bash
firecube ingest mtg_fci_l1c \
    --input-data /path/to/fci-zips \
    --target file:///path/to/output.zarr \
    --output-format zarr \
    --write-mode staged \
    --option include_pixel_time=false
```

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

### Defaults are nc_part-aligned

FCI L1C ZIPs contain 40 nc_parts per acquisition. The default chunk Y-dim
matches the nc_part row count so each nc_part write fills exactly one chunk
(no read-modify-write during streaming ingest):

| Resolution | Array size (y=x) | Default chunk Y |
|---|---|---|
| 500m | 22272 | 556 |
| 1km | 11136 | 278 |
| 2km | 5568 | 139 |

Default shards are byte-budgeted (128 MiB target). For 1 km uint16 this
groups ~21 chunks along Y, giving a shard shape of approximately
`(1, 5838, 11136, 1)`.

### When to override chunks

- **Larger Y chunks** (e.g. `zarr_chunk_y=2784`) reduce Zarr object count by
  grouping 10 nc_parts per chunk. Cost: read-modify-write on each nc_part write.
  Safe in `--write-mode staged`; avoid in `--write-mode direct` to S3.
- **Smaller Y chunks** enable finer spatial subsetting but multiply object count.

### When to override shards

- **Full-disk shards** (`zarr_shard_overrides`): minimise S3 object count
  (~1 object per `(time, channel)` per array).
- **Smaller shards**: bound shard size for object-storage size limits.

### Full-disk-per-shard recipe

| Resolution | Chunk override | Shard override |
|---|---|---|
| 500m | `(1, 5568, 22272, 1)` | `(1, 22272, 22272, 1)` |
| 1km | `(1, 2784, 11136, 1)` | `(1, 11136, 11136, 1)` |
| 2km | `(1, 1392, 5568, 1)` | `(1, 5568, 5568, 1)` |

```bash
firecube ingest mtg_fci_l1c \
  --input-data /path/to/zips \
  --target file:///path/to/output.zarr \
  --output-format zarr --write-mode staged \
  --option zarr_chunk_overrides='{"data_1km":[1,2784,11136,1]}' \
  --option zarr_shard_overrides='{"data_1km":[1,11136,11136,1]}'
```

### Resulting shard byte size (uint16 data arrays)

| Resolution | Full-disk shard | Byte size (uint16) |
|---|---|---|
| 500m | `(1, 22272, 22272, 1)` | ~990 MB |
| 1km | `(1, 11136, 11136, 1)` | ~248 MB |
| 2km | `(1, 5568, 5568, 1)` | ~62 MB |

For float64 `pixel_time` at 500m the shard would be ~3.96 GB. Disable
`pixel_time` via `--option include_pixel_time=false` if this is too large.

### Known caveats

- **Mid-store strategy change**: switching chunks or shards on an existing store
  raises `SchemaDriftError`. Re-ingest from source to apply a new layout.
- **`zarr_sharding=false` overrides everything**: `zarr_shard_overrides` shapes
  are ignored when sharding is disabled globally. Chunk overrides still apply.
- **`--write-mode direct` to S3**: avoid large chunks (>1 nc_part); each nc_part
  write becomes a download-modify-upload cycle. Use `--write-mode staged` instead.

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

**RAM scales linearly**: N workers × ~14.6 GiB per worker. Scale horizontally
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
| `ResumeConflictError` on restart | A previous run was interrupted (SIGKILL, OOM) and left a `started` record | 1. `firecube chunks runs list --product-name <name> --status started` to find the stale run ID. 2. `firecube chunks runs abandon --product-name <name> --run-id <id> --reason "crash recovery" --yes-i-really-mean-it` to clear the record. 3. Re-run the same slot range. Data written before the kill is intact; re-ingest overwrites the partial slot cleanly. |
| Some slots missing after a run | A pod exited with an error | Re-submit the missing slot range. Writes are idempotent with `force_reingest=true` (the default in `scripts/fci-ingest.sh`). |

## Operational Notes

- Use `pipeline_workers=1` per pod. Scale throughput via pod count, not thread count.
- Assign slot ranges by integer index. Convert dates to slots using
  `slot = (timestamp_utc - time_epoch_utc_midnight) / cadence` before setting
  `--slot-start`/`--slot-end`.
- Re-ingesting a completed slot overwrites it cleanly. Data integrity is
  preserved after abandon + re-ingest.
