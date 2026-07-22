# Production Ingestion Guide

## Purpose

Run [`scripts/fci-ingest.sh`](../../scripts/fci-ingest.sh) in production: pick
`PARALLELISM` to fit your host, configure the `obstore` driver for S3, and
split large time ranges across several hosts.

Full env-var reference for the script is in
[Customization → `scripts/fci-ingest.sh` environment variables](../customization.md#scriptsfci-ingestsh-environment-variables).

## Prerequisites

- Firecube ≥ 0.1.4 with the `mtg_fci_l1c` plugin installed.
- Read access to FCI L1C `.zip` files at `INPUT` (local path, `file://` URI,
  or `s3://` prefix). The same `INPUT` is passed to every pod.
- For local `TARGET`: any writable local filesystem.
- For S3 `TARGET`: Firecube installed with the `obstore` extra
  (`uv sync --extra obstore`) plus valid AWS credentials in the environment.
- When `GRIDS_FILE` is used: shared filesystem accessible from every pod.

## Time Axis Planning

For a full MTG FCI L1C cube, set `TIME_EPOCH=2024-09-24`, the first FCI L1C
availability date in the EUMETSAT Data Store. This makes slot 0 the first
possible full-cube slot.

FCI cadence is 10 minutes, so each day has 144 slots:

```text
slot = (timestamp_utc - time_epoch_utc_midnight) / cadence
```

`TIME_SLOTS` reserves the preallocated axis length. It can be larger than the
window you ingest now. Choose it from the full horizon you want to support; a
long sparse axis, such as several years, is valid if needed. Each run writes
only its requested `[SLOT_START, SLOT_END)` window.

## Two-Phase Workflow

**Phase 0: one-time setup**

- If `GEN_GRIDS=1`, `GRIDS_FILE` is set, and the file does not exist, generate
  shared geolocation grids via `firecube plugins mtg_fci_l1c geo generate`.
- Preallocate the Zarr store to `TIME_SLOTS` length via
  `firecube zarr preallocate` (skipped when `DO_PREALLOCATE=0`).

**Phase 1: parallel ingest**

- Fan out `firecube ingest` pods across disjoint `[SLOT_START, SLOT_END)`
  sub-ranges using `xargs -P`.
- Each pod ingests `SLOTS_PER_POD` slots at `pipeline_workers=1`;
  `PARALLELISM` pods run concurrently.
- Written slots are no-ops on re-run when `FORCE_REINGEST=1` (default), so
  retries are safe.

Use `DO_PREALLOCATE=0` when preallocation runs in a separate step (init
container, ahead-of-time job, or manual `firecube zarr preallocate` call).

## Shared Geolocation Grids

Use a shared geolocation grid file for production runs. Without `GRIDS_FILE`,
each pod can spend 1-20 seconds and 200 MB-4 GB recomputing the same
`latitude` and `longitude` arrays.

`GRIDS_FILE` must be a filesystem path visible inside every pod or host. It is
loaded with NumPy from a local path, not through Firecube storage drivers, so do
not use an S3 URI for this value.

Generate the file once before fan-out:

```bash
firecube plugins mtg_fci_l1c geo generate \
    --resolutions 1km,2km \
    --sub-satellite-lon 0.0 \
    --output /shared/fci_grids.npz
```

Then set `GRIDS_FILE=/shared/fci_grids.npz` in every `fci-ingest.sh`
invocation.

`GEN_GRIDS=1` is useful on one host when `GRIDS_FILE` is set and the file does
not exist yet. For multi-host runs, generate grids once before starting the
hosts and leave `GEN_GRIDS=0` on the host jobs. Do not let several hosts race to
create the same `.npz` file.

If the file is missing or lacks a requested resolution, ingestion logs a warning
and computes that grid on the fly. The run can still finish, but startup time
and memory pressure increase.

This script wraps Firecube's direct-region Zarr flow. Firecube owns
preallocation, slot-range validation, chunk claims, and run records; the plugin
owns the FCI-specific schema and write intents. See Firecube's public
[Direct Region Zarr](https://eumetsat.github.io/firecube/concepts/output-formats/zarr/direct-region/),
[Parallel Zarr Writes](https://eumetsat.github.io/firecube/concepts/output-formats/zarr/parallel-writes/),
and [Direct Zarr Plugins](https://eumetsat.github.io/firecube/concepts/plugins/direct-zarr/)
docs for the core behavior.

## Sizing the Fan-Out

Each pod is a single `firecube ingest` process with `pipeline_workers=1`:

| Resource | Per pod | Notes |
|---|---|---|
| RAM | **~14.6 GiB** peak RSS | Measured baseline for FDHSI, 1 km + 2 km, all 16 channels, `pixel_time_dtype=float64`. Feature-flag reductions in [Performance Tuning → Memory](../performance-tuning.md#memory-considerations). |
| CPU | ~1 physical core | numpy / BLAS / HDF5 may internally spawn threads. When packing many pods on one host, set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` to avoid oversubscription. |

Total host requirement is roughly **`PARALLELISM × 14.6 GiB` RAM** plus one
core per pod:

| Host RAM | Safe `PARALLELISM` |
|---|---|
| 32 GiB | 2 |
| 64 GiB | 4 |
| 128 GiB | 8 |
| 256 GiB | 16 |

For higher throughput, scale across multiple hosts (see [Multi-Host Scaling](#multi-host-scaling)).

## From Manual Commands To The Script

The README shows the same production flow as separate commands: generate grids,
preallocate the store, then ingest a slot window. `scripts/fci-ingest.sh` wraps
those steps and adds fan-out, logging, and idempotent re-runs.

| Manual value | Script variable |
|---|---|
| `--input-data` | `INPUT` |
| `--target` | `TARGET` |
| `--product-name` | `PRODUCT_NAME` |
| `--option time_epoch=...` | `TIME_EPOCH` |
| `--option time_slots=...` | `TIME_SLOTS` |
| `--option fci_grids_file=...` | `GRIDS_FILE` |
| `--slot-start`, `--slot-end` | `SLOT_START`, `SLOT_END` |
| number of parallel processes | `PARALLELISM` |

Use manual commands for a small first window. Use the script once the values are
known and you need repeated windows, logs, retries, or multi-host fan-out.

## Examples

### Local disk

Ingest the first week of a full-cube axis to local storage, 1-hour pods, 8
concurrent:

```bash
TIME_EPOCH=2024-09-24 \
TIME_SLOTS=1008 \
SLOT_START=0 SLOT_END=1008 \
INPUT=file:///data/fci-zips \
TARGET=file:///data/fci_l1c.zarr \
PRODUCT_NAME=mtg-fci-l1c \
GEN_GRIDS=1 \
GRIDS_FILE=/shared/fci_grids.npz \
SLOTS_PER_POD=6 \
PARALLELISM=8 \
bash scripts/fci-ingest.sh
```

Host requirement: `8 × 14.6 GiB ≈ 117 GiB` RAM plus 8 cores.

### S3 with the `obstore` driver

For S3 targets, use `STORAGE_DRIVER=obstore`; the default `fsspec` driver
is much slower on parallel writes. The example below ingests 12 slots
(2 hours) as 12 concurrent 1-slot pods, with preallocation and grid
generation already done (`DO_PREALLOCATE=0`):

```bash
STORAGE_DRIVER=obstore \
STORAGE_TYPE=s3 \
WRITE_MODE=direct \
DO_PREALLOCATE=0 \
INPUT=file:///root/data/fci-zips \
TARGET=s3://firecube/mtg-fci-l1c.zarr/ \
PRODUCT_NAME=mtg-fci-l1c \
PRODUCT_TYPE=FDHSI \
TIME_EPOCH=2024-09-24 TIME_SLOTS=1008 \
SLOT_START=0 SLOT_END=12 \
SLOTS_PER_POD=1 PARALLELISM=12 \
GRIDS_FILE=/root/data/fci_grids.npz \
LOG_ROOT=/root/logs \
bash scripts/fci-ingest.sh
```

Host requirement: `12 × 14.6 GiB ≈ 175 GiB` RAM plus 12 cores. Reduce
`PARALLELISM` for smaller hosts, or split the window across multiple hosts.

## Multi-Host Scaling

For throughput beyond one host, generate shared grids once, preallocate the
store once, then run one `fci-ingest.sh` per host with disjoint
`SLOT_START`/`SLOT_END` windows against the same store. Every host must use
identical `TIME_EPOCH`, `TIME_SLOTS`, and `GRIDS_FILE`.

```bash
# One-time grid generation on a shared filesystem
firecube plugins mtg_fci_l1c geo generate \
    --resolutions 1km,2km \
    --sub-satellite-lon 0.0 \
    --output /shared/fci_grids.npz

# One-time preallocation (from any host)
firecube zarr preallocate mtg_fci_l1c \
    --product-name mtg-fci-l1c \
    --target s3://firecube/mtg-fci-l1c.zarr/ \
    --storage-type s3 \
    --storage-driver obstore \
    --write-mode direct \
    --input-data file:///data/fci-zips \
    --option time_epoch=2024-09-24 \
    --option time_slots=1008 \
    --option fci_grids_file=/shared/fci_grids.npz

# Host 1: slots 0–503
TIME_EPOCH=2024-09-24 TIME_SLOTS=1008 \
SLOT_START=0 SLOT_END=504 \
DO_PREALLOCATE=0 PARALLELISM=8 \
STORAGE_DRIVER=obstore STORAGE_TYPE=s3 \
INPUT=file:///data/fci-zips \
TARGET=s3://firecube/mtg-fci-l1c.zarr/ \
PRODUCT_NAME=mtg-fci-l1c \
GRIDS_FILE=/shared/fci_grids.npz \
bash scripts/fci-ingest.sh

# Host 2: slots 504–1007 (runs concurrently)
TIME_EPOCH=2024-09-24 TIME_SLOTS=1008 \
SLOT_START=504 SLOT_END=1008 \
DO_PREALLOCATE=0 PARALLELISM=8 \
STORAGE_DRIVER=obstore STORAGE_TYPE=s3 \
INPUT=file:///data/fci-zips \
TARGET=s3://firecube/mtg-fci-l1c.zarr/ \
PRODUCT_NAME=mtg-fci-l1c \
GRIDS_FILE=/shared/fci_grids.npz \
bash scripts/fci-ingest.sh
```

## Failure Recovery

The full symptom → recovery table (`ClaimConflictError`, `ResumeConflictError`,
missing slots after a run) is in
[Performance Tuning → Failure Recovery](../performance-tuning.md#failure-recovery).

Production-specific failures:

| Symptom | Cause | Recovery |
|---|---|---|
| `FAIL` lines in `$LOGDIR/results.txt` | Individual pods exited non-zero | Read the per-pod log at `$LOGDIR/pod_<start>_<end>.log`, fix the cause, then re-run with the same `SLOT_START`/`SLOT_END`. Successful slots are no-ops with `FORCE_REINGEST=1` |
| Host OOM during phase 1 | `PARALLELISM × 14.6 GiB` exceeds available RAM | Lower `PARALLELISM` or split the window across more hosts |
| Slow S3 writes with default driver | Using `fsspec` for parallel S3 | Switch to `STORAGE_DRIVER=obstore` and install the `obstore` extra |

## Operational Notes

- `pipeline_workers=1` per pod (the script sets this automatically). Scale via
  `PARALLELISM` and host count, not thread count.
- `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` in the environment when
  packing many pods on one host, to prevent BLAS/HDF5 thread oversubscription.
- Preallocate once, ingest many times. The time axis shape is fixed at
  preallocation; `TIME_EPOCH` cannot be shifted later without starting over.
- `ASSUME_YES=1` in CI or scheduled runs to skip the interactive prompt.
- `LOG_ROOT` to direct log output at a persistent location.
