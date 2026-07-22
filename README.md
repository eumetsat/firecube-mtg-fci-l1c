# Firecube MTG FCI L1C Ingestor Plugin

Firecube plugin for ingesting MTG FCI (Flexible Combined Imager) Level 1C
products into direct-region Zarr stores.

## Contents

| Need | Read |
|---|---|
| Install the plugin | [Installation](#installation) |
| Run one local ingest | [Quickstart](#quickstart) |
| Run production ingestion | [Production Ingestion](#production-ingestion) |
| Navigate all docs | [docs/index.md](docs/index.md) |
| Understand the Zarr layout, variables, channel names, and quality bits | [FCI Data in Zarr](docs/fci-data-in-zarr.md) |
| Configure plugin options, script variables, grids, chunks, or shards | [Customization](docs/customization.md) |
| Run day/week/month production ingestion windows | [Production Ingestion Guide](docs/guides/production-ingestion.md) |
| Size memory, choose chunks/shards, or reason about concurrency | [Performance Tuning](docs/performance-tuning.md) |
| Review benchmark plots and workload notes | [Performance Benchmarks](docs/reference/performance-benchmarks.md) |
| Check SBOM and dependency-license information | [Software Bill Of Materials](#software-bill-of-materials) |
| Contribute code or docs | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Installation

Requirements:

- [uv](https://docs.astral.sh/uv/)
- Firecube 0.1.4 or newer

Clone the plugin and install it into the Firecube environment:

```bash
git clone https://github.com/eumetsat/firecube-mtg-fci-l1c.git
cd firecube-mtg-fci-l1c
firecube plugins install -e .
```

Check that Firecube can discover the plugin:

```bash
firecube plugins list
firecube plugins describe mtg_fci_l1c
```

## Quickstart

The plugin processes two MTG FCI L1C product families:

- **FDHSI**: Full Disk High Spectral Imagery (collection `EO:EUM:DAT:0662`)
- **HRFI**: High-Resolution Fast Imagery (collection `EO:EUM:DAT:0665`)

Download products from the [EUMETSAT Data Store](https://data.eumetsat.int/)
with [eumdac](https://user.eumetsat.int/resources/user-guides/eumetsat-data-access-client-eumdac-guide):

```bash
eumdac download -c EO:EUM:DAT:0662 \
    -s 2024-09-24T12:00:00 \
    -e 2024-09-24T13:00:00 \
    -o /path/to/fci-zips
```

Ingest a local FDHSI slot into a local Zarr store:

```bash
firecube ingest mtg_fci_l1c \
    --input-data /path/to/fci-zips \
    --target file:///path/to/output.zarr \
    --output-format zarr \
    --write-mode staged
```

To ingest only one channel, pass the `channels` option:

```bash
firecube ingest mtg_fci_l1c \
    --input-data /path/to/fci-zips \
    --target file:///path/to/output-vis-06.zarr \
    --output-format zarr \
    --write-mode staged \
    --option channels=vis_06
```

See [Customization](docs/customization.md) for all plugin options and
`scripts/fci-ingest.sh` environment variables.

The output has one Zarr group per resolution:

```
output.zarr/
├── data_500m/     # HRFI only
├── data_1km/      # FDHSI and HRFI
└── data_2km/      # FDHSI only
```

The main arrays are:

| Variables | Shape | Notes |
|---|---|---|
| `counts`, `pixel_quality`, `pixel_time` | `(time, y, x, channel)` | per-pixel data |
| `slope`, `offset` | `(time, channel)` | `radiance = counts * slope + offset` |
| `latitude`, `longitude` | `(y, x)` | static, computed once per group |
| `x`, `y` | `(x,)`, `(y,)` | GEOS projection angles |
| `time`, `channel_name`, `spatial_ref` | `(time,)`, `(channel,)`, `()` | coordinates and CRS metadata |

## Production Ingestion

Production runs write into a preallocated Zarr store and ingest explicit slot
windows. The time axis is fixed by three values:

- **Cadence**: FCI repeats every 10 minutes, so there are 144 slots per day.
- **`time_epoch`**: UTC-midnight date that maps to slot 0. For a full MTG FCI
  L1C cube, use `2024-09-24`, the first FCI L1C availability date in the
  EUMETSAT Data Store.
- **`time_slots`**: total length of the preallocated time axis. Use `144` for
  one day, `1008` for one week, or `4320` for 30 days. For a full cube, choose
  the horizon up-front; a long sparse axis, such as several years, is valid if
  you need it.

Slot indices use:

```
slot = (timestamp_utc - time_epoch_utc_midnight) / cadence
```

For example, with `time_epoch=2024-09-24`, `slot_start=0` and `slot_end=144`
cover 2024-09-24 00:00 UTC through, but not including, 2024-09-25 00:00 UTC.

Start with a small manual window. Once the values are proven, use
[`scripts/fci-ingest.sh`](scripts/fci-ingest.sh) for day, week, or month
windows.

### Manual Steps

Set the shared values:

```bash
STORE=file:///data/fci_l1c.zarr
INPUT=/data/fci-zips
GRIDS=/shared/fci_grids.npz
EPOCH=2024-09-24
SLOTS=1008      # one week at 10-minute cadence; increase for your full horizon
```

1. Generate shared geolocation grids once. In production, put this file on a
   filesystem path visible to every pod or host.

```bash
firecube plugins mtg_fci_l1c geo generate \
    --resolutions 1km,2km \
    --sub-satellite-lon 0.0 \
    --output "$GRIDS"
```

2. Preallocate the store before any pod writes data:

```bash
firecube zarr preallocate mtg_fci_l1c \
    --product-name mtg-fci-l1c \
    --target "$STORE" \
    --storage-type local \
    --storage-driver fsspec \
    --write-mode direct \
    --input-data "$INPUT" \
    --option product_type=FDHSI \
    --option time_epoch="$EPOCH" \
    --option time_slots="$SLOTS" \
    --option fci_grids_file="$GRIDS"
```

3. Ingest a small slot window:

```bash
firecube ingest mtg_fci_l1c \
    --input-data "$INPUT" \
    --target "$STORE" \
    --storage-type local \
    --storage-driver fsspec \
    --write-mode direct \
    --option product_type=FDHSI \
    --option time_epoch="$EPOCH" \
    --option time_slots="$SLOTS" \
    --option fci_grids_file="$GRIDS" \
    --slot-start 0 \
    --slot-end 6
```

4. Verify the store has the expected groups:

```bash
python - <<'PY'
import zarr

root = zarr.open_group("/data/fci_l1c.zarr", mode="r")
print(sorted(root.group_keys()))
PY
```

For larger windows, use the production helper with the same axis values and
grid file:

```bash
TIME_EPOCH="$EPOCH" TIME_SLOTS="$SLOTS" \
INPUT="$INPUT" TARGET="$STORE" \
GRIDS_FILE="$GRIDS" GEN_GRIDS=0 \
SLOT_START=0 SLOT_END=144 \
SLOTS_PER_POD=6 PARALLELISM=8 \
bash scripts/fci-ingest.sh
```

See the [Production Ingestion Guide](docs/guides/production-ingestion.md) for
S3 setup, multi-host runs, logging, and recovery. See
[Performance Tuning](docs/performance-tuning.md) before raising
`PARALLELISM`.

This plugin uses Firecube's direct-region Zarr path. The plugin declares FCI
schema and write intents. Firecube owns preallocation, chunk claims, run
records, and coordinated writes. See Firecube's public docs for the core model:
[Direct Region Zarr](https://eumetsat.github.io/firecube/concepts/output-formats/zarr/direct-region/),
[Parallel Zarr Writes](https://eumetsat.github.io/firecube/concepts/output-formats/zarr/parallel-writes/),
and [Direct Zarr Plugins](https://eumetsat.github.io/firecube/concepts/plugins/direct-zarr/).

## Software Bill Of Materials

Regenerate the CycloneDX 1.5 SBOM and the dependency-license report locally with:

```bash
mkdir -p .reports
uv export --format cyclonedx1.5 --all-groups --all-extras --output-file .reports/sbom.cdx.json
uv run --isolated --all-groups --all-extras --with hatchling --with pip-licenses pip-licenses --format=json --with-urls > .reports/dependency-licenses.json
```

The SBOM covers all uv dependency groups and extras. Use the dependency license
report and package metadata for license review; do not rely on CycloneDX license
fields alone.

### Included Components

The repository does not vendor third-party source or binary components.

### Direct Runtime Dependencies

The following dependencies are not included in the package but are required at
install/runtime:

| dependency | version | license | copyright | home_url | comments |
| --- | --- | --- | --- | --- | --- |
| `firecube` | 0.1.4 | Apache-2.0 |  | https://github.com/eumetsat/firecube | Required runtime dependency |
| `xarray` | 2026.4.0 | Apache-2.0 |  | https://xarray.dev/ | Direct dependency. |
| `numpy` | 2.5.0 | BSD-3-Clause |  | https://numpy.org | Direct dependency. |
| `h5netcdf` | 1.8.1 | BSD-3-Clause |  | https://h5netcdf.org | Direct dependency; NetCDF/HDF5 reader for FCI nc_parts. |

### Direct Build, Edit, And Test Dependencies

The following dependencies are only required for building, editing, or testing:

| dependency | version | sw type | license | copyright | home_url | comments |
| --- | --- | --- | --- | --- | --- | --- |
| `hatchling` | 1.30.1 | Development tools | MIT |  | https://hatch.pypa.io/latest/ | Build backend (`[build-system].requires`). |
| `pytest` | 9.1.1 | Development tools | MIT |  | https://docs.pytest.org/en/latest/ | Direct `dev` dependency. |
| `ruff` | 0.15.20 | Development tools | MIT |  | https://docs.astral.sh/ruff | Direct `dev` dependency. |
| `mypy` | 2.1.0 | Development tools | MIT |  | https://www.mypy-lang.org/ | Direct `dev` dependency. |
| `matplotlib` | 3.11.0 | Development tools | PSF-2.0 |  | https://matplotlib.org | Direct `dev` dependency group entry (notebooks/plots); PSF-based, BSD-compatible. |

## Copyright and License

Copyright © EUMETSAT 2025-2026

The provided code and instructions are licensed under [Apache License, Version 2.0](./LICENSE).

Contact [EUMETSAT](http://www.eumetsat.int) for details on the usage and distribution terms.

## Contributing

See the [Contributor Guide](docs/contributing/index.md) for development setup,
module architecture, pre-commit checks, commit conventions, Zarr variable
recipes, and contributor-only implementation notes.

## Authors

See [AUTHORS.md](./AUTHORS.md).
