# Firecube MTG FCI L1C Plugin Documentation

Use this index to choose the right page for the task.

## User Path

| Task | Page |
|---|---|
| Install the plugin and run one local ingest | [README](../README.md) |
| Understand groups, variables, channel names, calibration, and quality bits | [FCI Data in Zarr](fci-data-in-zarr.md) |
| Configure plugin options, script environment variables, grids, chunks, and shards | [Customization](customization.md) |

## Operator Path

| Task | Page |
|---|---|
| Run production ingestion with `scripts/fci-ingest.sh` | [Production Ingestion Guide](guides/production-ingestion.md) |
| Size memory, choose layout options, and avoid concurrency traps | [Performance Tuning](performance-tuning.md) |
| Review scaling plots and benchmark workload notes | [Performance Benchmarks](reference/performance-benchmarks.md) |

## Contributor Path

| Task | Page |
|---|---|
| Prepare a pull request | [Contributing](../CONTRIBUTING.md) |
| Find plugin internals and contributor-only references | [Contributor Notes](contributing/index.md) |
| Add or change Zarr variables | [How to Add a Zarr Variable](contributing/add-zarr-variable.md) |
| Find Python module ownership | [Python Module Responsibilities](contributing/python-modules.md) |
