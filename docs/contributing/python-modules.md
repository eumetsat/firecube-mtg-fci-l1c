# Python Module Responsibilities

Use this reference before changing code in the MTG FCI L1C plugin. It defines
where each kind of behavior belongs.

## Core edit surfaces

| File | Responsibility | Edit when | Do not put here |
|------|----------------|-----------|-----------------|
| `src/firecube_mtg_fci_l1c/schema.py` | Source functions + `VARIABLES` list; **single user-facing edit target** for adding variables. Source functions are **pure projections** onto pre-loaded `VariableContext` payloads and perform no I/O. Imports and re-exports `Variable`, `VariableContext`, `build_specs` from `_variable.py` | Add or modify a Zarr variable (source function + `Variable(...)` entry) | Spec-building logic, chunk/shard math, dataclass definitions, any I/O |
| `src/firecube_mtg_fci_l1c/ingestor.py` | DirectZarr orchestration: product resolution, slot model, batch groups, static intent emission, per-nc_part intent emission | Wire schema variables to real data, add guarded static emission, add per-batch write flow | Low-level NetCDF parsing details, projection math, constants tables |
| `src/firecube_mtg_fci_l1c/config.py` | Operator-facing `--option` fields and parsing helpers | Add a user-visible option or validate config | Schema attrs, data reads, write intent construction |
| `src/firecube_mtg_fci_l1c/plugin_cli.py` | Plugin CLI commands under `firecube plugins mtg_fci_l1c` | Add operator/developer commands | Runtime ingest logic |

## Data and support modules

| File | Responsibility | Edit when | Do not put here |
|------|----------------|-----------|-----------------|
| `src/firecube_mtg_fci_l1c/_data.py` | Input ZIP identification: product type, observation timestamp, filename validation, mixed-product rejection | Change how source ZIPs are recognized before streaming | Full-array decoding, NetCDF nc_part reading, product tables |
| `src/firecube_mtg_fci_l1c/_constants.py` | Product constants: product identifiers, channel/resolution tables, collection ids, repeat-cycle constants | Add or correct immutable product metadata | Filename parsing, validation functions, I/O, schema construction |
| `src/firecube_mtg_fci_l1c/_group_plan.py` | `GroupPlan` frozen dataclass + `resolve_group_plans()` factory. Single source of truth for `(product_type, resolution, group, dimsize, logical_channels, nc_channels)`. Used by `zarr_schema()`, `build_write_intents()`, `get_batch_groups()`, `global_expected_time_count()` | Add a new per-resolution attribute to the resolved plan; change how config + product_type resolve into groups | Source functions, payload construction, data reads |
| `src/firecube_mtg_fci_l1c/_variable.py` | Variable primitive and spec infrastructure: `Variable`, `VariableContext` frozen dataclasses, `variable_enabled()`, `_build_array_spec()`, `build_specs()` implementation, chunk/shard helpers. Also owns inlined spec helpers: `_validate_shard_override`, `_static_2d_chunks`, `_byte_budgeted_4d_shard`, `_copy_attrs` | Change the Variable dataclass fields, how dims map to ZarrArraySpec shapes, or how build_specs constructs groups | Source functions, VARIABLES list, product constants, data reads |
| `src/firecube_mtg_fci_l1c/_streaming.py` | Low-level FCI nc_part access: `NCPartReader`, `TimeMapAccumulator`, `expand_pixel_time`. Also exposes `ChannelSlicePayload` (pre-loaded counts/pixel_quality/pixel_time arrays) and `load_channel_slice()` factory used by the ingestor's spatial phase. | Change how NetCDF BODY/TRAIL data, row ranges, calibration, or pixel-time maps are read | Zarr specs, Firecube write orchestration, config parsing |
| `src/firecube_mtg_fci_l1c/_scratch.py` | Per-batch ZIP extraction and cleanup (`BatchScratch`, stdlib only) | Change scratch root naming, extraction safety, or cleanup | NetCDF parsing, schema, Firecube public APIs |

## Geolocation subpackage

| File | Responsibility | Edit when | Do not put here |
|------|----------------|-----------|-----------------|
| `src/firecube_mtg_fci_l1c/geolocation/__init__.py` | Re-exports `LatLonProvider` | Change the public surface of the subpackage | Projection math, grid loading |
| `src/firecube_mtg_fci_l1c/geolocation/projection.py` | FCI GEOS projection math (`compute_latlon`) | Correct projection formulas or constants | Ingest orchestration, file discovery |
| `src/firecube_mtg_fci_l1c/geolocation/grids.py` | NPZ grid file loader (`FciGrids`) | Change the generated-grid file format or metadata handling | Projection formulas, Zarr writes |
| `src/firecube_mtg_fci_l1c/geolocation/provider.py` | Runtime latitude/longitude provider (`LatLonProvider`): Stateless per-(grids_file, resolution_m) cache with `threading.Lock`; NPZ lookup or computed fallback, result memoized per key | Change how static lat/lon grids are loaded or computed for intents | Direct Zarr writes, schema attrs |

## Test files

| File | Responsibility |
|------|----------------|
| `tests/test_variable.py` | Variable + VariableContext dataclass behavior, VARIABLES registry invariants, build_specs output for each flag combo |
| `tests/test_ingestor.py` | Ingestor hooks, config wiring, source filtering, batch grouping |
| `tests/test_integration.py` | End-to-end Zarr output behavior and store read-back checks |
| `tests/test_data.py` | ZIP filename/product/timestamp helper behavior |

## Boundary rules

- Put immutable product tables in `_constants.py`.
- Put source-file recognition in `_data.py`.
- Put NetCDF nc_part reads in `_streaming.py`.
- Put source functions and the `VARIABLES` list in `schema.py`.
- Put the Variable/VariableContext dataclasses and spec-building logic in `_variable.py`.
- Put cross-hook resolution planning (which groups, channels, dims) in `_group_plan.py`.
- Put pre-loaded channel payload construction in `_streaming.py` (`ChannelSlicePayload`, `load_channel_slice`).
- Put batch orchestration and "when to emit what" in `ingestor.py`.
- Put geolocation math and grid loading in the `geolocation/` subpackage.

## Related

- [Contributor Notes](index.md)
- [How to Add a Zarr Variable](add-zarr-variable.md)
- [Documentation Index](../index.md)
