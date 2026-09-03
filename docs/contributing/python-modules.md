# Python Module Responsibilities

Use this reference before changing code in the MTG FCI L1C plugin. It defines
where each kind of behavior belongs.

## How one ZIP becomes Zarr regions

The hooks below run in this order. Line numbers move; the names do not.
Everything not listed is a helper called from one of these.

| # | Hook | Module | What happens |
|---|------|--------|--------------|
| 1 | `discover_source_files` | **core** | Not overridden. Core walks `--input-data` (local path or remote URI) and collects candidate files. |
| 2 | `filter_item` | `ingestor.py` -> `_data.py` | `is_valid_fci_zip()` keeps only `FCI-1C-RRAD` ZIPs with a parseable timestamp. |
| 3 | `get_batch_groups` | `ingestor.py` | Groups filtered items into batches. |
| 4 | `index_spec` | `ingestor.py` -> `_group_plan.py` | Declares one `TimeAxis.observed` axis per resolution group. `slot_count` is `None` when no extent is configured (serial mode). |
| 5 | `zarr_schema` | `_variables.py` -> `_schema.py` | `build_specs()` walks `VARIABLES`; `_build_array_spec()` dispatches each `dims` signature to its builder. |
| 6 | `prepare_batch_data` | `ingestor.py` | Registers the batch id before any decoding. |
| 7 | `build_write_intents` | `ingestor.py` | Extracts the batch's ZIPs in parallel, then per ZIP calls `_intents_for_zip()`, which calls `_intents_for_plan()` per resolution group. Those call `_emit_static_intents`, `_emit_time_channel_intents`, `_emit_spatial_intents`. All nc_part reads go through one `SharedNcPartReader`. |
| 8 | *compile* | **core** | Core resolves each `IndexedWrite` coordinate to a slot index and auto-emits the time-coordinate write. The plugin never computes a slot index. |
| 9 | `cleanup_batch_data` | `ingestor.py` -> `_scratch.py` | Pops the batch, then closes its resources **outside** the lock via `BatchResourceRegistry.teardown()`. |

Two rules follow from this shape, and most confusion comes from missing them:

- **Payloads are lazy.** `_emit_spatial_intents` attaches a callable, not an
  array. Core resolves it at dispatch time, so the reader's file handles must
  stay open until the batch is torn down — that is why the shared reader is
  registered as a batch resource rather than scoped with `with`.
- **Core owns the time axis.** The plugin declares the axis and emits
  `IndexedWrite` keyed by `coordinate=`; slot arithmetic and the time
  coordinate write are core's, not the plugin's.

## Core edit surfaces

| File | Responsibility | Edit when | Do not put here |
|------|----------------|-----------|-----------------|
| `src/firecube_mtg_fci_l1c/_variables.py` | Source functions + `VARIABLES` list; **single user-facing edit target** for adding variables. Source functions are **pure projections** onto pre-loaded `VariableContext` payloads and perform no I/O. Imports and re-exports `Variable`, `VariableContext`, `build_specs` from `_schema.py` | Add or modify a Zarr variable (source function + `Variable(...)` entry) | Spec-building logic, chunk/shard math, dataclass definitions, any I/O |
| `src/firecube_mtg_fci_l1c/ingestor.py` | DirectZarr orchestration: product resolution, slot model, batch groups, static intent emission, per-nc_part intent emission. `build_write_intents()` delegates per-ZIP work to `_intents_for_zip()` and per-resolution-group work to `_intents_for_plan()`. Emits `IndexedWrite` keyed by `coordinate=` so core resolves the slot; per-batch resources are held in core's `BatchResourceRegistry` | Wire schema variables to real data, add guarded static emission, add per-batch write flow | Low-level NetCDF parsing details, projection math, constants tables |
| `src/firecube_mtg_fci_l1c/config.py` | Operator-facing `--option` fields and parsing helpers | Add a user-visible option or validate config | Schema attrs, data reads, write intent construction |
| `src/firecube_mtg_fci_l1c/plugin_cli.py` | Plugin CLI commands under `firecube plugins mtg_fci_l1c` | Add operator/developer commands | Runtime ingest logic |

## Data and support modules

| File | Responsibility | Edit when | Do not put here |
|------|----------------|-----------|-----------------|
| `src/firecube_mtg_fci_l1c/_data.py` | Input ZIP identification: product type, observation timestamp, filename validation, mixed-product rejection | Change how source ZIPs are recognized before streaming | Full-array decoding, NetCDF nc_part reading, product tables |
| `src/firecube_mtg_fci_l1c/_constants.py` | Product constants: product identifiers, channel/resolution tables, collection ids, repeat-cycle constants | Add or correct immutable product metadata | Filename parsing, validation functions, I/O, schema construction |
| `src/firecube_mtg_fci_l1c/_group_plan.py` | `GroupPlan` frozen dataclass + `resolve_group_plans()` factory. Single source of truth for `(product_type, resolution, group, dimsize, logical_channels, nc_channels)`. Used by `zarr_schema()`, `build_write_intents()`, `get_batch_groups()`, `global_expected_time_count()` | Add a new per-resolution attribute to the resolved plan; change how config + product_type resolve into groups | Source functions, payload construction, data reads |
| `src/firecube_mtg_fci_l1c/_schema.py` | Variable primitive and spec infrastructure: `Variable`, `VariableContext` frozen dataclasses, `variable_enabled()`, `build_specs()`, and array-spec construction: `_build_array_spec()` resolves per-variable inputs into `_ArraySpecInputs`, then dispatches through the `_ARRAY_SPEC_BUILDER` mapping (dims signature -> one `_build_array_*` builder). Add a dims signature by writing a builder and adding one mapping entry. Chunk/shard helpers live here too. Also owns inlined spec helpers: `_validate_shard_override`, `_static_2d_chunks`, `_byte_budgeted_4d_shard`, `_copy_attrs` | Change the Variable dataclass fields, how dims map to ZarrArraySpec shapes, or how build_specs constructs groups | Source functions, VARIABLES list, product constants, data reads |
| `src/firecube_mtg_fci_l1c/_decode.py` | Low-level FCI nc_part access: `NCPartReader`, `TimeMapAccumulator`, `expand_pixel_time`, `list_fci_nc_parts`. `SharedNcPartReader` is the batch-scoped file-handle cache every read phase goes through (time-map, row-range, calibration, spatial) — do not open `NCPartReader` directly from the ingestor. `ChunkOwnedAssembler` assembles output chunks from at most two nc_parts with bounded caches. Also exposes `ChannelSlicePayload` and the `load_channel_slice()` factory used by the spatial phase. | Change how NetCDF BODY/TRAIL data, row ranges, calibration, or pixel-time maps are read | Zarr specs, Firecube write orchestration, config parsing |
| `src/firecube_mtg_fci_l1c/_scratch.py` | Per-batch ZIP extraction and cleanup (`BatchScratch`). `extract_zips_parallel()` is a thin wrapper over core's `extract_all_from_zips`; `close()` is the `BatchResourceRegistry` alias that hands scratch removal to a daemon thread | Change scratch root naming, extraction safety, or cleanup | NetCDF parsing, schema, re-implementing extraction that core already provides |

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
| `tests/test_schema.py` | Variable + VariableContext dataclass behavior, VARIABLES registry invariants, build_specs output for each flag combo |
| `tests/test_ingestor.py` | Ingestor hooks, config wiring, source filtering, batch grouping |
| `tests/test_integration.py` | End-to-end Zarr output behavior and store read-back checks |
| `tests/test_data.py` | ZIP filename/product/timestamp helper behavior |
| `tests/test_batch_lifecycle.py` | Per-batch resource registration and teardown |
| `tests/test_scratch.py` | Scratch extraction, zip-slip guard, cleanup |
| `tests/test_slot_index.py`, `tests/test_slot_index_non_parallel.py` | Slot resolution for the parallel and serial paths |
| `tests/test_golden_schema.py`, `tests/test_golden_output.py` | Schema and output snapshots; a diff here means the emitted cube changed |
| `tests/test_core_import_boundary.py` | Guards which core symbols the plugin may import |

The repository has ~25 test modules; the table lists the ones you are most
likely to need. Run the full suite rather than a subset before proposing a
change.

## Boundary rules

- Put immutable product tables in `_constants.py`.
- Put source-file recognition in `_data.py`.
- Put NetCDF nc_part reads in `_decode.py`.
- Put source functions and the `VARIABLES` list in `_variables.py`.
- Put the Variable/VariableContext dataclasses and spec-building logic in `_schema.py`.
- Put cross-hook resolution planning (which groups, channels, dims) in `_group_plan.py`.
- Put pre-loaded channel payload construction in `_decode.py` (`ChannelSlicePayload`, `load_channel_slice`).
- Put batch orchestration and "when to emit what" in `ingestor.py`.
- Put geolocation math and grid loading in the `geolocation/` subpackage.
- Before writing a helper, check `firecube.core.api` and core's
  [Core Utilities reference](https://eumetsat.github.io/firecube/reference/core-utilities/)
  for an existing one. Re-implementing core behaviour is how the plugin
  silently loses features: a hand-rolled `discover_source_files` override
  dropped the `storage_config` argument and so could not read remote (S3)
  sources at all. Override a core hook only for genuinely FCI-specific
  behaviour, such as `filter_item` recognising FCI-1C-RRAD ZIPs.

## Related

- [Contributor Notes](index.md)
- [How to Add a Zarr Variable](add-zarr-variable.md)
- [Documentation Index](../index.md)
