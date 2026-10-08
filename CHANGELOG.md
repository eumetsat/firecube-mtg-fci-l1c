# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Loose chunk input: `--input-data` may be a directory or prefix (local or S3, through Firecube source discovery) of unpacked BODY chunk `.nc` files, and the TRAIL chunk if present, instead of ZIP files. The chunk files of one repeat cycle form a scene; a scene may be partial. A scene is labelled with the nominal start of its repeat cycle (cycle 1 is `00:00:00`); ZIPs keep the sensing start floored to the minute. Quicklook and `.xml` files next to the chunks are ignored. See [Customization](docs/customization.md#input-forms).
- `partial_chunk` option (`fill` default, or `error`): `fill` leaves rows that no input file covers unwritten, so they keep the fill value and rows from earlier ingests survive; `error` fails the batch of a partial scene before it writes anything. See [Customization](docs/customization.md#partial-scenes).
- `body_chunks=[first,last]` option and the stripe store: with inclusive BODY chunk numbers (1 to 40), each group holds only those chunks' rows, widened to the group's output-chunk grid, at full width. Groups carry `body_chunks`, `disk_row_start` and `disk_row_stop` attributes, the window is part of the store identity, pixel data is read only from files inside the window, and every file's rows are checked against a row table. A stripe store cannot be grown yet. See [Customization](docs/customization.md#stripe-stores).
- `scripts/fci-ingest.sh`: `BODY_CHUNKS` and `PARTIAL_CHUNK` add `--option body_chunks=...` and `--option partial_chunk=...` to preallocation and every pod when set; invalid values stop the script before it calls Firecube. The script also stops the shell from expanding `[...]` in option values as file globs.
- Satellite position and Sun–Earth distance (issue #15): `subsatellite_latitude`, `subsatellite_longitude`, `platform_altitude` and `sun_earth_distance` as `(time,)` variables in every group. The satellite position is the mean over the repeat cycle of the `state/platform` tables (`NaN` when a product lacks them). `sun_earth_distance` is in AU, computed with astropy for the slot time; the product's own `earth_sun_distance` is not used because it holds the Sun–satellite distance. Adds `astropy>=7.0` as a runtime dependency. Golden snapshots regenerated.
- Radiance conversion constants as per-acquisition `(time, channel)` variables in every group (issue #14): `radiance_unit_conversion_coefficient`, `radiance_to_bt_conversion_coefficient_{wavenumber,a,b}`, `radiance_to_bt_conversion_constant_{c1,c2}`, `channel_effective_solar_irradiance`. Read from the `data/<channel>/measured` scalars of each product, like `slope`/`offset`. float32, `NaN` where the product marks a constant not applicable to a channel. Enabled with `include_calibration`. Slots ingested before a store had these variables read as `NaN`; every product ingested afterwards fills them. Golden snapshots regenerated.
- IR 3.8 dual-gain calibration (issue #16): `warm_slope` and `warm_offset` `(time, channel)` variables from the `warm_scale_factor`/`warm_add_offset` attributes of `effective_radiance`. They convert `ir_38` counts above 4095, exist only in groups that contain `ir_38` (FDHSI `data_2km`, HRFI `data_1km`), and are `NaN` for the other channels there. Enabled with `include_calibration`. Golden snapshots regenerated.
- [#13](https://github.com/eumetsat/firecube-mtg-fci-l1c/issues/13) `flat_store` option (default `false`): with exactly one effective resolution, the variables are written at the store root instead of `data_<res>/`, so `xr.open_zarr(store)` works without `group=`. More than one effective resolution fails with a configuration error before anything is written. Chunk and shard override keys stay `data_<res>`. See [Customization](docs/customization.md#flat-store-layout).
- [#13](https://github.com/eumetsat/firecube-mtg-fci-l1c/issues/13) `scripts/fci-ingest.sh`: `FLAT_STORE=1` adds `--option flat_store=true` to preallocation and every pod.
- [#13](https://github.com/eumetsat/firecube-mtg-fci-l1c/issues/13) `fix-fillvalue` works on flat stores; it detects the layout from the store and refuses empty, non-FCI, or mixed stores.
- `_schema.py`: time coordinate spec now uses `chunks=None` (dense chunk resolution delegated to core `preallocate`) and declares CF standard_name, long_name, axis attributes. Golden snapshots regenerated.

### Changed

- **BREAKING:** `product_type` is now a required option (`FDHSI` or `HRFI`). The plugin no longer detects it from the input and no longer falls back to `FDHSI`; without it `firecube ingest` and `firecube zarr preallocate` stop with `product_type is required`. Add `--option product_type=...` to existing commands. `scripts/fci-ingest.sh` already passed `PRODUCT_TYPE` (default `FDHSI`).
- **BREAKING:** the supported floor is now `firecube>=0.1.7` (was `>=0.1.5`).
- Source discovery drops files that are neither FCI ZIPs nor chunk files, and stops with a configuration error when the source holds ZIP and chunk files together (use `--input-filters '["!*.zip"]'` or `'["!*.nc"]'`) or files of the other product type (use `--input-filters`).
- ZIP member selection now accepts any satellite name in FCI chunk names, as loose input already did.
- `body_chunks` joins the resume slice identity. Spans written by earlier versions carry no value for it, so a single-pod run (no `--slot-start`/`--slot-end`) into such a store can report `ResumeConflictError`; continue with `--option resume_existing=true` or overwrite with `--option force_reingest=true`.
- The `time` coordinate now holds each product's start time floored to the full minute (issue #20), for example `12:20:00` instead of `12:20:06`, so `ds.sel(time="2026-09-28T12:20:00")` works. Slots without a product stay `NaT`. Slot placement is unchanged. **Existing stores with unfloored labels need a fresh ingest**: on a preallocated store, `firecube zarr preallocate` and `direct`-mode ingests stop with a `time slot … drift` error; a store ingested without preallocation ends up with mixed labels.
- Channel names are now the text coordinate `channel` of the `channel` dimension (issue #21), stored as the Zarr v3 `string` data type, so `ds.sel(channel="ir_105")` works without `assign_coords`. **The bytes variable `channel_name` is removed from new stores**; read `ds.channel` instead. Existing stores keep their `channel_name` array and gain `channel` on their next ingest in `direct` write mode; in `staged` write mode they are left unchanged. Golden snapshots regenerated.
- The lockfile and CI use firecube 0.1.7.
- [#13](https://github.com/eumetsat/firecube-mtg-fci-l1c/issues/13) `fix-fillvalue` now exits with an error on a store that holds no FCI arrays or groups, or that mixes root arrays with `data_<res>/` groups; it used to report such stores as "missing" and exit 0.
- A `resolutions` or `channels` selection that leaves no resolution to write is rejected with a configuration error naming both options, instead of failing inside Firecube.
- Plugin simplification pass. Behaviour-preserving except where noted:
  - `build_write_intents` split into `_intents_for_zip` / `_intents_for_plan` (182 -> 100 lines; maximum nesting 7 blocks -> 4).
  - Write intents are now `IndexedWrite` keyed by `coordinate=`; core resolves the slot index and auto-emits the time-coordinate write. `_emit_timestamp_intents` and all `ts_index` plumbing removed.
  - `index_spec()` no longer returns `None` in serial mode; it always declares the axis with `slot_count=None`. **This adds `firecube_resolved_index` and `firecube_resolved_index_identity_hash` to the store's root attrs on serial runs, which previously had none.** Parallel runs are unaffected. A serial ingest into a cube built with a configured extent will present `size: null` and fail identity verification; re-ingest to a new store in that case.
  - `RegularTimeAxis(...)` replaced by the equivalent `TimeAxis.observed(...)` (identical index identity).
  - Per-batch resources now use core's `BatchResourceRegistry`; the second `_retained_batch_scratches` registry and its cleanup path are gone.
  - All nc_part reads (time-map, row-range, calibration, spatial) route through `SharedNcPartReader`; the time-map and row-range phases no longer open their own `NCPartReader` (~120 fewer file opens per ZIP).
  - `_build_array_spec` is now a `dims -> builder` dispatch table over `_ArraySpecInputs` instead of a 125-line if-chain.
  - `config.__post_init__` split into per-concern `_validate_*` methods.

### Removed

- Product-type auto-detection from the input and the silent `FDHSI` fallback (see Changed).
- `FCI_PROJ_OFFSET_RAD` from `_constants.py` (internal, obsolete with the #8 fix).
- The former `discover_source_files` override, which omitted `storage_config` and so could not read remote (S3) sources. The current override calls Firecube's discovery and only classifies what it returns.

### Fixed

- The performance docs quoted a stale 14.6 GiB peak per worker. A full-disk FDHSI slot peaks at about 2.0 GiB (measured, see [Performance Tuning](docs/performance-tuning.md#memory-considerations)), and the docs no longer list per-option savings that were not re-measured.
- The time-axis index now honours the `channels` filter: a channel selection that leaves a resolution without channels no longer declares that resolution's group.
- `_scratch.register_cleanup_thread` now prunes finished threads. One thread was registered per batch and never released, so the registry grew by one entry for every batch a process handled; only threads still running are retained now.
- Batch resource teardown no longer runs while `_batch_resources_lock` is held. Closing a batch's readers closes ~40 NetCDF handles; holding the lock across that blocked concurrent `prepare_batch_data` calls and dispatch-time payload lookups.
- `fix-fillvalue` documentation now describes the command as a repair tool for cubes written before core stamped `_FillValue` itself, rather than a workaround for current behaviour. Current core emits the attribute during ingest; the command remains for older stores.
- [#8](https://github.com/eumetsat/firecube-mtg-fci-l1c/issues/8) `x`/`y` projection coordinates were shifted west/south by 1.00 px (1 km), 0.75 px (2 km) and 1.50 px (500 m); they now land on the pixel centres of `latitude`/`longitude` and match the L1C files' own `x`/`y`. Data and lat/lon were never affected. Existing stores keep the old values (static arrays are write-once, re-ingest raises `SchemaDriftError`): ingest into a new store.
- `compute_latlon` docstring: grids are in FCI native order (row 0 = south, col 0 = west), not north-to-south.

## [0.1.5] - 2026-08-13

### Added

- `projection_units` config option: `meter` (default), `metre` (alias), `radian`. Applies to both `x` and `y` coordinate arrays. See [Customization](docs/customization.md) for usage.
- `firecube plugins mtg_fci_l1c fix-fillvalue --store <path> [--yes-i-really-mean-it]` — post-ingest offline workaround for [#2](https://github.com/eumetsat/firecube-mtg-fci-l1c/issues/2). Stamps CF `_FillValue` on numeric arrays so xarray masks fill pixels to NaN on read. The `_FillValue` attribute is reserved by firecube-core and cannot be set through the plugin schema, so operators need to run this after each ingest until the upstream auto-emit fix ships.
- `MTG_PERSPECTIVE_POINT_HEIGHT_M` constant in `_constants.py` consolidating the geostationary altitude value.

### Changed

- Default projection units for `x`/`y` are now metres. `standard_name` becomes `projection_x_coordinate` / `projection_y_coordinate`; `units` becomes `m`. Set `--option projection_units=radian` for the angular convention.

### Fixed

- [#1](https://github.com/eumetsat/firecube-mtg-fci-l1c/issues/1) x-axis: `x` projection coordinate is now east-positive (was positive-westward). Consumers georeferencing via `x` + `spatial_ref` (rioxarray, cartopy, GDAL, satpy) render the disk correctly.
- [#3](https://github.com/eumetsat/firecube-mtg-fci-l1c/issues/3) time attrs: removed redundant `units` and `calendar`; xarray manages these via encoding on native `datetime64[s]`. Unblocks `xr.open_zarr(...).to_zarr(...)` round-trips.

## [0.1.4] - 2026-06-28

### Added

- `zarr_chunk_overrides` config option: per-group rank-4 chunk shape override `(time, y, x, channel)`. Mirrors the existing `zarr_shard_overrides` pattern. Takes precedence over `zarr_chunk_y` and the nc_part-aligned defaults.
- `docs/performance-tuning.md`: orthogonal-knob explanation, recommendations per resolution, full-disk-per-shard power-user recipe ("one full disk per shard").

### Changed

- `_validate_shard_override` error message now hints that the chunk shape may come from `zarr_chunk_overrides`, `zarr_chunk_y`, or defaults, helping users diagnose divisibility conflicts when combining overrides.
- README "Performance Notes" section now contains a unified "Tuning Chunks and Shards" subsection with an orthogonal-knobs table covering plugin-tier, template-tier, and engine-tier options.

> **Note:** Template-tier `zarr_compression` is documented but end-to-end flow through `DirectZarrIngestor` has not been verified at this release. File an issue or test before relying on it.

## [0.1.3] - 2026-06-27

### Added

- `x(x,)` and `y(y,)` projection coordinate variables in all resolution groups (`data_500m`, `data_1km`, `data_2km`). Geostationary angular coordinates in radians. CF attrs: `standard_name=projection_x_angular_coordinate` / `projection_y_angular_coordinate`, `units=radian`, `axis=X/Y`.
- `grid_mapping="spatial_ref"` attribute on `counts`, `pixel_quality`, `pixel_time` (CF §5.6 georeferencing).
- `ancillary_variables="pixel_quality pixel_time"` attribute on `counts` (CF §3.4).
- `coordinates="latitude longitude"` attribute on spatial data variables when `include_geolocation=True` (conditional; omitted when geolocation is disabled).
- CF `flag_masks` and `flag_meanings` attributes on `pixel_quality` (CF §3.5).
- CF `standard_name="time"` and `calendar="standard"` attributes on `pixel_time`.
- `crs_wkt` and `spatial_ref` WKT attributes on `spatial_ref` grid-mapping container for rioxarray/GDAL compatibility.
- `pixel_time_dtype` config option now accepts `"int32"` and `"int64"` in addition to `"float64"` (default) and `"float32"`.

### Changed

- **BREAKING:** Schema attributes changed in a backward-incompatible way. Existing Zarr stores written by version 0.1.2 will raise `SchemaDriftError` on re-ingest after upgrading. **Migration:** re-ingest from source ZIPs into a new target store.
- `spatial_ref` grid-mapping container attrs cleaned: removed CF §5.6 violations (`units`, `coordinates`); projection parameters preserved.
- `pixel_time` now carries `grid_mapping`, `standard_name`, and `calendar` CF attributes.
- Plugin now declares CF-1.8 compliance and passes `firecube advise compliance --profile cf-18` with zero errors on all resolution groups.

### Fixed

- `spatial_ref` previously carried spurious `units="m"` and `coordinates="y x"` attrs on a CRS container, which violates CF §5.6. These are removed.

[Unreleased]: https://github.com/eumetsat/firecube-mtg-fci-l1c/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/eumetsat/firecube-mtg-fci-l1c/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/eumetsat/firecube-mtg-fci-l1c/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/eumetsat/firecube-mtg-fci-l1c/releases/tag/v0.1.3
