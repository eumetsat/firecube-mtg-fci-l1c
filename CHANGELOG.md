# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
