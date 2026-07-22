# Changelog

## [0.1.4] - 2026-06-28

### Added

- `zarr_chunk_overrides` config option: per-group rank-4 chunk shape
  override `(time, y, x, channel)`. Mirrors the existing
  `zarr_shard_overrides` pattern. Takes precedence over `zarr_chunk_y`
  and the nc_part-aligned defaults.
- `docs/performance-tuning.md` (new): orthogonal-knob explanation,
  recommendations per resolution, full-disk-per-shard power-user recipe
  ("one full disk per shard").

### Changed

- `_validate_shard_override` error message now hints that the chunk
  shape may come from `zarr_chunk_overrides`, `zarr_chunk_y`, or
  defaults, helping users diagnose divisibility conflicts when
  combining overrides.
- README "Performance Notes" section now contains a unified
  "Tuning Chunks and Shards" subsection with an orthogonal-knobs table
  covering plugin-tier, template-tier, and engine-tier options.

### Compatibility

- Default behavior is identical to v0.1.3. Existing stores remain
  readable and appendable when using the same chunk and shard
  configuration. Changing chunk or shard shape mid-store is rejected
  by firecube core's `SchemaDriftError` (correct behavior: re-ingest
  from source to apply new layouts).

### Known Caveats

- Template-tier `zarr_compression` is documented but end-to-end flow
  through `DirectZarrIngestor` has not been verified at this release.
  File an issue or test before relying on it.

---

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

- `spatial_ref` grid-mapping container attrs cleaned: removed CF §5.6 violations (`units`, `coordinates`); projection parameters preserved.
- `pixel_time` now carries `grid_mapping`, `standard_name`, and `calendar` CF attributes.
- Plugin now declares CF-1.8 compliance and passes `firecube advise compliance --profile cf-18` with zero errors on all resolution groups.

### Fixed

- `spatial_ref` previously carried spurious `units="m"` and `coordinates="y x"` attrs on a CRS container, which violates CF §5.6. These are removed.

### Breaking Changes

- Schema attributes changed in a backward-incompatible way. Existing Zarr stores written by version 0.1.2 will raise `SchemaDriftError` on re-ingest after upgrading. **Migration:** re-ingest from source ZIPs into a new target store.
