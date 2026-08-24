# Done

Dated log of design decisions and release-completion notes. New entries appended to the top.
Related documents: [DESIGN.md](DESIGN.md), [TODO.md](TODO.md), [IDEAS.md](IDEAS.md).

## 2026-08-24 — decode_spatial payload cache: shipped then retired for chunk-owned assembly

### What shipped
A one-entry payload cache on SharedNcPartReader.decode_spatial keyed on
(part, channel, index2time_ref_by_is, dtype), together with a read-only
ChannelSlicePayload contract and behavior tests, on the
feat/firecube-directzarr-adoption-and-callable-payloads branch (later folded
into the DirectZarr adoption commit).

Measured effect on 8-slot FDHSI reference fixture: 62.4 s -> 25.6 s decode
CPU (~60% CPU reduction), 2/3 hit ratio driven by the ingestor's
per-nc_part emission of three back-to-back decode_spatial calls
(counts, pixel_quality, pixel_time).

### Why it is being retired

The chunk-owned assembly design (2026-08-24) supersedes the one-entry
cache. Under
chunk-owned assembly each output chunk is assembled from <= 2 nc_parts
and emitted as one region intent per (variable, channel, output_chunk).
The three-back-to-back decode pattern the one-entry cache exploited no
longer occurs. A separate two-source + one-assembled cache in the
ChunkOwnedAssembler owns caching for the optimized path.

Empirical justification for retiring instead of preserving as a fallback:
- Default zarr_chunk_y is aligned 1:1 to nc_part row count
  (_constants.py:129-135: 556 / 278 / 139 for 500m / 1km / 2km).
  Every output chunk covers exactly one nc_part under the default schema.
- The only setting that breaks the <=2 nc_parts precondition is
  zarr_chunk_y > 2 x nc_part_rows. Chunk-owned assembly adds a config-time
  assertion (MtgFciL1cConfig.__post_init__) that rejects such values
  with a clear error message. Bad configs fail loudly at config parse
  time; the runtime never has to fall back.
- Keeping the one-entry cache as fallback-only retained ~21 MiB per
  SharedNcPartReader x 12 pods = ~252 MiB for a path the production
  pipeline would never exercise.

### Scope of the retirement

The chunk-owned assembly change deletes:
- Cache fields on SharedNcPartReader
  (_cache_item, _cache_channel, _cache_index2time, _cache_dtype,
  _cache_payload) at approximately _streaming.py:281-288
- Cache read/write logic in decode_spatial() at approximately
  _streaming.py:304-347
- Cache-behavior tests in tests/test_streaming.py for hit / miss /
  close-clears / identity / dtype
- test_cache_hit_ratio_on_reference_fixture proxy test

This entry is the durable record of the cache design. The read-only
ChannelSlicePayload contract shipped alongside it is retained because
it applies to the assembled payloads too.

### Plan file disposition

The internal planning file for the decode-payload cache was deleted on
2026-08-24. This DONE.md entry is the durable record.

## 2026-08-23 — DirectZarr adoption + callable payloads + decode cache

### Shipped

**API migration:**
- `slot_index_model()` removed; replaced by `index_spec(ctx)` and `inspect_item(item, ctx)` (firecube DirectZarr contract).
- `MtgFciL1cConfig.zarr_sharding` field removed; flag now read from `template_config.zarr_sharding`.

**Performance:**
- `SharedNcPartReader` extracted from `_streaming.py`: one open file handle per nc_part per batch instead of one per decode phase.
- One-entry `ChannelSlicePayload` cache in `SharedNcPartReader.decode_spatial`: the three spatial variables (`counts`, `pixel_quality`, `pixel_time`) share one decode per `(nc_part, nc_channel)` slice. Reduces decode calls by 2/3 on the spatial hotpath. Cache key holds actual references; `index2time` compared with `is` to avoid stale-id hazard.
- `_emit_spatial_intents` emits callable `WriteIntent` payloads via `functools.partial` + module-scope `_decode_spatial_payload`.

**Bug fixes:**
- None-source parity: `_emit_spatial_intents` probes each variable source before emitting; skips intents whose source returns `None` (e.g. `pixel_time` when no time map exists). Restores old eager-path behavior.
- Deterministic reader/scratch cleanup: `SharedNcPartReader` retained alongside `BatchScratch` and closed at pipeline start, not prematurely at `build_write_intents` return.

**Docs:**
- `docs/customization.md`: `zarr_compression` and `zarr_codecs` template options documented with FCI-tuned examples and codec-lock-in warning.
- `docs/performance-tuning.md`: codec-choice section added (archive vs analysis trade-offs).

### Decisions

**Q: Should `zarr_compression` / `zarr_codecs` docs be gated on a specific firecube release?**

A: No. The options are available now and the docs describe current behavior. Version numbers in user-facing docs create maintenance debt and confuse operators who are already on the right version.

---

## 2026-08-13 — v0.1.5 release

### Shipped

**Bug fixes:**
- **X-axis sign** (issue #1): FCI GEOS x-coordinate is now east-positive (was west-positive due to inverted offset). `x/y` sources in `schema.py` regenerated; golden snapshots refreshed.
- **FillValue encoding pitfall** (issue #2): Added `firecube plugins mtg_fci_l1c fix-fillvalue --store PATH` post-ingest CLI. Dry-run default; apply with `--yes-i-really-mean-it`. This is a workaround for a firecube-core coordinate encoding gap that requires a cold-migration fix in a future core release.
- **Time attrs collision** (issue #3): Removed `units` and `calendar` from `time` variable attrs — they collided with xarray's encoding layer on round-trip. xarray's encoding now owns these exclusively.

**New features:**
- **`projection_units` config option**: accepts `"meter"` (default), `"metre"` (alias), or `"radian"`. Meters are more usable downstream; radians preserved for backward compat.
- **`MTG_PERSPECTIVE_POINT_HEIGHT_M` constant**: single source of truth for the FCI perspective point height; cross-checked against WKT at import time.


**Dependencies:**
- Added `pyproj>=3.6` as a `dev` dependency (used by the new projection CRS oracle tests). No runtime dependency change.

**Docs:**
- Documented `projection_units`, `fix-fillvalue` CLI, x-axis semantics, and time attr policy in `docs/customization.md` and `docs/fci-data-in-zarr.md`.
- Adopted firecube-core's `plans/*.md` schema: DESIGN.md (invariants), TODO.md (deferred work), DONE.md (this file), IDEAS.md (speculative), STYLE.md (repo conventions), TESTING_STANDARDS.md (test-quality rules).


### Verification

All Firecube invariants respected:
- No direct writes to `.firecube/` control-plane from plugin code
- No deep imports past `firecube.ingestor.api` and `firecube.core.api`
- Slot ranges disjoint for parallel ingestion (`pipeline_workers=1` per pod)
- `ZarrArraySpec.shards` used for per-array sharding declarations (plugin-owned shape)

Full test suite: 215 passed, 4 pre-existing CF advisor test failures acknowledged as unrelated (not caused by this release). No new tests added for the field rename — per [TESTING_STANDARDS.md](TESTING_STANDARDS.md), guarding against a re-introduced field name that never shipped in a tagged release is a "static archaeology" anti-pattern.
