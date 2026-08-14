# Done

Dated log of design decisions and release-completion notes. New entries appended to the top.
Related documents: [DESIGN.md](DESIGN.md), [TODO.md](TODO.md), [IDEAS.md](IDEAS.md).

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
