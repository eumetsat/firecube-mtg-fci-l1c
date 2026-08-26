# Ideas

Speculative or investigation-required work. Ideas may promote to [TODO.md](TODO.md)
after design discussion or concrete investigation, or move to [DONE.md](DONE.md)
with a rejection note. Keep entries brief; expand only when promoting.

Every idea here must be grounded in an observed pain point or documented user
workflow, and described from first principles.

## Idea 1 — Migrate byte-budgeted shard derivation to firecube-core mixin

**What:** Once firecube-core ships an opt-in `DirectZarrIngestor` mixin
surface for shard-shape derivation (mirroring the `CadenceSlotAllocation`
mixin pattern in firecube-core TODO §33), migrate this plugin's
`_byte_budgeted_4d_shard()` derivation into a core mixin and delete the local
implementation.

**Anchor:** `_byte_budgeted_4d_shard()` in
`src/firecube_mtg_fci_l1c/_schema.py:117-133` is FCI-shaped but the "byte
budget → group whole chunks along Y up to target" policy generalises to any
`DirectZarrIngestor` plugin writing large multi-dimensional data arrays.

**Trigger:** firecube-core exposes a `ByteBudgetedSharding` mixin (or
equivalent), verifiable by grepping the released firecube version for the
class in `firecube.ingestor.templates.direct_zarr` or a sibling module.

**Migration shape (post-trigger):**
- Plugin `MtgFciL1cIngestor` composes the mixin:
  `class MtgFciL1cIngestor(ByteBudgetedSharding, DirectZarrIngestor)`
- `_byte_budgeted_4d_shard()`, `_validate_shard_override`, and the byte-budget
  branch in `_build_array_spec()` are deleted
- Plugin retains `zarr_shard_target_bytes` and `zarr_shard_overrides` as thin
  config that feeds the mixin (if the mixin exposes those as tuning knobs) OR
  migrates fully to mixin defaults

**Open questions:**
- Does core also want to own the static-lat/lon exemption, or is that
  plugin-owned per array via `ZarrArraySpec.shards=None`?
- Is "grow Y only" the right default, or should the mixin accept a
  growth-axis parameter for time-dominant plugins?

**Not yet because:** firecube-core hasn't shipped the mixin surface. When it
does, promote this to TODO with a concrete migration checklist.

---

## Idea 2 — Investigate whether source netCDF chunk hints beat CHUNK_DEFAULTS_BY_RESOLUTION

**What:** Determine empirically whether FCI L1C netCDF chunk shapes (readable
via `h5netcdf` at streaming-read time) would produce measurably better read or
ingest throughput than the current `CHUNK_DEFAULTS_BY_RESOLUTION` values.

**Anchor:** `src/firecube_mtg_fci_l1c/_constants.py` defines
`CHUNK_DEFAULTS_BY_RESOLUTION` for `data_500m` / `data_1km` / `data_2km`.
These were chosen for parallel time-slot ingestion. Whether they align with
source HDF5 chunk boundaries is unverified.

**Investigation before deciding:**
1. Open a representative FDHSI and HRFI netCDF; dump chunk shapes for
   `pixel_values`, `pixel_quality`, `pixel_time` via `h5py` (or `h5netcdf`'s
   underlying handle).
2. Compare source chunk shape vs current `CHUNK_DEFAULTS_BY_RESOLUTION` for
   each resolution group.
3. Measure ingest throughput on a small window (e.g., 6 slots) with the
   current defaults vs source-aligned overrides via `zarr_chunk_overrides`.
4. If source-aligned wins by >10 % on realistic hardware, consider changing
   defaults OR adding an opt-in `chunk_source=hint` mode.

**Not yet because:** the investigation is unpriced. Current defaults are
proven in production; changing them without measurement risks worse
throughput or awkward interactions with `zarr_shard_target_bytes`.

---

## Idea 3 — Channel-image export helpers for visualisation and labeling

**What:** Add a plugin-owned utility that reads a channel from a plugin-written
Zarr cube, crops to a lat/lon bounding box (or explicit pixel window), applies
a radiometric stretch and optional display gamma, corrects the FCI south-up
grid to north-up, and writes 8-bit grayscale images locally or to object
storage. Ship as either a script under `scripts/` or a first-class subcommand
(`firecube plugins mtg_fci_l1c export-image`) alongside `geo generate` and
`fix-fillvalue`.

**Anchor:** The plugin ships `scripts/fci-ingest.sh` for cube creation, and
`geo generate` + `fix-fillvalue` subcommands for setup and repair, but has no
built-in path from cube to viewable imagery. Downstream workflows that need
per-channel imagery for visual inspection, quality review, or ML labeling
pipelines currently reinvent the same read-crop-stretch-flip loop against the
Zarr store, using knowledge that properly belongs to the plugin (channel
layout, `UINT16_FILL` sentinel, south-up grid convention, calibration formula
`radiance = counts * slope + offset`).

**Capabilities the helper should own:**
- Channel selection by logical name (e.g., `vis_04`, `ir_105`) with the right
  resolution group resolved from the plugin's channel-to-resolution map
- AOI selection via lat/lon bounding box (using the cube's `latitude` /
  `longitude` arrays) OR an explicit pixel window
- Calibration: apply `slope` + `offset` per slot per channel; mask the fill
  sentinel to NaN before stretching
- Radiometric stretch: locked `vmin` / `vmax` reproducibly across a batch
  (percentile-derived from a reference slot when not given explicitly)
- Optional display gamma for perceptual brightness
- Grid orientation: flip south-up (native FCI row order) to north-up for
  display
- Batch iteration over a slot range with skip-on-empty behaviour

**Open questions:**
- Ship as a script (matches the existing `fci-ingest.sh` pattern) or as a
  discoverable subcommand (`firecube plugins describe` picks it up, but adds
  imaging and object-storage optional deps to the plugin surface)?
- What imaging library? `pillow` is the obvious choice; is a runtime dep or
  an optional-extra the right shape?
- Should the object-storage write path reuse firecube-core's
  `firecube.core.api` filesystem helpers so the plugin does not add a
  separate fsspec/obstore surface?
- Does the bbox → pixel-window logic belong in the plugin's public API as a
  reusable helper (e.g., `bbox_to_pixel_window(lat, lon, bbox)`) so
  downstream tooling can reuse it without re-implementing lat/lon lookup?
- Which output format(s)? 8-bit grayscale JPEG is the labeling-pipeline
  default, but PNG (lossless) and Cloud Optimized GeoTIFF (georeferenced)
  are natural extensions if the initial helper generalises well.

**Not yet because:** need to decide script-vs-subcommand and dependency scope
before promoting. Once decided, promotes to TODO with a concrete surface
spec.
