# Design

This document records the plugin's architecture rules, locked design decisions, and the questions that drove them. New decisions are appended to [DONE.md](DONE.md) with a date. When a rule changes, update both files.

This plugin ingests MTG FCI Level 1C FDHSI and HRFI products into direct-region Zarr stores. It is a Firecube `DirectZarrIngestor` plugin: Firecube owns preallocation, chunk claims, run records, and coordinated writes. The plugin owns FCI-specific schema declarations and write intents.

## Architectural Invariants

These rules are non-negotiable. Violating any of them requires a `plans/DONE.md` entry explaining the exception.

- **Direct-region only.** This plugin uses Firecube's direct-region Zarr write path (`DirectZarrIngestor`), never the template append path. See Firecube's [Direct Region Zarr](https://eumetsat.github.io/firecube/concepts/output-formats/zarr/direct-region/).
- **Plugin owns FCI schema.** Every variable declared in `src/firecube_mtg_fci_l1c/schema.py` is FCI-specific: CF attributes, chunking hints, sharding shape, static-vs-time-indexed classification. Core owns none of this.
- **Sharding shape is plugin-owned.** Per firecube DESIGN.md:84-86, plugins declare per-array shard shape via `ZarrArraySpec.shards`. The plugin's `_byte_budgeted_4d_shard()` derivation applies FCI data physics (byte budgeting from resolution, per-group scoping, static-lat/lon exemption). Core cannot own this without hardcoding FCI knowledge.
- **Sharding enable/disable flag defers to Firecube template config when core wires it.** As of firecube 0.1.4.post1 the plugin owns `zarr_sharding` locally. Once core wires `ZarrTemplateConfig.zarr_sharding` to `DirectZarrIngestor` (mirroring PR #42 codec parity), the plugin will migrate to read the flag from `template_config` and remove its own field. Tracked in [TODO.md](TODO.md) §1.
- **Source functions in schema variables are pure and picklable.** They project from `VariableContext`; I/O belongs in `_streaming.py` and `ingestor.py`. No lambdas, no nested functions. Enforced by `AGENTS.md` and process-worker execution model.
- **No direct writes to Firecube control-plane state.** The plugin does not touch `.firecube/` directory. All state mutation goes through Firecube's `ChunkManager` facade (which the plugin does not import directly — it emits `WriteIntent` and lets Firecube coordinate).
- **Slot ranges are disjoint for parallel ingestion.** `pipeline_workers=1` per pod. Scale through separate slot-range pods, not through worker concurrency inside one pod. See `docs/guides/production-ingestion.md`.
- **Firecube baseline pinned to a released version.** The plugin depends on `firecube>=0.1.4` (or newer as bumped in DONE.md). Unreleased core versions are never pinned as a dependency. Adoption of new core features waits for their release tag.

## Plugin Contract Compliance

This plugin imports ONLY:
- `firecube.ingestor.api` — `PluginConfig`, `ZarrArraySpec`, `ZarrGroupSpec`, `WriteIntent`, `SlotAxis`, `SlotIndexModel`, `normalize_epoch_iso`, `DirectZarrIngestor`
- `firecube.core.api` — filesystem/URI helpers (only where actually used)

Deep imports into `firecube.runtime.*` or `firecube.core.*` internals are FORBIDDEN and would be caught by future test isolation.

## Config Ownership

Every field on `MtgFciL1cConfig` is one of:
1. **FCI domain**: product type, resolutions, channels, geolocation, time axis (owned by plugin)
2. **Sharding derivation inputs**: `zarr_shard_target_bytes`, `zarr_shard_overrides` (owned by plugin — byte budgeting is FCI-specific)
3. **Sharding enable/disable flag**: `zarr_sharding` (owned by plugin TODAY; will migrate to `template_config.zarr_sharding` when core wires it — see TODO.md §1)
4. **Chunk overrides**: `zarr_chunk_overrides`, `zarr_chunk_y` (owned by plugin — FCI resolution-dependent)

There is no field on `MtgFciL1cConfig` that overlaps semantically with `ZarrTemplateConfig` in a way that would confuse operators. If core adds a field that overlaps with a plugin field, we either migrate to core (preferred) or rename to avoid `--option` namespace collision.

## Zarr Sharding Model

- **Shape derivation**: `_byte_budgeted_4d_shard()` computes shard shape per group, targeting `zarr_shard_target_bytes` (default 128 MiB). It groups whole chunks along the y-axis up to the byte budget. It never splits a chunk. Static lat/lon arrays are never sharded.
- **Per-group override**: `zarr_shard_overrides: dict[str, tuple[int,int,int,int]] | None` lets operators pin explicit shard shapes per group (`data_1km`, `data_2km`, `data_500m`). Validated at schema build time to be a whole multiple of each array's chunk shape.
- **Enable/disable**: `zarr_sharding: bool = True` (True by default because FCI 4-D data arrays are large and sharding is the sane default for parallel ingestion; disable only when writing tiny test cubes).

## Compression Model

- **Firecube 0.1.4.post1 behavior**: `DirectZarrIngestor` does not consume `ZarrTemplateConfig` — all arrays are compressed with zarr v3 default `ZstdCodec(level=0)` regardless of any template setting.
- **Firecube 0.1.5+ behavior (once released)**: `DirectZarrIngestor` consumes `zarr_compression` and `zarr_codecs` from template (PR #42). Default `zarr_compression=True` preserves the `ZstdCodec(level=0)` behavior for existing cubes. Operators can opt out with `--option zarr_compression=false` (uncompressed) or set `--option zarr_codecs='[...]'` for custom pipelines.
- **Plugin does not set explicit compression**: per-array `filters`, `serializer`, `compressors` on `ZarrArraySpec` are all `None` (inherit template default). This is deliberate — compression is an operator concern, not a plugin decision.

## Decided Questions

Brief answers to recurring design questions. Full history with dates is in [DONE.md](DONE.md).

- **Which Firecube ingest path?** `DirectZarrIngestor` (direct-region), never `GenericZarrIngestor` (template append). FCI cubes are large, multi-resolution, and require explicit per-array schema control that only DirectZarr provides.
- **What if core adds a field with the same name as a plugin field?** Prefer migration to core's field (if semantics match). Otherwise rename plugin field to avoid `--option` namespace collision. Never maintain a semantically-different field with a colliding name.
- **Where does sharding derivation live?** Always in the plugin. Byte budgeting is FCI-specific (data physics, resolution scaling). Core cannot own this without hardcoding FCI knowledge.
- **Which projection units for x/y coords?** Configurable via `projection_units: str = "meter"`. Accepts `"meter"`, `"metre"` (alias), or `"radian"`. Meter is default because it's more usable downstream.
- **How do we handle FillValue encoding pitfalls?** Post-ingest CLI workaround: `firecube plugins mtg_fci_l1c fix-fillvalue --store PATH`. Dry-run by default; apply with `--yes-i-really-mean-it`. See `docs/guides/production-ingestion.md`.
- **Time coordinate attrs?** No CF `units` or `calendar` attrs on the `time` variable. xarray's encoding layer manages these; setting them as attrs causes round-trip collisions.
