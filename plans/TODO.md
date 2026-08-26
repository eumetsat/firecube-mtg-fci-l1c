# TODO

Accepted work only. Speculative ideas go to [IDEAS.md](IDEAS.md) until a design discussion promotes them.
New decisions are recorded in [DONE.md](DONE.md) with a date.

## Active Work

### §1 Adopt firecube-core `ZarrTemplateConfig.zarr_sharding` when core wires it

**Goal:** Once firecube-core wires `ZarrTemplateConfig.zarr_sharding` to `DirectZarrIngestor` (mirroring PR #42 codec parity — see `HANDOFF.md` in firecube-core repo), migrate this plugin to read the enable/disable flag from `template_config.zarr_sharding` and remove the plugin's own `zarr_sharding: bool` field.

**Trigger:** firecube-core release includes the sharding wiring change. Verify by checking:
- Released firecube version consumes `template.zarr_sharding` in `direct_zarr.py` `_setup_global_zarr_schema` (grep for `zarr_sharding` in installed firecube's `direct_zarr.py`)
- `DirectZarrIngestor.template_config_class = ZarrTemplateConfig` is set
- `validate_zarr_specs_against_template()` extended to include sharding validation

**Decision (2026-08-13):** Accepted. Plugin currently owns `zarr_sharding: bool = True` locally; migrate to template-owned flag once core support ships in a released firecube.

**Design constraints:**
1. **Byte-budgeted derivation stays plugin-owned.** Even after migration, `_byte_budgeted_4d_shard()` remains in the plugin. Only the enable/disable flag moves to core. This preserves DESIGN.md:84-86 boundary.
2. **No dependency bump ahead of core release.** The plugin dependency `firecube>=X.Y.Z` may only pin to a released firecube version. If core work slips, this TODO waits.
3. **Cold migration (no compatibility shim).** Once removed, the plugin's `zarr_sharding` field disappears. Operators must use `--option zarr_sharding=...` which routes to the template config. Existing cubes work unchanged (default `True` preserves `ZstdCodec(level=0)` compression behavior).
4. **Rename operator surface stays stable.** The field name `zarr_sharding` is already the target name in both the plugin and core template config; the migration is a re-owning, not a rename.
5. **Docs update mandatory.** `docs/customization.md` moves the `zarr_sharding` row from "Plugin options" to "Firecube template options" (or unified table with a source column).

**Acceptance criteria:**
- Plugin `MtgFciL1cConfig.zarr_sharding` field removed
- `_variable.py` reads `ctx.config.template_config.zarr_sharding` (or equivalent access path core provides)
- All existing `zarr_sharding=false` operator invocations continue to work via template config parsing
- Test suite passes with `firecube` bumped to the release that includes the wiring
- `docs/customization.md` documents the new source of the flag

**References:**
- Firecube-core sharding-wiring proposal (filed by the plugin maintainer as a `§N` item in firecube-core `plans/TODO.md`) — the parity work that unlocks this migration
- Firecube-core PR #42 (commit `5599826`) — codec parity pattern to mirror
- Plugin `plans/DESIGN.md` — invariants that must survive the migration

---

### §3 Diagnose the four pre-existing CF advisor test failures

**Goal:** Root-cause and resolve the four parametrisations of
`tests/test_integration.py::test_cf_advisor_zero_errors_per_group` that fail
with `AttributeError: 'CFFinding' object has no attribute 'path'`. Currently
marked `@pytest.mark.xfail(strict=False)` referencing this item so CI stays
green; xfail must be removed once the root cause is fixed.

**Trigger:** none — accepted work, schedule when maintainer capacity allows.

**Decision (2026-08-14):** Accepted. Promoted from IDEAS.md §3 to unblock CI
adoption (the xfail markers require an accepted TODO item per
`plans/TESTING_STANDARDS.md`).

**Design constraints:**
1. **Diagnose before fixing.** Identify whether the failure is (a) a bug in
   the upstream CF advisor, (b) the plugin passing the wrong shape to the
   advisor, or (c) an advisor API change. Only after root-cause is known can
   we pick the right fix path.
2. **No silent fix.** Do not delete the tests or convert to skip; either fix
   the invocation, patch upstream, or delete the tests with a documented
   rationale in DONE.md.
3. **Remove the xfail marker.** Once resolved, the `@pytest.mark.xfail` on
   `test_cf_advisor_zero_errors_per_group` must be removed so a future
   regression fails loudly.

**Acceptance criteria:**
- Root cause identified and documented in DONE.md
- All four parametrisations either pass, or the tests are removed with a
  documented reason
- `@pytest.mark.xfail` marker removed from
  `tests/test_integration.py::test_cf_advisor_zero_errors_per_group`

**References:**
- `plans/TESTING_STANDARDS.md` § Forbidden Test Patterns (Permanent xfail
  drift): xfail requires an accepted TODO item, owner, and removal condition —
  this §4 satisfies that requirement.

---

### §4 Sharding drift detection for FCI cubes — v0.1.6+

**Goal:** When firecube adds sharding drift detection (analogous to codec drift in PR #42), verify FCI cubes trigger the correct behavior: cubes preallocated with `zarr_sharding=true` cannot be resumed with `zarr_sharding=false` (and vice versa) without triggering `SchemaDriftError`.

**Trigger:** firecube-core adds sharding drift detection (tracked in `HANDOFF.md` Design Constraint 4).

**Decision (2026-08-13):** Accepted. Verification only — no plugin behavior change needed.

**Design constraints:**
1. Verification test only; do NOT re-implement drift detection in the plugin.
2. Test uses a small FCI cube fixture; skip in default suite (mark `@pytest.mark.slow`).

**Acceptance criteria:**
- `tests/test_sharding_drift.py` created; verifies resume rejects flipped sharding flag
- Test skipped when firecube version < the release that includes drift detection

**References:**
- Firecube-core `HANDOFF.md` — proposed drift detection scope

## Deferred / Not Started

(None currently.)

## Recently Completed

See [DONE.md](DONE.md) for full history.
