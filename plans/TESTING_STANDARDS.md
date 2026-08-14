# Testing Standards

Contributor and agent guidance for writing, reviewing, pruning, and running
tests in this plugin repo. This document defines what good tests are.

This standard is adopted from and kept in sync with firecube-core
`plans/TESTING_STANDARDS.md`. When core updates its standard, review the diff
and mirror the durable changes here. Plugin-specific adaptations are called out
inline.

## Audience

This is an internal contributor standard for humans and agentic workloads that
change plugin source, tests, plans, or docs. It is not a user guide.

## Core Rule

Tests must prove behavior that matters to this plugin's contracts. A test is
worth keeping only if it would fail when a real regression breaks one of these
things:

- FCI schema correctness or persisted Zarr output layout
- ingest idempotency, resume, and slot-range write coordination (via firecube)
- `ZarrArraySpec` per-array declarations (chunks, shards, fill values, CF attrs)
- public CLI, `MtgFciL1cConfig` option, or persisted-format contract
- projection and geolocation numerics (grids, CRS oracle, radians vs meters)
- interactions with firecube's `DirectZarrIngestor` public API surface

Tests that mainly pin formatting, implementation shape, or old incident history
must be rare, clearly marked, and kept out of the default bug-catching loop
unless they protect a current public contract.

## Required Test Shape

Every new or changed non-trivial behavior needs tests in the smallest useful
combination of these categories.

| Category | Required When | What It Must Prove |
|---|---|---|
| Contract | Public CLI, `MtgFciL1cConfig` field, persisted Zarr schema, or plugin export changes | Accepted inputs, rejected inputs, stable output shape, clear error behavior |
| Boundary | Code handles ranges, dimensions, chunks, empty data, optional fields, or time spans | Zero/empty, one, max, just past max, invalid, and mixed valid/invalid cases |
| Failure mode | Any streaming reader, config validator, or writer can fail | The failure is loud, typed when possible, and does not commit partial state |
| Integration | Behavior crosses source netCDF, `_variable.py`, `schema.py`, or the Zarr store | At least one realistic end-to-end path with real filesystem/Zarr objects |
| Numerics | Projection, WKT, CRS, grid arithmetic, or NaN-at-limb handling | Values match an independent oracle (e.g., `pyproj`), not a mirror of the same code |

Do not use line coverage as proof. Coverage is only a discovery tool for
finding untested behavior.

## Forbidden Test Patterns

Reject these patterns in review:

- **Mirror tests**: recomputing expected values with the same logic as the code
  under test.
- **Happy-path-only tests**: testing a successful run without the adjacent
  invalid, empty, conflicting, or partial state.
- **Mock-first tests**: mocking internal collaborators so heavily that the test
  verifies the mock choreography instead of plugin behavior.
- **Assertion-light tests**: only checking `exit_code == 0`, `is not None`,
  "does not raise", or "method was called" when a meaningful output, file,
  array, or error can be asserted.
- **Snapshot sprawl**: adding full-output snapshots for broad surfaces like
  help text when a semantic assertion would catch the real contract. Golden
  schema and output snapshots (`tests/golden/*.json`) are the exception — they
  are the plugin's persisted-schema contract.
- **Static archaeology**: scanning for historical strings, phase names, or old
  private paths — including guarding against a re-introduced field name that
  never shipped in a release — unless the test protects an active architecture
  invariant.
- **Dead fixture tests**: tests tied to removed config fields, product names,
  or sample paths that no longer represent a current contract.
- **Permanent xfail drift**: adding `xfail` for a known bug without an accepted
  TODO item, owner, and removal condition.

## Static And Architecture Tests

Static tests are allowed only for repository invariants where runtime coverage
is weak or too expensive:

- forbidden deep imports from `firecube.runtime.*` or `firecube.core.*` past
  the public `firecube.ingestor.api` / `firecube.core.api` surface (Plugin
  Contract, see [DESIGN.md](DESIGN.md))
- lambdas or nested functions inside schema variable source functions (must
  stay picklable for process workers, see [DESIGN.md](DESIGN.md))
- direct writes to `.firecube/` control-plane state from plugin code

Static tests must name the invariant they protect and prefer AST or structured
inspection over fragile substring matching.

## CLI, Docs, And Snapshot Tests

CLI and docs tests are contract tests only when they protect behavior a user
or operator depends on.

Prefer:

- command exits and error classes for required/invalid arguments (e.g.,
  `fix-fillvalue` requires `--store`; dry-run vs `--yes-i-really-mean-it`)
- JSON or machine-readable output shape checks
- semantic help assertions for required flags and safety warnings
- schema-drift error paths on config changes between ingests

Avoid:

- full golden snapshots for every command in the default test lane
- checking the same help text through multiple test files
- treating prose wrapping changes as product regressions

Golden schema/output snapshots under `tests/golden/` ARE the persisted-format
contract for this plugin and must be regenerated deliberately from code, never
hand-edited.

## Mocks, Fakes, And Fixtures

Use real objects wherever practical:

- real local temporary directories for storage behavior
- real Zarr stores for writer, schema, and metadata behavior
- real source netCDF fixtures for streaming-read behavior
- `pyproj` as an independent CRS oracle for projection tests

Mocks are acceptable at external boundaries or to force rare failures:

- clock/time when deterministic timestamps matter
- low-level filesystem failures that are hard to trigger otherwise

When using a fake, assert on plugin-visible effects, not only fake method
calls. Examples: output arrays, CF attrs, `ZarrArraySpec` fields, or
user-facing errors.

## Dependency Rules

- Runtime deps are minimal and pinned in `pyproject.toml` `[project]`. Do not
  add new runtime deps without a design note in [DONE.md](DONE.md).
- Dev/test deps live under `[project.optional-dependencies]` (`dev` group).
- Never pin an unreleased firecube-core version. `firecube>=X.Y.Z` may only
  reference a tagged PyPI release.
- `pyproj` is a `dev` dependency, used solely by CRS oracle tests. It is not
  a runtime requirement.

## Standard Test Commands

Primary agentic workload commands:

```bash
uv run --frozen pytest tests/ -q
uv run --frozen ruff check src/ tests/
uv run --frozen mypy src/
```

Run a focused subset during development:

```bash
uv run --frozen pytest tests/test_variable.py -k "byte_budget" -q
```

## Agent Workload Rules

Before an agent adds or edits tests, it must:

1. Read [AGENTS.md](../AGENTS.md), [DESIGN.md](DESIGN.md), and this file.
2. Name the behavior and risk category the test protects.
3. Prefer a behavior test over a static scan or snapshot.
4. Check whether an existing test already protects the behavior (avoid mirror
   duplicates).
5. Never write a "regression guard" for a config key, field name, or code path
   that never shipped in a tagged release.
6. Run the smallest relevant test first, then the full suite for the touched
   module.

Agents must not add broad snapshot, static grep, or xfail tests to make
progress look measurable. Test count is not a quality metric.

## Review Checklist

Before accepting new or changed tests:

- [ ] The test name describes the behavior and expected outcome.
- [ ] Expected values are independent and concrete (not recomputed with the
      same logic as the code under test).
- [ ] The test would fail for a realistic regression.
- [ ] Failure modes and boundaries are covered when the behavior is risky.
- [ ] Mocks are limited to external boundaries or forced rare failures.
- [ ] The test asserts plugin-visible effects, not internal mock calls.
- [ ] Static checks are justified by a current invariant.
- [ ] The test does not depend on stale field names, sample paths, or removed
      config options.
- [ ] The test does not guard against a state that never shipped (no static
      archaeology).
