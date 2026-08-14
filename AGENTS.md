# Firecube MTG FCI L1C Plugin

This repo is a Firecube plugin for ingesting MTG FCI Level 1C FDHSI and HRFI
products into direct-region Zarr stores.

## Quickstart

- Install all groups: `uv sync --all-groups`
- Run tests: `uv run --frozen pytest tests/ -q`
- Lint: `uv run --frozen ruff check src/ tests/`
- Type check: `uv run --frozen mypy src/`
- Describe plugin: `firecube plugins describe mtg_fci_l1c`
- Run helper script: `bash scripts/fci-ingest.sh`

If `uv run` tries to resolve from scratch and cannot find `firecube`, use the
existing lockfile with `--frozen`.

## Documentation Rules

Before creating or substantially rewriting docs, read
[`.prompts/docs-policy.md`](.prompts/docs-policy.md). Choose the matching prompt:

- User docs: `.prompts/write-user-doc.md`
- Operator docs: `.prompts/write-operator-doc.md`
- Plugin-author docs: `.prompts/write-plugin-doc.md`
- Internal docs: `.prompts/write-internal-doc.md`
- Notebooks: `.prompts/write-example-notebook.md`
- Fact checking: `.prompts/doc-fact-check.md`

Keep public docs focused on reader tasks, commands, expected output,
verification, and troubleshooting. Do not put implementation history, audit
notes, private module paths, or raw evidence logs in public task pages.

Contributor material belongs under
[`docs/contributing/index.md`](docs/contributing/index.md).
Keep the root README user/operator focused.

## QA And Asset Metadata

- Do not edit or remove SBOM, dependency-license, ABOUT, AUTHORS, or LICENSE
  material unless the task explicitly asks for it.
- Generated SBOM and dependency-license reports belong under git-ignored `.reports/`
  and are regenerated locally; they are not committed.
- Public binary assets copied into `docs/assets/` need sibling `.ABOUT` files.
  Follow the existing `docs/assets/performance/*.png.ABOUT` format.
- Published benchmark plots and workload notes belong in
  `docs/reference/performance-benchmarks.md`.

## Firecube And Zarr Invariants

- Current plugin version is `0.1.5`. Firecube baseline is `0.1.4`.
- The plugin uses Firecube direct-region Zarr behavior. Link core mechanics to
  Firecube public docs instead of duplicating them:
  - Direct Region Zarr: `https://eumetsat.github.io/firecube/concepts/output-formats/zarr/direct-region/`
  - Parallel Zarr Writes: `https://eumetsat.github.io/firecube/concepts/output-formats/zarr/parallel-writes/`
  - Direct Zarr Plugins: `https://eumetsat.github.io/firecube/concepts/plugins/direct-zarr/`
- Firecube owns preallocation, schema setup, chunk claims, run records, and
  coordinated writes. This plugin owns FCI-specific schema declarations and
  write intents.
- Keep slot ranges disjoint for parallel ingestion. Use `pipeline_workers=1`
  per pod because increasing workers inside one pod can create same-slot write
  conflicts; scale through separate slot-range pods.
- Do not write Firecube control-plane state directly from plugin code.

## Where Things Live

- Plugin config: `src/firecube_mtg_fci_l1c/config.py`
- Zarr schema and variable declarations: `src/firecube_mtg_fci_l1c/schema.py`
- Ingest orchestration and write intents: `src/firecube_mtg_fci_l1c/ingestor.py`
- NetCDF streaming reads: `src/firecube_mtg_fci_l1c/_streaming.py`
- Geolocation helpers and grid CLI: `src/firecube_mtg_fci_l1c/geolocation/`
  and `src/firecube_mtg_fci_l1c/plugin_cli.py`
- Production fan-out helper: `scripts/fci-ingest.sh`
- User/operator docs: `README.md`, `docs/customization.md`,
  `docs/fci-data-in-zarr.md`, `docs/guides/production-ingestion.md`,
  `docs/performance-tuning.md`
- Contributor docs: `docs/contributing/`

## Code Style

- Prefer explicit, readable Python with type hints over clever abstractions.
- Keep source functions for schema variables pure. They project from
  `VariableContext`; I/O belongs in streaming/ingestor code.
- Do not use lambdas or nested functions in schema variable sources. They must
  remain picklable for process workers.
- Add tests with behavior changes. Follow [`plans/TESTING_STANDARDS.md`](plans/TESTING_STANDARDS.md):
  behavior-first, no mirror tests, no static archaeology for state that never
  shipped, no assertion-light happy-path guards. For docs-only edits, run
  link/path checks and the standard project checks when practical.
