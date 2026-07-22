# Contributing

Thank you for contributing to the Firecube MTG FCI L1C plugin.

This plugin accepts contributions through pull requests. This document outlines
the process to help you prepare a focused change, run the right checks, and get
your contribution reviewed.

## Prerequisites

Use Python 3.12 and `uv`.

```bash
uv sync --all-groups
uv run --frozen firecube plugins describe mtg_fci_l1c
```

If `uv run` tries to resolve from scratch and cannot find `firecube`, use the
existing lockfile with `--frozen`.

For documentation work, read [`AGENTS.md`](AGENTS.md) and
[`.prompts/docs-policy.md`](.prompts/docs-policy.md) before editing public docs.

## Branches And Scope

Create one branch per logical change. Use a short, conventional prefix:

```text
feat/
fix/
docs/
test/
refactor/
ci/
chore/
perf/
build/
```

Keep pull requests focused. Separate generated artifacts, dependency updates,
and broad refactors from behavior changes when that makes review clearer.

## Development Rules

Prefer small, explicit changes that match the existing module boundaries.

- Use public Firecube plugin imports from `firecube.ingestor.api` and
  `firecube.core.api`.
- Keep FCI-specific config validation in `src/firecube_mtg_fci_l1c/config.py`.
- Keep Zarr variable declarations in `src/firecube_mtg_fci_l1c/schema.py`.
- Keep schema source functions pure. They should project from
  `VariableContext`; I/O belongs in streaming and ingestor code.
- Do not use lambdas or nested source functions in schema declarations. They
  must remain picklable for process workers.
- Keep slot ranges disjoint for parallel ingestion. Use `pipeline_workers=1`
  per pod and scale with separate slot-range pods.
- Do not write Firecube control-plane state directly from plugin code.

For plugin-specific contributor notes, see
[`docs/contributing/index.md`](docs/contributing/index.md).

## Tests

Tests should prove user-visible behavior, persisted state, public contracts, or
important failure modes. Avoid tests that only mirror implementation details.

Use the smallest useful loop first:

```bash
uv run --frozen pytest tests/path_or_file.py -q
```

Run the plugin test suite before review:

```bash
uv run --frozen pytest tests/ -q
```

## Lint, Format, And Types

The source-level gates are:

```bash
uv run --frozen ruff check src/ tests/
uv run --frozen mypy src/
```

Run formatting only when needed for files you changed:

```bash
uv run --frozen ruff format src/ tests/
```

## Documentation

Before editing public documentation, read `.prompts/docs-policy.md`.

Use the matching prompt in `.prompts/` for substantial documentation work:

- `.prompts/write-user-doc.md` for user tasks.
- `.prompts/write-plugin-doc.md` for plugin author guidance.
- `.prompts/write-operator-doc.md` for production operations.
- `.prompts/write-internal-doc.md` for contributor-only design or maintenance notes.
- `.prompts/write-example-notebook.md` for notebooks.

Public docs should give commands, expected outcomes, verification steps, and
recovery steps. Keep implementation history, audit notes, private module paths,
and raw evidence logs out of public task pages.

Contributor material belongs under
[`docs/contributing/index.md`](docs/contributing/index.md).
Keep the root README user/operator focused.

For docs-only changes, at minimum check repository-local links in the touched
Markdown files. If public binary assets are copied into `docs/assets/`, add a
sibling `.ABOUT` file for each asset.

## Dependencies, Licenses, And SBOM

Do not treat SBOM, dependency-license, ABOUT, AUTHORS, or LICENSE material as
cleanup noise.

When changing dependencies, extras, build tooling, or dependency-license
evidence, regenerate the local reports and update the README tables as needed:

```bash
mkdir -p .reports
uv export --format cyclonedx1.5 --all-groups --all-extras --output-file .reports/sbom.cdx.json
uv run --isolated --all-groups --all-extras --with hatchling --with pip-licenses pip-licenses --format=json --with-urls > .reports/dependency-licenses.json
```

`.reports/` is git-ignored and is not committed. Review
`.reports/dependency-licenses.json` for GPL, LGPL, AGPL, SSPL, unknown runtime
licenses, and missing license evidence. Do not rely on SBOM license fields
alone.

## Pull Request Checklist

Before opening a pull request:

```bash
git status --short
git diff --stat
git diff --check
uv run --frozen ruff check src/ tests/
uv run --frozen mypy src/
uv run --frozen pytest tests/ -q
```

Use the pull request description to explain:

- what changed;
- why it changed;
- which user, operator, plugin, or maintainer behavior is affected;
- which checks you ran;
- any known follow-up work.

Do not include local virtual environments, caches, generated products, test data
outputs, logs, credentials, build directories, or benchmark artifacts in
a pull request.

## Sign Your Work

Sign every commit with your real name and a reachable email address. The
sign-off is a line at the end of the commit message certifying that you wrote
the contribution or otherwise have the right to submit it under this project's
license.

If your Git identity is configured, `git commit -s` adds the sign-off
automatically:

```bash
git config --global user.name "Jane Smith"
git config --global user.email "jane.smith@example.com"
git commit -s -m "fix: reject ambiguous FCI slot configuration"
```

The commit log should include matching author and sign-off identity:

```text
Author: Jane Smith <jane.smith@example.com>
Date:   Thu Feb 2 11:41:15 2026 +0000

fix: reject ambiguous FCI slot configuration

Signed-off-by: Jane Smith <jane.smith@example.com>
```

## AI-Assisted Contributions

AI tools may help draft or review a change, but the contributor remains
responsible for the submitted work. Before opening a pull request, review every
change yourself, remove unused scaffolding, run the relevant checks, and be ready
to explain the design, tests, and failure behavior.

If AI assistance materially shaped the change, disclose that in the pull request
description. Do not use generated responses as a substitute for engaging with
review comments yourself.

If AI assistance materially shaped a commit, record that with an `Assisted-by`
trailer in the commit message. Keep your own `Signed-off-by` line as the final
certification that you have the right to submit the work.

```bash
git commit -s -m "fix: reject ambiguous FCI slot configuration

Assisted-by: GitHub Copilot <copilot@github.com>"
```

The resulting commit log should keep both trailers:

```text
Author: Jane Smith <jane.smith@example.com>
Date:   Thu Feb 2 11:41:15 2026 +0000

fix: reject ambiguous FCI slot configuration

Assisted-by: GitHub Copilot <copilot@github.com>
Signed-off-by: Jane Smith <jane.smith@example.com>
```

## Commit Messages

Use clear conventional-commit-style subjects:

```text
feat: add FCI grid metadata validation
fix: reject ambiguous FCI slot configuration
docs: update production ingestion guide
test: cover staged geolocation resume
ci: add plugin smoke gate
chore: regenerate SBOM artifacts
```

Each commit should represent one logical unit. If a review asks for changes, it
is fine to add follow-up commits while review is active; maintainers may squash
or ask for cleanup before merge.
