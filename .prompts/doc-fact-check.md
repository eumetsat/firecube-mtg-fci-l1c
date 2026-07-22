---
description: Run MTG FCI L1C plugin docs code examples and verify prose claims against the codebase
agent: build
---

# Documentation Fact-Check

Run code examples in the docs and verify prose claims against the actual
codebase and maintained EUMETSAT references.

## Arguments

`$ARGUMENTS` - optional file paths relative to `docs/` to narrow the scope.

Examples:

- `/doc-fact-check guides/production-ingestion.md`
- `/doc-fact-check reference/`
- `/doc-fact-check`

If no arguments are given, check all `.md` files under `docs/` plus
notebook-facing claims in `README.md`.

## Setup

```bash
uv sync --all-groups
uv run firecube --help
uv run firecube plugins describe mtg_fci_l1c
```

If any setup step fails, report it and continue with checks that can still run.
Blocks that fail because of missing credentials, missing private data, or
external services are setup problems, not documentation bugs.

## 1. Run Code Examples

For each fenced code block in the selected docs:

### Classify Each Block

- **Runnable**: self-contained Python or Bash with no placeholders, no ellipsis,
  and no user-specific files.
- **Continuation**: follows a runnable block on the same page and reuses its
  variables; concatenate with its predecessor.
- **Skip**: partial snippets, signatures-only snippets, output samples,
  unlabeled blocks, install commands, destructive commands, cloud/S3 commands
  without local fixtures, or commands requiring private data/plugins.

### Execute Runnable Blocks

- **Python**: write to a temp `.py` file and run with `uv run python`.
- **Bash**: run only commands that are safe locally and can complete without
  credentials.
- **CLI help**: run documented `firecube ... --help` commands and verify exit
  code 0.

Never invent missing imports, variables, files, credentials, or plugin names.
Record pass/fail/skip for every block.

## 2. Check Claims Against Code

Read each selected doc file and find verifiable factual claims. Focus on inline
code spans, command flags, config keys, table cells, defaults, feature names,
product names, channel names, group names, and numbers.

Check:

- **CLI flags and subcommands**: run `uv run firecube ... --help`.
- **Config keys**: verify against config parsing and documented examples.
- **Public SDK imports**: verify names are exported from `firecube.ingestor.api`
  or `firecube.core.api`.
- **Plugin entry points**: verify entry point group names and loading behavior.
- **Default values**: verify against dataclasses/config classes.
- **Metric names and environment variables**: verify against source.
- **MTG FCI L1C facts**: verify against maintained EUMETSAT user guides or
  catalogue pages.

Do not check subjective claims such as "fast", "simple", or "production-ready"
unless the doc attaches a measurable value.

## 3. Report

For each finding:

```text
[ERROR|STALE|DRIFT] docs/path.md:LINE - Summary
Docs say:
Code or source says:
Suggested fix:
```

- **ERROR**: code example fails, command is invalid, API does not exist, or
  documented behavior is wrong.
- **STALE**: outdated value, renamed API, removed flag, old config, or old
  external product fact.
- **DRIFT**: minor mismatch such as wording, count, default, or incomplete
  example.

End with a summary:

- files checked
- blocks run/passed/failed/skipped
- claim issues found

For findings, propose fixes but do not apply them unless the user asked for
edits.
