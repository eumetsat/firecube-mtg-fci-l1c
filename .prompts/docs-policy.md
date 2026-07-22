---
description: Apply Firecube MTG FCI L1C documentation audience and public/private boundary rules
agent: build
---

# Documentation Policy

Use this prompt before creating, reviewing, or substantially rewriting
documentation for the Firecube MTG FCI L1C plugin.

## Goal

Keep published plugin documentation useful for users and keep implementation
history out of the default reading path.

## Arguments

`$ARGUMENTS` - optional doc paths to review or rewrite. If omitted, apply this
policy to the current documentation task.

## 1. Identify The Audience

Choose one primary audience before writing:

- **User**: installs Firecube and this plugin, downloads MTG FCI L1C inputs,
  runs ingestion, and checks Zarr outputs.
- **Plugin author**: extends or uses plugin-facing behavior through Firecube's
  public SDK.
- **Operator**: runs MTG FCI L1C ingestion in production, CI, Argo, KFP, cron,
  or S3-backed environments.
- **Contributor**: changes this plugin's internals, tests, performance model,
  geolocation handling, or architecture.

If a page serves more than one audience, split it or move the lower-level detail
behind a short "Learn more" link.

## 2. Public Docs Standard

Public docs explain what a reader can do, what command or code to use, what
result to expect, and how to recover from common failures.

Public docs should usually answer at least one of these questions:

- What can I do?
- What do I need before starting?
- What command or code do I run?
- What output should I expect?
- How do I verify it worked?
- What do I do when it fails?

Architecture belongs in public docs only when it changes a user decision.
Explain the user consequence first, then add the smallest necessary mental
model.

## 3. Internal Detail Boundary

Do not put these in public task pages unless they are required for a user
action:

- phase history, audit findings, reviewer names, commit labels, or evidence logs
- line numbers, private module paths, or source-file archaeology
- private Firecube runtime modules or private plugin helper modules
- design invariants, rationale, tradeoff matrices, or implementation debates
- project-management notes, planning docs, or scratch tracking files

Use these homes instead:

- `README.md`: first-run user path, installation, quick start, and links
- `docs/guides/`: task-oriented user and operator guides
- `docs/reference/`: complete factual surfaces such as CLI, config, schema, and
  output layout references
- `notebooks/`: runnable tutorials and explorations with concrete inputs and
  outputs
- `docs/contributing/`: architecture, design rationale, maintenance policy, and
  implementation history

## 4. Page Types

Use one page type explicitly:

- **Tutorial**: teaches through a working example.
- **How-to**: solves one practical task.
- **Reference**: lists the complete surface without narrative.
- **Explanation**: gives a user-facing mental model.
- **Internal note**: records architecture, design rationale, or maintenance
  policy.

Do not mix tutorial, reference, and internal design history in one page.

## 5. Writing Rules

- Start with the task or decision, not with architecture.
- Prefer runnable commands, short code examples, expected output, verification,
  and troubleshooting.
- Use public CLI flags and public SDK imports only.
- Name required flags explicitly when a command will fail without them.
- Keep troubleshooting entries actionable and paste-runnable where possible.
- Link to internals only after the user-facing path is complete.
- Keep section names consistent across related public pages. When a recurring
  section has an established heading, reuse it instead of inventing synonyms.
- Link maintained EUMETSAT resources for MTG FCI L1C facts instead of copying
  large tables into plugin docs.

## 6. Template Selection

Use the prompt templates under `.prompts/` when creating or rewriting pages:

- `/write-user-doc` for user tasks.
- `/write-plugin-doc` for public SDK or plugin-author pages.
- `/write-operator-doc` for production operations.
- `/write-internal-doc` for architecture and design rationale.
- `/write-example-notebook` for notebooks under `notebooks/` or `examples/`.

Templates are scaffolds. Remove empty sections before publishing.
