---
description: Create or rewrite a Jupyter notebook for the MTG FCI L1C plugin
agent: build
---

# Write Example Notebook

Use this prompt when adding or rewriting a notebook under `notebooks/` or
`examples/`. These notebooks are EUMETSAT-style training material for
JupyterHub/JupyterLab, not scratch demos. Apply `.prompts/docs-policy.md` first.

## Arguments

`$ARGUMENTS` - target notebook path and optional topic description.

## Structure

Use this order:

1. **Title and navigation** - one markdown cell with `# Title` and relative
   links to an index, previous notebook, or next notebook when those files
   exist.
2. **Prerequisites alert box** - `<div class="alert alert-block alert-warning">`
   listing prerequisite notebooks, installs, credentials, and required input
   data. A linked prerequisite is the only assumed knowledge.
3. **Learning outcomes** - 3 to 5 bullets under "What you will learn".
4. **Outline** - an anchor-link table of contents for longer notebooks.
5. **Data access** - say what MTG FCI L1C data is used, how the notebook obtains
   it end-to-end, and how the reader would access other data. Include Data Store
   collection IDs, `eumdac` credentials, and maintained EUMETSAT links.
6. **Body** - the workflow, narrated.
7. **Cleanup** - a final code cell that removes everything the notebook created,
   using plain `if`/`else` branches and printing each removal.
8. **Wrap-up and bottom navigation** - 2 or 3 sentences interpreting what was
   built and a `previous | index | next` line when navigation exists.

## Rules

- Use literate programming: at least 2x markdown cells vs code cells, with text
  distributed through the notebook. Every code cell gets a markdown lead-in
  saying why it exists, and every visible output gets a short interpretation.
- Keep it self-contained. The notebook must run top-to-bottom from a fresh
  kernel in the learner's checkout. Do not depend on files from other repos,
  internal CI, runbooks, or hidden local state.
- Derive constants in view of the reader. For example, explain that FCI full
  disk repeat cycles are every 10 minutes, so `SLOTS_PER_DAY = 24 * 6`.
- Avoid chained one-liners. Use named intermediates and print or display useful
  results so each cell teaches one concept.
- Calls with several keyword arguments should read like forms: one argument per
  line, with full-word variable names.
- Use package or documented helper APIs for UI and display. Do not build
  one-off notebook widgets unless the notebook is explicitly about widget
  behavior.
- Preserve Firecube invariants: use public `firecube` commands or public SDK
  APIs, do not hand-edit Firecube state, and make recovery point at re-running
  idempotent ingestion.
- Use alert boxes for asides: `alert-warning` for prerequisites or caveats,
  `alert-info` for notes, `alert-success` for checkpoints, and `alert-danger`
  before destructive cleanup.
- Attach PNG images, not SVG.
- Anchor paths on `Path.home()` or the notebook's directory, not the kernel's
  current working directory.

## Reproducibility Check

Before shipping:

- Restart kernel, then run all cells headlessly with `nbclient` when available.
- Verify claimed outputs by inspecting them, not by assuming success.
- Ensure dependency instructions match notebook imports.
- Confirm the cleanup cell leaves the machine as it was found.

## Report

Summarize the notebook path, cell count and markdown/code ratio, navigation links
added, index entry updates, external references removed or replaced, and the
headless-execution result.
