# PARALLAX 0.1.0 release notes

**Status.** Release candidate in the working tree. No GitHub Release
has been created. The package version is 0.1.0. A workshop manuscript
is included. It is not a published paper and it has no DOI.

These notes describe what the tree implements and what was measured.
They do not describe the 24-scenario evaluation, a second model, or a
hosted agent runner. Those have not been built as results.

## What you can run

- A stdio MCP server with three read-only tools over eight synthetic
  incidents.
- Paired factual and counterfactual continuations from one prefix, with
  schema-checked edits in the invariance, adaptation, and instruction
  families.
- Decision X-Ray over an enumerated integer candidate list.
- A null, intervention, and negative control from one checkpoint draw.
- Scripted LangGraph and CrewAI examples through a local adapter.
  LangGraph resumes a checkpoint. CrewAI starts a new crew per arm.
- A Streamlit view of the sanitized catalog, and a static site that
  repeats the published figures.
- Offline tests and dry-runs without a provider key.

## What was measured

Three provider-backed studies, not pooled, all on `gpt-6-luna` with low
reasoning effort:

- Protocol pilot `exp-3a-da9fc3022a77b55b`: 15/15 valid pairs. H1 5/5,
  H2 5/5, H3 rejection 5/5, H3 contamination 0/5.
- Decision X-Ray `xray-547963e428ed516b`: INVESTIGATE at 500 basis
  points, MONITOR at 499, INVESTIGATE at 501. The minimum is over that
  candidate pair only.
- Controls `ctl-3fcf26ebf34c7ce4`: null 0/3 flips, intervention 3/3,
  negative 0/3, from one checkpoint draw. No significance test.

A mini-pilot and a one-trial custom schedule are in the catalog and are
not part of those rates. The mini-pilot's H3 denominator is 0.
Figures, intervals, and token totals are in `docs/RESULTS.md`.

Scripted BYOA PASS reports are example executions. They are not these
rates.

## What you should not infer

The sample is small, the fixtures are synthetic, and the labels are a
closed set. Control repeats share a checkpoint. A fresh provider run
can differ. The software does not certify an agent, and it does not
claim a safety property for MCP in general.

## Install

Python 3.12 and uv. Commands are in `docs/REPRODUCIBILITY.md`.
The `live` and `crewai` extras conflict. Install them in separate
environments. CI does not call a paid model.

## License

No `LICENSE` file is present. Reuse is not granted yet. The
recommendation and the list of what stays blocked are in
`docs/LICENSING.md`.

## Publishing the site

GitHub Pages deploys only from a manual workflow dispatch on `main`,
after Pages is set to GitHub Actions. See `docs/DEPLOYMENT.md`. This
release does not publish the site and does not name a verified URL.
