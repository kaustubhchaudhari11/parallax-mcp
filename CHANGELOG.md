# Changelog

All notable changes to PARALLAX are documented here. The format follows
the spirit of Keep a Changelog. The package version in `pyproject.toml`
is the version named below.

## 0.1.0 — unreleased

First research-software release candidate. No GitHub Release has been
published. The version stays 0.1.0 until an owner publishes one.

### Added

- Synthetic incident fixtures F000–F111 and a stdio MCP server with
  read-only `get_metrics`, `get_logs`, and `get_deployment`.
- One agent loop, canonical observation capture, schema-checked
  H1/H2/H3 edits, paired continuation from a pre-fork prefix, and a
  decision oracle that is applied after the model answers.
- Protocol-pilot, Decision X-Ray, and research-control runners.
  Dry-runs print plans. Provider calls require `--live`.
- Sanitized summaries for the recorded studies, listed in
  `docs/RESULTS.md`.
- A read-only Streamlit dashboard and a static research site in
  `site/`, with a manual GitHub Pages workflow.
- Local LangGraph and CrewAI adapters, with scripted examples. The
  adapters are not provider-backed reliability studies.
- This changelog, the manuscript in `paper/`, and the release documents
  under `docs/`.

### Notes

- The 24-scenario exploratory evaluation has not been run.
- No license file is included. See `docs/LICENSING.md`.
- `live` and `crewai` extras conflict and are installed in separate
  environments.
