# PARALLAX

**Test Your AI Agents Before You Trust Their Decisions.**

PARALLAX tests whether a tool-using AI agent's decisions actually depend on the evidence it receives. Change one observation, replay the decision, and inspect what changed.

**Python 3.12 · MCP · LangGraph · CrewAI · Streamlit**  
*v0.1.0 research preview*

## One number. Two decisions.

In a recorded synthetic incident experiment, changing one metric produced a different decision:

| Tool evidence | Agent decision |
| --- | --- |
| **500 bps** (original) | **INVESTIGATE** |
| **499 bps** (one-point edit) | **MONITOR** |
| **501 bps** (control candidate) | **INVESTIGATE** |

The result comes from one synthetic fixture and one model configuration, not a general reliability guarantee.

## Built to test the decision, not just the output

- **Counterfactual replay** — Capture MCP tool evidence, make one validated edit, and compare agent decisions.
- **Decision X-Ray** — Explore observed decision boundaries over an explicitly defined candidate set.
- **LangGraph + CrewAI** — Evaluate trusted local agents with checkpoint continuation (LangGraph) or controlled fresh reruns (CrewAI).
- **Research controls + reports** — Run repeatable comparisons and explore sanitized results in a local Streamlit dashboard.

## Quickstart

Requires **Python 3.12** and [uv](https://docs.astral.sh/uv/). From the repository root:

```powershell
uv sync --all-groups
uv run python -m parallax.experiment --protocol-pilot --dry-run
uv run pytest -q
```

The dry-run and default tests do **not** call a paid model. To explore the research dashboard:

```powershell
uv run --extra dashboard streamlit run streamlit_app.py
```

**Bring your own agent:** Follow the [LangGraph and CrewAI quickstarts](docs/BRING_YOUR_OWN_AGENT.md). Factories execute **trusted local Python**; PARALLAX does not host or sandbox uploaded agents.

## Recorded research

The protocol pilot yielded **15/15 valid pairs** (5/5 for each of H1, H2, and H3). The separate X-Ray and control studies examine a decision boundary and repeated continuations. These are small, exploratory experiments; scripted framework examples are not provider-backed reliability measurements.

[Results](docs/RESULTS.md) · [Reproduce](docs/REPRODUCIBILITY.md) · [Architecture](docs/ARCHITECTURE.md) · [Research manuscript](paper/parallax-paper.md) · [Security](SECURITY.md)

## License

[Apache License 2.0](LICENSE) © 2026 Kaustubh Chaudhari.

Third-party libraries and materials retain their own licenses. See [licensing details](docs/LICENSING.md).
