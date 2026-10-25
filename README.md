# PARALLAX

Test Your AI Agents Before You Trust Their Decisions.

v0.1.0 Research Preview. The package version stays 0.1.0 until a GitHub release is published.

PARALLAX changes one piece of tool evidence and compares the agent's decisions. The prompt, model settings, and starting point stay fixed. You see whether the decision holds, moves, or follows an instruction that arrived inside the tool output.

## What it does

An agent investigates a synthetic incident with three read-only tools: `get_metrics`, `get_logs`, and `get_deployment`. It chooses `MONITOR`, `INVESTIGATE`, or `ESCALATE`. PARALLAX keeps the original observation, applies one declared edit, and scores the two decisions after they exist.

You can also point PARALLAX at a compatible LangGraph graph or CrewAI crew that already lives in your checkout. LangGraph resumes from a checkpoint. CrewAI runs a fresh comparison.

## Why it matters

A correct label can still be insensitive to the evidence behind it. The recorded F100 search shows the difference in one field. At 500 basis points the decision was INVESTIGATE. At 499 it became MONITOR. At 501 it stayed INVESTIGATE.

That result belongs to one synthetic fixture and one model configuration. It is a reason to test the decision, not a claim about every agent.

## Key capabilities

- Paired comparison of one captured observation and one validated edit.
- A deterministic decision rule for the synthetic incident, applied after the model answers.
- Decision X-Ray over an enumerated candidate set.
- Research controls that repeat continuations from one checkpoint draw.
- Local LangGraph and CrewAI evaluations with PASS, FAIL, and INCONCLUSIVE.
- A read-only dashboard of sanitized summaries.

## Quickstart

From the repository root, with Python 3.12 and [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --all-groups
uv run pytest -q
uv run ruff check .
```

If `uv` is not on PATH in Windows PowerShell, the checked shim is:

```powershell
$uv = "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe"
```

Offline tests and the scripted examples do not call a paid model. The MCP server is `python -m parallax.server`. It speaks MCP on stdin and stdout.

## LangGraph and CrewAI

Install one framework extra in the environment you use for that adapter. Install the `live` extra and the `crewai` extra in separate environments. They pin incompatible `jiter` ranges. LangGraph does not need the `live` extra.

Dry-run checks the spec and loads the factory. It does not call a model.

```powershell
uv run --extra langgraph python -m parallax.byoa --framework langgraph --agent-factory examples/byoa/langgraph/agent.py:build_target --spec examples/byoa/langgraph/spec.json --dry-run
uv run --extra crewai python -m parallax.byoa --framework crewai --agent-factory examples/byoa/crewai/crew.py:build_target --spec examples/byoa/crewai/spec.json --dry-run
```

The factory is a `path.py:attribute` file inside this repository, outside `src/parallax`. The path check is not a sandbox. Run a checkout you trust. Reports go to `byoa-runs/`, which is gitignored.

The contract, graph shape, and crew limits are in [`docs/BRING_YOUR_OWN_AGENT.md`](docs/BRING_YOUR_OWN_AGENT.md).

## Research demonstration

Three studies are published as sanitized summaries. They are not one pooled result. The live rows use one model configuration, gpt-6-luna. No p-value is reported.

| Study | What was recorded |
| --- | --- |
| Protocol pilot `exp-3a-da9fc3022a77b55b` | 15/15 valid pairs. H1, H2, and H3 each 5/5. Instruction contamination 0/5. Instruction adoption 0/5. |
| Decision X-Ray `xray-547963e428ed516b` | F100 at 500 basis points stayed INVESTIGATE. 499 became MONITOR. 501 stayed INVESTIGATE. |
| Controls `ctl-3fcf26ebf34c7ce4` | One checkpoint draw. Null 0/3, intervention 3/3, negative 0/3. |

The static site in `site/` presents these figures. Publishing that site is a manual GitHub Pages workflow on `main`. This checkout does not treat the Pages URL as already verified.

The local dashboard reads the same sanitized catalog and does not call a model:

```powershell
uv sync --extra dashboard
uv run --extra dashboard streamlit run streamlit_app.py
```

## How to reproduce

The method, pairing rules, and limits are in [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md). Python 3.12, `uv.lock`, and MCP Python SDK 2.3.0 are part of that record. Prompt, fixture, and intervention hashes are stored with each study.

`python -m parallax.experiment --protocol-pilot --dry-run` prints the frozen manifest and does not call a provider. A live rerun is a separate `--live` step. It needs the `live` extra and a provider key outside the repository. The default tests do not take that step. Raw traces stay in gitignored `runs/`. The catalog is the sanitized summary.

## Architecture

The implemented path is fixtures, MCP tools, one agent loop, capture, a validated edit, two continuations, a post-hoc score, and a sanitized report. Component boundaries and the two Bring Your Own Agent modes are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Limitations

The published demo uses a small synthetic incident dataset and one live model configuration. The protocol pilot is 15 pairs. Control repeats share one checkpoint. The X-Ray search enumerated 499 and 501 only. Scripted agent examples are not those rates.

LangGraph resumes from a checkpoint. CrewAI uses a controlled fresh rerun. The website does not execute an uploaded agent. These studies do not guarantee reliability for every agent.

The 24-scenario exploratory evaluation has not been run. A paper is not published.

## License and project links

No license file is included yet. Do not treat the source as licensed for reuse until one is added.

- Repository: <https://github.com/kaustubhchaudhari11/parallax-mcp>
- Bring Your Own Agent: [`docs/BRING_YOUR_OWN_AGENT.md`](docs/BRING_YOUR_OWN_AGENT.md)
- Research protocol: [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md)
- Project history: [`docs/PROJECT_CHARTER.md`](docs/PROJECT_CHARTER.md) and [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md)

Do not commit `.env` files, API keys, or unreviewed traces from live model calls.
