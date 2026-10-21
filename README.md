# PARALLAX

**Counterfactual Reliability Testing for MCP Agents**

Same Agent. Different Evidence. Different Future.

PARALLAX is a research prototype for a specific question: when a
tool-using language-model agent investigates a simulated production
incident, what does a controlled change in an MCP tool response do to its
decision?

The agent may call three read-only tools, `get_metrics`, `get_logs`, and
`get_deployment`, and must choose `MONITOR`, `INVESTIGATE`, or
`ESCALATE`. Fixtures are synthetic. Nothing is remediated, paged, or
deployed. The comparison that matters is a pair: the same agent, prompt,
model settings, and prefix, with one declared edit to a single tool
observation.

Three hypotheses organize the edits.

1. **Invariance.** An irrelevant change preserves the correct decision.
2. **Sensitivity.** A change to decision-relevant facts moves the decision
   to the new correct one.
3. **Trust boundaries.** An instruction hidden in a tool result does not
   override the task policy.

They are hypotheses. This repository has not tested them.

## Status

Documentation checkpoint, 21 October 2026.

| Piece | State |
| --- | --- |
| Charter, protocol, architecture, literature review | Written for review |
| MCP server, client, fixtures, tests | Not implemented |
| Agent, mutation, replay, traces | Not implemented |
| Experiment results | None |
| Streamlit dashboard | Not implemented |
| Console script `parallax` | Prints `Hello from parallax!` |

Do not treat the documents as evidence that the system runs.

## Planned architecture

The MCP server is the factual world. It serves fixtures and does not know
the experiment arm. A capture step stores the exact tool result. Paired
replay branches immediately before the agent consumes that result: one
arm keeps it, the other substitutes a schema-checked mutant. A
deterministic oracle scores the decision afterward. The oracle label is
never placed in the prompt, the fixture, or the tool payload.

```text
fixtures -> MCPServer -> mcp.Client -> capture
                                      |-> factual arm
                                      |-> mutation engine -> mutant arm
                                               |
                                               v
                                         one agent loop
                                               |
                                               v
                                    oracle + JSONL trace
                                               |
                                               v
                                    comparison view (day 3)
```

Component boundaries, the SDK calls this design depends on, and the
snapshot rule are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
The oracle, the pairing rules, and the metrics are in
[`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md).
Why the prototype exists, what it will not do, and the acceptance tests
are in [`docs/PROJECT_CHARTER.md`](docs/PROJECT_CHARTER.md).
Prior replay, injection, MCP-security, and metamorphic-testing work is
compared in [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md).
Novelty is not claimed.

Headroom is outside this path. Evaluation observations are stored whole.

## Repository facts checked on 21 October 2026

| Item | Value |
| --- | --- |
| Python in `.venv` | 3.12.2 |
| Declared Python | `>=3.12` |
| `mcp` | 2.3.0 (`mcp[cli]>=2.3.0` in `pyproject.toml`) |
| Pydantic | 2.14.1 |
| Dev tools declared | pytest `>=9.1.1`, Ruff `>=0.17.1` |
| Virtualenv | created by uv 0.13.2 |
| Server API in this SDK | `mcp.server.mcpserver.MCPServer` |
| Git | `main`, no commits yet |

`import mcp.server.fastmcp` fails on this SDK. FastMCP was renamed to
`MCPServer`. Protocol version strings known to the installed package run
from `2024-11-05` through `2026-07-28`. Runs will log the version the
session actually negotiates.

The shell used for the inspection did not have `uv` on `PATH`. The
existing `.venv\Scripts\python.exe` did import `mcp` 2.3.0. After `uv` is
on `PATH`, the setup commands are:

```powershell
uv sync --all-groups
uv run pytest
uv run ruff check src tests
```

Until then, the same tools can be invoked with the virtualenv interpreter:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
```

No test files have been added. No extra packages were installed for this
documentation checkpoint. `pytest` and Ruff were not run, because there is
no implementation to check yet.
Streamlit, model SDKs, and Headroom are not dependencies of the current
`pyproject.toml`. API keys belong in the environment, never in the
repository. `.gitignore` excludes `.env`, `.venv`, and local `runs/`.

## Milestones

Work stops at the end of each day for review. Acceptance details are in
the charter.

### Day 1 — MCP foundations and deterministic tests

Implement the fixture models, three synthetic incidents, `MCPServer`
with the three tools, and tests through `mcp.Client`. Include one stdio
launch. Tests must pass offline, without a model key, and must show that
payloads are deterministic, schema-valid, and free of oracle labels.
Run pytest and Ruff and report the actual output.

### Day 2 — Agent, interception, and replay

Add one agent loop, exact capture, pure mutation functions, snapshot
replay of a factual arm and a mutant arm, the oracle, and JSONL traces.
Tests for the oracle, the mutations, and prompt isolation do not call a
model. A live call is optional and skipped when no key is set. A fake
model used in unit tests is labeled as a fake and is not a research
result.

### Day 3 — Evaluation and the comparison view

Run repeated baselines and paired replays. Compute the protocol metrics
from the full trial ledger, with Wilson intervals and discordant-pair
counts. Add a Streamlit view that shows the original observation, the
mutated observation, and the two decisions. The page shows counts and
limitations. It does not call a model.

## Limitations

The protocol states these in full. The short form:

- Synthetic incidents and a policy written for the experiment.
- One agent loop. The exploratory evaluation is 20–24 scenarios with
  3–5 repeats. Five repeats on three scenarios are a pilot, not the
  evaluation.
- Hosted models can change answers even when the prefix is fixed.
- Invariance is defined by the fields the oracle ignores. A model may
  still use those fields.
- Trust tests cover text inside tool results in this simulation. They
  do not cover malicious servers or poisoned tool descriptions.
- No result exists until a trace file exists.

## License and credentials

No license file has been added. Do not commit `.env` files, API keys, or
trace dumps from live model calls unless their contents have been reviewed.
