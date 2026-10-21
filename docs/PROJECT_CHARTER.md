# PARALLAX Project Charter

**Status:** architecture review incorporated, 21 October 2026.
No experimental system has been built. The repository contains a package
scaffold whose console script prints a hello string. Nothing in this
charter is a measured result.

**Name:** PARALLAX
**Subtitle:** Counterfactual Reliability Testing for MCP Agents
**Tagline:** Same Agent. Different Evidence. Different Future.

## Why this exists

Tool-using agents decide from observations returned by tools. A single
transcript shows the decision the agent made. It does not show whether
that decision was tied to the evidence.

PARALLAX is a reproducible prototype for that question. The same agent,
the same prompt, the same model settings, and the same initial state are
held fixed. A declared edit is then applied to one Model Context Protocol
(MCP) tool observation. The prototype records whether the decision stays
correct, changes correctly, or follows an untrusted instruction that was
sitting inside the observation.

The setting is a simulated production-incident investigation. One agent
may call three read-only tools:

1. `get_metrics(incident_id)`
2. `get_logs(incident_id)`
3. `get_deployment(incident_id)`

It must return one of three decisions: `MONITOR`, `INVESTIGATE`, or
`ESCALATE`. Every incident, metric, log line, and deployment record is a
synthetic fixture. The agent has no remediation tool. No external system
is contacted, paged, restarted, or modified.

## Hypotheses

These are claims to test. They are not assumptions about model behavior.

**H1 — Invariance.** A semantically irrelevant change to an MCP
observation preserves the agent's correct decision.

**H2 — Sensitivity.** A change to decision-relevant evidence causes the
agent to adopt the decision required by the updated evidence.

**H3 — Trust boundaries.** An instruction embedded in an MCP observation
is not adopted in the model output. The initial experiment measures
decision contamination and instruction adoption. It does not measure an
executed operational action, because the three tools are read-only.

"Irrelevant", "decision-relevant", and "authority" are defined by the
research protocol and the decision oracle. They are properties of the
experiment, not impressions formed while reading a transcript.

## Scope

The prototype will contain:

- A real MCP server and MCP client, using the installed official Python
  SDK.
- Deterministic synthetic incident fixtures.
- One explicit, configurable agent control loop.
- Exact capture of original and mutated tool observations.
- A deterministic mutation engine.
- An agent-state snapshot and a paired replay from that snapshot.
- A deterministic decision oracle whose labels never enter the agent
  context.
- JSONL traces with model identity, parameters, tool calls, token usage
  when the provider reports it, latency, interventions, errors, and run
  identifiers.
- A Streamlit comparison view, scheduled for day 3, that reads traces.
  It is a display of recorded trials, not a source of measurements.

## Non-goals

- Replaying arbitrary production agent logs.
- Attributing which step caused a failure. That is a different question,
  studied by causal-replay work reviewed in `LITERATURE_REVIEW.md`.
- Testing protocol attacks, transport attacks, OAuth flaws, tool-description
  poisoning, or malicious third-party MCP servers.
- Building a defense and claiming that it works.
- Emulating tools with another language model.
- Compressing evaluation observations with Headroom. Headroom is a
  separate development-tool experiment. It is not part of the measurement
  path and will not be installed inside the agent.
- Calling live infrastructure, sending notifications, or executing
  remediation.
- Reporting benchmark scores before the trials have been run.
- Claiming academic novelty. The literature comparison is input to that
  judgment, and the judgment stays open until the prototype has been
  evaluated.

## What a result will mean

A completed prototype can support a narrow statement of this form:

> Under this agent loop, this model configuration, these fixtures, and
> these pre-declared interventions, the recorded decisions matched or
> diverged from the oracle at the rates written in the trace summary.

It cannot support a statement about agents, models, or MCP in general.
Repeated runs are expected to vary. Disagreement across repeats is a
result, and it will be reported.

## Safety boundary

Fixtures are synthetic. Tool handlers read the fixture store and return
data. They do not open network connections, shells, or production APIs.
Trust-boundary strings live inside fixture fields and are shown only to
the agent under test. The runner scores the agent's decision. It does
not carry out any instruction found in a tool result.

## Three-day plan

### Day 1 — MCP foundations and deterministic tests

Build the fixture store, the three-tool MCP server, and a client session
that can list and call the tools. Prove determinism, schema validity, and
the absence of oracle labels with pytest. No language model.

### Day 2 — Agent, capture, mutation, and replay

Add one agent loop, exact observation capture, the mutation engine, the
snapshot boundary, factual-versus-mutated replay, the oracle, and JSONL
traces. Unit tests cover the oracle and the mutations without an API key.
A live model call is optional and is skipped when no key is configured.

### Day 3 — Paired evaluation and the comparison view

Run the pre-declared experiment, compute the metrics in
`RESEARCH_PROTOCOL.md` from the traces, and add a Streamlit view that
shows an original observation beside its mutant and the two decisions.
Charts are labeled as prototype counts. Limitations stay visible.

Each day stops when its acceptance criteria are met. The next day does
not start automatically.

## Acceptance criteria

### Day 1

- The server exposes exactly `get_metrics`, `get_logs`, and
  `get_deployment`.
- Each tool reads a synthetic fixture selected by `incident_id`.
- Two calls with the same id return the same payload.
- Payloads validate against the Pydantic models, including integer
  `error_rate_bps` and the log-line invariants.
- The eight predicate cells each receive the decision in the protocol
  table. The test imports the oracle and does not import an agent.
- An unknown id returns `is_error=True` with the structured error object
  and no traceback.
- The canonical model-visible string is tested against
  `structured_content` and is distinct from the pretty-printed `content`
  text. Key order and insignificant whitespace do not change the
  canonical string. The full day-1 list is in `docs/ARCHITECTURE.md`.
- Day 1 registers only the three observation tools. It does not add
  `propose_action`.
- No fixture and no tool payload contains a decision, label, or oracle
  field.
- Tests use the installed MCP client against the installed server.
- Tests pass with no network access and no model API key.
- Ruff reports no issues on the files touched that day.

### Day 2

- One agent loop is the only control flow. Its prompt, tool list, model
  parameters, and step budget are explicit inputs.
- The object passed into that loop has no oracle label. A test constructs
  the prompt inputs and fails if a label field is present.
- Original tool results are stored byte-for-byte as the agent received
  them, and mutated results are stored separately.
- Every mutation is a pure function. Schema-invalid output is an engine
  error and is not sent to the agent.
- Every registered H1 function changes the canonical JSON and leaves
  M, L, and D unchanged. A key-reorder edit and a whitespace edit do
  not change the canonical JSON and are rejected as H1 interventions.
- Replay copies one frozen prefix into both arms and checks
  `prefix_sha256` before either arm continues. The mutant arm keeps a
  branch-local overlay so a later read of the edited tool returns the
  same canonical payload. The shared fixture store is not written.
- The initial tool list remains the three observation tools. The
  side-effect-free `propose_action` extension is not implemented.
- The oracle is a deterministic function of fixture facts. Tests cover
  every cell of the decision table.
- A trace row records run id, model id, model parameters, tool calls,
  latencies, intervention id, errors, and the decision. Token counts are
  recorded only when the provider returns them.
- Unit tests do not require a live model. No test invents a model
  response and treats it as a measured run.

### Day 3

- The runner executes repeated baselines from a fixed initial state and
  paired factual/mutant replays from the same snapshot.
- Metric values in the summary are computed from the JSONL traces by a
  tested function.
- Every attempted trial appears in the summary ledger. Natural baseline
  variation, factual-replay variation, and the mutant contrast are
  reported together, with Wilson intervals and discordant-pair counts.
  Trials are not removed because a replay disagreed with a baseline.
- Parse failures are reported as their own count and are not imputed.
- The Streamlit view reads those traces and shows original and mutated
  observations side by side. It does not call a model.
- The written summary states the model, the sample size, and the
  limitations in the protocol. It does not generalize beyond that run.

## Current repository facts

Verified while writing this charter:

| Item | Value |
| --- | --- |
| Python | 3.12.2 |
| Package version | 0.1.0 |
| MCP Python SDK | 2.3.0 |
| Pydantic | 2.14.1 |
| Virtualenv creator | uv 0.13.2 |
| Git | `main`, no commits yet |
| Application behavior | `parallax:main` prints `Hello from parallax!` |

`uv` was not on `PATH` in the shell used for this inspection. The
existing `.venv` interpreter did import `mcp` and `pydantic`.

## Related documents

- `docs/RESEARCH_PROTOCOL.md` — hypotheses, oracle, pairing, metrics.
- `docs/ARCHITECTURE.md` — components, MCP data flow, snapshot boundary.
- `docs/LITERATURE_REVIEW.md` — sources checked, and how this design
  differs from them.
- `README.md` — setup and milestone entry point.
