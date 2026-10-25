# Latency agent

This is a second local agent. It is not the incident example.

The edited field is `metrics.latency_p99_ms`. The error rate stays 100
in both arms. The graph's own policy is:

- below 400 ms: `KEEP`
- below 1000 ms: `WATCH`
- otherwise: `PAGE`

The spec expects `KEEP` at 200 ms and `PAGE` at 1500 ms. An error-rate
policy would not emit those labels. The model id is
`scripted-latency-policy`. Resuming the graph does not call a provider.
`comparison_mode` is `checkpoint_continuation`.

The factory path is repository-local and uses forward slashes. A path
outside this checkout is rejected. That check executes the file; it is
not a sandbox.

## Command

```powershell
uv run --extra langgraph python -m parallax.byoa --framework langgraph --agent-factory examples/byoa/latency/agent.py:build_target --spec examples/byoa/latency/spec.json --dry-run
```

```powershell
uv run --extra langgraph python -m parallax.byoa --framework langgraph --agent-factory examples/byoa/latency/agent.py:build_target --spec examples/byoa/latency/spec.json --live --output byoa-runs
```

## What the factory implements

`build_target` returns a `LangGraphTarget` from the public contract:
a compiled graph, `evidence_key`, `decision_key`, `decision_node`, and
`side_effects="read_only"`. The graph is compiled with a checkpointer
and `interrupt_before=["decide"]`. No adapter module was edited for
this agent.
