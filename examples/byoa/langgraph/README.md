# LangGraph example

This graph is a real `StateGraph`. `collect` reads the evidence channel.
The compiled graph pauses before `decide`. `decide` maps
`metrics.error_rate_bps` to `MONITOR`, `INVESTIGATE`, or `ESCALATE`.

The model id is `scripted-incident-policy`. Compiling and resuming this
graph does not call a provider.

## Command

Dry-run:

```powershell
uv run --extra langgraph python -m parallax.byoa --framework langgraph --agent-factory examples/byoa/langgraph/agent.py:build_target --spec examples/byoa/langgraph/spec.json --dry-run
```

Scripted live run:

```powershell
uv run --extra langgraph python -m parallax.byoa --framework langgraph --agent-factory examples/byoa/langgraph/agent.py:build_target --spec examples/byoa/langgraph/spec.json --live --output byoa-runs
```

Offline tests:

```powershell
uv run --extra langgraph pytest -q tests/test_byoa_langgraph.py
```

## Expected report

`comparison_mode` is `checkpoint_continuation`. For the spec in
`spec.json`, a valid run reports factual `MONITOR`, counterfactual
`ESCALATE`, `shared_checkpoint` true, and verdict `PASS` under
`escalate-when-the-error-rate-is-raised`. The original checkpoint id is
recorded. Token usage is null because this graph does not call a model.

## Limitations

The evidence channel has no reducer. A reducer that keeps the old value
makes the run inconclusive. The graph has no checkpointer path other
than the in-memory saver used here. Hidden side effects are not detected.
