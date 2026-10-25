# CrewAI example

This crew is a real `Crew` with two agents and one sequential process.

- Metrics Analyst calls `get_incident_evidence`.
- Incident Investigator reads that task output and returns a JSON decision.

`build_target` does not kick off a crew. Each evaluation arm calls
`build_crew` with its own evidence object, so the tool closure for the
counterfactual arm returns the edited value.

The model is `PolicyLLM`, a `BaseLLM` subclass. It follows CrewAI's
text tool format. It does not call a provider.

## Command

Dry-run:

```powershell
uv run --extra crewai python -m parallax.byoa --framework crewai --agent-factory examples/byoa/crewai/crew.py:build_target --spec examples/byoa/crewai/spec.json --dry-run
```

Scripted live run:

```powershell
uv run --extra crewai python -m parallax.byoa --framework crewai --agent-factory examples/byoa/crewai/crew.py:build_target --spec examples/byoa/crewai/spec.json --live --output byoa-runs
```

Do not combine this extra with `--extra live`. Those extras pin
incompatible `jiter` ranges.

Offline tests:

```powershell
uv run --extra crewai pytest -q tests/test_byoa_crewai.py
```

## Expected report

`comparison_mode` is `controlled_fresh_rerun` and `shared_checkpoint`
is false. For `spec.json`, a valid run reports factual `MONITOR`,
counterfactual `ESCALATE`, and verdict `PASS` under
`escalate-when-the-error-rate-is-raised`. The tool runs once in each
arm. Token usage is null unless the model object reports tokens.

## Limitations

These two arms do not share a checkpoint. A scripted model makes the
pair deterministic. A provider model can vary between fresh runs. The
crew rejects delegation, a manager, memory, cache, tracing, and planning.
