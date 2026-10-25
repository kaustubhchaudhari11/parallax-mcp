# Bring Your Own Agent

PARALLAX can evaluate one compatible local agent. You declare the
evidence, one edit, and the behavior you expect. A framework adapter
runs the agent twice and writes a versioned report. The verdict is
PASS, FAIL, or INCONCLUSIVE against that contract.

This is not a claim that every agent is supported. A run is valid only
when the agent matches the contract below and the checks in the adapter
succeed.

## How an evaluation works

1. You point the CLI at a spec file and a factory inside this repository.
2. `--dry-run` checks the spec and loads the factory. It does not call the model.
3. `--live` is required before the adapter executes the agent.
4. The adapter keeps the original evidence and builds one modified copy.
5. It reads a decision from each run and applies only the rules in the spec.
6. The report is written under the spec output directory, by default `byoa-runs/`.

The public Streamlit site explains this flow and can display a report
you upload. It does not load a factory, execute Python, or start an agent.

BYOA reports are a separate study from the H1/H2/H3 pilot, Decision
X-Ray, and research controls. One report is one trial. It is not a
reliability rate, and a decision change is not automatically correct.

## Evaluation contract

The spec is JSON, schema version 1. `load_spec` rejects a file that
fails these rules before any agent code runs.

| Field | Meaning |
| --- | --- |
| `framework` | `langgraph` or `crewai` |
| `agent_factory` | `relative/path.py:attribute` inside this repository |
| `task_id` | Name of the task you are scoring |
| `evidence_source` | Tool or evidence name. For CrewAI this must be the tool name |
| `evidence` | Original JSON object |
| `evidence_schema` | Leaf path to `int`, `str`, or `bool` |
| `editable_fields` | Leaves the adapter is allowed to change |
| `interventions` | Allowed values for each editable leaf |
| `selected_intervention` | The one value applied to the counterfactual arm |
| `allowed_decisions` | Labels the adapter will accept |
| `expected_behavior` | Your factual rule and counterfactual rule |
| `max_model_requests` | Ceiling for model calls across both arms |
| `max_tool_requests` | Ceiling for tool calls across both arms |
| `timeout_seconds` | Limit for one graph resume or one crew kickoff |
| `output_dir` | Report directory. It is not part of the contract hash |
| `comparison_mode` | Fixed by the framework. See below |
| `side_effects` | Only `read_only` is accepted |
| `credentials` | `none`, or `env` when a live provider key is required |

A factory path is a relative `.py` file and one attribute. `..`,
absolute paths, and files under `src/parallax` are rejected. The
dashboard never calls the loader.

The selected value must be one of the declared intervention values and
must change the original field. A missing observation is not treated as
a successful edit. A decision that cannot be parsed is not PASS.

`expected_behavior` contains two rules, `factual` and `counterfactual`.
Each rule is `equals` with one `value`, or `one_of` with `values`.
Those labels must belong to `allowed_decisions`. This policy is yours.
It does not call the PARALLAX incident oracle.

LangGraph specs must set `comparison_mode` to `checkpoint_continuation`.
CrewAI specs must set it to `controlled_fresh_rerun`.

## Connect a LangGraph graph

The factory returns a `LangGraphTarget`:

- `graph`: a compiled `langgraph` graph
- `evidence_key`: state channel that holds the evidence object
- `decision_key`: state channel that holds the decision label
- `decision_node`: node that reads the evidence and writes the decision
- `side_effects`: `read_only`
- `model_id`: optional label stored in the report

The checked APIs, against LangGraph 1.2.17, are:

- `langgraph.graph.StateGraph`
- `StateGraph.compile(checkpointer=InMemorySaver(), interrupt_before=[...])`
- `CompiledStateGraph.invoke`
- `CompiledStateGraph.get_state`
- `CompiledStateGraph.get_state_history`
- `CompiledStateGraph.update_state`

The graph must be compiled with a checkpointer and must pause before
`decision_node`. The adapter invokes the graph once, finds the
checkpoint whose `next` is that node, and checks that the evidence
channel still matches the spec. It then:

1. Calls `update_state` with a deep copy of the edited evidence.
2. Reads the original checkpoint again and checks that it was not overwritten.
3. Reads the forked checkpoint and checks that the edited field replaced the old value.
4. Resumes the original checkpoint with `invoke(None, pre.config)`.
5. Resumes the fork with `invoke(None, fork_config)`.

`update_state` creates a new checkpoint. It does not edit the checkpoint
you pass in. A channel without a reducer is replaced. A channel with a
reducer can keep or accumulate the old value. If the forked field is not
the selected value, the report is INCONCLUSIVE and names the reducer.
If there is no checkpoint before the decision node, the report says the
graph shape is unsupported. That result is not described as replay.

The reference graph is `examples/byoa/langgraph/agent.py`. Its decision
node maps `metrics.error_rate_bps` to MONITOR, INVESTIGATE, or ESCALATE.
The offline spec expects MONITOR at 100 and ESCALATE at 800. That model
is scripted. `examples/byoa/latency/agent.py` is a separate graph: it
edits `metrics.latency_p99_ms` and emits KEEP, WATCH, or PAGE.

## Connect a CrewAI crew

The factory returns a `CrewAITarget` whose `build_crew(evidence)` returns
a new `crewai.Crew` for that evidence object. The checked APIs, against
CrewAI 1.6.1, are:

- `crewai.Agent`
- `crewai.Task`
- `crewai.Crew`
- `Crew.kickoff`
- `crewai.Process.sequential`
- `crewai.llms.base_llm.BaseLLM` and `BaseLLM.call`
- `crewai.tools.tool`

The reference crew has two agents. The metrics analyst calls
`get_incident_evidence`. The incident investigator reads that task
output and returns `{"decision": "..."}`. The tool closure holds the
evidence for that arm. The adapter calls the tool once before kickoff
and requires the returned text to contain the evidence value. If it does
not, kickoff does not run.

A supported crew also has to satisfy all of these:

- `process` is sequential
- at least two agents and no more than four
- no manager agent and no manager model
- `allow_delegation` is false and code execution is false
- `memory`, `cache`, `planning` are false and `tracing` is false
- `max_iter` is at most 6 and `max_retry_limit` is at most 1
- the named evidence tool is attached exactly once

The reference is `examples/byoa/crewai/crew.py`.

## Evidence fields

Leaf names in `evidence` must match `evidence_schema`. A boolean is not
an integer. Empty objects are rejected. Only leaves listed in
`editable_fields` can be edited, and the run applies one of them.

For LangGraph, put the evidence object in the declared state channel.
For CrewAI, the tool named by `evidence_source` must return that object.
Prompt text alone is not the evidence boundary.

## Expected behavior

PASS means both parsed labels satisfy your two rules and the execution
checks passed.

FAIL means the execution was valid and at least one label violates your
rule. MONITOR followed by ESCALATE is FAIL when your factual rule
required ESCALATE, even though the decision changed.

INCONCLUSIVE means the adapter cannot score the pair. Timeout, a provider
exception, a missing decision, a label outside `allowed_decisions`, a
missing tool result, a reducer that drops the edit, an unsupported graph,
or a crew that delegates are all INCONCLUSIVE. None of those are PASS.

## Dry-run

Dry-run loads the factory and calls it. The LangGraph factory compiles
the graph. The CrewAI factory returns the builder and does not kick off
a crew. The command prints the plan, the budgets, and
`live_execution=false`. It sends no model request.

```powershell
uv run --extra langgraph python -m parallax.byoa --framework langgraph --agent-factory examples/byoa/langgraph/agent.py:build_target --spec examples/byoa/langgraph/spec.json --dry-run
```

```powershell
uv run --extra langgraph python -m parallax.byoa --framework langgraph --agent-factory examples/byoa/latency/agent.py:build_target --spec examples/byoa/latency/spec.json --dry-run
```

```powershell
uv run --extra crewai python -m parallax.byoa --framework crewai --agent-factory examples/byoa/crewai/crew.py:build_target --spec examples/byoa/crewai/spec.json --dry-run
```

Install one framework extra at a time. `live` and `crewai` cannot be
installed together: CrewAI 1.6.1 and the OpenAI extra pin incompatible
`jiter` ranges. LangGraph does not use that extra.

## Separate studies

These are separate records. A scripted BYOA pair is not a paid
provider trial, and it is not pooled with the research studies.

- Native research experiments use the PARALLAX agent loop. A paid
  provider call happens only when the `live` extra is installed and an
  experiment command is run. The dashboard shows the recorded pilot,
  X-Ray, and control summaries.
- `examples/byoa/langgraph` is a scripted LangGraph graph. Its model id
  is `scripted-incident-policy`. `comparison_mode` is
  `checkpoint_continuation`. `--live` resumes that graph. It does not
  call a provider.
- `examples/byoa/latency` is a second scripted LangGraph graph. It
  edits `metrics.latency_p99_ms` and emits `KEEP`, `WATCH`, or `PAGE`.
  It also uses checkpoint continuation. It is not the incident policy.
- `examples/byoa/crewai` is a scripted CrewAI crew. Its model is a
  `BaseLLM` subclass. `comparison_mode` is `controlled_fresh_rerun`.
  The arms do not share a checkpoint. `--live` calls `Crew.kickoff`.
  It does not call a provider.

## What you implement

The loader imports one repository-local file. It does not load an
arbitrary project outside this checkout. Use forward slashes in
`agent_factory`. Backslashes and `..` are rejected.

For LangGraph, the factory returns a `LangGraphTarget`. The graph is
compiled with a checkpointer and pauses before `decision_node`. The
decision node reads the evidence channel and writes one allowed label.
`examples/byoa/latency/agent.py` is the pattern for a policy that is
not the incident example.

For CrewAI, the factory returns a `CrewAITarget`. `build_crew(evidence)`
returns a new sequential crew whose named tool returns that evidence
object. The final task returns JSON with the decision key. The reference
crew is `examples/byoa/crewai/crew.py`.

The spec declares the field, the one edit, the allowed labels, and the
two rules. The adapter does not supply a policy. Dry-run, then pass
`--live` to execute. A factory outside the repository is not a supported
path.

## Factory loading is not a sandbox

`load_factory` checks that the path stays inside the repository and
outside `src/parallax`, then imports the file. Importing a factory
executes that file with the same permissions as the CLI. The path check
is not a security sandbox. Run BYOA only on a checkout you trust.
The public dashboard does not call the loader.

`side_effects` is a declaration on the target. The adapter rejects any
value other than `read_only` before it runs the graph or the crew. It
does not inspect node or tool source for hidden effects.

## Timeouts, retries, and budgets

`timeout_seconds` bounds one graph `invoke` or one `Crew.kickoff`. The
wait uses a worker thread. When the wait expires, the adapter records a
timeout and stops waiting for the result. The pool then shuts down by
waiting for that worker. The thread is not killed, and the process is
not cancelled.

The adapter does not retry a failed arm. A crew agent with
`max_retry_limit` above 1 is rejected. `max_iter` above 6 is rejected.

Model and tool ceilings are counted from graph state fields, and from
`llm.call` and the tool function the crew exposes. A call hidden inside
a node that never updates those fields is not counted. A count over the
ceiling makes the verdict INCONCLUSIVE.

## Paid execution

`--live` is the only switch that executes an agent. The reference specs
set `credentials` to `none` because their models are scripted and make
no provider call. A factory that needs a key sets `credentials` to
`env`. The CLI then loads the project environment. The report stores
the model id you put on the target. It does not store the key.

```powershell
uv run --extra langgraph python -m parallax.byoa --framework langgraph --agent-factory examples/byoa/langgraph/agent.py:build_target --spec examples/byoa/langgraph/spec.json --live --output byoa-runs
```

Imports, tests, dry-runs, and the dashboard do not pass `--live`.

## Checkpoint comparison

A LangGraph report sets `shared_checkpoint` true only when the forked
checkpoint's parent id is the pre-decision checkpoint and the original
checkpoint still holds the original evidence. Both decisions then come
from resuming that shared prefix. The report records `checkpoint_id`.

This is one pair. It does not by itself prove that the edit caused the
change, and it does not estimate a rate.

## Why CrewAI is a fresh rerun

CrewAI 1.6.1, as used here, builds a new crew for each arm. There is no
shared pre-decision checkpoint. The report sets:

- `comparison_mode` = `controlled_fresh_rerun`
- `shared_checkpoint` = false
- `checkpoint_id` = null

The tool boundary is controlled. The model and the crew runtime can
still vary between arms. Do not read this mode as replay. A later
checkpoint implementation would be a different mode, not a rename of
this one.

## Limitations

- Only the two frameworks and the target objects above are supported.
  A second example does not make arbitrary agents compatible.
- The adapter does not scan arbitrary Python for hidden side effects.
  Mail, databases, payments, deploys, and production writes are out of
  scope. `side_effects` must be `read_only`. That flag is a declaration.
- Request counts are enforced when the graph writes `model_requests`
  and `tool_requests`, or when the crew exposes `llm.call` and the tool
  function. A graph that hides calls inside a node is not fully counted.
- A timeout stops the wait. It is not hard cancellation. The worker
  thread can continue until it returns, and the pool shutdown waits
  for it.
- Scripted reference models do not report provider token bills. Token
  fields are recorded when the model object reports them. A scripted
  call is not paid provider inference.
- The `live` and `crewai` extras conflict. Do not install them together.
  CrewAI 1.6.1 and the OpenAI extra pin incompatible `jiter` ranges.
- No causal claim follows from one pair. A fresh CrewAI rerun can vary.
  A LangGraph pair shares only the recorded pre-decision checkpoint.

## Privacy and execution safety

Reports are JSON. They include hashes of the evidence, the decision
labels, usage counts, short error strings, and the limitations. They
do not include API keys or raw model transcripts. Text containing
`sk-`, `api_key`, `bearer `, or `authorization` is rejected.

The loader reads a local file you named. It does not accept uploaded
Python. The Streamlit page accepts one JSON report up to 256,000 bytes,
validates it, and does not store it. Do not upload secrets or private
data. There is no Run Agent button.

`byoa-runs/` is gitignored.

## Extending an adapter

Add a framework only when you can name the real object that produces the
decision and the comparison mode that object actually supports. Keep the
mode explicit. Do not wrap `parallax.agent` and label the result as that
framework. Validate the spec before execution, keep customer policies in
the spec, and leave the pilot, X-Ray, and control studies untouched.
