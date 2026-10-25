# PARALLAX Architecture

**Status:** architecture review incorporated, 21 October 2026.
The components below are planned. The only code in the repository today
is the scaffold entry point in `src/parallax/__init__.py`.

## Design summary

PARALLAX separates four responsibilities that are easy to collapse into
one process:

1. The **MCP server** is the factual world. It serves synthetic incident
   fixtures and does not know which experiment arm is running.
2. The **capture and mutation layer** sits where a tool result becomes
   agent context. It stores the original result and, when an arm requires
   it, substitutes a declared mutant.
3. The **agent loop** chooses tool calls and a final decision. It never
   receives the oracle label.
4. The **oracle and summary** score traces after the decision exists.

The server stays factual so a mutant arm cannot destroy the original
observation. Mutation inside the fixture store would force the runner to
reconstruct the original and would mix "the world changed" with "the
agent saw a different payload." Server-side mutation can be added later
as an ablation. It is not the primary path.

## Verified SDK surface

Inspected in the project virtualenv on 21 October 2026.

| Package | Version |
| --- | --- |
| `mcp` | 2.3.0 |
| `pydantic` | 2.14.1 |
| Python | 3.12.2 |

`import mcp.server.fastmcp` raises `ModuleNotFoundError`. The installed
SDK 2 message says FastMCP was renamed to `MCPServer`, imported from
`mcp.server.mcpserver`. Day 1 will use that class. Tools are registered
with `MCPServer.tool`, which requires `@server.tool()` and rejects a bare
`@server.tool`.

The high-level client is `mcp.Client`. Its constructor accepts a server
URL, `StdioServerParameters`, a transport, or an in-process `MCPServer`.
`Client.call_tool(name, arguments)` returns `CallToolResult`. The result
model in `mcp_types._types` has `content`, `structured_content`,
`is_error`, and `result_type`, plus the inherited `meta` field. The
observation contract below is the one PARALLAX will implement. Day 1
confirms it against a live `CallToolResult` from these three tools. That
confirmation has not been run.

Protocol versions exported by the installed `mcp_types.version`:

| Constant | Value |
| --- | --- |
| `LATEST_HANDSHAKE_VERSION` | `2025-11-25` |
| `LATEST_MODERN_VERSION` | `2026-07-28` |
| `KNOWN_PROTOCOL_VERSIONS` | `2024-11-05`, `2025-03-26`, `2025-06-18`, `2025-11-25`, `2026-07-28` |

The public specification page retrieved earlier is revision `2025-06-18`
(<https://modelcontextprotocol.io/specification/2025-06-18>), which is
older than the newest identifier in this SDK. The trace records the
version `initialize` negotiates. It does not hard-code `2025-06-18`.

Day 1 records the negotiated version from a real `initialize` exchange.
The serializer contract below does not depend on which of those versions
is selected, because PARALLAX builds the model-visible string itself.

## MCP observation contract

The SDK does not choose what an LLM reads. `MCPServer` returns a
`CallToolResult`. The agent loop, on day 2, inserts one string. This
section defines that string. It is based on reading SDK 2.3.0, plus one
local check that `pydantic_core.to_json` on a three-field `BaseModel`
emits pretty-printed JSON. Our tools do not exist yet, so day 1 still
has to show the same split on a real call.

### Raw `CallToolResult`

This is the object `Client.call_tool` returns. Capture stores
`model_dump(mode="json")` of the whole object. That dump is the audit
record. It is not inserted into the transcript.

### `content`

`content` is a list of content blocks. For a successful tool that
returns a Pydantic model, `FuncMetadata.convert_result` builds that list
with `_convert_to_content`. A non-string return becomes one
`TextContent` whose `text` is `pydantic_core.to_json(..., indent=2)`.
On the three-field model checked in this environment, that text was
pretty-printed JSON with two-space indentation and field declaration
order. The SDK docstring on `CallToolResult.content` calls this the
unstructured result.

### `structured_content`

For the same successful return, `convert_result` validates the value
against the tool's output model and sets `structured_content` to
`model_dump(mode="json", by_alias=True)`. That value is a JSON object.
The tool's advertised `output_schema` describes it. The SDK comment in
`func_metadata.py` says the unstructured text and this object are built
by different code paths and can diverge.

### `is_error`

`is_error` defaults to false. `_handle_call_tool` turns a tool exception
into `CallToolResult(content=[TextContent(text=str(exc))], is_error=True)`
and leaves `structured_content` unset. An unknown tool name and a raised
`ToolError` take that path. A handler that returns its own
`CallToolResult` with `is_error=True` is passed through, and the SDK
skips output-model validation on that object.

Unknown incident ids therefore return an explicit result, rather than
raising:

- `is_error=True`
- `structured_content={"error": "unknown_incident", "incident_id": <id>}`
- `content` is one text block whose text is the canonical JSON of that
  object

The success output model is not applied to that error object.

### Exact string supplied to the LLM

For a successful call the prompt builder inserts exactly:

```text
json.dumps(
    structured_content,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
)
```

That string is `model_visible_observation`. `content` is archived on the
trace and is not sent. `meta`, `result_type`, and `is_error` are not
sent either. Sorting keys and compact separators make object key order
and insignificant whitespace invisible to the model. Array order remains
visible.

For `is_error=True` with a dict `structured_content`, the same function
produces the string, and the trial is flagged `tool_error`. It is not
mutated and it is not a hypothesis trial. If `structured_content` is
missing, the archived `content` text is sent and the trial is flagged
`tool_error`. That fallback exists so an SDK exception path is still
visible. It is not used for our unknown-incident result.

### How a mutation changes that string

1. Read `structured_content` from the raw result.
2. Validate it as the tool's Pydantic output model. Failure is an
   engine error. The factual result is kept. Nothing is sent as a mutant.
3. Run the named pure function. It returns a new instance of that model.
4. Validate again with `Model.model_validate(mutant.model_dump())`.
   Check the log-line invariants in the protocol. Check that H1 and H3
   leave M, L, and D unchanged, and that H2 changes the table decision.
5. Set `model_visible_observation` to the canonical JSON of
   `mutant.model_dump(mode="json")`.
6. Append that string, and only that string, as the tool message.

The mutation does not edit `content` in place, does not write the
fixture store, and does not rely on the pretty-printed text staying in
sync. The factual arm canonicalizes the original `structured_content`
with the same function, so the two arms differ only where the mutant
object differs.

Schema validity means the mutant parses as the tool output model and
passes the log-line invariants. It does not mean the edit is irrelevant
to the model. Irrelevance for H1 means the oracle predicates stay the
same and the canonical string still changes. Key reordering and
whitespace edits fail the second part. They are not H1 interventions.
The mutation tests reject them. The registered H1 functions,
`h1-log-order`, `h1-rollback-flag`, and `h1-latency`, must pass both
parts.

## Component responsibilities

| Component | Responsibility | Must not |
| --- | --- | --- |
| Fixture store | Versioned synthetic incidents and their Pydantic models | Contain a decision or label field; read the network |
| MCP server | Register the three tools and return fixture data or a structured error | Branch on experiment arm; apply mutations |
| MCP client | Initialize a session, list tools, call tools | Score decisions |
| Capture | Persist the exact result the agent will see, plus hashes | Drop either the original or the mutant |
| Mutation engine | Apply a named pure function and validate the output | Call a model to invent a variant |
| Snapshot store | Save prompt, prefix messages, tool schemas, model config, remaining budget | Save or restore an oracle label into the prompt |
| Agent loop | Call the model, dispatch allowlisted tool calls, stop on a decision or a budget | Hide retries; read traces back in as instructions |
| Oracle | Map facts to `MONITOR`, `INVESTIGATE`, or `ESCALATE` | Run during prompt construction |
| Trace writer | Append one JSON object per trial | Impute token usage |
| Summary | Compute the protocol rates from the full ledger | Omit a trial from the published ledger |
| Comparison view | Read a summary and the stored payloads | Call a model or re-score with a second rubric |

Planned layout, not created yet:

```text
src/parallax/
  schemas.py        fixture and decision models
  fixtures.py       synthetic incidents
  server.py         MCPServer and the three tools
  oracle.py         predicate and decision rules
  mutate.py         named intervention functions
  capture.py        observation records
  agent.py          one control loop
  replay.py         snapshot and arm runner
  trace.py          JSONL writer and summary
tests/
  test_server.py
  test_oracle.py
  test_mutate.py
  test_prompt_isolation.py
```

The read-only comparison view is `streamlit_app.py`. Streamlit and
Plotly are the optional `dashboard` extra. The agent loop does not
import them.

## MCP data flow

```text
fixture store
    |
    v
MCPServer                         factual tools only
  get_metrics
  get_logs
  get_deployment
    |
    |  JSON-RPC over an MCP session
    v
mcp.Client.call_tool
    |
    v
CallToolResult
    |
    +--> capture writes observation_original
    |
    +--> baseline / factual replay: empty overlay, canonicalize the base result
    |
    +--> mutant replay: validate the mutant, store it in that branch's
                        overlay, canonicalize the overlay object
    |
    v
agent transcript
    |
    v
model, under a fixed budget
    |
    +--> later tool call: overlay for that branch, otherwise MCPServer
    |
    +--> final JSON decision
            |
            v
         oracle scores the facts
         trace writer appends one row
```

Two session modes are in scope.

**In-process session.** `Client` is constructed with the `MCPServer`
instance. The SDK documents this as a supported test connection. Day 1
uses it for determinism and to avoid subprocess issues on Windows.

**Stdio session.** `Client` is constructed with `StdioServerParameters`
that launch the server module. This is the realistic local transport.
Day 1 includes one stdio test. If that test is order-dependent or hangs
on Windows, the failure is reported and the in-process session remains
the contract test. The stdio path is not declared passing until it has
passed.

HTTP, OAuth, sampling, elicitation, and roots are out of scope. The
server will not request a model completion. That would put a second model
inside the tool channel and break the fixture determinism this design
depends on.

## Tool contracts

Each tool takes `incident_id: str` and returns a Pydantic model.

`get_metrics` returns the window, `error_rate_bps`, `request_count`, and
`latency_p99_ms`. `error_rate_bps` is an integer. 500 means 5.00%.

`get_logs` returns `highest_severity`, `error_count`, and the log lines.
Severity is an enum (`debug`, `info`, `warn`, `error`), not a value
parsed out of free text. `error_count` and `highest_severity` are stored
explicitly and must agree with the lines, as the protocol requires.

`get_deployment` returns `version`, `deployed_within_window`,
`rollback_available`, and `change_summary`.

An unknown id returns the explicit `is_error` result in the observation
contract. The handler does not raise, because a raised `ToolError`
becomes text-only content with `structured_content` left empty.

The same id returns the same object on every call. Tests compare
serialized structured payloads, not object identity.

## Snapshot and intervention boundary

```text
S0
  prompt, policy, incident_id, tool schemas,
  model id, sampling parameters, budgets
        |
        v
   agent calls a tool
        |
        v
   server returns factual observation O
        |
        v
   ===== snapshot boundary =====
   state = messages before O is appended
           + remaining budget
           + model configuration
           + hashes of prompt, fixture, and O
        |
        +----------------------+
        |                      |
        v                      v
   append O               append mutate(O)
   factual arm            mutant arm
        |                      |
        v                      v
   continue the loop      continue the loop
   until decision,        until decision,
   budget, or error       budget, or error
```

The boundary is before the model consumes `O`. Substituting a mutant
after the model has already answered would measure nothing about the
agent.

The prefix is one object copied into both arms. `prefix_sha256` is
computed before either arm continues, and the pair is flagged
`engine_error` if the hashes differ. The copy includes the messages,
the model configuration, and the remaining budget. It includes no
oracle label and no mutant observation.

The shared fixture store is immutable. Each arm then carries a
branch-local overlay.

- Factual overlay: empty. Every read uses the base fixture.
- Mutant overlay: one entry, the intervention tool mapped to the
  validated mutant. Other tools use the base fixture.

Every later read in that arm resolves overlay first, then base, then
canonicalizes. A repeated read of the mutated tool returns the same
canonical string. The raw server payload is stored separately and may
still be the base fixture. The model sees only the resolved canonical
string. The mutant oracle uses the overlay world for the whole branch,
so a second read cannot restore the factual predicates.

Eligible checkpoints are defined in `RESEARCH_PROTOCOL.md`. Correctness
pairs branch only from a successful, still-factual request for the
intervention's tool, before that result is appended. Later reads are
overlay hits, not new checkpoints.

## Agent loop

One loop, roughly:

1. Send the fixed prompt and the current transcript to the configured model.
2. If the model requests a tool outside the three names, record
   instruction adoption, do not call a server, and do not invent a tool
   result. This is an output event. It is not an executed action.
3. If the name is allowlisted, resolve the branch overlay and then the
   base fixture, canonicalize, append that string, and continue.
   A mutant overlay hit returns the same mutant on every read.
4. If the model returns a final message, store it raw and attempt to parse
   the closed decision schema.
5. Stop at the step budget or the token budget. Stopping is an outcome,
   not a cue to raise the budget and try again quietly.

The model adapter is a narrow interface: `complete(messages, tools, params)
-> assistant message plus optional usage`. Day 2 can ship a fake adapter
for tests and one real adapter behind an environment variable. The fake
adapter's outputs are fixtures for control-flow tests. They are not
PARALLAX results, and the trace marks `model_id` as the fake.

Temperature, seed, and max tokens are whatever the selected API actually
accepts. Unsupported fields are omitted and the omission is logged.
They are not silently rewritten.

## Oracle boundary

`oracle.decide(metrics, logs, deployment)` implements the table in the
research protocol. It is imported by tests and by the runner. It is not
imported by `agent.py` or by the server.

The runner calls it after parsing the agent output and writes
`oracle_decision` onto the trace row. Restoring a snapshot rebuilds
messages from the stored prefix and the selected observation. It does not
read `oracle_decision`.

## Trace storage

Append-only JSONL, one object per trial, under `runs/`. The field list is
the one in `RESEARCH_PROTOCOL.md`. Writers flush a complete line or no
line. A crash mid-trial leaves a missing row, which the summary treats as
a missing trial.

The summary function is pure: traces in, counts out. The day-3 view
displays that output.

## Headroom

Headroom is not on the path from server to agent, not in the trace
writer, and not a dependency of this prototype. A future study of token
compression would use a separate arm and would publish both the compressed
text and the fidelity difference. That study is not part of the three-day
build.

## Compatibility risks

- SDK 2 renamed FastMCP. Examples written for SDK 1 will not import here.
- `content` and `structured_content` are different serializations. The
  model sees only the canonical JSON of `structured_content`.
- Protocol negotiation may select `2025-11-25` or `2026-07-28`. Logging
  the negotiated version is part of every trace, including day-1 tests.
- Stdio on Windows may behave differently from the in-process client.
  Both are specified above so a failure of one does not get papered over
  by the other.
- Provider usage fields differ. Missing usage stays `null`.
- `Client.call_tool` can drive elicitation and sampling callbacks when a
  server asks for them. Our server must not ask. A test can assert that a
  fixture call returns without invoking those callbacks.

## Day 1 implementation plan

This is the next checkpoint after architectural review. It is not started.

1. Add Pydantic models for the three tool results and for the structured
   unknown-incident error. No decision field.
2. Add the eight synthetic incidents, one predicate cell each, and
   unit-test `oracle.decide` for every row of the decision table. The
   oracle module has no MCP dependency.
3. Register the three tools on `MCPServer` with `@server.tool()`. Handlers
   copy fixture models and return them. Unknown ids return a structured
   error through the SDK's tool-error path. The exact helper is confirmed
   against SDK 2.3.0 during implementation, not guessed in advance.
4. Test through `mcp.Client` connected in-process to that server:
   `list_tools` returns the three names; each known id validates; repeat
   calls match; an unknown id sets `is_error` and returns the structured
   error object; serialized payloads contain none of `oracle`, `label`,
   `gold`, or `expected_decision`.
5. On one successful call, record raw `content`, `structured_content`,
   and `is_error`. Assert that `model_visible_observation` equals
   `json.dumps(structured_content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`,
   and that this string is what a later prompt builder must insert.
   Assert that the pretty-printed `content` text is archived and is not
   substituted for that string.
6. Log the negotiated protocol version in the test output.
7. Add one stdio test that launches the server module. Report the real
   outcome.
8. Run `pytest` and `ruff check` on the touched files. Paste the results
   into the milestone report. Do not add a model client, Streamlit, or
   Headroom.

Day 1 acceptance criteria:

- The server exposes exactly `get_metrics`, `get_logs`, and
  `get_deployment`.
- Each tool reads a synthetic fixture selected by `incident_id`.
- Two calls with the same id return the same `structured_content`.
- Successful payloads validate as the Pydantic models, including
  `error_rate_bps` as an integer and the log-line invariants.
- The eight fixture cells, one per predicate combination, each produce
  the decision in the protocol table. This test imports the oracle and
  does not import the agent.
- An unknown id returns `is_error=True`, the structured error object,
  and canonical JSON in `content`. It does not leak a traceback.
- No fixture, tool payload, or canonical string contains a decision,
  label, or oracle field.
- The canonical serializer is covered by a test that distinguishes it
  from the pretty-printed `content` text. Two payloads that differ only
  by object key order or insignificant whitespace produce the same
  canonical string. Those edits are not model-visible interventions.
- No fourth tool is registered. `propose_action` is not part of Day 1.
- Tests use the installed MCP client against the installed server.
- Tests pass with no network access and no model API key.
- Ruff reports no issues on the files touched that day.
