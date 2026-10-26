# Reproducibility

This is the install and check path for PARALLAX 0.1.0. An unfamiliar
engineer can verify the software without a paid model call. Recorded
decisions are not recreated by these commands. They live in the
sanitized catalog and are tabulated in [`RESULTS.md`](RESULTS.md).

A dry-run prints a plan. It does not call a provider. A command with
`--live` is a separate, user-authorized execution. The protocol pilot,
Decision X-Ray, and control studies call a provider only when `--live`
is passed and a key is available. Do not run those live commands as
part of ordinary verification.

## Requirements

- Python 3.12. The recorded studies used 3.12.2. The package requires
  `>=3.12`.
- [uv](https://docs.astral.sh/uv/) 0.13.2 is the version pinned by CI.
  A newer uv may resolve differently. `uv.lock` is the lockfile.
- Git, for the checkout. The release commit under review is `4a09e59`
  plus the uncommitted release documents.
- No API key for the commands in this file unless a section says the
  command is live.

On Windows, if `uv` is not on `PATH`:

```powershell
$uv = "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe"
```

Use `$uv` in place of `uv` below. The examples use `uv`.

From the repository root:

```powershell
uv python install 3.12
uv sync --all-groups
uv run python --version
```

`uv sync --all-groups` installs the core package and the development
group (`pytest`, `ruff`). It does not install the optional extras.

## Extras

| Extra | What it adds | Needed for |
| --- | --- | --- |
| none | MCP SDK, Pydantic, the server, the oracle, mutations, replay | Offline unit tests that do not import the provider adapter |
| `dev` group | pytest, ruff | `uv run pytest`, `uv run ruff check .` |
| `live` | `openai`, `python-dotenv` | Research dry-runs of `experiment`, `xray`, `controls`, and `live`. Those modules import the adapter even when they do not call it. The import is not a paid request. |
| `dashboard` | Streamlit, Plotly | `streamlit run streamlit_app.py` |
| `langgraph` | LangGraph | The LangGraph adapter and its example |
| `crewai` | CrewAI | The CrewAI adapter and its example |

### `live` and `crewai` cannot share an environment

`pyproject.toml` declares a conflict between the `live` extra and the
`crewai` extra. CrewAI's instructor dependency and the OpenAI extra pin
incompatible `jiter` ranges. `uv sync` refuses to install both extras
together. LangGraph does not use that extra and can be installed beside
`live`.

Use one environment for native live runs and a second environment for
CrewAI. The CrewAI commands below use `uv run --extra crewai --isolated`
so they do not rewrite the project environment that holds `live`.

## Offline tests

```powershell
uv sync --all-groups --extra live --extra dashboard
uv run --extra live --extra dashboard pytest -q
uv run --extra live --extra dashboard ruff check .
```

CI runs the same pair, then `uv build`. The default tests do not send a
provider request. They inject scripted completions where a model
adapter is required.

Expected result on the v0.1.0 tree: the suite passes and Ruff reports
no findings. This document does not paste a count that has not just
been produced by that command. The release report records the count
from the run that prepared v0.1.0.

## MCP fixture verification

The fixture store and the server are covered by tests that speak MCP
to the in-process server. No network and no key:

```powershell
uv run pytest -q tests/test_fixtures.py tests/test_server.py tests/test_oracle.py tests/test_mutate.py tests/test_prompt_isolation.py
```

`python -m parallax.server` is the stdio server. It waits for an MCP
client. The tests above are the verification. Starting the server in a
terminal and leaving it open does not add evidence.

What those tests establish: eight fixtures, three read-only tools,
schema validity, the decision table, rejection of illegal edits, and a
prompt construction that does not carry the oracle label. They do not
establish a model rate.

## Research dry-run

Install the `live` extra first. The dry-run imports the adapter module.
It prints the frozen protocol-pilot manifest and returns. It does not
construct a provider client.

```powershell
uv sync --extra live
uv run --extra live python -m parallax.experiment --protocol-pilot --dry-run
```

The manifest's scenario bindings are S01, S11, and S19, with five
repeats. If `PARALLAX_MODEL` is unset, the printed model id is
`(unset)`. A local `.env` may fill that field. The key value is not
printed. Do not add `--live`.

The recorded study `exp-3a-da9fc3022a77b55b` is the catalog, not the
text this command prints.

## Decision X-Ray dry-run

The default candidate list is 499, 501, 498, 502, 495, and 505, and the
default strategy is `early-stop`. That plan is not the published search.
The published search is exhaustive and uses only 499 and 501:

```powershell
uv run --extra live python -m parallax.xray --dry-run --strategy exhaustive --candidates 499 501 --fixture F100 --tool get_metrics --field error_rate_bps
```

The command prints the plan, including the minimality definition. It
does not continue a checkpoint and it does not recreate
`xray-547963e428ed516b`.

## Controls dry-run

The defaults match the published design: fixture F100, field
`error_rate_bps`, intervention 499, negative 501, three repetitions,
seed 0. The seed does not resample a model.

```powershell
uv run --extra live python -m parallax.controls --dry-run
```

The printed study id is new. It is not `ctl-3fcf26ebf34c7ce4`. The
command does not call a provider.

## LangGraph dry-run

```powershell
uv run --extra langgraph python -m parallax.byoa --framework langgraph --agent-factory examples/byoa/langgraph/agent.py:build_target --spec examples/byoa/langgraph/spec.json --dry-run
```

Dry-run loads the factory and compiles the graph. It prints
`live_execution=false`. It does not resume the graph. A second example,
`examples/byoa/latency`, is the same kind of check with a different
spec. See [`BRING_YOUR_OWN_AGENT.md`](BRING_YOUR_OWN_AGENT.md).

The reference graph's decision function is scripted. Running it with
`--live` executes that function. It still does not call a paid
provider, because the spec sets `credentials` to `none`. That `--live`
switch is an agent execution, not a provider bill. Reports land in
`byoa-runs/`, which is gitignored. A PASS line from that example is not
a reliability rate.

## CrewAI dry-run

Use an isolated environment so the project virtualenv can keep the
`live` extra:

```powershell
uv run --extra crewai --isolated python -m parallax.byoa --framework crewai --agent-factory examples/byoa/crewai/crew.py:build_target --spec examples/byoa/crewai/spec.json --dry-run
```

Dry-run loads the factory and does not call `Crew.kickoff`. The
reference crew is scripted. `--live` on this spec kickoff the crew and
does not call a paid provider. It is still not a measurement of a
hosted model. Do not install `crewai` into the same environment as
`live`.

## Streamlit

```powershell
uv sync --extra dashboard
uv run --extra dashboard streamlit run streamlit_app.py
```

The app reads the bundled catalog. It does not read `OPENAI_API_KEY`.
If a local `runs/` directory exists, the sidebar can select it. That
view is still read-only. Public release behavior is the bundled
catalog, which withholds checkpoint identifiers.

`.streamlit/config.toml` sets `gatherUsageStats` to false.

There is no hosted dashboard URL in this release.

## Report format

Native traces are append-only JSONL under `runs/`. A protocol run
writes `manifest.json`, `summary.json`, `trials.jsonl`, and
`traces.jsonl`. X-Ray writes `manifest.json`, `baseline.json`,
`candidates.jsonl`, `result.json`, and `traces.jsonl`. Controls write
`manifest.json`, `baseline.json`, `continuations.jsonl`, `result.json`,
and `traces.jsonl`. All of `runs/` is gitignored.

The committed research record is the sanitized catalog. Its provenance
string says provider response identifiers, fork-call identifiers,
checkpoints, prompts, and raw traces are omitted.

A Bring Your Own Agent report is one JSON file, schema version 1, under
`byoa-runs/` unless `--output` says otherwise. One file is one trial.
It is not pooled with the pilot.

## Execution budgets

Defaults in the research CLIs:

| Budget | Default |
| --- | --- |
| Model calls per agent loop | 4 (`PARALLAX_MAX_MODEL_CALLS`) |
| Tool calls | 6 (`PARALLAX_MAX_TOOL_CALLS`) |
| Protocol-pilot repeats | 5, fixed |
| Protocol-pilot pair model-call ceiling | 10 |
| Protocol-pilot worst-case model requests | 150 |
| X-Ray published candidates | 2, when passed explicitly |
| X-Ray default candidates, if you omit `--candidates` | 6, early-stop |
| Control repetitions | 3 |
| Control worst-case model requests at the defaults | 31, printed by the dry-run |

The protocol's planning token ceilings are in
[`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md). They are ceilings, not
the usage that was billed. Billed token totals for the recorded studies
are in [`RESULTS.md`](RESULTS.md).

BYOA specs carry their own ceilings. The LangGraph reference spec allows
0 model requests and 4 tool requests, with a 10-second timeout. The
CrewAI reference spec allows 12 model requests and 4 tool requests,
with a 20-second timeout. A timeout stops the wait. It does not kill
the worker thread.

## Provenance and validation

Each native manifest stores prompt hashes, fixture hashes, the
intervention source hash, the SDK version, and the Python version.
The catalog copies those fields. `docs/RESULTS.md` quotes the hashes
for the recorded studies.

To check that a local summary still matches the catalog proportions,
compare `summary.json` for an experiment id with the same id inside
`release.json`. Do not rebuild a summary by hand from `traces.jsonl`
and treat the rebuild as the published figure. The dashboard's tested
summary function is the scoring path. Display code reads that summary.

`uv build` produces an sdist and a wheel under `dist/`. `dist/` is
gitignored. The package version in `pyproject.toml` is `0.1.0`.

## Commands that spend money

These require the `live` extra, `OPENAI_API_KEY`, `PARALLAX_MODEL`, and
an explicit `--live`. They are not part of CI. They are not required to
verify the release.

- `python -m parallax.experiment --protocol-pilot --live`
- `python -m parallax.xray --live`
- `python -m parallax.controls --live`
- `python -m parallax.live --live`
- `python -m parallax.live_pair --live`

A live rerun is a new sample. It does not have to match the catalog.
Do not overwrite `runs/` for a published id unless you intend to resume
that directory with `--resume`.

Copy [`.env.example`](../.env.example) to `.env` only on a machine that
will make those calls. `.env` is gitignored.
