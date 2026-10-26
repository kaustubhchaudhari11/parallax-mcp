# Security

PARALLAX is a research prototype. It evaluates agents against synthetic
fixtures and displays sanitized summaries. It is not a production
control plane, and it is not a security boundary for untrusted code
submitted by the public.

## Supported versions

| Version | Status |
| --- | --- |
| 0.1.0 (this tree, not yet a GitHub Release) | The only line. Fixes land on the main branch after review. |

## Reporting a vulnerability

Open a private GitHub security advisory on this repository if that
feature is available. Otherwise write to the maintainer address
recorded as the package author in `pyproject.toml`. Do not open a
public issue that contains a key, a trace, or an exploit.

Please include the version, the path, and what an attacker can do.
You do not need to include a copied secret. Say which file and which
kind of value.

## Secrets and local artifacts

- `.env`, API keys, and provider transcripts do not belong in git.
- `runs/`, `byoa-runs/`, and `*.jsonl` are gitignored because traces
  can hold prompts and provider metadata.
- `.env.example` contains empty placeholders. It is not a credential.
- The committed catalog omits provider response identifiers, fork-call
  identifiers, and checkpoint identifiers.
- Tests use obvious placeholder strings to prove that real markers are
  rejected or redacted. Those strings are fixtures, not credentials.

The package metadata includes the maintainer's email on purpose, so
that a published package has an author. That address is not a secret.
An owner who does not want it in a public package should change it
before publishing.

A scan of the tracked tree and of history that introduced `sk-` found
test fixtures and rejection checks, not a committed provider key.
Details of that review stay in the release notes for the people
merging v0.1.0. This file does not quote fixture values.

## Native tools

`get_metrics`, `get_logs`, and `get_deployment` read synthetic fixtures.
They do not open a network connection, a shell, or a production API.
Trust-boundary strings live inside fixture fields and are scored as
model output. The runner does not perform them.

Live provider calls happen only after `--live`, with a key from the
environment. Continuous integration does not pass `--live`.

## Bring Your Own Agent

The adapter runs code that already lives in the checkout the operator
chose.

- The loader accepts a relative Python file inside the repository and
  outside `src/parallax`. That check is not a sandbox. Importing the
  factory executes it with the operator's permissions.
- There is no hosted endpoint that runs an uploaded agent. The website
  does not accept Python. The dashboard does not call the loader.
- `side_effects` must be `read_only`. That declaration is not a
  sandbox. The adapter does not inspect node or tool source for hidden
  effects such as mail, payments, or deploys.
- `timeout_seconds` bounds one graph resume or one crew kickoff by
  waiting on a worker thread. When the wait expires, the adapter
  records a timeout and stops waiting for the result. The thread is
  not killed. Pool shutdown waits for that worker. A timeout is not
  cancellation.
- LangGraph resumes a checkpoint and can re-execute nodes after that
  checkpoint. CrewAI builds a new crew for each arm and does not share
  a checkpoint. Those are different execution models. A CrewAI
  difference can come from the edit or from running the crew again.
- Request ceilings count the calls the adapter can see. A call hidden
  inside a node that never updates the counted fields is not counted.
- Run BYOA only on a checkout you trust. Do not point it at a
  repository you have not read.

## Dashboard and site

The static site does not call the network from its script. The
dashboard does not attach a provider client. Viewing results requires
no API key. Do not deploy the dashboard with a provider key in its
environment if the only purpose is to show the catalog.
