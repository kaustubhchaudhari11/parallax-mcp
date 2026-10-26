# Licensing decision

**Status.** No license has been chosen. This file is a recommendation.
It is not a grant. Do not add a `LICENSE` file, a `license` field in
`pyproject.toml`, or a GitHub license badge until the owner accepts a
license in writing.

Until that happens, the default position is that the copyright holder
has not granted reuse rights. Readers can look at the repository.
They cannot rely on a right to copy, modify, or ship it.

## Recommendation

Apache License, Version 2.0.

Use it for this project if the owner wants an open-source research
release that other labs and companies can run, modify, and
redistribute, with an explicit patent grant from contributors and a
requirement to keep notices intact.

Why this license rather than a shorter permissive text:

- The code is research software that may sit next to papers and
  industrial prototypes. Apache-2.0 is widely used for that mix and is
  understood by university technology-transfer offices.
- The patent grant matters more here than it does for a small script.
  An evaluation harness can be reimplemented, and the grant reduces
  ambiguity about contributor patent claims.
- The notice and change-tracking requirements are light, and they fit
  a repository that already keeps a changelog and a manuscript.
- Dependencies used by the core install, including the MCP Python SDK
  and Pydantic, are themselves under permissive licenses. Apache-2.0
  does not force those dependencies onto a different license, and it
  does not require PARALLAX to be copyleft.

MIT is a reasonable alternative if the owner wants the shortest
permissive grant and does not want the patent and notice machinery.
GPL-3.0 is a poor fit if the goal is a library that closed downstream
tools can call. No license was selected between these options.

CC BY 4.0 is appropriate for the manuscript if the owner wants a
documentation license that is separate from the code. It is not a
software license and should not be the only grant on `src/`.

## What stays blocked

Until a `LICENSE` file is added and the package metadata names it:

- PyPI publication should wait. Uploading states a license.
- Downstream packagers should not redistribute the code.
- The README must keep saying that reuse is not granted. It does.
- The GitHub license detector will report no license. That is accurate.
- Figures in `paper/figures/` and the manuscript travel with the code.
  They are not separately licensed.
- Third-party snippets are not an issue in the current tree. The
  manuscript quotes short method phrases from this repository. It does
  not copy papers.

Accepting Apache-2.0 later is a one-file addition plus the `license`
field. It does not require rewriting history. It should be its own
commit, made by the owner, not folded silently into a feature change.

## Owner action

Reply with the license name to use. The recommended text is Apache-2.0.
No `LICENSE` file will be added without that reply.
