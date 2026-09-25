---
name: eg-fix-bug
description: Fix a bug using the elephant/goldfish workflow. Enforces a problem doc, goldfish diagnosis check, failing test, fix, and test gate.
---

Fix a bug using the elephant/goldfish workflow. The aim: write a problem doc, goldfish-check the diagnosis (so we are not anchored to the first hypothesis), capture the bug as a failing test, fix it, then run the test gate.

If the user provided a GitHub issue URL or `#<number>`, fetch it first with `run_shell_command` using `gh issue view <number> --json title,body,labels,comments` and seed the problem doc from it.

## Step 0: Triviality gate

**Skip the goldfish/test ceremony for:** typo fixes in copy or comments, dead-code removal, version bumps, formatter-only diffs, single-line config tweaks. Go straight to Step 5 (review/test gate).

**Run the full loop for everything else.**

## Step 1: Write the problem doc (in this conversation)

Print a tight problem doc to the user:

```
PROBLEM DOC
- Symptom: <what the user observes>
- Repro: <steps to reproduce, or "user did not provide; need to derive">
- Suspected area: <file/module/route/worker/job/screen>
- Hypothesised root cause: <one sentence>
- Blast radius: <which other surfaces could be affected>
- "Fixed" means: <specific test passes / specific behavior / specific output>
```

If the user gave no repro and the bug is not obvious, **stop and ask** for a repro path. Do NOT guess.

For interactive Marimo notebook UI bugs, the repro path involves inspecting `marimo check` or navigating the Marimo app at `http://localhost:2718` (start via `uv run --no-project --python /opt/micromamba/envs/janus/bin/python marimo edit` or `marimo run <notebook.py>`). For atomistic simulation / janus-core bugs, repro is a failing `pytest` test or a minimal headless script run with `--no-tracker`.

## Step 2: Goldfish diagnosis check

Spawn a fresh agent with `invoke_agent` (`agent_name="generalist"` or `"codebase_investigator"`):

The goldfish gets ONLY the symptom + repro from Step 1. It does NOT get your hypothesised root cause.

**Prompt body to send:**

```
Independent diagnosis of a bug in this repo (interactive Marimo notebooks and workflows for STFC janus-core atomistic machine-learning interatomic potential (MLIP) simulations; AGENTS.md at the repo root has the full architecture, or check legacy GEMINI.md if present).

Symptom: <FILL IN from Step 1>
Repro: <FILL IN from Step 1>

Investigate. Where in the codebase is the bug most likely to live? Cite specific file:line locations. List the top 1-3 candidate root causes ranked by likelihood. For each candidate, name what evidence in the code supports it and what would falsify it. Do NOT propose a fix yet — just diagnose.
```

**Compare goldfish output to your Step 1 hypothesis.**
- Convergence: proceed to Step 3 with confidence.
- Divergence: re-investigate. Update the problem doc if the goldfish is right.

## Step 3: Capture the bug as a failing test BEFORE fixing

The verification criterion lives in code, not in chat. Pick the right tier:

- **Unit / simulation logic (`tests/`)**: Add or update a `pytest` test in `tests/` run via `uv run --no-project --python /opt/micromamba/envs/janus/bin/python pytest` (always disable emission tracking with `tracker=False` or `--no-tracker`).
- **Marimo notebook DAG / cell execution**: Validate reactive cell dependencies with `uv run --no-project --python /opt/micromamba/envs/janus/bin/python marimo check` or import and test cell functions directly in `pytest`.

Write the test using `write_file` or `replace`. Run it via `run_shell_command`. Confirm it fails for the reason described in the problem doc. If it fails for a different reason, fix the test before touching the implementation.

For purely visual Marimo widget/layout issues that cannot be asserted in `pytest`, add a regression test for the underlying state/helper function and verify the interactive notebook behavior at `http://localhost:2718`.

## Step 4: Fix it

Implement the smallest change that turns the failing test green and matches the "Fixed means" criterion using `replace` or `write_file`.

Avoid adjacent refactors. Bug fix scope is the bug, nothing else.

Re-run the failing test. It must go green.

## Step 5: Hand off to review

Run the `eg-precommit-review` workflow explicitly if the user requested a review, or review your own diff critically.

## Step 6: Test gate

Execute the test gate via `run_shell_command`:

```sh
uv run --no-project --python /opt/micromamba/envs/janus/bin/python ruff check .
uv run --no-project --python /opt/micromamba/envs/janus/bin/python marimo check
uv run --no-project --python /opt/micromamba/envs/janus/bin/python pytest
```

All required tiers must pass.

## Step 7: Final report

Print to the user:
- Bug summary (one line)
- Root cause (one line)
- Fix (file:line)
- Test that captures it (file:test name)
- Goldfish-vs-elephant agreement
- Test gate status

**STOP.** Do NOT commit. Wait for the user's literal commit instruction. Follow Conventional Commits (`fix:`, `test:`, etc.) when authorized.
