# AGENTS.md — janus-core-marimo

## Working with Gemini CLI (Elephant/Goldfish Skills)

This project has installed five [Gemini CLI Skills](https://github.com/google/gemini-cli) in `.agents/skills/` that wrap an "elephant/goldfish" workflow inspired by [this article](https://drensin.medium.com/elephants-goldfish-and-the-new-golden-age-of-software-engineering-c33641a48874): the "elephant" is the working session with full context; the "goldfish" is a fresh subagent spawned via `invoke_agent` with no prior context.

For implementation work, the goldfish stress-tests a problem/design doc or a diff. For brainstorming and PRD writing, multiple goldfish run in parallel with different lenses to generate divergent ideas or research findings the elephant synthesizes.

These are activated contextually based on your intent, but you can explicitly trigger them by naming the skill:

| Intent | Skill Trigger | Description |
|---|---|---|
| Early-stage concept design | "Let's run `eg-brainstorm` for <rough idea>" | Multiple goldfish in parallel generate divergent ideas. All questions via `ask_user`. Hands off to `eg-prd` or `eg-new-feature`. |
| Product Requirements Doc | "Let's write an `eg-prd` for <feature>" | Codebase grounding → structured gap-filling → deep research with parallel goldfish → synthesized PRD. Saves to `docs/prds/`. |
| Bug fix flow | "Run `eg-fix-bug` for <description \| #issue>" | Bug fix flow: problem doc → goldfish diagnosis check → failing test → fix → `eg-precommit-review` → test gate. |
| Feature flow | "Run `eg-new-feature` for <description>" | Feature flow: scope confirm → design doc → three-goldfish design check (comprehension + critic + readiness) → implement → `eg-precommit-review` → test gate. Checks Marimo reactive DAG rules and `janus-core` simulation guardrails. |
| Independent review | "Run `eg-precommit-review`" | Local independent-review loop on the pending diff (`ruff check .`, `marimo check`, `pytest`). |

You provide a brief instruction; Gemini writes the doc back at you. Examples:

> Let's do an `eg-brainstorm` for an interactive MLIP benchmark comparison dashboard in Marimo
> Please run the `eg-new-feature` flow for a reactive geometry optimization notebook with MACE and FrechetCellFilter
> Time for an `eg-precommit-review`

Each skill workflow stops short of committing. Authorize the commit explicitly when ready. Follow Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).

---

## Project Architecture & Execution Rules

### 1. Python Environment & Package Management
- Always run Python commands, tests, and notebooks inside the `janus` micromamba environment (`/opt/micromamba/envs/janus/bin/python`) using `uv`:
  ```sh
  uv run --no-project --python /opt/micromamba/envs/janus/bin/python <command>
  ```
- Install packages and run tests via `uv`.

### 2. Linting, Notebook Validation & Testing
- **Linter**: `uv run --no-project --python /opt/micromamba/envs/janus/bin/python ruff check .`
- **Marimo DAG Check**: `uv run --no-project --python /opt/micromamba/envs/janus/bin/python marimo check`
- **Unit Tests**: `uv run --no-project --python /opt/micromamba/envs/janus/bin/python pytest`
- **Interactive Dev Server**: `uv run --no-project --python /opt/micromamba/envs/janus/bin/python marimo edit` (default `http://localhost:2718`)

### 3. STFC `janus-core` Simulation Rules
- **Headless Execution & Emission Tracking**: In automated scripts, notebooks run as scripts, or test pipelines, always pass `--no-tracker` (or `tracker=False` in Python API calls) to prevent unnecessary CodeCarbon overhead or background `pynvml` warnings.
- **Cell Relaxation Filter**: For crystal geometry optimizations involving unit cell changes, always prefer `--filter FrechetCellFilter` (or `filter_func="FrechetCellFilter"`) over older unit cell filters for stable metric convergence.
- **Structure Verification**: Always verify structure coordinates and periodic boundary conditions (e.g., via `tools/structure_inspector.py <structure>`) before executing heavy simulations.
- **Live Documentation**: Query `https://context7.com/stfc/janus-core` or `janus_mcp.context7_client.Context7Client` for current `janus-core` API signatures.

### 4. Marimo Reactive Notebook Conventions
- **DAG Integrity**: Never redefine a variable across multiple cells or introduce cyclic cell dependencies.
- **Execution Guards**: Always gate computationally expensive MLIP inference, geometry optimization, MD trajectories, or phonon calculations behind `mo.ui.run_button` and `mo.stop(...)` so notebooks load immediately without blocking the kernel.
- **State Management**: Avoid mutating objects in-place across cells; return new values from cells or use `mo.state` only when strictly necessary.
