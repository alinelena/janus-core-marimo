# janus-core-marimo

An interactive, reactive 8-tab **Marimo** workbench for [STFC `janus-core`](https://github.com/stfc/janus-core) atomistic machine-learned interatomic potential (MLIP) simulations, featuring **Chemiscope 3D** molecular/crystalline visualization, **Altair 2D** interactive charts, a cross-tab **Structure Relay Ledger**, and on-demand **isolated `uv` environments** for conflicting MLIP backends.

---

## Features & Calculation Modes

Each calculation/inference mode from `janus --help` (excluding training/preprocessing) has a dedicated reactive tab in [`janus_workbench.py`](janus_workbench.py):

| Tab | `janus` CLI Mode | Altair 2D Chart (`mo.ui.altair_chart`) | Chemiscope 3D View (`chemiscope.show`) |
|---|---|---|---|
| **1. Single-Point** | `janus singlepoint` | Per-atom force magnitude histogram ($\|\mathbf{F}_i\|$ in eV/Å) & virial stress bar chart | 3D crystal/molecule viewer with per-atom force vector arrows (`ase_vectors_to_arrows`) |
| **2. Geometry Optimization** | `janus geomopt` | Dual-axis energy ($E$) & max force ($F_{\max}$) convergence vs. optimization step | Optimization relaxation trajectory with force vectors and `"Promote Relaxed Structure to Ledger"` relay |
| **3. Molecular Dynamics** | `janus md` | Multi-panel time series of $E_{\text{pot}}$, $E_{\text{kin}}$, $E_{\text{tot}}$, $T(t)$, $P(t)$, and $V(t)$ | Subsampled MD trajectory viewer with `"Promote Frame to Ledger"` relay |
| **4. Phonons** | `janus phonons` | Phonon dispersion band structure ($\omega(\mathbf{q})$ in THz) + vibrational DOS & thermal properties ($C_V, S, F$) | Equilibrium supercell viewer (`hdf5=False` YAML band output for lightweight execution) |
| **5. Equation of State (EOS)** | `janus eos` | Raw $E(V)$ lattice points + Birch-Murnaghan fitted curve & bulk modulus $B_0$ summary | Volumetric compression/expansion structures across strains |
| **6. Elasticity** | `janus elasticity` | Interactive $6 \times 6$ Voigt stiffness tensor heatmap ($C_{ij}$ in GPa) + $B_V, B_R, B_{VRH}, G_V, G_R, G_{VRH}, E, \nu$ | Deformed shear/normal strain state structures |
| **7. Nudged Elastic Band (NEB)** | `janus neb` | Minimum Energy Path (MEP) barrier $\Delta E$ (eV) vs. reaction coordinate (Å) + cubic spline fit | Interpolated band images from reactant $\rightarrow$ transition state $\rightarrow$ product (`write_band=True`) |
| **8. MLIP Descriptors** | `janus descriptors` | Per-element invariant descriptor distribution (`invariants_only=True`, `calc_per_atom=True`) | 3D structure colored by local atomic environment descriptors |

---

## Architecture Highlights

- **Shared MLIP Configuration, Existing/Custom Env Selector & Isolated `uv` Environments (`src/janus_marimo/envs.py`)**:
  - Supports all 14 `janus-core` MLIP architectures (`mace_mp`, `mace_off`, `mace_omol`, `mace_polar`, `mace`, `sevennet`, `chgnet`, `orb`, `mattersim`, `fairchem`, `nequip`, `dpa3`, `grace`, and `upet`) across 10 isolated environment groups.
  - Auto-discovers existing local `micromamba`, `conda`, and `.venvs/janus-*` environments on disk and allows specifying any custom environment root or `bin/python` path with fast readiness validation and environment-aware SHA-256 run caching.
  - Provides a one-click **"Provision `janus-<group>` Environment via `uv`"** button in the header to install conflicting MLIP dependencies into `.venvs/janus-<group>` without polluting the Marimo frontend kernel.
- **Cross-Tab Structure Relay Ledger & Pre-Flight Chemiscope 3D Preview (`src/janus_marimo/ledger.py` & `src/janus_marimo/viz.py`)**:
  - Ships with built-in crystal/defect presets (NaCl rocksalt, Si diamond, Cu FCC $2\times 2\times 2$ vacancy initial/final pair for NEB, and Fe BCC) plus instant auto-active file upload (`.extxyz`, `.xyz`, `.cif`, `.vasp`, `.poscar`, `.traj`).
  - Uploaded and selected structures immediately render in a lazy **Chemiscope 3D Pre-Flight X-Ray Viewer** displaying per-atom nearest-neighbor distances ($d_{\min, i}$), coordination numbers, clash flags, fractional coordinates, and optional $2\times 2\times 1$ / $2\times 2\times 2$ visual supercell tiling before running any calculation.
  - Relaxed structures from **GeomOpt**, frames from **MD**, or transition-state images from **NEB** can be promoted directly into the shared ledger for downstream **Phonons**, **EOS**, **Elasticity**, or **Descriptors** calculations.
- **Simulation Guardrails (`src/janus_marimo/runner.py` & `src/janus_marimo/inspector.py`)**:
  - Pre-flight structure & periodic boundary condition (PBC) validation before every calculation.
  - Enforces `--no-tracker` (`tracker: False`) in all subprocess executions and `FrechetCellFilter` for variable-cell relaxations.
  - Content-addressed SHA-256 run caching under `runs/<mode>/<run_id>/`.

---

## Quick Start (Local with `uv`)

### Prerequisites
- [`uv`](https://docs.astral.sh/uv/) (`>= 0.10.0`)
- Base `janus` Python environment at `/opt/micromamba/envs/janus/bin/python` (or a local virtual environment created via `uv`)

### 1. Launch the Interactive Marimo Workbench (Edit Mode)
```bash
uv run --no-project --python /opt/micromamba/envs/janus/bin/python marimo edit janus_workbench.py
```
Then open **`http://localhost:2718`** in your browser.

### 2. Launch as a Read-Only Web App
```bash
uv run --no-project --python /opt/micromamba/envs/janus/bin/python marimo run janus_workbench.py --host 0.0.0.0 --port 2718
```

### 3. Inspect a Crystal Structure via CLI before Simulation
```bash
uv run --no-project --python /opt/micromamba/envs/janus/bin/python python tools/structure_inspector.py <path-to-structure> --require-pbc
```

### 4. Provision an Isolated MLIP Environment Manually (Optional)
You can provision any conflicting MLIP backend directly from the notebook UI header, or from the terminal:
```bash
# Example: Provision an isolated SevenNet environment in .venvs/janus-sevennet
uv venv .venvs/janus-sevennet --python 3.12
uv pip install --python .venvs/janus-sevennet/bin/python "janus-core[sevennet]" "janus-core[d3]"
```

---

## Running with Docker

Container definitions live in [`containers/`](containers/README.md) (`containers/Dockerfile`, `containers/Dockerfile.dockerignore`, and `containers/docker-compose.yml`) and pre-configure `/opt/micromamba/envs/janus/bin/python` with `janus-core[mace,d3]`, `marimo`, `chemiscope`, `altair`, and `uv`.

### Option A: Docker Compose (Recommended)
```bash
docker compose -f containers/docker-compose.yml up --build
```
Open **`http://localhost:2718`** in your browser.
- Persistent volumes are automatically mounted for `./runs` (cached simulation outputs) and `./.venvs` (on-demand provisioned MLIP virtual environments).

### Option B: Direct `docker build` & `docker run`

1. **Build the image from the repository root**:
   ```bash
   docker build -f containers/Dockerfile -t janus-core-marimo:latest .
   ```

2. **Run on CPU**:
   ```bash
   docker run --rm -it \
     -p 2718:2718 \
     -v "$(pwd)/runs:/workspace/runs" \
     -v "$(pwd)/.venvs:/workspace/.venvs" \
     janus-core-marimo:latest
   ```

3. **Run with NVIDIA GPU Acceleration** (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):
   ```bash
   docker run --rm -it \
     --gpus all \
     -p 2718:2718 \
     -v "$(pwd)/runs:/workspace/runs" \
     -v "$(pwd)/.venvs:/workspace/.venvs" \
     janus-core-marimo:latest
   ```

4. **Run in Read-Only App Mode**:
   ```bash
   docker run --rm -it \
     -p 2718:2718 \
     -v "$(pwd)/runs:/workspace/runs" \
     -v "$(pwd)/.venvs:/workspace/.venvs" \
     janus-core-marimo:latest \
     uv run --no-project --python /opt/micromamba/envs/janus/bin/python \
     marimo run janus_workbench.py --host 0.0.0.0 --port 2718
   ```

See [`containers/README.md`](containers/README.md) for full container documentation.

---

## Development, Building & Testing

This project uses **`uv`** with the **`uv_build`** backend and **`ruff`** for linting and formatting.

```bash
# Build sdist and wheel packages into dist/ using uv_build
uv build

# Run Ruff linter
uv run --no-project --python /opt/micromamba/envs/janus/bin/python ruff check .

# Check code formatting with Ruff
uv run --no-project --python /opt/micromamba/envs/janus/bin/python ruff format --check .

# Validate Marimo reactive DAG integrity (zero redefined variables or cycles)
uv run --no-project --python /opt/micromamba/envs/janus/bin/python marimo check janus_workbench.py

# Run the pytest test suite
uv run --no-project --python /opt/micromamba/envs/janus/bin/python pytest
```

---

## Repository Structure

```text
├── janus_workbench.py                 # Main 8-tab reactive Marimo notebook
├── pyproject.toml                     # uv_build package config, dependencies, pytest & ruff settings
├── README.md                          # Project overview & quick start guide
├── AGENTS.md                          # Project architecture & agent execution rules
├── containers/
│   ├── Dockerfile                     # Container image with uv + /opt/micromamba/envs/janus
│   ├── Dockerfile.dockerignore        # BuildKit context exclusions
│   ├── docker-compose.yml             # Compose service with persistent .venvs/ and runs/ mounts
│   └── README.md                      # Container build & deployment guide
├── src/janus_marimo/
│   ├── __init__.py                    # Package exports
│   ├── envs.py                        # MLIP environment catalog, readiness probe & uv provisioner
│   ├── inspector.py                   # Pre-flight structure & PBC validation implementation
│   ├── ledger.py                      # Structure Relay Ledger, presets & SHA-256 run caching
│   ├── runner.py                      # Per-mode YAML config compiler & stateless CLI runner
│   ├── parsers.py                     # Output parsers for all 8 janus-core calculation modes
│   └── viz.py                         # Chemiscope 3D widget & Altair 2D chart builders
├── tools/
│   └── structure_inspector.py         # CLI wrapper for pre-flight structure inspection
├── tests/                             # Unit test suite (pytest)
└── docs/                              # Brainstorm briefs & technical design specifications
```
