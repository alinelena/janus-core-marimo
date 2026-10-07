# DESIGN DOC: Janus-Core 8-Tab Marimo Workbench (`janus-core-marimo`)

**Date:** 2026-09-25  
**Brainstorm Reference:** `docs/eg-brainstorms/janus-core-marimo-workbench-2026-09-25.md`

---

## 1. Why

Computational materials scientists using STFC [`janus-core`](https://github.com/stfc/janus-core) need an interactive, reproducible Marimo notebook that exposes all 8 calculation/inference modes of `janus --help` (`singlepoint`, `geomopt`, `md`, `phonons`, `eos`, `elasticity`, `neb`, `descriptors`) in dedicated tabs, with 3D structure/trajectory/vector visualization powered by **Chemiscope** (`chemiscope`) and 2D diagnostic plots powered by **Altair** (`altair`).

Two fundamental obstacles prevent building this as a naive single-environment Marimo notebook:
1. **Conflicting MLIP Ecosystems**: Foundation potentials supported by `janus-core` (`mace`/`mace_mp`/`mace_off`/`mace_omol`/`mace_polar`, `sevennet`, `chgnet`, `orb`, `mattersim`, `fairchem`, `nequip`, `dpa3`, `grace`, `upet`) require mutually incompatible versions of `torch`, `e3nn`, `dgl`, `cuequivariance`, or `deepmd-kit` and cannot all be installed into a single Python environment.
2. **Marimo Reactive Blast Radius & Browser WebGL Limits**: Wiring a global MLIP configuration dropdown directly into 8 calculation tabs in a static DAG invalidates and resets already-completed tabs whenever the user switches models, while mounting 8 simultaneous Chemiscope (3Dmol.js) viewers exhausts the browser's 16-WebGL-context limit.

This design solves both problems via a **Stateless Content-Addressed YAML-Artifact Subprocess Bus + Structure Relay Ledger**, with **single-active-MLIP configuration** and **user-chosen on-demand `uv` virtual environment provisioning**.

---

## 2. Scope

### In Scope
- **8 `janus-core` Calculation Modes (1 Tab per Mode)**:
  1. `singlepoint` — Static potential energy, atomic forces, Cauchy stress tensor, and optional Hessian evaluation.
  2. `geomopt` — Atomic coordinate and unit-cell relaxation (`LBFGS`, `BFGS`, `FIRE`), defaulting to `FrechetCellFilter` when cell optimization is enabled.
  3. `md` — Molecular dynamics across `nvt`, `npt`, `nve`, `nph`, `nvt-nh`, `nvt-csvr`, `npt-mtk` ensembles, with optional on-the-fly RDF/VAF post-processing.
  4. `phonons` — Harmonic force constants, phonon band dispersion (`bands`), vibrational density of states (`dos`, `pdos`), and thermal properties (`thermal`: $F$, $S$, $C_v$) via Phonopy.
  5. `eos` — Equation of state ($E(V)$ curve and bulk modulus $B_0$) across `birchmurnaghan`, `murnaghan`, `vinet`, `pouriertarantola`, etc.
  6. `elasticity` — Full $6 \times 6$ Voigt stiffness tensor $C_{ij}$ (GPa) and Voigt-Reuss-Hill ($K_{\text{VRH}}, G_{\text{VRH}}, E, \nu$, universal anisotropy) mechanical moduli.
  7. `neb` — Nudged Elastic Band (`ase` IDPP or `pymatgen` interpolation, optional Climbing Image `climb`, forward/reverse energy barriers $\Delta E^\ddagger$ and reaction pathway).
  8. `descriptors` — MLIP invariant/equivariant descriptor calculation (structure-mean, per-element mean, and per-atom descriptors).
- **Shared MLIP Configuration & User-Chosen `uv` Environment Provisioner**:
  - Top-level configuration selecting **one active MLIP at a time** (`arch`, `model`, `device`, `dispersion`, `calc_kwargs`).
  - Interactive **MLIP Environment Provisioner** allowing the user to select which conflicting MLIP environment groups (`mace`, `sevennet`, `chgnet`, `orb`, `mattersim`, `fairchem`, `nequip`, `dpa3`, `grace`, `upet`) to create/install on demand under `.venvs/janus-<group>` via `uv`, while defaulting to `/opt/micromamba/envs/janus/bin/python` when the requested MLIP is already available in the base `janus` environment.
- **Structure Inspection & Cross-Tab Structure Relay Ledger**:
  - `tools/structure_inspector.py` (CLI + importable function) verifying atomic coordinates, periodic boundary conditions (PBC), cell volume, and minimum interatomic distance ($d_{\min}$) before running calculations.
  - Persistent **Structure Relay Ledger** (`#0: Seed Structure` $\rightarrow$ `#1: geomopt relaxed` $\rightarrow$ `#2: promoted MD/NEB frame`) allowing downstream tabs (`phonons`, `eos`, `elasticity`, `md`, `neb`, `descriptors`) to consume structures produced by upstream tabs via an explicit **"Promote to Structure Ledger"** action button (`on_change` callback) that never triggers reactive loops.
- **Interactive Visualizations (Altair 2D + Chemiscope 3D)**:
  - Interactive Altair charts (`mo.ui.altair_chart`) in every tab for 2D curves, histograms, parity plots, and heatmaps.
  - Chemiscope 3D structure/trajectory/vector/environment widgets (`chemiscope.show` + `chemiscope.ase_vectors_to_arrows`) wrapped in `mo.lazy` and `mo.ui.tabs(..., lazy=True)` so only the currently viewed tab mounts a WebGL context.

### Explicitly Out of Scope
- **MLIP Training & Dataset Preprocessing**: `janus train` and `janus preprocess` are excluded at this stage.
- **Persistent Background Socket Worker Daemons**: All calculations execute via stateless `uv run` CLI subprocesses cached in content-addressed run folders (`runs/<mode>/<run_id>/`).
- **Multi-MLIP Concurrent Ensemble Dispatch**: Only one MLIP is active per run click (historical runs remain inspectable in the ledger).
- **Remote HPC Job Schedulers (SLURM/PBS)** and **Non-`janus-core` DFT codes**.

---

## 3. Surfaces Touched

Because this is a greenfield repository (currently containing only `AGENTS.md`, `.agents/skills/`, and `docs/eg-brainstorms/janus-core-marimo-workbench-2026-09-25.md`), the following files will be created. All internal package imports use `janus_marimo.*` and `tools.structure_inspector`, with `pythonpath = ["src", "."]` configured in `pyproject.toml` (and bootstrapped via `sys.path` in `janus_workbench.py` so `marimo check janus_workbench.py` and `marimo edit janus_workbench.py` work without requiring an editable install).

1. `pyproject.toml` — Project metadata using `uv_build` backend (`[tool.uv.build-backend]` with `module-name = "janus_marimo"`, `module-root = "src"`), `pythonpath = ["src", "."]` for `pytest`, and `ruff` linter/formatter configuration (`line-length = 100`, `target-version = "py312"`).
2. `tools/__init__.py`, `tools/structure_inspector.py` & `src/janus_marimo/inspector.py` — Standalone CLI script and importable module for structure & PBC validation (`inspect_structure`, `validate_neb_endpoints`, `StructureReport`).
3. `src/janus_marimo/__init__.py` — Package exports.
4. `src/janus_marimo/envs.py` — MLIP architecture catalog (`MLIP_ENV_CATALOG`, `ARCH_TO_ENV_GROUP`, `DEFAULT_MODEL_BY_ARCH`), environment readiness probe (`resolve_env_status`), and on-demand `uv venv` + `uv pip install` provisioner (`build_provision_commands`, `provision_mlip_env`).
5. `src/janus_marimo/ledger.py` — Data models (`MLIPConfig`, `StructureEntry`, `RunRecord`, `WorkspaceLedger`), deterministic 16-hex-character SHA-256 run-directory hashing (`compute_run_id`), and built-in ASE structure presets (`build_preset_structures`).
6. `src/janus_marimo/runner.py` — Per-mode `janus-core` YAML configuration compiler (`build_janus_yaml_config`), CLI command builder (`build_cli_command`), and stateless subprocess executor (`execute_janus_mode`).
7. `src/janus_marimo/parsers.py` — Artifact parsers (`parse_singlepoint_run`, `parse_geomopt_run`, `parse_md_run`, `parse_phonons_run`, `parse_eos_run`, `parse_elasticity_run`, `parse_neb_run`, `parse_descriptors_run`) reading `.extxyz`, `.yml`, and `.dat` outputs into `pandas.DataFrame` tables and `ase.Atoms` lists.
8. `src/janus_marimo/viz.py` — Chemiscope 3D widget builder (`sanitize_atoms_for_chemiscope`, `build_chemiscope_widget`) and 8 Altair 2D chart builders.
9. `janus_workbench.py` — The main Marimo reactive notebook with the Shared MLIP Config & Environment Manager, Structure Relay Ledger, and 8 calculation tabs.
10. `tests/test_inspector.py`, `tests/test_envs.py`, `tests/test_ledger.py`, `tests/test_runner.py`, `tests/test_parsers.py`, `tests/test_viz.py` — Unit test suite.

---

## 4. Interfaces

### 4.1 `tools/structure_inspector.py`

Validates an `ase.Atoms` object or structure file path before heavy simulations, as mandated by `AGENTS.md`.

```python
from dataclasses import dataclass, field
from pathlib import Path
from ase import Atoms

@dataclass(frozen=True)
class StructureReport:
    formula: str
    n_atoms: int
    pbc: tuple[bool, bool, bool]
    volume: float | None
    cell_lengths: tuple[float, float, float]
    cell_angles: tuple[float, float, float]
    min_distance: float | None
    has_overlapping_atoms: bool
    has_non_finite_coords: bool
    is_valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

def inspect_structure(
    struct: Atoms | str | Path,
    *,
    min_dist_threshold: float = 0.5,
    require_pbc: bool = False,
) -> StructureReport:
    """
    Inspect atomic coordinates, periodic boundary conditions, and interatomic distances.
    - Reads `struct` via `ase.io.read(struct, index=-1)` if a `str` or `Path` is given.
    - Checks `len(atoms) > 0` and `bool(np.isfinite(atoms.positions).all())`.
    - Computes `volume = float(atoms.get_volume())` if `any(atoms.pbc)` and `abs(np.linalg.det(atoms.cell)) > 1e-8` else `None`.
    - Computes `cell_lengths = tuple(float(x) for x in atoms.cell.lengths())` and `cell_angles = tuple(float(x) for x in atoms.cell.angles())`.
    - When `len(atoms) > 1` and not `has_non_finite_coords`, computes pairwise distance matrix `d = atoms.get_all_distances(mic=any(atoms.pbc))`, masks the diagonal with `np.inf`, and sets `min_distance = float(np.min(d))`; otherwise `min_distance = None`.
    - Sets `has_overlapping_atoms = (min_distance is not None and min_distance < min_dist_threshold)`.
    - Appends to `errors`:
      - `"Structure contains 0 atoms."` if `len(atoms) == 0`.
      - `"Structure contains NaN or infinite atomic coordinates."` if `has_non_finite_coords`.
      - `f"Overlapping atoms detected: min distance {min_distance:.3f} Å < {min_dist_threshold:.3f} Å."` if `has_overlapping_atoms`.
      - `"3D periodic boundary conditions (PBC: True, True, True) and non-zero cell volume are required for this calculation."` if `require_pbc and (not all(atoms.pbc) or volume is None or volume <= 0.0)`.
    - Appends to `warnings`:
      - `"Non-periodic boundary conditions detected in at least one direction."` if `not all(atoms.pbc)`.
      - `f"Short interatomic distance detected ({min_distance:.3f} Å < 0.800 Å)."` if `min_distance is not None and min_dist_threshold <= min_distance < 0.8`.
    - Sets `is_valid = (len(errors) == 0)`.
    """

def validate_neb_endpoints(
    init_atoms: Atoms,
    final_atoms: Atoms | None,
    *,
    min_dist_threshold: float = 0.5,
) -> tuple[bool, list[str]]:
    """
    Validate initial and final NEB structures pre-flight:
    1. If `final_atoms is None`, return `(False, ["NEB requires a final structure endpoint (`final_struct_entry`)."])`.
    2. Run `inspect_structure(init_atoms, min_dist_threshold=min_dist_threshold)` and `inspect_structure(final_atoms, min_dist_threshold=min_dist_threshold)`; collect any errors prefixed with `"Initial structure: "` / `"Final structure: "`.
    3. Verify `len(init_atoms) == len(final_atoms)`; if not, append `f"NEB endpoint atom count mismatch: initial has {len(init_atoms)} atoms, final has {len(final_atoms)} atoms."`.
    4. If lengths match, verify `list(init_atoms.numbers) == list(final_atoms.numbers)`; if not, append `"NEB endpoint chemical species/ordering mismatch between initial and final structures."`.
    5. If lengths match, verify `not np.allclose(init_atoms.positions, final_atoms.positions, atol=1e-5)`; if identical, append `"NEB initial and final atomic coordinates are identical (zero reaction pathway displacement)."`.
    6. Return `(len(errors) == 0, errors)`.
    """

def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point: `python tools/structure_inspector.py <structure_path> [--min-dist 0.5] [--require-pbc] [--json]`.
    Prints human-readable summary or JSON serialized `StructureReport` and returns exit code 0 if `is_valid` else 1.
    """
```

---

### 4.2 `src/janus_marimo/envs.py`

Maps all 14 `janus-core` MLIP architectures (`janus_core.helpers.janus_types.Architectures`) to 10 isolated virtual environment conflict groups and handles on-demand `uv` provisioning.

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ArchType = Literal[
    "mace", "mace_mp", "mace_off", "mace_omol", "mace_polar",
    "sevennet", "chgnet", "orb", "mattersim", "fairchem",
    "nequip", "dpa3", "grace", "upet",
]

EnvGroupName = Literal[
    "mace", "sevennet", "chgnet", "orb", "mattersim",
    "fairchem", "nequip", "dpa3", "grace", "upet",
]

DEFAULT_JANUS_PYTHON = Path("/opt/micromamba/envs/janus/bin/python")
DEFAULT_VENVS_ROOT = Path(".venvs")

@dataclass(frozen=True)
class MLIPEnvSpec:
    group: EnvGroupName
    architectures: tuple[ArchType, ...]
    import_probe: str
    janus_extra: str
    pip_packages: tuple[str, ...]
    default_model: str | None
    conflict_note: str

# Exact catalog for all 10 conflict groups:
MLIP_ENV_CATALOG: dict[EnvGroupName, MLIPEnvSpec] = {
    "mace": MLIPEnvSpec(
        group="mace",
        architectures=("mace_mp", "mace_off", "mace_omol", "mace_polar", "mace"),
        import_probe="mace",
        janus_extra="janus-core[mace]",
        pip_packages=("janus-core[mace]",),
        default_model="small",
        conflict_note="MACE pins e3nn==0.4.4 and specific torch/cuequivariance versions.",
    ),
    "sevennet": MLIPEnvSpec(
        group="sevennet",
        architectures=("sevennet",),
        import_probe="sevenn",
        janus_extra="janus-core[sevennet]",
        pip_packages=("janus-core[sevennet]",),
        default_model="SevenNet-0_11July2024",
        conflict_note="SevenNet (sevenn) requires e3nn>=0.5.0 which conflicts with older MACE/NequIP pins.",
    ),
    "chgnet": MLIPEnvSpec(
        group="chgnet",
        architectures=("chgnet",),
        import_probe="chgnet",
        janus_extra="janus-core[chgnet]",
        pip_packages=("janus-core[chgnet]",),
        default_model=None,
        conflict_note="CHGNet sets global torch float32 default dtype and pins specific pymatgen/dgl dependencies.",
    ),
    "orb": MLIPEnvSpec(
        group="orb",
        architectures=("orb",),
        import_probe="orb_models",
        janus_extra="janus-core[orb]",
        pip_packages=("janus-core[orb]",),
        default_model="orb_v3_conservative_20_omat",
        conflict_note="Orb-models requires dm-tree/pynanoflann and modern torch>=2.4 pins.",
    ),
    "mattersim": MLIPEnvSpec(
        group="mattersim",
        architectures=("mattersim",),
        import_probe="mattersim",
        janus_extra="janus-core[mattersim]",
        pip_packages=("janus-core[mattersim]",),
        default_model="mattersim-v1.0.0-5M",
        conflict_note="MatterSim pins torch_geometric/torch_scatter/e3nn versions that clash with MACE/FAIRChem.",
    ),
    "fairchem": MLIPEnvSpec(
        group="fairchem",
        architectures=("fairchem",),
        import_probe="fairchem.core",
        janus_extra="janus-core[fairchem]",
        pip_packages=("janus-core[fairchem]",),
        default_model="uma-m-1p1",
        conflict_note="FAIRChem-core v2 (UMA/eSEN) pins strict torch, hydra, and cluster graph library versions.",
    ),
    "nequip": MLIPEnvSpec(
        group="nequip",
        architectures=("nequip",),
        import_probe="nequip",
        janus_extra="janus-core[nequip]",
        pip_packages=("janus-core[nequip]",),
        default_model=None,
        conflict_note="NequIP pins e3nn and TorchScript compiler constraints incompatible with SevenNet/Orb.",
    ),
    "dpa3": MLIPEnvSpec(
        group="dpa3",
        architectures=("dpa3",),
        import_probe="deepmd",
        janus_extra="janus-core[dpa3]",
        pip_packages=("janus-core[dpa3]",),
        default_model=None,
        conflict_note="DeepMD-kit (DPA3) installs custom C++/CUDA op libraries that can clash with PyG/DGL.",
    ),
    "grace": MLIPEnvSpec(
        group="grace",
        architectures=("grace",),
        import_probe="tensorpotential",
        janus_extra="janus-core[grace]",
        pip_packages=("janus-core[grace]",),
        default_model="GRACE-2L-OMAT",
        conflict_note="GRACE (tensorpotential) requires TensorFlow/JAX backend dependencies isolated from PyTorch MLIPs.",
    ),
    "upet": MLIPEnvSpec(
        group="upet",
        architectures=("upet",),
        import_probe="upet",
        janus_extra="janus-core[upet]",
        pip_packages=("janus-core[upet]",),
        default_model="pet-mad-s",
        conflict_note="UPET pins specific metatensor/metatomic versions.",
    ),
}

# Exact mapping from all 14 architectures to their EnvGroupName:
ARCH_TO_ENV_GROUP: dict[ArchType, EnvGroupName] = {
    "mace": "mace",
    "mace_mp": "mace",
    "mace_off": "mace",
    "mace_omol": "mace",
    "mace_polar": "mace",
    "sevennet": "sevennet",
    "chgnet": "chgnet",
    "orb": "orb",
    "mattersim": "mattersim",
    "fairchem": "fairchem",
    "nequip": "nequip",
    "dpa3": "dpa3",
    "grace": "grace",
    "upet": "upet",
}

# Exact default model label per architecture (matching janus_core.helpers.mlip_calculators.choose_calculator):
# Note: None means either janus-core uses its built-in version default (chgnet) or requires a user-supplied model path (mace, nequip, dpa3).
DEFAULT_MODEL_BY_ARCH: dict[ArchType, str | None] = {
    "mace_mp": "small",
    "mace_off": "small",
    "mace_omol": "extra_large",
    "mace_polar": "polar-1-m",
    "mace": None,
    "sevennet": "SevenNet-0_11July2024",
    "chgnet": None,
    "orb": "orb_v3_conservative_20_omat",
    "mattersim": "mattersim-v1.0.0-5M",
    "fairchem": "uma-m-1p1",
    "nequip": None,
    "dpa3": None,
    "grace": "GRACE-2L-OMAT",
    "upet": "pet-mad-s",
}

ARCH_REQUIRES_EXPLICIT_MODEL: frozenset[ArchType] = frozenset({"mace", "nequip", "dpa3"})

@dataclass(frozen=True)
class EnvStatus:
    group: EnvGroupName
    python_executable: Path
    is_ready: bool
    source: Literal["base_janus", "isolated_venv", "missing"]
    detail: str

def _probe_python_imports(python_exe: Path, modules: tuple[str, ...]) -> bool:
    """
    Check if `python_exe` exists and can import all `modules`:
    1. If `python_exe == Path(sys.executable)` (same interpreter), check via `importlib.util.find_spec(mod) is not None` for each top-level/submodule name.
    2. Otherwise run `subprocess.run([str(python_exe), "-c", f"import {', '.join(modules)}"], capture_output=True, text=True, timeout=15)` and return `returncode == 0` (if the OS blocks subprocess spawning in a restricted test sandbox, fall back to checking whether the module package directory exists under `<venv_prefix>/lib/python*/site-packages/`).
    """

def resolve_env_status(
    arch: ArchType,
    *,
    venvs_root: Path = DEFAULT_VENVS_ROOT,
    base_python: Path = DEFAULT_JANUS_PYTHON,
) -> EnvStatus:
    """
    Determine which Python executable to use for `arch`:
    1. Look up `group = ARCH_TO_ENV_GROUP[arch]` and `spec = MLIP_ENV_CATALOG[group]`.
    2. Check isolated venv first: `venv_python = venvs_root / f"janus-{group}" / "bin" / "python"`.
       If `venv_python.exists()` and `_probe_python_imports(venv_python, ("janus_core", spec.import_probe))`:
       return `EnvStatus(group=group, python_executable=venv_python, is_ready=True, source="isolated_venv", detail=f"Isolated venv ready ({venv_python})")`.
    3. Next check `base_python`:
       If `base_python.exists()` and `_probe_python_imports(base_python, ("janus_core", spec.import_probe))`:
       return `EnvStatus(group=group, python_executable=base_python, is_ready=True, source="base_janus", detail=f"Base janus env ready ({base_python})")`.
    4. Otherwise return `EnvStatus(group=group, python_executable=venv_python, is_ready=False, source="missing", detail=f"Environment 'janus-{group}' is not provisioned yet. {spec.conflict_note}")`.
    """

def build_provision_commands(
    group: EnvGroupName,
    *,
    venvs_root: Path = DEFAULT_VENVS_ROOT,
    python_version: str = "3.12",
    include_d3: bool = True,
) -> list[list[str]]:
    """
    Return the deterministic `uv` command sequence to provision `.venvs/janus-<group>`:
    - `venv_dir = venvs_root / f"janus-{group}"`
    - `venv_python = venv_dir / "bin" / "python"`
    - `packages = list(MLIP_ENV_CATALOG[group].pip_packages) + (["janus-core[d3]"] if include_d3 else [])`
    - Returns:
      1. `["uv", "venv", str(venv_dir), "--python", python_version]`
      2. `["uv", "pip", "install", "--python", str(venv_python), *packages]`
    """

def provision_mlip_env(
    group: EnvGroupName,
    *,
    venvs_root: Path = DEFAULT_VENVS_ROOT,
    python_version: str = "3.12",
    include_d3: bool = True,
    timeout_s: int = 900,
) -> tuple[bool, str]:
    """
    Execute `build_provision_commands(...)` sequentially via `subprocess.run(..., capture_output=True, text=True, timeout=timeout_s)`.
    If any command exits non-zero or raises `OSError`/` subprocess.SubprocessError`, stop and return `(False, combined_log)`.
    Otherwise return `(True, combined_log)`.
    """
```

---

### 4.3 `src/janus_marimo/ledger.py`

Manages content-addressed run directories (`runs/<mode>/<run_id>/`), run history, and the **WorkspaceLedger** (`#0`, `#1`, `#2`...) so tabs can pass structures to downstream tabs without mutating globals or invalidating completed tabs.

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from ase import Atoms
from tools.structure_inspector import StructureReport, inspect_structure

JanusMode = Literal[
    "singlepoint", "geomopt", "md", "phonons",
    "eos", "elasticity", "neb", "descriptors",
]

@dataclass(frozen=True)
class MLIPConfig:
    arch: str = "mace_mp"
    model: str | None = "small"
    device: Literal["cpu", "cuda", "mps", "xpu"] = "cpu"
    dispersion: bool = False
    calc_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_calc_kwargs(self) -> dict[str, Any]:
        """Return a shallow copy of `calc_kwargs` merged with `{"dispersion": True}` when `self.dispersion` is True."""

@dataclass(frozen=True)
class StructureEntry:
    entry_id: str                  # e.g. "struct_0", "struct_1"
    label: str                     # e.g. "#0: NaCl (seed) [8 atoms]" or "#1: NaCl (geomopt mace_mp)"
    atoms: Atoms
    source_mode: str               # "seed", "upload", "geomopt", "md", "neb", "eos", etc.
    is_relaxed: bool               # True if produced by geomopt or minimized before calc
    report: StructureReport
    provenance_run_id: str | None = None

@dataclass(frozen=True)
class RunRecord:
    run_id: str                    # 16-hex-char prefix of SHA-256 digest (matches compute_run_id)
    mode: JanusMode
    mlip_config: MLIPConfig
    struct_entry_id: str
    final_struct_entry_id: str | None  # Used by NEB when two endpoints are supplied
    mode_params: dict[str, Any]
    run_dir: Path                  # e.g. Path("runs/geomopt/<run_id>")
    config_yaml_path: Path         # run_dir / "config.yml"
    summary_yaml_path: Path        # run_dir / f"{mode}-summary.yml"
    output_files: dict[str, Any]
    cli_command: list[str]
    status: Literal["cached", "succeeded", "failed"]
    duration_s: float
    stdout_stderr: str

@dataclass
class WorkspaceLedger:
    structures: dict[str, StructureEntry] = field(default_factory=dict)
    runs: dict[str, RunRecord] = field(default_factory=dict)

    @classmethod
    def from_presets(cls) -> "WorkspaceLedger":
        """
        Create a new `WorkspaceLedger` populated with all entries from `build_preset_structures()`
        as `struct_0`, `struct_1`, ..., `struct_4` (`source_mode="seed"`, `is_relaxed=False`).
        """

    def register_structure(
        self,
        atoms: Atoms,
        *,
        name: str,
        source_mode: str,
        is_relaxed: bool = False,
        provenance_run_id: str | None = None,
    ) -> StructureEntry:
        """
        Inspect `atoms` via `inspect_structure(atoms)`, assign `entry_id = f"struct_{len(self.structures)}"`
        and `label = f"#{len(self.structures)}: {name} ({atoms.get_chemical_formula()}, {len(atoms)} atoms)"`,
        store a copy of `atoms` in `self.structures[entry_id]`, and return the new `StructureEntry`.
        """

    def register_run(self, record: RunRecord) -> None:
        """Store `record` in `self.runs[record.run_id]`."""

    def get_structure(self, entry_id: str) -> StructureEntry:
        """Return `self.structures[entry_id]`, or fall back to the first structure in `self.structures` if `entry_id` is not found."""

    def structure_options(self) -> dict[str, str]:
        """Return `{entry.label: entry.entry_id for entry in self.structures.values()}` suitable for `mo.ui.dropdown(options=...)`."""

    def latest_run_for_mode(self, mode: JanusMode) -> RunRecord | None:
        """Return the most recently registered `RunRecord` in `self.runs.values()` with `r.mode == mode`, or `None`."""

def compute_run_id(
    mode: JanusMode,
    mlip_config: MLIPConfig,
    atoms: Atoms,
    mode_params: dict[str, Any],
    final_atoms: Atoms | None = None,
) -> str:
    """
    Compute a deterministic 16-hex-char SHA-256 digest (`hashlib.sha256(payload).hexdigest()[:16]`) over canonical JSON (`sort_keys=True`) of:
    - `mode`: `str`
    - `mlip_config`: `{"arch": mlip_config.arch, "model": mlip_config.model, "device": mlip_config.device, "dispersion": mlip_config.dispersion, "calc_kwargs": mlip_config.calc_kwargs}`
    - `atoms`: `{"numbers": atoms.numbers.tolist(), "positions": np.round(atoms.positions, 6).tolist(), "cell": np.round(atoms.cell.array, 6).tolist(), "pbc": atoms.pbc.tolist()}`
    - `final_atoms`: same dict as `atoms` if `final_atoms is not None` else `None`
    - `mode_params`: `mode_params`
    """

def build_preset_structures() -> dict[str, Atoms]:
    """
    Return 5 built-in ASE crystal presets so all 8 tabs (including NEB) work out-of-the-box without requiring file uploads:
    1. `"NaCl (rocksalt)"`: `ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=True)` (8 atoms, 3D PBC)
    2. `"Si (diamond)"`: `ase.build.bulk("Si", "diamond", a=5.43, cubic=True)` (8 atoms, 3D PBC)
    3. `"Cu (fcc)"`: `ase.build.bulk("Cu", "fcc", a=3.61, cubic=True)` (4 atoms, 3D PBC)
    4. `"Li (bcc vacancy hop - initial)"`: `li_init = ase.build.bulk("Li", "bcc", a=3.49, cubic=True) * (2, 2, 2); del li_init[0]` (15 atoms, vacancy at `(0.0, 0.0, 0.0)`)
    5. `"Li (bcc vacancy hop - final)"`: same 15-atom $2 \times 2 \times 2$ bcc Li supercell where the nearest-neighbor body-center Li atom at `(1.745, 1.745, 1.745)` (index 0 of `li_init`) has hopped into the vacancy site `(0.0, 0.0, 0.0)`
    """
```

---

### 4.4 `src/janus_marimo/runner.py`

Compiles validated `typer-config` YAML files for each of the 8 `janus` CLI commands and executes them statelessly via `uv run`.

```python
from pathlib import Path
from typing import Any
from janus_marimo.ledger import JanusMode, MLIPConfig, RunRecord, StructureEntry

def build_janus_yaml_config(
    mode: JanusMode,
    mlip_config: MLIPConfig,
    *,
    struct_path: Path,
    run_dir: Path,
    mode_params: dict[str, Any],
    final_struct_path: Path | None = None,
) -> dict[str, Any]:
    """
    Build a `janus <mode> --config` dictionary matching exact `janus_core.cli.<mode>` parameter names:
    - Common keys injected in all 8 modes:
      - `arch`: `mlip_config.arch`
      - `device`: `mlip_config.device`
      - `model`: `mlip_config.model` (included only if `mlip_config.model` is truthy/non-None so `janus-core` uses its architecture default otherwise)
      - `calc_kwargs`: `mlip_config.to_calc_kwargs()`
      - `file_prefix`: `str(run_dir / "job")`
      - `log`: `str(run_dir / f"{mode}-log.yml")`
      - `summary`: `str(run_dir / f"{mode}-summary.yml")`
      - `tracker`: `False` (always False per AGENTS.md rule #2)
    - Mode-specific keys and guardrails:
      1. `singlepoint`:
         - `struct`: `str(struct_path)`
         - `properties`: `list(mode_params.get("properties", ["energy", "forces", "stress"]))`
         - `out`: `str(run_dir / "job-results.extxyz")`
         - `progress_bar`: `False`
      2. `geomopt`:
         - `struct`: `str(struct_path)`
         - `optimizer`: `str(mode_params.get("optimizer", "LBFGS"))`
         - `fmax`: `float(mode_params.get("fmax", 0.01))`
         - `steps`: `int(mode_params.get("steps", 500))`
         - `opt_cell_fully`: `bool(mode_params.get("opt_cell_fully", True))`
         - `opt_cell_lengths`: `bool(mode_params.get("opt_cell_lengths", False))` (if `opt_cell_fully` is True, `opt_cell_lengths` is forced to `False`)
         - `filter_class`: `str(mode_params.get("filter_class", "FrechetCellFilter"))` when `opt_cell_fully` or `opt_cell_lengths` is True (per AGENTS.md rule #3); omitted when both cell flags are False because `janus geomopt` raises `ValueError` if `--filter` is passed without `--opt-cell-fully` or `--opt-cell-lengths`
         - `pressure`: `float(mode_params.get("pressure", 0.0))`
         - `symmetrize`: `bool(mode_params.get("symmetrize", False))`
         - `write_traj`: `True`
         - `out`: `str(run_dir / "job-opt.extxyz")`
      3. `md`:
         - `struct`: `str(struct_path)`
         - `ensemble`: `str(mode_params.get("ensemble", "nvt"))`
         - `temp`: `float(mode_params.get("temp", 300.0))`
         - `steps`: `int(mode_params.get("steps", 200))`
         - `timestep`: `float(mode_params.get("timestep", 1.0))`
         - `stats_every`: `int(mode_params.get("stats_every", 10))`
         - `traj_every`: `int(mode_params.get("traj_every", 10))`
         - `stats_file`: `str(run_dir / "job-stats.dat")`
         - `traj_file`: `str(run_dir / "job-traj.extxyz")`
         - `final_file`: `str(run_dir / "job-final.extxyz")`
         - `friction`: `float(mode_params.get("friction", 0.005))`
         - `pressure`: `float(mode_params.get("pressure", 0.0))`
         - `post_process_kwargs`: `{"rdf_compute": True, "rdf_rmax": float(mode_params.get("rdf_rmax", 6.0)), "rdf_nbins": int(mode_params.get("rdf_nbins", 60))}` if `mode_params.get("rdf_compute", False)` else `{}`
         - `progress_bar`: `False`
      4. `phonons`:
         - `struct`: `str(struct_path)`
         - `supercell`: `str(mode_params.get("supercell", "2 2 2"))`
         - `displacement`: `float(mode_params.get("displacement", 0.01))`
         - `mesh`: `list(mode_params.get("mesh", [10, 10, 10]))`
         - `symmetrize`: `bool(mode_params.get("symmetrize", False))`
         - `minimize`: `bool(mode_params.get("minimize", False))`
         - `fmax`: `float(mode_params.get("fmax", 0.01))`
         - `bands`: `bool(mode_params.get("bands", True))`
         - `dos`: `bool(mode_params.get("dos", True))`
         - `pdos`: `bool(mode_params.get("pdos", False))`
         - `thermal`: `bool(mode_params.get("thermal", True))`
         - `temp_min`: `float(mode_params.get("temp_min", 0.0))`
         - `temp_max`: `float(mode_params.get("temp_max", 1000.0))`
         - `temp_step`: `float(mode_params.get("temp_step", 50.0))`
         - `hdf5`: `False` (ensures `job-auto_bands.yml` is written in YAML format alongside `job-dos.dat` and `job-thermal.yml`)
         - `write_full`: `True`
         - `plot_to_file`: `False`
         - `progress_bar`: `False`
      5. `eos`:
         - `struct`: `str(struct_path)`
         - `min_volume`: `float(mode_params.get("min_volume", 0.95))`
         - `max_volume`: `float(mode_params.get("max_volume", 1.05))`
         - `n_volumes`: `int(mode_params.get("n_volumes", 7))`
         - `eos_type`: `str(mode_params.get("eos_type", "birchmurnaghan"))`
         - `minimize`: `bool(mode_params.get("minimize", False))`
         - `minimize_all`: `bool(mode_params.get("minimize_all", False))`
         - `fmax`: `float(mode_params.get("fmax", 0.05))`
         - `write_structures`: `True`
         - `plot_to_file`: `False`
      6. `elasticity`:
         - `struct`: `str(struct_path)`
         - `normal_magnitude`: `float(mode_params.get("normal_magnitude", 0.01))`
         - `shear_magnitude`: `float(mode_params.get("shear_magnitude", 0.06))`
         - `n_strains`: `int(mode_params.get("n_strains", 4))`
         - `minimize`: `bool(mode_params.get("minimize", False))`
         - `minimize_all`: `bool(mode_params.get("minimize_all", False))`
         - `fmax`: `float(mode_params.get("fmax", 0.05))`
         - `write_voigt`: `True`
         - `write_structures`: `True`
      7. `neb`:
         - `init_struct`: `str(struct_path)`
         - `final_struct`: `str(final_struct_path)`
         - `n_images`: `int(mode_params.get("n_images", 5))`
         - `interpolator`: `str(mode_params.get("interpolator", "ase"))`
         - `neb_class`: `str(mode_params.get("neb_class", "NEB"))`
         - `neb_kwargs`: `{"climb": bool(mode_params.get("climb", True)), "k": float(mode_params.get("k", 0.1))}`
         - `optimizer`: `str(mode_params.get("optimizer", "NEBOptimizer"))`
         - `fmax`: `float(mode_params.get("fmax", 0.05))`
         - `steps`: `int(mode_params.get("steps", 100))`
         - `minimize`: `bool(mode_params.get("minimize", False))`
         - `write_band`: `True`
         - `plot_band`: `False`
      8. `descriptors`:
         - `struct`: `str(struct_path)`
         - `invariants_only`: `bool(mode_params.get("invariants_only", True))`
         - `calc_per_element`: `bool(mode_params.get("calc_per_element", True))`
         - `calc_per_atom`: `bool(mode_params.get("calc_per_atom", True))`
         - `out`: `str(run_dir / "job-descriptors.extxyz")`
         - `progress_bar`: `False`
    """

def build_cli_command(
    python_executable: Path,
    mode: JanusMode,
    config_yaml_path: Path,
) -> list[str]:
    """
    Return the exact command list:
    `["uv", "run", "--no-project", "--python", str(python_executable), "-m", "janus_core.cli.janus", mode, "--config", str(config_yaml_path), "--no-tracker"]`
    """

def execute_janus_mode(
    mode: JanusMode,
    mlip_config: MLIPConfig,
    struct_entry: StructureEntry,
    mode_params: dict[str, Any],
    *,
    python_executable: Path,
    runs_root: Path = Path("runs"),
    final_struct_entry: StructureEntry | None = None,
    force_rerun: bool = False,
    timeout_s: int = 1800,
) -> RunRecord:
    """
    1. Pre-flight structure validation:
       - If `mode == "neb"`, call `ok, errors = validate_neb_endpoints(struct_entry.atoms, final_struct_entry.atoms if final_struct_entry else None)`. If `not ok`, return `RunRecord(..., status="failed", stdout_stderr="\\n".join(errors))` without spawning a subprocess.
       - Otherwise call `report = inspect_structure(struct_entry.atoms, require_pbc=(mode in {"phonons", "eos", "elasticity"}))`. If `not report.is_valid`, return `RunRecord(..., status="failed", stdout_stderr="\\n".join(report.errors))` without spawning a subprocess.
       - Also verify that if `mlip_config.arch in ARCH_REQUIRES_EXPLICIT_MODEL` (`"mace"`, `"nequip"`, `"dpa3"`) and `not mlip_config.model`, return `RunRecord(..., status="failed", stdout_stderr=f"Architecture '{mlip_config.arch}' requires an explicit model path or checkpoint.")`.
    2. Compute `run_id = compute_run_id(mode, mlip_config, struct_entry.atoms, mode_params, final_struct_entry.atoms if final_struct_entry else None)` and `run_dir = runs_root / mode / run_id`.
    3. If `not force_rerun` and `(run_dir / f"{mode}-summary.yml").exists()`:
       read `summary_data = yaml.safe_load((run_dir / f"{mode}-summary.yml").read_text())` and return `RunRecord(..., output_files=summary_data.get("output_files", {}), status="cached", duration_s=0.0, stdout_stderr="Loaded from content-addressed cache.")`.
    4. Create `run_dir.mkdir(parents=True, exist_ok=True)`. Write `struct_entry.atoms` to `run_dir / "input.extxyz"` via `ase.io.write`. If `mode == "neb"` and `final_struct_entry is not None`, write `final_struct_entry.atoms` to `run_dir / "final_input.extxyz"`.
    5. Build config dict via `build_janus_yaml_config(...)` and dump to `run_dir / "config.yml"` using `yaml.safe_dump(..., sort_keys=False)`.
    6. Execute `build_cli_command(python_executable, mode, run_dir / "config.yml")` via `subprocess.run(..., capture_output=True, text=True, timeout=timeout_s)` and record wall-clock `duration_s`.
    7. If `returncode == 0` and `(run_dir / f"{mode}-summary.yml").exists()`, parse `output_files` from `f"{mode}-summary.yml"` and return `RunRecord(..., status="succeeded")`; else return `RunRecord(..., status="failed")`.
    """
```

---

### 4.5 `src/janus_marimo/parsers.py`

Parses the disk artifacts in `run_dir` produced by each `janus <mode>` run into strongly typed dataclasses containing `list[ase.Atoms]` and `pandas.DataFrame` tables.

#### Helper for Extracting `janus-core` Namespaced Energy / Forces / Stress
In `janus-core` (`janus_core.helpers.struct_io.results_to_info`), properties written to `.extxyz` files are stored under `f"{arch}_energy"`, `f"{arch}_forces"`, `f"{arch}_stress"`, or `f"{arch}_d3_energy"` / `f"{arch}_d3_forces"` when D3 dispersion is active, or fallback `"energy"` / `"forces"` / `atoms.calc.results`.
- `_extract_energy(atoms: Atoms, arch: str) -> float`: checks `atoms.info` for `f"{arch}_d3_energy"`, `f"{arch}_energy"`, `"energy"`, or any key ending with `"_energy"`, then `atoms.calc.results.get("energy")` if `atoms.calc` is attached, defaulting to `0.0`.
- `_extract_forces(atoms: Atoms, arch: str) -> np.ndarray`: checks `atoms.arrays` for `f"{arch}_d3_forces"`, `f"{arch}_forces"`, `"forces"`, or any key ending with `"_forces"`, then `atoms.calc.results.get("forces")`, defaulting to `np.zeros((len(atoms), 3))`.
- `_extract_stress(atoms: Atoms, arch: str) -> list[float] | None`: checks `atoms.info` for `f"{arch}_d3_stress"`, `f"{arch}_stress"`, `"stress"`, or any key ending with `"_stress"`, converting a 6-element Voigt vector or $3 \times 3$ matrix (`[s[0,0], s[1,1], s[2,2], s[1,2], s[0,2], s[0,1]]`) to a 6-float list, or `None` if absent.

```python
from dataclasses import dataclass
from pathlib import Path
from ase import Atoms
import pandas as pd

@dataclass(frozen=True)
class ParsedSinglePoint:
    structures: list[Atoms]
    energy_ev: float
    energy_per_atom_ev: float
    max_force_ev_ang: float
    stress_voigt_ev_ang3: list[float] | None
    atom_df: pd.DataFrame          # columns: ["atom_index", "symbol", "x", "y", "z", "fx", "fy", "fz", "force_norm"]

@dataclass(frozen=True)
class ParsedGeomOpt:
    initial_atoms: Atoms
    optimized_atoms: Atoms
    trajectory: list[Atoms]
    traj_df: pd.DataFrame          # columns: ["step", "energy_ev", "delta_energy_mev_atom", "max_force_ev_ang", "volume_ang3"]
    converged: bool

@dataclass(frozen=True)
class ParsedMD:
    trajectory: list[Atoms]
    final_atoms: Atoms
    stats_df: pd.DataFrame         # columns: ["step", "time_fs", "temp_k", "epot_ev", "ekin_ev", "etot_ev", "pressure_gpa", "volume_ang3"]
    rdf_df: pd.DataFrame | None    # columns: ["r_ang", "g_r", "pair"] if RDF files present else None

@dataclass(frozen=True)
class ParsedPhonons:
    bands_df: pd.DataFrame | None  # columns: ["q_index", "distance", "band_index", "frequency_thz", "q_label"]
    dos_df: pd.DataFrame | None    # columns: ["frequency_thz", "dos"]
    thermal_df: pd.DataFrame | None # columns: ["temperature_k", "free_energy_kj_mol", "entropy_j_k_mol", "heat_capacity_j_k_mol"]
    has_imaginary_modes: bool
    min_frequency_thz: float | None
    band_structures: list[Atoms]

@dataclass(frozen=True)
class ParsedEOS:
    raw_df: pd.DataFrame           # columns: ["lattice_scalar", "volume_ang3", "energy_ev"]
    fit_curve_df: pd.DataFrame     # 100-point curve: ["volume_ang3", "energy_ev"]
    bulk_modulus_gpa: float
    v0_ang3: float
    e0_ev: float
    structures: list[Atoms]

@dataclass(frozen=True)
class ParsedElasticity:
    c_ij_matrix: list[list[float]] # 6x6 Voigt matrix in GPa
    c_ij_df: pd.DataFrame          # columns: ["i_label", "j_label", "c_ij_gpa"] (36 rows, C_11..C_66)
    moduli_df: pd.DataFrame        # columns: ["property", "method", "value_gpa"] (Bulk/Shear Reuss, Voigt, VRH + Young's VRH)
    poisson_ratio: float
    universal_anisotropy: float
    is_mechanically_stable: bool
    structures: list[Atoms]

@dataclass(frozen=True)
class ParsedNEB:
    band_images: list[Atoms]
    barrier_ev: float
    delta_e_ev: float
    max_force_ev_ang: float
    image_df: pd.DataFrame         # columns: ["image_index", "rxn_coord_ang", "energy_ev", "rel_energy_ev", "max_force_ev_ang"]
    saddle_image_index: int

@dataclass(frozen=True)
class ParsedDescriptors:
    structures: list[Atoms]
    mean_descriptor: float | None
    element_descriptors: dict[str, float]
    atom_df: pd.DataFrame          # columns: ["structure_index", "atom_index", "symbol", "x", "y", "z", "descriptor_value", "force_norm"]
```

#### Exact File Schemas & Parsing Rules for All 8 `parse_*_run` Functions
1. `parse_singlepoint_run(run_dir: Path, arch: str) -> ParsedSinglePoint`:
   - Reads `run_dir / "job-results.extxyz"` via `ase.io.read(..., index=":")`.
   - Extracts `energy_ev = _extract_energy(atoms, arch)`, `forces = _extract_forces(atoms, arch)`, `stress_voigt_ev_ang3 = _extract_stress(atoms, arch)` from the first frame `atoms = structures[0]`.
   - Builds `atom_df` with `atom_index` (`0..N-1`), `symbol`, `x, y, z` from `atoms.positions`, `fx, fy, fz` from `forces`, and `force_norm = np.linalg.norm(forces, axis=1)`.
2. `parse_geomopt_run(run_dir: Path, arch: str, fmax_target: float = 0.01) -> ParsedGeomOpt`:
   - Reads `optimized_atoms = ase.io.read(run_dir / "job-opt.extxyz", index=-1)`.
   - Reads `trajectory = ase.io.read(run_dir / "job-traj.extxyz", index=":")` if `(run_dir / "job-traj.extxyz").exists()` else `[optimized_atoms]`.
   - Reads `initial_atoms = ase.io.read(run_dir / "input.extxyz", index=-1)` if `(run_dir / "input.extxyz").exists()` else `trajectory[0]`.
   - For each frame `i, frame` in `enumerate(trajectory)`, computes `e = _extract_energy(frame, arch)`, `fmax = float(np.linalg.norm(_extract_forces(frame, arch), axis=1).max())` (or `float(frame.info.get("max_force", 0.0))` if forces array is absent), `vol = float(frame.get_volume()) if any(frame.pbc) else 0.0`, and `delta_energy_mev_atom = (e - e0) * 1000.0 / len(frame)` where `e0` is step 0 energy.
   - Determines `converged: bool`: checks `optimized_atoms.info.get("converged")` if present as a boolean; otherwise checks `float(traj_df["max_force_ev_ang"].iloc[-1]) <= fmax_target`.
3. `parse_md_run(run_dir: Path, arch: str) -> ParsedMD`:
   - Reads `trajectory = ase.io.read(run_dir / "job-traj.extxyz", index=":")` and `final_atoms = ase.io.read(run_dir / "job-final.extxyz", index=-1)` (falling back to `trajectory[-1]`).
   - Parses `run_dir / "job-stats.dat"`: `janus-core` writes a header line starting with `# Step | Real_Time [s] | Time [fs] | Epot/N [eV] | EKin/N [eV] | T [K] | ETot/N [eV] | Density [g/cm^3] | Volume [Å^3] | P [GPa] | ...` followed by whitespace-separated numeric rows. Strips `#`, splits header names on `|`, strips unit suffixes `[...]`, and maps columns to `stats_df` (`step = Step`, `time_fs = Time`, `temp_k = T`, `epot_ev = Epot/N * n_atoms`, `ekin_ev = EKin/N * n_atoms`, `etot_ev = ETot/N * n_atoms`, `pressure_gpa = P`, `volume_ang3 = Volume`).
   - Scans `run_dir.glob("*rdf.dat")`: if any exist, reads each 2-column (`r_ang`, `g_r`) whitespace-delimited file (skipping `#` comments), attaches `pair` inferred from the filename stem (e.g., `"Na_Cl"` or `"total"`), and concatenates into `rdf_df`.
4. `parse_phonons_run(run_dir: Path, arch: str) -> ParsedPhonons`:
   - **Bands (`job-auto_bands.yml` or `job-bands.yml`)**: Loads YAML via `yaml.safe_load`. Iterates over `q_index, qpt in enumerate(data["phonon"])`: extracts `distance = float(qpt["distance"])`, `q_label = str(qpt.get("label") or "")`, and for each `band_idx, band in enumerate(qpt["band"])` records `frequency_thz = float(band["frequency"])`.
   - **DOS (`job-dos.dat`)**: Parses whitespace-separated 2-column numeric rows (ignoring `#` comment lines) into `dos_df` with columns `["frequency_thz", "dos"]`.
   - **Thermal (`job-thermal.yml`)**: Loads YAML via `yaml.safe_load`, iterates over `data["thermal_properties"]`, and maps `temperature -> temperature_k`, `free_energy -> free_energy_kj_mol`, `entropy -> entropy_j_k_mol`, `heat_capacity -> heat_capacity_j_k_mol`.
   - Computes `min_frequency_thz` across `bands_df["frequency_thz"]` (or `dos_df["frequency_thz"]` if `bands_df` is None) and sets `has_imaginary_modes = bool(min_frequency_thz is not None and min_frequency_thz < -0.05)`.
   - Reads `band_structures`: reads `run_dir / "input.extxyz"` (and `2x2x2` supercell copy) for 3D Chemiscope visualization.
5. `parse_eos_run(run_dir: Path, arch: str, eos_type: str = "birchmurnaghan") -> ParsedEOS`:
   - Parses `run_dir / "job-eos-raw.dat"` (header: `#Lattice Scalar | Energy [eV] | Volume [Å^3]`, whitespace-separated rows of `lattice_scalar, energy_ev, volume_ang3`) into `raw_df`.
   - Parses `run_dir / "job-eos-fit.dat"` (header: `#Bulk modulus [GPa] | Energy [eV] | Volume [Å^3]`, single row of `bulk_modulus_gpa, e0_ev, v0_ang3`).
   - Computes `fit_curve_df` (100 points from `raw_df["volume_ang3"].min()` to `raw_df["volume_ang3"].max()`): fits `ase.eos.EquationOfState(raw_df["volume_ang3"].to_numpy(), raw_df["energy_ev"].to_numpy(), eos=eos_type)` and evaluates its fitted polynomial/curve function across `np.linspace(v_min, v_max, 100)` (falling back to cubic/quadratic interpolation over `raw_df` if `EquationOfState.fit()` fails).
   - Reads `structures` from `run_dir / "job-generated.extxyz"` via `ase.io.read(..., index=":")` (falling back to `input.extxyz`).
6. `parse_elasticity_run(run_dir: Path, arch: str) -> ParsedElasticity`:
   - Parses `run_dir / "job-elastic_tensor.dat"`: per `janus_core/calculations/elasticity.py:297-339`, this file has a comment line followed by a single space-separated line of **45 floats**:
     - indices `0..5`: `k_reuss`, `k_voigt`, `k_vrh`, `g_reuss`, `g_voigt`, `g_vrh` (all in GPa)
     - index `6`: `youngs_modulus_gpa`
     - index `7`: `universal_anisotropy`
     - index `8`: `poisson_ratio`
     - indices `9..44`: 36 floats representing the row-major $6 \times 6$ Voigt stiffness matrix $C_{ij}$ in GPa.
   - Reshapes `values[9:45]` into a $6 \times 6$ array `c_ij`, builds `c_ij_df` with labels `i_label, j_label in ["C1", "C2", "C3", "C4", "C5", "C6"]`, builds `moduli_df` with rows for Bulk Modulus (`Reuss`, `Voigt`, `VRH`), Shear Modulus (`Reuss`, `Voigt`, `VRH`), and Young's Modulus (`VRH`), and sets `is_mechanically_stable = bool(np.all(np.linalg.eigvalsh(c_ij) > 0.0))`.
   - Reads `structures` from `run_dir / "job-elasticity-generated.extxyz"` if present else `[ase.io.read(run_dir / "input.extxyz")]`.
7. `parse_neb_run(run_dir: Path, arch: str) -> ParsedNEB`:
   - Parses `run_dir / "job-neb-results.dat"` (header: `#Barrier [eV] | delta E [eV] | Max force [eV/Å]`, single row of 3 floats: `barrier_ev`, `delta_e_ev`, `max_force_ev_ang`).
   - Reads `band_images = ase.io.read(run_dir / "job-neb-band.extxyz", index=":")`.
   - Computes cumulative reaction coordinate `rxn_coord_ang` along `band_images` (`0.0` for image 0, accumulating `np.linalg.norm( band_images[i].get_positions() - band_images[i-1].get_positions() )`), extracts per-image `energy_ev = _extract_energy(img, arch)`, `rel_energy_ev = energy_ev - energy_0`, and `max_force_ev_ang`, and sets `saddle_image_index = int(image_df["rel_energy_ev"].idxmax())`.
8. `parse_descriptors_run(run_dir: Path, arch: str) -> ParsedDescriptors`:
   - Reads `structures = ase.io.read(run_dir / "job-descriptors.extxyz", index=":")`.
   - Extracts `mean_descriptor = structures[0].info.get(f"{arch}_descriptor")` (or any `*_descriptor` key in `info`).
   - Extracts `element_descriptors = {el: float(val) for key, val in structures[0].info.items() if key.endswith("_descriptor") and key != f"{arch}_descriptor" ...}`.
   - Extracts per-atom descriptor array from `atoms.arrays.get(f"{arch}_descriptors")` (or any `*_descriptors` key in `atoms.arrays`, defaulting to `np.full(len(atoms), mean_descriptor or 0.0)`) and builds `atom_df` with columns `["structure_index", "atom_index", "symbol", "x", "y", "z", "descriptor_value", "force_norm"]`.

---

### 4.6 `src/janus_marimo/viz.py`

Constructs interactive **Altair** charts (`alt.Chart`) and **Chemiscope** widgets (`chemiscope.show`) with sanitized ASE properties and force arrow shapes.

```python
from collections.abc import Sequence
from typing import Any, Literal
import altair as alt
from ase import Atoms
import pandas as pd

def sanitize_atoms_for_chemiscope(
    structures: Sequence[Atoms],
    *,
    max_frames: int = 200,
) -> tuple[list[Atoms], dict[str, dict[str, Any]], str | None]:
    """
    Prepare a list of `ase.Atoms` for `chemiscope.show` without triggering Chemiscope property warnings or browser memory bloat:
    1. If `len(structures) > max_frames`, uniformly stride-slice `structures` down to `<= max_frames` (always retaining the first and last frame).
    2. Deep-copy each `Atoms` and clear `atoms.calc = None` and `atoms.info = {}` on the copy after extracting scalar energy so non-scalar / rank>=2 arrays in `atoms.info` (such as 3x3 stress tensors, strains, or Hessian matrices) never trigger Chemiscope's `_ase_get_structure_properties` warnings.
    3. Detect any 3-vector force array in the original `atoms.arrays` (`f"{arch}_forces"`, `"forces"`, or any key ending in `"_forces"`) or `atoms.calc.results` and copy it to `cleaned.arrays["forces"]` (float64 `(N, 3)`) across all frames; also compute `cleaned.arrays["force_norm"] = np.linalg.norm(forces, axis=1)`. Keep only `"numbers"`, `"positions"`, `"forces"`, `"force_norm"`, and any 1D scalar float array ending in `"descriptors"` in `cleaned.arrays`.
    4. Build an explicit `properties` dict containing at least two properties at both structure and atom levels:
       - `"step": {"target": "structure", "values": list(range(len(cleaned_structures)))}`
       - `"energy_ev": {"target": "structure", "values": [float(e) for e in energies]}`
       - `"atom_index": {"target": "atom", "values": np.concatenate([np.arange(len(s)) for s in cleaned_structures]).tolist()}`
       - `"force_norm": {"target": "atom", "values": np.concatenate([s.arrays["force_norm"] for s in cleaned_structures]).tolist()}`
    5. Return `(cleaned_structures, properties, "forces" if has_forces else None)`.
    """

def build_chemiscope_widget(
    structures: Sequence[Atoms],
    *,
    mode: Literal["default", "structure", "map"] = "structure",
    show_force_arrows: bool = True,
    force_scale: float = 0.5,
    extra_properties: dict[str, dict[str, Any]] | None = None,
    environments_cutoff: float | None = None,
    max_frames: int = 200,
) -> Any:
    """
    Build and return a `chemiscope` anywidget (`ChemiscopeWidget` or `StructureWidget`):
    - Calls `sanitize_atoms_for_chemiscope(structures, max_frames=max_frames)`.
    - Merges `extra_properties` into `properties` if provided.
    - If `show_force_arrows` and `detected_force_key == "forces"`, builds `shapes = {"forces": chemiscope.ase_vectors_to_arrows(cleaned_structures, key="forces", scale=force_scale, radius=0.08)}` and sets `settings = {"structure": [{"atoms": True, "unitCell": True, "shape": "forces"}]}`.
    - If `environments_cutoff` is set (e.g. `4.0` Å when `mode == "default"` in the `descriptors` tab), passes `environments = chemiscope.all_atomic_environments(cleaned_structures, cutoff=environments_cutoff)`.
    - Calls `chemiscope.show(structures=cleaned_structures, properties=properties, shapes=shapes, environments=environments, settings=settings, mode=mode)`.
    """

# Altair 2D Chart Builders (all return `alt.Chart | alt.LayerChart | alt.HConcatChart | alt.VConcatChart`)
def chart_singlepoint_forces(atom_df: pd.DataFrame) -> alt.Chart: ...
def chart_geomopt_convergence(traj_df: pd.DataFrame) -> alt.HConcatChart: ...
def chart_md_thermodynamics(stats_df: pd.DataFrame, rdf_df: pd.DataFrame | None = None) -> alt.VConcatChart: ...
def chart_phonons_spectrum(
    bands_df: pd.DataFrame | None,
    dos_df: pd.DataFrame | None,
    thermal_df: pd.DataFrame | None,
) -> alt.VConcatChart: ...
def chart_eos_curve(raw_df: pd.DataFrame, fit_curve_df: pd.DataFrame, v0: float, e0: float, b0_gpa: float) -> alt.LayerChart: ...
def chart_elasticity_tensor(c_ij_df: pd.DataFrame, moduli_df: pd.DataFrame) -> alt.HConcatChart: ...
def chart_neb_barrier(image_df: pd.DataFrame) -> alt.LayerChart: ...
def chart_descriptors_distribution(atom_df: pd.DataFrame) -> alt.Chart: ...
```

---

## 5. UX Flow, Reactive State Design & Visual Layout (`janus_workbench.py`)

### 5.1 Visual Structure (ASCII Sketch)

```
+--------------------------------------------------------------------------------------------------+
|  JANUS-CORE MARIMO WORKBENCH (8-Mode Atomistic Simulation & Visual Analytics)                    |
+------------------------------------------------+-------------------------------------------------+
|  1. SHARED MLIP CONFIGURATION (Single Active)  |  2. STRUCTURE RELAY LEDGER & INSPECTOR          |
|  [Arch: mace_mp v] [Model: small            ]  |  [Active Structure: #0: NaCl (seed)        v]   |
|  [Device: cpu   v] [x] D3 Dispersion           |  Or Upload (.cif/.xyz/.poscar): [Upload File]   |
|  [Calc Kwargs JSON: {}                      ]  |  Inspector: PASS | PBC: (T,T,T) | d_min: 2.82 A |
|  Env Status: [READY: base_janus (/opt/...)]    |  Ledger Entries: #0 (NaCl)..#4 (Li NEB final)   |
|  > Accordion: Provision Conflicting MLIP Venvs |                                                 |
|    [ ] mace  [x] sevennet  [ ] orb  [ ] chgnet |                                                 |
|    [Provision Selected Venvs via uv]           |                                                 |
+------------------------------------------------+-------------------------------------------------+
|  3. CALCULATION TABS (`mo.ui.tabs(..., lazy=True)`)                                              |
|  [1. SinglePoint] [2. GeomOpt] [3. MD] [4. Phonons] [5. EOS] [6. Elasticity] [7. NEB] [8. Desc]  |
+--------------------------------------------------------------------------------------------------+
|  Active Tab Body (e.g., Tab 2: GeomOpt)                                                          |
|  +--------------------------------------------------------------------------------------------+  |
|  | Structure Source: [Use Active Header Structure (#0) v]  Optimizer: [LBFGS v]               |  |
|  | Cell Relaxation: [x] Full Cell (`FrechetCellFilter`)    fmax: [0.01]  Steps: [500]         |  |
|  | Scalar Pressure (GPa): [0.0]   [ ] Symmetrize           [ Run Geometry Optimization ]      |  |
|  +--------------------------------------------------------------------------------------------+  |
|  | Status Banner: Succeeded in 1.4s (Run ID: a9f3c1... | tracker=False | FrechetCellFilter)   |  |
|  | [ Promote Relaxed Structure to Ledger ]        [ View YAML Config & Reproducible CLI ]     |  |
|  +---------------------------------------------+----------------------------------------------+  |
|  | ALTAIR 2D INTERACTIVE DIAGNOSTIC            | CHEMISCOPE 3D VIEWER (`mo.lazy`)             |  |
|  | (`mo.ui.altair_chart`)                      | (`chemiscope.show` + force vector arrows)    |  |
|  | - Energy & Max Force vs. Optimization Step  | - Interactive 3D unit cell & trajectory      |  |
|  | - Brush/click steps to filter 3D frames     | - Toggle atomic force vectors (`forces`)     |  |
|  +---------------------------------------------+----------------------------------------------+  |
+--------------------------------------------------------------------------------------------------+
```

### 5.2 Cycle-Free Marimo Reactive DAG Architecture
To guarantee **zero reactive loops** and ensure that running a calculation or editing the shared MLIP dropdown never wipes out other tabs:
1. **`get_ledger, set_ledger = mo.state(WorkspaceLedger.from_presets())`**:
   - Calculation execution cells (`sp_exec_cell`, `opt_exec_cell`, `md_exec_cell`, `ph_exec_cell`, `eos_exec_cell`, `el_exec_cell`, `neb_exec_cell`, `desc_exec_cell`) **never call `set_ledger(...)` unconditionally during cell evaluation**.
   - Instead, when `geomopt`, `md`, or `neb` finishes, it renders an explicit `mo.ui.button(label="Promote Structure to Ledger", on_change=lambda _: set_ledger(...))`. Clicking that button explicitly appends the new `StructureEntry` (`#5`, `#6`, ...) to the ledger on user command, avoiding any automatic reactive cycle between `get_ledger()` and calculation cells.
2. **Per-Tab Run Caching & `mo.stop` Isolation**:
   - Each of the 8 tabs has its own UI parameter cell (`<mode>_controls_cell`) and execution/visualization cell (`<mode>_exec_cell`).
   - Each `<mode>_exec_cell` starts with `mo.stop(not <mode>_run_btn.value, mo.callout(mo.md("Configure parameters and click **Run** to execute."), kind="info"))`.
   - Because `execute_janus_mode` caches completed runs by their deterministic 16-char SHA-256 `run_id` under `runs/<mode>/<run_id>/`, re-clicking Run with the same structure and parameters returns the cached `RunRecord` in `<5 ms`.

---

## 6. Simulation & Reactive Execution Guards

1. **Headless Emission Tracking Disabled (`AGENTS.md` Rule #3)**:
   - `build_janus_yaml_config` unconditionally sets `tracker: False` in every generated YAML file, and `build_cli_command` passes `--no-tracker` on the CLI.
2. **Cell Relaxation Filter (`AGENTS.md` Rule #3)**:
   - In `geomopt`, whenever variable-cell relaxation (`opt_cell_fully` or `opt_cell_lengths`) is enabled, `build_janus_yaml_config` sets `filter_class: "FrechetCellFilter"`.
3. **Structure, PBC & NEB Endpoint Verification (`AGENTS.md` Rule #3)**:
   - `execute_janus_mode` calls `inspect_structure` (or `validate_neb_endpoints` for `neb`) before launching any subprocess. Bulk crystal modes (`phonons`, `eos`, `elasticity`) pass `require_pbc=True` so non-periodic structures are caught immediately before Phonopy or strain generators fail.
4. **Marimo Reactive DAG Integrity (`AGENTS.md` Rule #4)**:
   - **Zero Redefined Variables**: Every cell in `janus_workbench.py` uses unique, mode-prefixed exported variable names (`sp_*`, `opt_*`, `md_*`, `ph_*`, `eos_*`, `el_*`, `neb_*`, `desc_*`).
   - **Execution Guards**: All 8 calculation execution cells and the environment provisioning cell are gated by `mo.ui.run_button` and `mo.stop(not <btn>.value, ...)`.
   - **WebGL Context Protection**: `mo.ui.tabs(..., lazy=True)` and `mo.lazy(...)` wrap each tab's Chemiscope widget so at most 1 Chemiscope WebGL context is active in the DOM at a time.

---

## 7. Failure Modes

| Failure Scenario | Detection Layer | User-Visible Behavior & Recovery |
|---|---|---|
| Overlapping atoms ($d_{\min} < 0.5$ Å) or non-finite coordinates | `inspect_structure` pre-flight check | Calculation is blocked before spawning Python/CUDA; tab renders an `mo.callout(..., kind="danger")` listing the offending distance/coordinates. |
| Non-periodic structure (`pbc != (True, True, True)`) passed to `phonons`, `eos`, or `elasticity` | `inspect_structure(..., require_pbc=True)` | Blocked pre-flight with a clear error explaining that volumetric/strain/lattice-dynamics modes require 3D periodic boundary conditions. |
| Invalid NEB endpoints (missing `final_struct_entry`, mismatched atom counts/species, or identical initial & final coordinates) | `validate_neb_endpoints` pre-flight check | Blocked pre-flight with a clear `mo.callout(..., kind="danger")` detailing the endpoint mismatch. |
| Architecture requiring explicit model path (`mace`, `nequip`, `dpa3`) run with empty `model` | `execute_janus_mode` pre-flight check | Blocked pre-flight with an error prompting the user to supply a local checkpoint path or URL in the Shared MLIP Configuration. |
| Selected MLIP architecture not installed in base `janus` or `.venvs/janus-<group>` | `resolve_env_status(arch)` | Header badge turns amber (`Missing venv: janus-<group>`); clicking Run shows instructions to provision `.venvs/janus-<group>` in the Environment Manager accordion. |
| `janus` CLI subprocess exits non-zero (e.g., CUDA OOM, unsupported `hessian` or `descriptors` method on a specific MLIP) | `execute_janus_mode` returncode check | Host Marimo kernel stays alive; tab renders an `mo.callout(..., kind="danger")` with the exact CLI command, exit code, and collapsible `stdout/stderr` log. |
| Optimizer (`geomopt` or `neb`) reaches `steps` without hitting `fmax` | `parse_geomopt_run` / `parse_neb_run` | Warning banner highlights non-convergence (`max_force > fmax`), while still rendering the trajectory and Altair convergence curve. |
| Imaginary phonon modes ($\nu < -0.05$ THz) | `parse_phonons_run` (`has_imaginary_modes`) | Amber warning callout alerts the user to dynamical instability and highlights negative frequencies below the 0 THz dashed line in the Altair dispersion plot. |
| Mechanically unstable crystal ($C_{ij}$ has $\le 0$ eigenvalues) | `parse_elasticity_run` (`is_mechanically_stable`) | Amber warning callout flags Born mechanical instability alongside the $6 \times 6$ $C_{ij}$ heatmap. |

---

## 8. Verification Criteria

1. **Automated Static & Linter Checks**:
   - `uv run --no-project --python /opt/micromamba/envs/janus/bin/python ruff check .` passes with 0 errors.
   - `uv run --no-project --python /opt/micromamba/envs/janus/bin/python marimo check janus_workbench.py` passes with 0 DAG errors or warnings.
2. **Automated Unit Tests (`pytest`)**:
   - `uv run --no-project --python /opt/micromamba/envs/janus/bin/python pytest` passes all unit tests across:
     - `tests/test_inspector.py`: validates periodic bulk crystal pass, overlapping atom rejection, non-finite coordinate rejection, non-PBC rejection when `require_pbc=True`, `validate_neb_endpoints` checks, and CLI exit codes.
     - `tests/test_envs.py`: validates `ARCH_TO_ENV_GROUP` and `DEFAULT_MODEL_BY_ARCH` coverage of all 14 `janus-core` `Architectures`, `resolve_env_status` detection, and `build_provision_commands` `uv venv` / `uv pip install` command generation.
     - `tests/test_ledger.py`: validates deterministic 16-char `compute_run_id` hashing, `build_preset_structures` (all 5 presets including the 15-atom Li bcc vacancy hop initial/final pair), and `WorkspaceLedger` structure/run registration.
     - `tests/test_runner.py`: validates `build_janus_yaml_config` across all 8 modes (`tracker: False` in all 8, `filter_class: "FrechetCellFilter"` when `opt_cell_fully=True` and omitted when fixed-cell, `hdf5: False` in `phonons`, `write_band: True` in `neb`), and tests `execute_janus_mode` pre-flight validation, caching, and subprocess execution.
     - `tests/test_parsers.py`: validates all 8 parsers (`parse_singlepoint_run`, `parse_geomopt_run`, `parse_md_run`, `parse_phonons_run`, `parse_eos_run`, `parse_elasticity_run`, `parse_neb_run`, `parse_descriptors_run`) against synthetic `janus-core` artifact fixtures.
     - `tests/test_viz.py`: validates `sanitize_atoms_for_chemiscope` (stripping 3x3 matrices from `info`, striding $>200$ frames, computing `force_norm`), `build_chemiscope_widget` (in both `"structure"` and `"default"` modes with force arrows), and all 8 Altair chart builders returning valid Vega-Lite specifications (`chart.to_dict()`).

---

## 9. Out-of-Scope Follow-Ups

- **MLIP Training & Active Learning Tabs (`janus preprocess`, `janus train`)**: Can be added as a separate notebook or phase 2 tab group once dataset curating workflows are scoped.
- **Persistent UNIX-Socket Worker Pool (`LRU-1` GPU Lease)**: Can be slotted into `runner.py` behind the same `execute_janus_mode` interface if users want sub-second interactive turnaround without cold-starting PyTorch.
- **Multi-MLIP Overlay Mode**: Allowing users to select multiple cached `RunRecord`s from the ledger to overlay their EOS/NEB/phonon curves on a single Altair chart.
