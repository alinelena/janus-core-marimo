# Design Specification: Auto-Discovered & Custom Environment Selector + Auto-Active Upload & Chemiscope 3D Pre-Flight Preview

- **Date**: 2026-09-26
- **Status**: Revised (Addressing Round 1 Critic Gaps)
- **Brainstorm Reference**: [docs/eg-brainstorms/custom-env-and-chemiscope-upload-preview-2026-09-26.md](../eg-brainstorms/custom-env-and-chemiscope-upload-preview-2026-09-26.md)
- **Parent Spec Reference**: [docs/specs/janus-core-marimo-workbench-2026-09-25.md](janus-core-marimo-workbench-2026-09-25.md)

---

## 1. Why

1. **Custom & Existing Environment Execution**: Researchers using `janus-core` frequently maintain existing `micromamba`, `conda`, or `uv` environments on disk (e.g., `/opt/micromamba/envs/mace-gpu`, `~/.conda/envs/sevennet`, or custom editable `janus-core` checkouts). Currently, `resolve_env_status` in `src/janus_marimo/envs.py:244-283` only checks `.venvs/janus-<group>/bin/python` and `/opt/micromamba/envs/janus/bin/python`. Users need to be able to pick from auto-discovered local environments or type/paste any existing environment root directory or `bin/python` path to run calculations, with fast readiness validation and environment-aware run caching.
2. **Immediate 3D Chemiscope Inspection on Structure Upload/Selection**: Currently, uploading a structure file in Section 2 (`janus_workbench.py:379-413`) requires clicking a separate `"Register Uploaded ... in Ledger"` button and only displays a text summary banner (`Formula`, `PBC`, `Volume`, `d_min`) without any 3D visualization until a calculation is executed in one of the 8 tabs. Users need uploaded files to **immediately become the active structure** in `WorkspaceLedger` and render in an interactive **Chemiscope 3D Pre-Flight X-Ray Viewer** (with per-atom nearest-neighbor distances $d_{\min, i}$, coordination numbers, clash flags, fractional coordinates, and optional visual supercell tiling) before launching any calculation.

---

## 2. Scope

### In Scope
1. **Symlink-Safe Path Normalization & Static-First Environment Probing (`src/janus_marimo/envs.py`)**:
   - Add `normalize_python_executable(raw_path: str | Path) -> Path` that accepts either an environment root directory (e.g., `/opt/micromamba/envs/my-env`, `~/.venv`) or a direct Python executable path (`.../bin/python`, `.../Scripts/python.exe`) and normalizes via `os.path.abspath(os.path.expanduser(...))` **without** `Path.resolve()` symlink dereferencing (preserving `pyvenv.cfg` context for `uv venv` and `python -m venv`).
   - Fix `_probe_python_imports(python_exe: Path, modules: tuple[str, ...]) -> bool` (`src/janus_marimo/envs.py:211-242`) to compare `os.path.abspath` instead of `.resolve()`, and add a fast static filesystem check `_probe_modules_static(python_exe: Path, modules: tuple[str, ...]) -> bool | None` so standard `site-packages` directories without `.pth` overrides validate in `<5 ms` without spawning a cold PyTorch subprocess.
   - Add `discover_local_environments(*, venvs_root: Path = DEFAULT_VENVS_ROOT, base_python: Path = DEFAULT_JANUS_PYTHON, extra_env_roots: Sequence[Path] = ()) -> dict[str, str]` to auto-discover existing environments under `.venvs/janus-*`, `/opt/micromamba/envs/*`, `~/micromamba/envs/*`, `~/.micromamba/envs/*`, `~/.conda/envs/*`, `~/miniconda3/envs/*`, `~/anaconda3/envs/*`, and `~/.conda/environments.txt`.
   - Extend `EnvStatus.source` to `Literal["custom_env", "isolated_venv", "base_janus", "missing"]` and add `custom_env_path: str | Path | None = None` to `resolve_env_status(...)`. When `custom_env_path` is non-empty, it takes strict precedence and reports actionable errors if the executable is missing/broken or required packages (`janus_core` or the active MLIP backend) are missing, never silently falling back to `base_janus`.
2. **Environment-Aware Run Caching (`src/janus_marimo/ledger.py`)**:
   - Add `custom_env_path: str | None = None` to `MLIPConfig` (`src/janus_marimo/ledger.py:31-39`).
   - Update `compute_run_id` (`src/janus_marimo/ledger.py:91-115`) so that when `mlip_config.custom_env_path` is non-empty, `"custom_env_path": str(mlip_config.custom_env_path)` is included in `payload_dict["mlip_config"]` (preserving identical SHA-256 hashes when `custom_env_path is None`).
3. **Idempotent Auto-Registration of Uploaded Structures (`src/janus_marimo/ledger.py`, `janus_workbench.py`)**:
   - Add `WorkspaceLedger.register_or_get_structure(...) -> tuple[StructureEntry, bool]` (`src/janus_marimo/ledger.py:160-185`) that deduplicates against existing `self.structures.values()` by matching `entry.source_mode == source_mode`, `entry.label.split(": ", 1)[1].startswith(f"{name} (")`, and `_serialize_atoms_for_hash(entry.atoms) == _serialize_atoms_for_hash(atoms)`.
   - Update `structure_ledger_controls_cell` and `structure_ledger_panel_cell` in `janus_workbench.py:341-462` so that uploading a file via `upload_struct_ui` immediately parses, registers, and selects the uploaded structure as `active_struct_dropdown.value` on upload without requiring a secondary button click.
4. **Pre-Flight Chemiscope 3D X-Ray Widget (`src/janus_marimo/viz.py`, `janus_workbench.py`)**:
   - Add `compute_preflight_atom_properties(atoms: Atoms, *, supercell: tuple[int, int, int] = (1, 1, 1), cutoff: float = 4.0) -> tuple[Atoms, dict[str, dict[str, Any]]]` and `build_preflight_chemiscope_widget(atoms: Atoms, *, supercell: tuple[int, int, int] = (1, 1, 1), cutoff: float = 4.0) -> Any` in `src/janus_marimo/viz.py` computing per-atom `symbol`, nearest-neighbor distance `d_min_ang` (via `ase.neighborlist.neighbor_list("ijd", ...)` when 3D periodic so periodic images in small cells like FCC Cu are properly included), `coordination_number` (within `cutoff`), `clash_flag` (`1` if `0.0 < d_min_ang < 0.8` else `0`), and fractional coordinates (`frac_x`, `frac_y`, `frac_z`), plus non-destructive visual supercell tiling along periodic axes.
   - Render the active structure inside Section 2 (`structure_ledger_panel_cell` in `janus_workbench.py`) wrapped in `mo.lazy` and gated by a `"Show 3D Structure Preview (Chemiscope)"` checkbox (`value=True` by default) and a `"3D Preview Tiling"` dropdown.

### Explicitly Out of Scope
- MLIP training and dataset preprocessing (`janus train`, `janus preprocess`).
- Remote SSH/SLURM environment execution.
- Modifying any of the 8 calculation output parsers in `src/janus_marimo/parsers.py`.

---

## 3. Surfaces Touched

1. [`src/janus_marimo/envs.py`](../../src/janus_marimo/envs.py) (lines 189–285):
   - Update `EnvStatus` (`source: Literal["custom_env", "isolated_venv", "base_janus", "missing"]`).
   - Add `normalize_python_executable(raw_path: str | Path) -> Path`.
   - Add `_probe_modules_static(python_exe: Path, modules: tuple[str, ...]) -> bool | None` and update `_probe_python_imports(python_exe: Path, modules: tuple[str, ...]) -> bool`.
   - Add `discover_local_environments(...) -> dict[str, str]`.
   - Update `resolve_env_status(arch: ArchType, *, venvs_root: Path = DEFAULT_VENVS_ROOT, base_python: Path = DEFAULT_JANUS_PYTHON, custom_env_path: str | Path | None = None) -> EnvStatus`.
2. [`src/janus_marimo/ledger.py`](../../src/janus_marimo/ledger.py) (lines 30–116, 160–205):
   - Add `custom_env_path: str | None = None` to `MLIPConfig`.
   - Update `compute_run_id` to include `"custom_env_path"` in `payload_dict["mlip_config"]` when `mlip_config.custom_env_path` is truthy.
   - Add `WorkspaceLedger.register_or_get_structure(...) -> tuple[StructureEntry, bool]`.
3. [`src/janus_marimo/viz.py`](../../src/janus_marimo/viz.py) (after line 210):
   - Add `compute_preflight_atom_properties(atoms: Atoms, *, supercell: tuple[int, int, int] = (1, 1, 1), cutoff: float = 4.0) -> tuple[Atoms, dict[str, dict[str, Any]]]`.
   - Add `build_preflight_chemiscope_widget(atoms: Atoms, *, supercell: tuple[int, int, int] = (1, 1, 1), cutoff: float = 4.0) -> Any`.
4. [`src/janus_marimo/__init__.py`](../../src/janus_marimo/__init__.py) (lines 3–115):
   - Export `discover_local_environments`, `normalize_python_executable`, `compute_preflight_atom_properties`, and `build_preflight_chemiscope_widget`.
5. [`janus_workbench.py`](../../janus_workbench.py) (lines 24–103, 106–122, 124–311, 340–462):
   - Import `discover_local_environments`, `normalize_python_executable`, `compute_preflight_atom_properties`, and `build_preflight_chemiscope_widget` in `bootstrap_imports` (plus `hashlib`).
   - Add `get_last_upload_sig, set_last_upload_sig = mo.state(None)` in `state_init_cell`.
   - Add a dedicated `mlip_env_controls_cell(DEFAULT_JANUS_PYTHON, DEFAULT_VENVS_ROOT, discover_local_environments, get_env_refresh, mo)` that reads `get_env_refresh()` and creates `discovered_env_ui` (`mo.ui.dropdown`) and `custom_env_path_ui` (`mo.ui.text`), wiring both into `mlip_resolved_config_cell`.
   - Add `preview_supercell_ui` (`mo.ui.dropdown`) and `show_3d_preview_ui` (`mo.ui.checkbox(value=True, ...)`) in `structure_upload_control_cell`.
   - Update `structure_ledger_controls_cell` and `structure_ledger_panel_cell` so uploaded files auto-register and become active immediately within a single cell execution pass, and render the `mo.lazy` Chemiscope 3D pre-flight viewer in Section 2.
6. [`tests/test_envs.py`](../../tests/test_envs.py), [`tests/test_ledger.py`](../../tests/test_ledger.py), [`tests/test_viz.py`](../../tests/test_viz.py):
   - Add unit tests covering `normalize_python_executable`, `discover_local_environments`, `resolve_env_status(..., custom_env_path=...)`, `MLIPConfig.custom_env_path` in `compute_run_id`, `WorkspaceLedger.register_or_get_structure`, `compute_preflight_atom_properties`, and `build_preflight_chemiscope_widget`.

---

## 4. Interfaces

### 4.1 `src/janus_marimo/envs.py`

```python
@dataclass(frozen=True)
class EnvStatus:
    """Resolved Python environment status for an MLIP architecture."""

    group: EnvGroupName
    python_executable: Path
    is_ready: bool
    source: Literal["custom_env", "isolated_venv", "base_janus", "missing"]
    detail: str


def normalize_python_executable(raw_path: str | Path) -> Path:
    """
    Normalize a user-supplied environment prefix directory or Python binary path into a Path.
    - Expands `~` via `os.path.expanduser(str(raw_path).strip())` and normalizes relative segments
      via `os.path.abspath(...)` WITHOUT calling `Path.resolve()` (so venv `bin/python` symlinks
      are preserved while `p.exists()` still verifies the target binary exists).
    - If the resulting path `p` is an existing directory (`p.is_dir()`):
      - If `(p / "bin" / "python").exists()` or `(p / "bin" / "python").is_symlink()`, returns `p / "bin" / "python"`.
      - Else if `(p / "Scripts" / "python.exe").exists()`, returns `p / "Scripts" / "python.exe"`.
      - Else returns `p / "bin" / "python"`.
    - Otherwise returns `p`.
    """


def _probe_modules_static(python_exe: Path, modules: tuple[str, ...]) -> bool | None:
    """
    Fast filesystem check for `modules` inside `<prefix>/lib/python*/site-packages`.
    - Infers `venv_prefix = python_exe.parent.parent` if `python_exe.parent.name in {"bin", "Scripts"}`
      else `python_exe.parent`.
    - Finds `site_dirs = list(venv_prefix.glob("lib/python*/site-packages"))`.
    - If `not site_dirs`, returns `None` (unknown layout; caller falls back to subprocess).
    - If all `mod` in `modules` satisfy `_module_exists_on_disk(venv_prefix, mod)`, returns `True`.
    - If any `.pth` file exists under `site_dirs` (`any(any(sd.glob("*.pth")) for sd in site_dirs)`),
      returns `None` (editable installs may live outside `site-packages`; caller falls back to subprocess).
    - Otherwise returns `False`.
    """


def _probe_python_imports(python_exe: Path, modules: tuple[str, ...]) -> bool:
    """
    Check if `python_exe` exists (`python_exe.exists()`) and can import all `modules`.
    - Returns `False` immediately if `not python_exe.exists()` (including broken/dangling symlinks).
    - Checks `same_interp = (os.path.abspath(os.path.expanduser(str(python_exe))) == os.path.abspath(sys.executable))`
      without `.resolve()` symlink escape; if `same_interp`, uses `importlib.util.find_spec`.
    - Otherwise checks `static_res = _probe_modules_static(python_exe, modules)`; if `static_res is not None`,
      returns `static_res`.
    - Otherwise runs `subprocess.run([str(python_exe), "-c", f"import {', '.join(modules)}"], ...)`
      with `_module_exists_on_disk` fallback on `OSError`/`SubprocessError`.
    """


def discover_local_environments(
    *,
    venvs_root: Path = DEFAULT_VENVS_ROOT,
    base_python: Path = DEFAULT_JANUS_PYTHON,
    extra_env_roots: Sequence[Path] = (),
) -> dict[str, str]:
    """
    Discover existing Python/micromamba/conda/uv environments on disk.
    - Always includes `"Auto (.venvs/janus-<group> or base janus)": ""` as the first entry.
    - Includes `base_python` if `base_python.exists()`:
      `f"Base Janus ({base_python.parent.parent.name}: {base_python})": str(base_python)`.
    - Scans candidate environment directories under:
      - `sorted(venvs_root.glob("janus-*"))` if `venvs_root.is_dir()`
      - `sorted(Path("/opt/micromamba/envs").glob("*"))` if `/opt/micromamba/envs` is a dir
      - `sorted((Path.home() / "micromamba" / "envs").glob("*"))`
      - `sorted((Path.home() / ".micromamba" / "envs").glob("*"))`
      - `sorted((Path.home() / ".conda" / "envs").glob("*"))`
      - `sorted((Path.home() / "miniconda3" / "envs").glob("*"))`
      - `sorted((Path.home() / "anaconda3" / "envs").glob("*"))`
      - Non-empty, non-comment lines in `Path.home() / ".conda" / "environments.txt"` (if present)
      - Each path `root` in `extra_env_roots`: if `normalize_python_executable(root).exists()`, treats
        `root` itself as a candidate env directory; otherwise if `root.is_dir()`, scans `sorted(root.glob("*"))`.
    - For each candidate directory `env_dir`, computes `py_exe = normalize_python_executable(env_dir)`.
      If `py_exe.exists()` and `str(py_exe)` has not yet been added, inserts
      `f"{env_dir.name} ({py_exe})": str(py_exe)`.
    """


def resolve_env_status(
    arch: ArchType,
    *,
    venvs_root: Path = DEFAULT_VENVS_ROOT,
    base_python: Path = DEFAULT_JANUS_PYTHON,
    custom_env_path: str | Path | None = None,
) -> EnvStatus:
    """
    Determine which Python executable to use for `arch`.
    1. If `custom_env_path is not None and str(custom_env_path).strip() != ""`:
       - `custom_python = normalize_python_executable(custom_env_path)`
       - If `not custom_python.exists()`:
         returns `EnvStatus(group=group, python_executable=custom_python, is_ready=False, source="custom_env", detail=f"Custom Python executable not found at '{custom_python}'.")`
       - If `_probe_python_imports(custom_python, ("janus_core", spec.import_probe))` is `True`:
         returns `EnvStatus(group=group, python_executable=custom_python, is_ready=True, source="custom_env", detail=f"Custom environment ready ({custom_python})")`
       - Else returns `EnvStatus(group=group, python_executable=custom_python, is_ready=False, source="custom_env", detail=f"Custom environment '{custom_python}' is missing 'janus_core' or '{spec.import_probe}' for '{arch}'. Install '{spec.janus_extra}'.")`
    2. Otherwise checks `isolated_venv` (`venvs_root / f"janus-{group}" / "bin" / "python"`), then `base_janus` (`base_python`), then returns `source="missing"` as before.
    """
```

### 4.2 `src/janus_marimo/ledger.py`

```python
@dataclass(frozen=True)
class MLIPConfig:
    """Active MLIP calculator configuration."""

    arch: str = "mace_mp"
    model: str | None = "small"
    device: Literal["cpu", "cuda", "mps", "xpu"] = "cpu"
    dispersion: bool = False
    calc_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_env_path: str | None = None
```

In `compute_run_id`:
```python
    mlip_payload: dict[str, Any] = {
        "arch": mlip_config.arch,
        "model": mlip_config.model,
        "device": mlip_config.device,
        "dispersion": mlip_config.dispersion,
        "calc_kwargs": mlip_config.calc_kwargs,
    }
    if mlip_config.custom_env_path:
        mlip_payload["custom_env_path"] = str(mlip_config.custom_env_path)
```

In `WorkspaceLedger`:
```python
    def register_or_get_structure(
        self,
        atoms: Atoms,
        *,
        name: str,
        source_mode: str,
        is_relaxed: bool = False,
        provenance_run_id: str | None = None,
    ) -> tuple[StructureEntry, bool]:
        """
        Check if an entry in `self.structures.values()` already has `entry.source_mode == source_mode`,
        `f": {name} (" in entry.label`, and `_serialize_atoms_for_hash(entry.atoms) == _serialize_atoms_for_hash(atoms)`.
        If found, return `(existing_entry, False)`.
        Otherwise register via `self.register_structure(...)` and return `(new_entry, True)`.
        """
```

### 4.3 `src/janus_marimo/viz.py`

```python
def compute_preflight_atom_properties(
    atoms: Atoms,
    *,
    supercell: tuple[int, int, int] = (1, 1, 1),
    cutoff: float = 4.0,
) -> tuple[Atoms, dict[str, dict[str, Any]]]:
    """
    Compute a non-destructively tiled copy of `atoms` and Chemiscope atom-target diagnostic properties:
    - Tiling: For each axis `k` in `(0, 1, 2)`, repeats by `max(1, int(supercell[k]))` ONLY if
      `bool(atoms.pbc[k])` is True and `atoms.cell.lengths()[k] > 1e-6`; otherwise uses `1` along axis `k`
      (so non-periodic molecules never stack duplicate atoms at distance 0).
    - Neighbor & clash analysis on `display_atoms` (when `0 < len(display_atoms) <= 2000` and
      `np.isfinite(display_atoms.positions).all()`):
      - Let `cell_arr = np.asarray(display_atoms.cell.array, dtype=float)`,
        `cell_valid = bool(np.isfinite(cell_arr).all() and abs(float(np.linalg.det(cell_arr))) > 1e-8)`.
      - If `any(display_atoms.pbc) and cell_valid`:
        Uses `i_idx, j_idx, d_ij = ase.neighborlist.neighbor_list("ijd", display_atoms, cutoff=max(float(cutoff), 6.0))`
        so periodic image neighbors in small unit cells (e.g., 4-atom FCC Cu where `L/2 < 4.0 Å`) and self-image
        distances are accurately captured. For each atom `a in range(len(display_atoms))`, `d_min_ang[a]` is
        `round(float(np.min(d_ij[i_idx == a])), 4)` if `np.any(i_idx == a)` else `0.0`, and
        `coordination_number[a]` is `int(np.sum((i_idx == a) & (d_ij <= float(cutoff))))`.
      - Else if `len(display_atoms) > 1`:
        Uses `dist_mat = np.asarray(display_atoms.get_all_distances(mic=False), dtype=float)` with
        `np.fill_diagonal(dist_mat, np.inf)`, setting `d_min_ang[a] = round(float(np.min(dist_mat[a])), 4)`
        and `coordination_number[a] = int(np.sum(dist_mat[a] <= float(cutoff)))`.
      - Else: `d_min_ang = [0.0] * len(display_atoms)` and `coordination_number = [0] * len(display_atoms)`.
      - `clash_flag = [1 if (0.0 < d < 0.8) else 0 for d in d_min_ang]`.
    - Fractional coordinates (`frac_x`, `frac_y`, `frac_z`):
      - If `cell_valid and len(display_atoms) > 0 and np.isfinite(display_atoms.positions).all()`:
        `scaled = np.round(display_atoms.get_scaled_positions(wrap=False), 4)`
      - Else:
        `scaled = np.zeros((len(display_atoms), 3), dtype=float)`
    - Returns `(display_atoms, extra_properties)` where `extra_properties` contains keys
      `"symbol"`, `"d_min_ang"`, `"coordination_number"`, `"clash_flag"`, `"frac_x"`, `"frac_y"`, `"frac_z"`,
      each formatted as `{"target": "atom", "values": [...]}` (with `"units": "Å"` on `"d_min_ang"`).
    """


def build_preflight_chemiscope_widget(
    atoms: Atoms,
    *,
    supercell: tuple[int, int, int] = (1, 1, 1),
    cutoff: float = 4.0,
) -> Any:
    """
    Call `compute_preflight_atom_properties(atoms, supercell=supercell, cutoff=cutoff)` and return
    `build_chemiscope_widget([display_atoms], mode="structure", show_force_arrows=False, extra_properties=extra_props)`.
    """
```

---

## 5. UX Flow

### 5.1 Selecting or Specifying an Existing Environment (Section 1)
1. In **Section 1 (`1. Shared MLIP Configuration`)**:
   - A dedicated cell `mlip_env_controls_cell(DEFAULT_JANUS_PYTHON, DEFAULT_VENVS_ROOT, discover_local_environments, get_env_refresh, mo)` reads `_ = get_env_refresh()` (so provisioning a `.venvs/janus-<group>` environment automatically refreshes the discovered list) and creates:
     - `discovered_env_ui = mo.ui.dropdown(options=discover_local_environments(venvs_root=DEFAULT_VENVS_ROOT, base_python=DEFAULT_JANUS_PYTHON), value="Auto (.venvs/janus-<group> or base janus)", label="Existing Environment")`
     - `custom_env_path_ui = mo.ui.text(value="", placeholder="e.g. /opt/micromamba/envs/my-env or /path/to/bin/python", label="Custom Env Root or Python Path (override)")`
2. Resolution precedence in `mlip_resolved_config_cell`:
   - `_effective_custom_env = custom_env_path_ui.value.strip() or (discovered_env_ui.value or "").strip() or None`
   - Calls `active_env_status = resolve_env_status(arch_ui.value, custom_env_path=_effective_custom_env)`.
   - Sets `active_mlip_config = MLIPConfig(..., custom_env_path=_effective_custom_env)`.
   - The **Environment Status Callout** displays:
     - `kind="success"` when `active_env_status.is_ready` is `True`.
     - `kind="danger"` when `active_env_status.source == "custom_env"` and `not active_env_status.is_ready`.
     - `kind="warn"` when `active_env_status.source == "missing"`.

### 5.2 Auto-Active Upload & Immediate Chemiscope 3D Preview (Section 2)
1. In `structure_upload_control_cell(mo)`:
   - Defines `upload_struct_ui = mo.ui.file(filetypes=[".xyz", ".extxyz", ".cif", ".vasp", ".poscar", ".traj"], multiple=False, label="Upload Custom Structure (.cif / .extxyz / .xyz / POSCAR)")`.
   - Defines `show_3d_preview_ui = mo.ui.checkbox(value=True, label="Show 3D Structure Preview (Chemiscope)")`.
   - Defines `preview_supercell_ui = mo.ui.dropdown(options={"1×1×1 (Unit Cell)": (1, 1, 1), "2×2×1 (Surface / Slab Tiling)": (2, 2, 1), "2×2×2 (Bulk Supercell Tiling)": (2, 2, 2)}, value="1×1×1 (Unit Cell)", label="3D Preview Tiling")`.
2. In `structure_ledger_controls_cell(WorkspaceLedger, get_last_upload_sig, get_ledger, hashlib, load_uploaded_structure, mo, set_last_upload_sig, set_ledger, upload_struct_ui)`:
   - Reads `_ledger = get_ledger()` and initializes `upload_error_msg = None` and `_target_label = None`.
   - If `upload_struct_ui.value` is non-empty:
     - Extracts `_uploaded_file = upload_struct_ui.value[0]`, `_fname = _uploaded_file.name`, `_raw_bytes = _uploaded_file.contents`.
     - Computes `_upload_sig = (_fname, hashlib.sha256(_raw_bytes).hexdigest()[:16])`.
     - If `_upload_sig != get_last_upload_sig()`:
       - In a `try / except Exception as _exc` block:
         - Calls `_up_atoms = load_uploaded_structure(_fname, _raw_bytes)`.
         - Creates a shallow copy `_new_ledger = WorkspaceLedger(structures=dict(_ledger.structures), runs=dict(_ledger.runs))`.
         - Calls `_entry, _ = _new_ledger.register_or_get_structure(_up_atoms, name=f"Uploaded {_fname}", source_mode="upload", is_relaxed=False)`.
         - Updates `_ledger = _new_ledger`, calls `set_ledger(_new_ledger)`, calls `set_last_upload_sig(_upload_sig)`, and sets `_target_label = _entry.label`.
       - On exception: sets `upload_error_msg = f"Failed to parse uploaded file '{_fname}': {_exc}"` and calls `set_last_upload_sig(_upload_sig)`.
   - Builds `_opts = _ledger.structure_options()` and `_labels = list(_opts.keys())`.
   - Selects `_default_label = _target_label if (_target_label and _target_label in _opts) else (_labels[-1] if len(_labels) > 5 else _labels[0])`.
   - Creates `active_struct_dropdown = mo.ui.dropdown(options=_opts, value=_default_label, label="Active Structure from Relay Ledger")`.
   - Returns `(active_struct_dropdown, upload_error_msg)`.
3. In `structure_ledger_panel_cell(active_struct_dropdown, build_preflight_chemiscope_widget, get_ledger, mo, preview_supercell_ui, show_3d_preview_ui, upload_error_msg, upload_struct_ui)`:
   - Resolves `active_struct_entry = get_ledger().get_structure(active_struct_dropdown.value or "struct_0")`.
   - Renders `upload_struct_ui`, `active_struct_dropdown`, `show_3d_preview_ui`, and `preview_supercell_ui` in the header row, followed by an error callout if `upload_error_msg` is set, the `Inspector` callout, and (when `show_3d_preview_ui.value` is `True`) `mo.lazy(lambda: build_preflight_chemiscope_widget(active_struct_entry.atoms, supercell=preview_supercell_ui.value))`.

---

## 6. Simulation & Reactive Execution Guards

1. **Marimo Reactive DAG Integrity**:
   - Because `mo.state` uses `allow_self_loops=False`, `structure_ledger_controls_cell` updates its local `_ledger` reference in the same pass as `set_ledger(_new_ledger)` so `active_struct_dropdown` immediately contains and selects the newly uploaded structure without needing a self-loop re-trigger.
   - `mlip_env_controls_cell` depends on `get_env_refresh` so newly provisioned `.venvs/janus-<group>` environments appear in `discovered_env_ui` immediately after `env_provision_exec_cell` finishes.
2. **WebGL Context Budget**:
   - Only one pre-flight Chemiscope widget is ever rendered in Section 2, wrapped in `mo.lazy(...)` and gated by `show_3d_preview_ui`.
3. **STFC `janus-core` Guardrails**:
   - All subprocess invocations continue to pass `--no-tracker` (`tracker: False` in YAML config) and `FrechetCellFilter` for variable-cell relaxations.

---

## 7. Failure Modes

| Failure Scenario | User-Visible Behavior & Return Value |
|---|---|
| User enters a non-existent path or broken symlink (e.g., `/bad/path`) | `normalize_python_executable` resolves `/bad/path/bin/python`; `custom_python.exists()` is `False`; `resolve_env_status` returns `EnvStatus(is_ready=False, source="custom_env", detail="Custom Python executable not found at '/bad/path/bin/python'.")`; Section 1 displays a red `danger` callout. |
| User selects an existing environment missing `janus_core` or the active MLIP package | `resolve_env_status` returns `EnvStatus(is_ready=False, source="custom_env", detail="Custom environment '...' is missing 'janus_core' or '<probe>' for '<arch>'. Install '<extra>'.")`; Section 1 displays a red `danger` callout. |
| User uploads a corrupted or unparseable structure file | `structure_ledger_controls_cell` catches the exception, preserves `_ledger`, and returns `upload_error_msg`; `structure_ledger_panel_cell` displays a red `danger` callout. |
| User uploads a non-periodic molecule (`.xyz`) and selects `2×2×2` tiling | `compute_preflight_atom_properties` only tiles along axes where `pbc[k]` is `True` and `cell.lengths()[k] > 1e-6`, preventing artificial 0 Å atom stacking, and sets `frac_* = 0.0` when the cell is singular. |

---

## 8. Verification Criteria

1. **Unit Tests (`pytest`)**:
   - `tests/test_envs.py`:
     - `test_normalize_python_executable_dir_binary_and_symlink`: Verifies directory root (`tmp_path / "my_env"` -> `tmp_path / "my_env" / "bin" / "python"`), direct binary path, and symlink preservation (`normalize_python_executable` keeps the symlink path rather than dereferencing it).
     - `test_discover_local_environments`: Verifies discovery of `.venvs/janus-*` and `extra_env_roots` entries, exclusion of broken symlinks, and presence of `"Auto (.venvs/janus-<group> or base janus)"`.
     - `test_resolve_env_status_custom_env_path`: Verifies `source == "custom_env"` when ready, when missing package, and when binary does not exist (confirming no silent fallback to `base_janus`).
   - `tests/test_ledger.py`:
     - `test_compute_run_id_with_custom_env_path`: Verifies `custom_env_path=None` preserves existing hash and setting `custom_env_path="/custom/python"` produces a distinct 16-char hash.
     - `test_register_or_get_structure_idempotent`: Verifies calling `register_or_get_structure` twice with the same `Atoms`, `name`, and `source_mode` returns `(entry, True)` first and `(entry, False)` second without duplicating ledger entries.
   - `tests/test_viz.py`:
     - `test_preflight_chemiscope_properties_and_widget`: Calls `compute_preflight_atom_properties` and `build_preflight_chemiscope_widget` on periodic bulk Cu (`supercell=(2, 2, 1)`, verifying `len(display_atoms) == 16`, `coordination_number == 12` for FCC Cu at `cutoff=4.0`, and keys `symbol`, `d_min_ang`, `coordination_number`, `clash_flag`, `frac_x`, `frac_y`, `frac_z`) and on a non-periodic molecule (`H2O`, verifying `supercell=(2, 2, 2)` does not duplicate non-periodic atoms and all values are finite).
2. **Static & DAG Gates**:
   - `uv run --no-project --python /opt/micromamba/envs/janus/bin/python ruff check .`
   - `uv run --no-project --python /opt/micromamba/envs/janus/bin/python ruff format --check .`
   - `uv run --no-project --python /opt/micromamba/envs/janus/bin/python marimo check janus_workbench.py`
   - `uv run --no-project --python /opt/micromamba/envs/janus/bin/python pytest`

---

## 9. Out-of-Scope Follow-Ups

- Interactive atom coordinate editing or cell-vector dragging directly inside the browser 3D canvas.
- Multi-frame trajectory slider during `.extxyz` upload (currently selects the final frame `index=-1` per `load_uploaded_structure`).
- Remote SSH/SLURM environment probing.
