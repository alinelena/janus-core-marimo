"""MLIP architecture environment catalog, readiness probe, and on-demand uv provisioner."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ArchType = Literal[
    "mace",
    "mace_mp",
    "mace_off",
    "mace_omol",
    "mace_polar",
    "sevennet",
    "chgnet",
    "orb",
    "mattersim",
    "fairchem",
    "nequip",
    "dpa3",
    "deepmd",
    "grace",
    "upet",
]

EnvGroupName = Literal[
    "mace",
    "sevennet",
    "chgnet",
    "orb",
    "mattersim",
    "fairchem",
    "nequip",
    "dpa3",
    "grace",
    "upet",
]

DEFAULT_JANUS_PYTHON = Path("/opt/micromamba/envs/janus/bin/python")
DEFAULT_VENVS_ROOT = Path(".venvs")


@dataclass(frozen=True)
class MLIPEnvSpec:
    """Specification for an isolated MLIP virtual environment group."""

    group: EnvGroupName
    architectures: tuple[ArchType, ...]
    import_probe: str
    janus_extra: str
    pip_packages: tuple[str, ...]
    default_model: str | None
    conflict_note: str


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
        architectures=("dpa3", "deepmd"),
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
    "deepmd": "dpa3",
    "grace": "grace",
    "upet": "upet",
}

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
    "deepmd": None,
    "grace": "GRACE-2L-OMAT",
    "upet": "pet-mad-s",
}

ARCH_REQUIRES_EXPLICIT_MODEL: frozenset[ArchType] = frozenset({"mace", "nequip", "dpa3"})


@dataclass(frozen=True)
class EnvStatus:
    """Resolved Python environment status for an MLIP architecture."""

    group: EnvGroupName
    python_executable: Path
    is_ready: bool
    source: Literal["custom_env", "isolated_venv", "base_janus", "missing"]
    detail: str


def normalize_python_executable(raw_path: str | Path) -> Path:
    """Normalize a user-supplied environment prefix directory or Python binary path into a Path without dereferencing symlinks."""
    expanded = os.path.expanduser(str(raw_path).strip())
    p = Path(os.path.abspath(expanded))
    if p.is_dir():
        bin_py = p / "bin" / "python"
        if bin_py.exists() or bin_py.is_symlink():
            return bin_py
        scripts_py = p / "Scripts" / "python.exe"
        if scripts_py.exists():
            return scripts_py
        return bin_py
    return p


def _module_exists_on_disk(venv_prefix: Path, module_name: str) -> bool:
    """Fallback check if a module exists under `<venv_prefix>/lib/python*/site-packages`."""
    rel_parts = module_name.split(".")
    for site_pkg in venv_prefix.glob("lib/python*/site-packages"):
        candidate_dir = site_pkg.joinpath(*rel_parts)
        candidate_py = site_pkg.joinpath(*rel_parts[:-1], f"{rel_parts[-1]}.py")
        if candidate_dir.is_dir() or candidate_py.is_file():
            return True
    return False


def _probe_modules_static(python_exe: Path, modules: tuple[str, ...]) -> bool | None:
    """Fast filesystem check for `modules` inside `<prefix>/lib/python*/site-packages`."""
    venv_prefix = (
        python_exe.parent.parent
        if python_exe.parent.name in {"bin", "Scripts"}
        else python_exe.parent
    )
    site_dirs = list(venv_prefix.glob("lib/python*/site-packages"))
    if not site_dirs:
        return None
    if all(_module_exists_on_disk(venv_prefix, mod) for mod in modules):
        return True
    if any(any(sd.glob("*.pth")) for sd in site_dirs):
        return None
    return False


def _probe_python_imports(python_exe: Path, modules: tuple[str, ...]) -> bool:
    """Check if `python_exe` exists and can import all `modules`."""
    if not python_exe.exists():
        return False

    try:
        same_interp = os.path.abspath(os.path.expanduser(str(python_exe))) == os.path.abspath(
            sys.executable
        )
    except OSError:
        same_interp = str(python_exe) == sys.executable

    if same_interp:
        for mod in modules:
            try:
                if importlib.util.find_spec(mod) is None:
                    return False
            except (ModuleNotFoundError, ValueError):
                return False
        return True

    static_res = _probe_modules_static(python_exe, modules)
    if static_res is not None:
        return static_res

    try:
        proc = subprocess.run(
            [str(python_exe), "-c", f"import {', '.join(modules)}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        venv_prefix = python_exe.parent.parent
        return all(_module_exists_on_disk(venv_prefix, mod) for mod in modules)


def discover_local_environments(
    *,
    venvs_root: Path = DEFAULT_VENVS_ROOT,
    base_python: Path = DEFAULT_JANUS_PYTHON,
    extra_env_roots: Sequence[Path] = (),
) -> dict[str, str]:
    """Discover existing Python/micromamba/conda/uv environments on disk."""
    options: dict[str, str] = {"Auto (.venvs/janus-<group> or base janus)": ""}
    seen_executables: set[str] = set()

    if base_python.exists():
        norm_base = normalize_python_executable(base_python)
        base_str = str(norm_base)
        seen_executables.add(base_str)
        env_name = (
            norm_base.parent.parent.name if norm_base.parent.name in {"bin", "Scripts"} else "base"
        )
        options[f"Base Janus ({env_name}: {base_str})"] = base_str

    candidate_dirs: list[Path] = []
    if venvs_root.is_dir():
        candidate_dirs.extend(sorted(venvs_root.glob("janus-*")))

    home = Path.home()
    standard_parents = [
        Path("/opt/micromamba/envs"),
        home / "micromamba" / "envs",
        home / ".micromamba" / "envs",
        home / ".conda" / "envs",
        home / "miniconda3" / "envs",
        home / "anaconda3" / "envs",
    ]
    for parent in standard_parents:
        if parent.is_dir():
            candidate_dirs.extend(sorted(parent.glob("*")))

    conda_txt = home / ".conda" / "environments.txt"
    if conda_txt.is_file():
        try:
            for line in conda_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                clean = line.strip()
                if clean and not clean.startswith("#"):
                    candidate_dirs.append(Path(clean))
        except OSError:
            pass

    for root in extra_env_roots:
        root_path = Path(root)
        if normalize_python_executable(root_path).exists():
            candidate_dirs.append(root_path)
        elif root_path.is_dir():
            candidate_dirs.extend(sorted(root_path.glob("*")))

    for env_dir in candidate_dirs:
        py_exe = normalize_python_executable(env_dir)
        py_str = str(py_exe)
        if py_exe.exists() and py_str not in seen_executables:
            seen_executables.add(py_str)
            env_label_name = (
                py_exe.parent.parent.name
                if py_exe.parent.name in {"bin", "Scripts"}
                else env_dir.name
            )
            options[f"{env_label_name} ({py_str})"] = py_str

    return options


def resolve_env_status(
    arch: ArchType,
    *,
    venvs_root: Path = DEFAULT_VENVS_ROOT,
    base_python: Path = DEFAULT_JANUS_PYTHON,
    custom_env_path: str | Path | None = None,
) -> EnvStatus:
    """Determine which Python executable to use for `arch`."""
    group = ARCH_TO_ENV_GROUP[arch]
    spec = MLIP_ENV_CATALOG[group]

    if custom_env_path is not None and str(custom_env_path).strip() != "":
        custom_python = normalize_python_executable(custom_env_path)
        if not custom_python.exists():
            return EnvStatus(
                group=group,
                python_executable=custom_python,
                is_ready=False,
                source="custom_env",
                detail=f"Custom Python executable not found at '{custom_python}'.",
            )
        if _probe_python_imports(custom_python, ("janus_core", spec.import_probe)):
            return EnvStatus(
                group=group,
                python_executable=custom_python,
                is_ready=True,
                source="custom_env",
                detail=f"Custom environment ready ({custom_python})",
            )
        return EnvStatus(
            group=group,
            python_executable=custom_python,
            is_ready=False,
            source="custom_env",
            detail=(
                f"Custom environment '{custom_python}' is missing 'janus_core' or "
                f"'{spec.import_probe}' for '{arch}'. Install '{spec.janus_extra}'."
            ),
        )

    venv_python = venvs_root / f"janus-{group}" / "bin" / "python"

    if venv_python.exists() and _probe_python_imports(
        venv_python, ("janus_core", spec.import_probe)
    ):
        return EnvStatus(
            group=group,
            python_executable=venv_python,
            is_ready=True,
            source="isolated_venv",
            detail=f"Isolated venv ready ({venv_python})",
        )

    if base_python.exists() and _probe_python_imports(
        base_python, ("janus_core", spec.import_probe)
    ):
        return EnvStatus(
            group=group,
            python_executable=base_python,
            is_ready=True,
            source="base_janus",
            detail=f"Base janus env ready ({base_python})",
        )

    return EnvStatus(
        group=group,
        python_executable=venv_python,
        is_ready=False,
        source="missing",
        detail=f"Environment 'janus-{group}' is not provisioned yet. {spec.conflict_note}",
    )


def build_provision_commands(
    group: EnvGroupName,
    *,
    venvs_root: Path = DEFAULT_VENVS_ROOT,
    python_version: str = "3.12",
    include_d3: bool = True,
) -> list[list[str]]:
    """Return the deterministic `uv` command sequence to provision `.venvs/janus-<group>`."""
    spec = MLIP_ENV_CATALOG[group]
    venv_dir = venvs_root / f"janus-{group}"
    venv_python = venv_dir / "bin" / "python"
    packages = list(spec.pip_packages)
    if include_d3:
        packages.append("janus-core[d3]")

    return [
        ["uv", "venv", str(venv_dir), "--python", python_version],
        ["uv", "pip", "install", "--python", str(venv_python), *packages],
    ]


def provision_mlip_env(
    group: EnvGroupName,
    *,
    venvs_root: Path = DEFAULT_VENVS_ROOT,
    python_version: str = "3.12",
    include_d3: bool = True,
    timeout_s: int = 900,
) -> tuple[bool, str]:
    """Execute `build_provision_commands(...)` sequentially via `subprocess.run`."""
    commands = build_provision_commands(
        group,
        venvs_root=venvs_root,
        python_version=python_version,
        include_d3=include_d3,
    )
    logs: list[str] = []
    for cmd in commands:
        cmd_str = " ".join(cmd)
        logs.append(f"$ {cmd_str}")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logs.append(f"ERROR executing command: {exc}")
            return False, "\n".join(logs)

        if proc.stdout:
            logs.append(proc.stdout.strip())
        if proc.stderr:
            logs.append(proc.stderr.strip())
        if proc.returncode != 0:
            logs.append(f"Command exited with return code {proc.returncode}.")
            return False, "\n".join(logs)

    logs.append(f"Successfully provisioned environment janus-{group}.")
    return True, "\n".join(logs)
