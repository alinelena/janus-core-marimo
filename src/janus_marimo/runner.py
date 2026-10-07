"""Stateless janus-core YAML configuration compiler and CLI subprocess runner."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

import ase.io
import yaml

from janus_marimo.envs import ARCH_REQUIRES_EXPLICIT_MODEL
from janus_marimo.inspector import inspect_structure, validate_neb_endpoints
from janus_marimo.ledger import (
    JanusMode,
    MLIPConfig,
    RunRecord,
    StructureEntry,
    compute_run_id,
)


def build_janus_yaml_config(
    mode: JanusMode,
    mlip_config: MLIPConfig,
    *,
    struct_path: Path,
    run_dir: Path,
    mode_params: dict[str, Any],
    final_struct_path: Path | None = None,
) -> dict[str, Any]:
    """Build a `janus <mode> --config` dictionary matching exact `janus_core.cli.<mode>` parameter names."""
    cfg: dict[str, Any] = {
        "arch": mlip_config.arch,
        "device": mlip_config.device,
        "calc_kwargs": mlip_config.to_calc_kwargs(),
        "file_prefix": str(run_dir / "job"),
        "log": str(run_dir / f"{mode}-log.yml"),
        "summary": str(run_dir / f"{mode}-summary.yml"),
        "tracker": False,
    }
    if mlip_config.model:
        cfg["model"] = mlip_config.model

    if mode == "singlepoint":
        cfg.update(
            {
                "struct": str(struct_path),
                "properties": list(mode_params.get("properties", ["energy", "forces", "stress"])),
                "out": str(run_dir / "job-results.extxyz"),
                "progress_bar": False,
            }
        )
    elif mode == "geomopt":
        opt_cell_fully = bool(mode_params.get("opt_cell_fully", True))
        opt_cell_lengths = (
            False if opt_cell_fully else bool(mode_params.get("opt_cell_lengths", False))
        )
        cfg.update(
            {
                "struct": str(struct_path),
                "optimizer": str(mode_params.get("optimizer", "LBFGS")),
                "fmax": float(mode_params.get("fmax", 0.01)),
                "steps": int(mode_params.get("steps", 500)),
                "opt_cell_fully": opt_cell_fully,
                "opt_cell_lengths": opt_cell_lengths,
                "pressure": float(mode_params.get("pressure", 0.0)),
                "symmetrize": bool(mode_params.get("symmetrize", False)),
                "write_traj": True,
                "out": str(run_dir / "job-opt.extxyz"),
            }
        )
        if opt_cell_fully or opt_cell_lengths:
            cfg["filter_class"] = str(mode_params.get("filter_class", "FrechetCellFilter"))
    elif mode == "md":
        rdf_compute = bool(mode_params.get("rdf_compute", False))
        post_process_kwargs: dict[str, Any] = (
            {
                "rdf_compute": True,
                "rdf_rmax": float(mode_params.get("rdf_rmax", 6.0)),
                "rdf_nbins": int(mode_params.get("rdf_nbins", 60)),
            }
            if rdf_compute
            else {}
        )
        cfg.update(
            {
                "struct": str(struct_path),
                "ensemble": str(mode_params.get("ensemble", "nvt")),
                "temp": float(mode_params.get("temp", 300.0)),
                "steps": int(mode_params.get("steps", 200)),
                "timestep": float(mode_params.get("timestep", 1.0)),
                "stats_every": int(mode_params.get("stats_every", 10)),
                "traj_every": int(mode_params.get("traj_every", 10)),
                "stats_file": str(run_dir / "job-stats.dat"),
                "traj_file": str(run_dir / "job-traj.extxyz"),
                "final_file": str(run_dir / "job-final.extxyz"),
                "friction": float(mode_params.get("friction", 0.005)),
                "pressure": float(mode_params.get("pressure", 0.0)),
                "post_process_kwargs": post_process_kwargs,
                "progress_bar": False,
            }
        )
    elif mode == "phonons":
        cfg.update(
            {
                "struct": str(struct_path),
                "supercell": str(mode_params.get("supercell", "2 2 2")),
                "displacement": float(mode_params.get("displacement", 0.01)),
                "mesh": list(mode_params.get("mesh", [10, 10, 10])),
                "symmetrize": bool(mode_params.get("symmetrize", False)),
                "minimize": bool(mode_params.get("minimize", False)),
                "fmax": float(mode_params.get("fmax", 0.01)),
                "bands": bool(mode_params.get("bands", True)),
                "dos": bool(mode_params.get("dos", True)),
                "pdos": bool(mode_params.get("pdos", False)),
                "thermal": bool(mode_params.get("thermal", True)),
                "temp_min": float(mode_params.get("temp_min", 0.0)),
                "temp_max": float(mode_params.get("temp_max", 1000.0)),
                "temp_step": float(mode_params.get("temp_step", 50.0)),
                "hdf5": False,
                "write_full": True,
                "plot_to_file": False,
                "progress_bar": False,
            }
        )
    elif mode == "eos":
        cfg.update(
            {
                "struct": str(struct_path),
                "min_volume": float(mode_params.get("min_volume", 0.95)),
                "max_volume": float(mode_params.get("max_volume", 1.05)),
                "n_volumes": int(mode_params.get("n_volumes", 7)),
                "eos_type": str(mode_params.get("eos_type", "birchmurnaghan")),
                "minimize": bool(mode_params.get("minimize", False)),
                "minimize_all": bool(mode_params.get("minimize_all", False)),
                "fmax": float(mode_params.get("fmax", 0.05)),
                "write_structures": True,
                "plot_to_file": False,
            }
        )
    elif mode == "elasticity":
        cfg.update(
            {
                "struct": str(struct_path),
                "normal_magnitude": float(mode_params.get("normal_magnitude", 0.01)),
                "shear_magnitude": float(mode_params.get("shear_magnitude", 0.06)),
                "n_strains": int(mode_params.get("n_strains", 4)),
                "minimize": bool(mode_params.get("minimize", False)),
                "minimize_all": bool(mode_params.get("minimize_all", False)),
                "fmax": float(mode_params.get("fmax", 0.05)),
                "write_voigt": True,
                "write_structures": True,
            }
        )
    elif mode == "neb":
        cfg.update(
            {
                "init_struct": str(struct_path),
                "final_struct": str(final_struct_path) if final_struct_path else "",
                "n_images": int(mode_params.get("n_images", 5)),
                "interpolator": str(mode_params.get("interpolator", "ase")),
                "neb_class": str(mode_params.get("neb_class", "NEB")),
                "neb_kwargs": {
                    "climb": bool(mode_params.get("climb", True)),
                    "k": float(mode_params.get("k", 0.1)),
                },
                "optimizer": str(mode_params.get("optimizer", "NEBOptimizer")),
                "fmax": float(mode_params.get("fmax", 0.05)),
                "steps": int(mode_params.get("steps", 100)),
                "minimize": bool(mode_params.get("minimize", False)),
                "write_band": True,
                "plot_band": False,
            }
        )
    elif mode == "descriptors":
        cfg.update(
            {
                "struct": str(struct_path),
                "invariants_only": bool(mode_params.get("invariants_only", True)),
                "calc_per_element": bool(mode_params.get("calc_per_element", True)),
                "calc_per_atom": bool(mode_params.get("calc_per_atom", True)),
                "out": str(run_dir / "job-descriptors.extxyz"),
                "progress_bar": False,
            }
        )
    else:
        raise ValueError(f"Unsupported janus mode: {mode}")

    return cfg


def build_cli_command(
    python_executable: Path,
    mode: JanusMode,
    config_yaml_path: Path,
) -> list[str]:
    """Return the exact CLI command list for executing janus-core."""
    _ = python_executable
    return [
        "janus",
        mode,
        "--config",
        str(config_yaml_path),
        "--no-tracker",
    ]


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
    """Validate structure pre-flight, check content-addressed cache, and run janus CLI subprocess."""
    final_atoms = final_struct_entry.atoms if final_struct_entry is not None else None
    run_id = compute_run_id(
        mode,
        mlip_config,
        struct_entry.atoms,
        mode_params,
        final_atoms,
    )
    run_dir = runs_root / mode / run_id
    config_yaml_path = run_dir / "config.yml"
    summary_yaml_path = run_dir / f"{mode}-summary.yml"
    cli_command = build_cli_command(python_executable, mode, config_yaml_path)

    # 1. Pre-flight structure validation
    if mode == "neb":
        ok, neb_errors = validate_neb_endpoints(struct_entry.atoms, final_atoms)
        if not ok:
            return RunRecord(
                run_id=run_id,
                mode=mode,
                mlip_config=mlip_config,
                struct_entry_id=struct_entry.entry_id,
                final_struct_entry_id=(final_struct_entry.entry_id if final_struct_entry else None),
                mode_params=dict(mode_params),
                run_dir=run_dir,
                config_yaml_path=config_yaml_path,
                summary_yaml_path=summary_yaml_path,
                output_files={},
                cli_command=cli_command,
                status="failed",
                duration_s=0.0,
                stdout_stderr="\n".join(neb_errors),
            )
    else:
        opt_cell_fully = bool(mode_params.get("opt_cell_fully", True))
        opt_cell_lengths = (
            False if opt_cell_fully else bool(mode_params.get("opt_cell_lengths", False))
        )
        opt_var_cell = mode == "geomopt" and (opt_cell_fully or opt_cell_lengths)
        require_pbc = mode in {"phonons", "eos", "elasticity"} or opt_var_cell
        report = inspect_structure(struct_entry.atoms, require_pbc=require_pbc)
        if not report.is_valid:
            return RunRecord(
                run_id=run_id,
                mode=mode,
                mlip_config=mlip_config,
                struct_entry_id=struct_entry.entry_id,
                final_struct_entry_id=None,
                mode_params=dict(mode_params),
                run_dir=run_dir,
                config_yaml_path=config_yaml_path,
                summary_yaml_path=summary_yaml_path,
                output_files={},
                cli_command=cli_command,
                status="failed",
                duration_s=0.0,
                stdout_stderr="\n".join(report.errors),
            )

    if mlip_config.arch in ARCH_REQUIRES_EXPLICIT_MODEL and not mlip_config.model:
        return RunRecord(
            run_id=run_id,
            mode=mode,
            mlip_config=mlip_config,
            struct_entry_id=struct_entry.entry_id,
            final_struct_entry_id=(final_struct_entry.entry_id if final_struct_entry else None),
            mode_params=dict(mode_params),
            run_dir=run_dir,
            config_yaml_path=config_yaml_path,
            summary_yaml_path=summary_yaml_path,
            output_files={},
            cli_command=cli_command,
            status="failed",
            duration_s=0.0,
            stdout_stderr=(
                f"Architecture '{mlip_config.arch}' requires an explicit model path or checkpoint."
            ),
        )

    # 2. Content-addressed cache check
    if not force_rerun and summary_yaml_path.exists():
        try:
            summary_data = yaml.safe_load(summary_yaml_path.read_text()) or {}
        except Exception:
            summary_data = {}
        return RunRecord(
            run_id=run_id,
            mode=mode,
            mlip_config=mlip_config,
            struct_entry_id=struct_entry.entry_id,
            final_struct_entry_id=(final_struct_entry.entry_id if final_struct_entry else None),
            mode_params=dict(mode_params),
            run_dir=run_dir,
            config_yaml_path=config_yaml_path,
            summary_yaml_path=summary_yaml_path,
            output_files=dict(summary_data.get("output_files", {})),
            cli_command=cli_command,
            status="cached",
            duration_s=0.0,
            stdout_stderr="Loaded from content-addressed cache.",
        )

    # 3. Write input structures and YAML config
    run_dir.mkdir(parents=True, exist_ok=True)
    struct_path = run_dir / "input.extxyz"
    ase.io.write(str(struct_path), struct_entry.atoms, format="extxyz")

    final_struct_path: Path | None = None
    if mode == "neb" and final_struct_entry is not None:
        final_struct_path = run_dir / "final_input.extxyz"
        ase.io.write(str(final_struct_path), final_struct_entry.atoms, format="extxyz")

    cfg = build_janus_yaml_config(
        mode,
        mlip_config,
        struct_path=struct_path,
        run_dir=run_dir,
        mode_params=mode_params,
        final_struct_path=final_struct_path,
    )
    config_yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    # 4. Execute CLI subprocess
    t0 = time.perf_counter()
    env_bin_dir = os.path.abspath(os.path.expanduser(str(python_executable.parent)))
    subproc_env = {
        **os.environ,
        "PATH": f"{env_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    try:
        proc = subprocess.run(
            cli_command,
            env=subproc_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        duration_s = time.perf_counter() - t0
        combined_output = "\n".join(
            part for part in (proc.stdout.strip(), proc.stderr.strip()) if part
        )
        returncode = proc.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        duration_s = time.perf_counter() - t0
        combined_output = f"Subprocess execution error: {exc}"
        returncode = 1

    if returncode == 0 and summary_yaml_path.exists():
        try:
            summary_data = yaml.safe_load(summary_yaml_path.read_text()) or {}
        except Exception:
            summary_data = {}
        return RunRecord(
            run_id=run_id,
            mode=mode,
            mlip_config=mlip_config,
            struct_entry_id=struct_entry.entry_id,
            final_struct_entry_id=(final_struct_entry.entry_id if final_struct_entry else None),
            mode_params=dict(mode_params),
            run_dir=run_dir,
            config_yaml_path=config_yaml_path,
            summary_yaml_path=summary_yaml_path,
            output_files=dict(summary_data.get("output_files", {})),
            cli_command=cli_command,
            status="succeeded",
            duration_s=duration_s,
            stdout_stderr=combined_output,
        )

    return RunRecord(
        run_id=run_id,
        mode=mode,
        mlip_config=mlip_config,
        struct_entry_id=struct_entry.entry_id,
        final_struct_entry_id=(final_struct_entry.entry_id if final_struct_entry else None),
        mode_params=dict(mode_params),
        run_dir=run_dir,
        config_yaml_path=config_yaml_path,
        summary_yaml_path=summary_yaml_path,
        output_files={},
        cli_command=cli_command,
        status="failed",
        duration_s=duration_s,
        stdout_stderr=combined_output or f"janus {mode} exited with code {returncode}.",
    )
