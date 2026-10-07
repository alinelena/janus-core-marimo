"""Unit tests for janus_marimo.runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import ase.build
import yaml
from ase import Atoms

from janus_marimo.ledger import MLIPConfig, StructureEntry, WorkspaceLedger
from janus_marimo.runner import (
    build_cli_command,
    build_janus_yaml_config,
    execute_janus_mode,
)
from tools.structure_inspector import inspect_structure


def test_build_janus_yaml_config_all_8_modes(tmp_path: Path) -> None:
    mlip = MLIPConfig(arch="mace_mp", model="small", device="cpu", dispersion=True)
    struct_path = tmp_path / "input.extxyz"
    final_path = tmp_path / "final_input.extxyz"

    modes: list[str] = [
        "singlepoint",
        "geomopt",
        "md",
        "phonons",
        "eos",
        "elasticity",
        "neb",
        "descriptors",
    ]
    for mode in modes:
        cfg = build_janus_yaml_config(
            mode,  # type: ignore[arg-type]
            mlip,
            struct_path=struct_path,
            run_dir=tmp_path,
            mode_params={},
            final_struct_path=final_path if mode == "neb" else None,
        )
        assert cfg["tracker"] is False
        assert cfg["arch"] == "mace_mp"
        assert cfg["model"] == "small"
        assert cfg["calc_kwargs"] == {"dispersion": True}

    # Geomopt variable-cell vs fixed-cell filter_class check
    opt_var = build_janus_yaml_config(
        "geomopt",
        mlip,
        struct_path=struct_path,
        run_dir=tmp_path,
        mode_params={"opt_cell_fully": True},
    )
    assert opt_var["filter_class"] == "FrechetCellFilter"
    assert opt_var["opt_cell_lengths"] is False

    opt_fixed = build_janus_yaml_config(
        "geomopt",
        mlip,
        struct_path=struct_path,
        run_dir=tmp_path,
        mode_params={"opt_cell_fully": False, "opt_cell_lengths": False},
    )
    assert "filter_class" not in opt_fixed

    # Phonons hdf5: False check
    ph_cfg = build_janus_yaml_config(
        "phonons",
        mlip,
        struct_path=struct_path,
        run_dir=tmp_path,
        mode_params={},
    )
    assert ph_cfg["hdf5"] is False

    # NEB write_band: True check
    neb_cfg = build_janus_yaml_config(
        "neb",
        mlip,
        struct_path=struct_path,
        run_dir=tmp_path,
        mode_params={"climb": True, "k": 0.15},
        final_struct_path=final_path,
    )
    assert neb_cfg["write_band"] is True
    assert neb_cfg["neb_kwargs"] == {"climb": True, "k": 0.15}


def test_build_cli_command(tmp_path: Path) -> None:
    cmd = build_cli_command(
        Path("/opt/micromamba/envs/janus/bin/python"),
        "singlepoint",
        tmp_path / "config.yml",
    )
    assert cmd == [
        "janus",
        "singlepoint",
        "--config",
        str(tmp_path / "config.yml"),
        "--no-tracker",
    ]


def test_execute_janus_mode_preflight_and_caching(tmp_path: Path) -> None:
    ledger = WorkspaceLedger.from_presets()
    nacl_entry = ledger.get_structure("struct_0")
    mlip = MLIPConfig(arch="mace_mp", model="small", device="cpu")
    py_exe = Path("/opt/micromamba/envs/janus/bin/python")

    # 1. Pre-flight rejects non-PBC structure on EOS
    mol = ase.build.molecule("H2O")
    mol_entry = StructureEntry(
        entry_id="mol_0",
        label="H2O",
        atoms=mol,
        source_mode="upload",
        is_relaxed=False,
        report=inspect_structure(mol),
    )
    rec_non_pbc = execute_janus_mode(
        "eos",
        mlip,
        mol_entry,
        {},
        python_executable=py_exe,
        runs_root=tmp_path,
    )
    assert rec_non_pbc.status == "failed"
    assert "3D periodic boundary conditions" in rec_non_pbc.stdout_stderr

    rec_opt_non_pbc = execute_janus_mode(
        "geomopt",
        mlip,
        mol_entry,
        {"opt_cell_fully": True},
        python_executable=py_exe,
        runs_root=tmp_path,
    )
    assert rec_opt_non_pbc.status == "failed"
    assert "3D periodic boundary conditions" in rec_opt_non_pbc.stdout_stderr

    # 2. Pre-flight rejects architecture requiring explicit model when model=None
    rec_no_model = execute_janus_mode(
        "singlepoint",
        MLIPConfig(arch="nequip", model=None),
        nacl_entry,
        {},
        python_executable=py_exe,
        runs_root=tmp_path,
    )
    assert rec_no_model.status == "failed"
    assert "requires an explicit model path" in rec_no_model.stdout_stderr

    # 3. Subprocess execution and subsequent cache hit
    def _fake_subprocess_run(cmd: list[str], **kwargs: object):
        cfg_idx = cmd.index("--config") + 1
        cfg_path = Path(cmd[cfg_idx])
        cfg_data = yaml.safe_load(cfg_path.read_text())
        summary_path = Path(cfg_data["summary"])
        summary_path.write_text(yaml.safe_dump({"output_files": {"results": cfg_data["out"]}}))

        class _Res:
            returncode = 0
            stdout = "Simulation complete"
            stderr = ""

        return _Res()

    with patch("subprocess.run", side_effect=_fake_subprocess_run) as mock_run:
        rec_first = execute_janus_mode(
            "singlepoint",
            mlip,
            nacl_entry,
            {"properties": ["energy", "forces"]},
            python_executable=py_exe,
            runs_root=tmp_path,
        )
        assert rec_first.status == "succeeded"
        assert mock_run.call_count == 1

        rec_cached = execute_janus_mode(
            "singlepoint",
            mlip,
            nacl_entry,
            {"properties": ["energy", "forces"]},
            python_executable=py_exe,
            runs_root=tmp_path,
        )
        assert rec_cached.status == "cached"
        assert mock_run.call_count == 1
        assert rec_cached.run_id == rec_first.run_id

    # 4. Pre-flight rejects overlapping atoms
    bad_atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    bad_entry = StructureEntry(
        entry_id="bad_0",
        label="H2 bad",
        atoms=bad_atoms,
        source_mode="upload",
        is_relaxed=False,
        report=inspect_structure(bad_atoms),
    )
    rec_bad = execute_janus_mode(
        "singlepoint",
        mlip,
        bad_entry,
        {},
        python_executable=py_exe,
        runs_root=tmp_path,
    )
    assert rec_bad.status == "failed"
    assert "Overlapping atoms" in rec_bad.stdout_stderr


def test_build_cli_command_invokes_typer_app(tmp_path: Path) -> None:
    import os
    import subprocess

    py_exe = Path("/opt/micromamba/envs/janus/bin/python")
    cmd = build_cli_command(py_exe, "singlepoint", tmp_path / "config.yml")
    assert cmd[0] == "janus"
    help_cmd = [*cmd[:-3], "--help"]
    env = {
        **os.environ,
        "PATH": f"{py_exe.parent}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    proc = subprocess.run(help_cmd, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert "Perform single point calculations" in proc.stdout


def test_execute_janus_mode_singlepoint_default_mace(tmp_path: Path) -> None:
    from janus_marimo.envs import resolve_env_status
    from janus_marimo.parsers import parse_singlepoint_run

    ledger = WorkspaceLedger.from_presets()
    nacl_entry = ledger.get_structure("struct_0")
    mlip = MLIPConfig()  # default mace_mp, model="small", device="cpu"
    env_status = resolve_env_status(mlip.arch, custom_env_path=mlip.custom_env_path)
    assert env_status.is_ready is True

    rec = execute_janus_mode(
        "singlepoint",
        mlip,
        nacl_entry,
        {"properties": ["energy", "forces", "stress"]},
        python_executable=env_status.python_executable,
        runs_root=tmp_path,
        force_rerun=True,
    )
    assert rec.status == "succeeded", f"Singlepoint failed: {rec.stdout_stderr}"
    assert rec.summary_yaml_path.exists()

    parsed = parse_singlepoint_run(rec.run_dir, mlip.arch)
    assert parsed.energy_ev < 0.0
    assert len(parsed.atom_df) == len(nacl_entry.atoms)
    assert parsed.stress_voigt_ev_ang3 is not None
    assert len(parsed.stress_voigt_ev_ang3) == 6


