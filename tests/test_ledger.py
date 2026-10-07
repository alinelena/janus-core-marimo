"""Unit tests for janus_marimo.ledger."""

from __future__ import annotations

from pathlib import Path

import ase.build
import ase.io

from janus_marimo.ledger import (
    MLIPConfig,
    RunRecord,
    WorkspaceLedger,
    build_preset_structures,
    compute_run_id,
    load_uploaded_structure,
)


def test_compute_run_id_deterministic_and_16_chars() -> None:
    nacl = ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    cfg = MLIPConfig(arch="mace_mp", model="small", device="cpu", dispersion=False)
    params = {"optimizer": "LBFGS", "fmax": 0.01}

    id1 = compute_run_id("geomopt", cfg, nacl, params)
    id2 = compute_run_id("geomopt", cfg, nacl.copy(), dict(params))
    assert len(id1) == 16
    assert id1 == id2

    cfg_d3 = MLIPConfig(arch="mace_mp", model="small", device="cpu", dispersion=True)
    id3 = compute_run_id("geomopt", cfg_d3, nacl, params)
    assert id3 != id1
    assert cfg_d3.to_calc_kwargs() == {"dispersion": True}


def test_build_preset_structures() -> None:
    presets = build_preset_structures()
    assert len(presets) == 5
    assert "NaCl (rocksalt)" in presets
    assert "Si (diamond)" in presets
    assert "Cu (fcc)" in presets
    assert "Li (bcc vacancy hop - initial)" in presets
    assert "Li (bcc vacancy hop - final)" in presets

    li_init = presets["Li (bcc vacancy hop - initial)"]
    li_final = presets["Li (bcc vacancy hop - final)"]
    assert len(li_init) == 15
    assert len(li_final) == 15
    assert list(li_init.numbers) == list(li_final.numbers)
    assert not (li_init.positions == li_final.positions).all()


def test_workspace_ledger_registration(tmp_path: Path) -> None:
    ledger = WorkspaceLedger.from_presets()
    assert len(ledger.structures) == 5
    assert "struct_0" in ledger.structures
    assert "struct_4" in ledger.structures

    opts = ledger.structure_options()
    assert len(opts) == 5

    cu_relaxed = ledger.get_structure("struct_2").atoms.copy()
    cu_relaxed.cell *= 1.01
    new_entry = ledger.register_structure(
        cu_relaxed,
        name="Cu (geomopt relaxed)",
        source_mode="geomopt",
        is_relaxed=True,
        provenance_run_id="0123456789abcdef",
    )
    assert new_entry.entry_id == "struct_5"
    assert new_entry.is_relaxed is True
    assert ledger.get_structure("struct_5") == new_entry
    assert ledger.get_structure("nonexistent_id") == ledger.structures["struct_0"]

    rec = RunRecord(
        run_id="0123456789abcdef",
        mode="geomopt",
        mlip_config=MLIPConfig(),
        struct_entry_id="struct_2",
        final_struct_entry_id=None,
        mode_params={"fmax": 0.01},
        run_dir=tmp_path,
        config_yaml_path=tmp_path / "config.yml",
        summary_yaml_path=tmp_path / "geomopt-summary.yml",
        output_files={},
        cli_command=["uv", "run"],
        status="succeeded",
        duration_s=1.2,
        stdout_stderr="ok",
    )
    ledger.register_run(rec)
    assert ledger.latest_run_for_mode("geomopt") == rec
    assert ledger.latest_run_for_mode("phonons") is None


def test_load_uploaded_structure_all_formats(tmp_path: Path) -> None:
    si = ase.build.bulk("Si", "diamond", a=5.43, cubic=True)
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    up_dir = tmp_path / "uploads"

    formats = [
        ("si.extxyz", "extxyz"),
        ("si.xyz", "xyz"),
        ("si.cif", "cif"),
        ("si.vasp", "vasp"),
        ("si.poscar", "vasp"),
        ("si.traj", "traj"),
    ]
    for fname, fmt in formats:
        fpath = src_dir / fname
        ase.io.write(str(fpath), si, format=fmt)
        parsed = load_uploaded_structure(
            fname,
            fpath.read_bytes(),
            upload_dir=up_dir,
        )
        assert len(parsed) == 8
        assert parsed.get_chemical_formula() == "Si8"


def test_compute_run_id_with_custom_env_path() -> None:
    nacl = ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    cfg_default = MLIPConfig(arch="mace_mp", model="small", device="cpu", custom_env_path=None)
    cfg_custom = MLIPConfig(
        arch="mace_mp",
        model="small",
        device="cpu",
        custom_env_path="/custom/env/bin/python",
    )
    id_default = compute_run_id("singlepoint", cfg_default, nacl, {})
    id_custom = compute_run_id("singlepoint", cfg_custom, nacl, {})
    assert len(id_default) == 16
    assert len(id_custom) == 16
    assert id_default != id_custom


def test_register_or_get_structure_idempotent() -> None:
    ledger = WorkspaceLedger.from_presets()
    si = ase.build.bulk("Si", "diamond", a=5.43, cubic=True)

    entry1, created1 = ledger.register_or_get_structure(
        si,
        name="Uploaded si.cif",
        source_mode="upload",
        is_relaxed=False,
    )
    assert created1 is True
    assert len(ledger.structures) == 6

    entry2, created2 = ledger.register_or_get_structure(
        si.copy(),
        name="Uploaded si.cif",
        source_mode="upload",
        is_relaxed=False,
    )
    assert created2 is False
    assert entry2.entry_id == entry1.entry_id
    assert len(ledger.structures) == 6


def test_register_intermediate_geomopt_structure() -> None:
    ledger = WorkspaceLedger.from_presets()
    cu = ase.build.bulk("Cu", "fcc", a=3.61)
    entry = ledger.register_structure(
        cu,
        name="GeomOpt Step 3 (mace_mp)",
        source_mode="geomopt_intermediate",
        is_relaxed=False,
        provenance_run_id="run-test",
    )
    assert entry.source_mode == "geomopt_intermediate"
    assert entry.is_relaxed is False
    assert "GeomOpt Step 3 (mace_mp)" in entry.label
    assert entry.provenance_run_id == "run-test"
