"""Unit tests for tools.structure_inspector."""

from __future__ import annotations

from pathlib import Path

import ase.build
import ase.io
import numpy as np
from ase import Atoms

from janus_marimo.ledger import build_preset_structures
from tools.structure_inspector import (
    inspect_structure,
    main,
    validate_neb_endpoints,
)


def test_inspect_periodic_bulk_crystal_passes() -> None:
    nacl = ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    report = inspect_structure(nacl, require_pbc=True)
    assert report.is_valid is True
    assert report.n_atoms == 8
    assert report.pbc == (True, True, True)
    assert report.volume is not None and report.volume > 100.0
    assert report.min_distance is not None and report.min_distance > 2.5
    assert report.has_overlapping_atoms is False
    assert report.has_non_finite_coords is False
    assert report.errors == []


def test_inspect_overlapping_atoms_rejected() -> None:
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    report = inspect_structure(atoms, min_dist_threshold=0.5)
    assert report.is_valid is False
    assert report.has_overlapping_atoms is True
    assert any("Overlapping atoms detected" in err for err in report.errors)


def test_inspect_non_finite_coordinates_rejected() -> None:
    atoms = ase.build.bulk("Cu", "fcc", a=3.61, cubic=True)
    atoms.positions[0, 1] = np.nan
    report = inspect_structure(atoms)
    assert report.is_valid is False
    assert report.has_non_finite_coords is True
    assert any("NaN or infinite" in err for err in report.errors)


def test_inspect_non_pbc_rejected_when_require_pbc_true() -> None:
    water = ase.build.molecule("H2O")
    report_optional = inspect_structure(water, require_pbc=False)
    assert report_optional.is_valid is True
    assert any("Non-periodic" in w for w in report_optional.warnings)

    report_required = inspect_structure(water, require_pbc=True)
    assert report_required.is_valid is False
    assert any("3D periodic boundary conditions" in err for err in report_required.errors)


def test_validate_neb_endpoints_checks() -> None:
    presets = build_preset_structures()
    li_init = presets["Li (bcc vacancy hop - initial)"]
    li_final = presets["Li (bcc vacancy hop - final)"]

    ok, errors = validate_neb_endpoints(li_init, li_final)
    assert ok is True
    assert errors == []

    # Missing final endpoint
    ok_none, err_none = validate_neb_endpoints(li_init, None)
    assert ok_none is False
    assert any("requires a final structure" in e for e in err_none)

    # Atom count mismatch
    ok_len, err_len = validate_neb_endpoints(li_init, presets["NaCl (rocksalt)"])
    assert ok_len is False
    assert any("atom count mismatch" in e for e in err_len)

    # Species mismatch
    li_mutated = li_final.copy()
    li_mutated.numbers[0] = 11  # Na instead of Li
    ok_spec, err_spec = validate_neb_endpoints(li_init, li_mutated)
    assert ok_spec is False
    assert any("species/ordering mismatch" in e for e in err_spec)

    # Identical coordinates
    ok_ident, err_ident = validate_neb_endpoints(li_init, li_init.copy())
    assert ok_ident is False
    assert any("identical" in e for e in err_ident)


def test_cli_main_exit_codes(tmp_path: Path) -> None:
    valid_path = tmp_path / "si.extxyz"
    si = ase.build.bulk("Si", "diamond", a=5.43, cubic=True)
    ase.io.write(str(valid_path), si, format="extxyz")

    rc_ok = main([str(valid_path), "--require-pbc", "--json"])
    assert rc_ok == 0

    bad_path = tmp_path / "bad.extxyz"
    bad_atoms = Atoms(
        "He2",
        positions=[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    ase.io.write(str(bad_path), bad_atoms, format="extxyz")

    rc_fail = main([str(bad_path)])
    assert rc_fail == 1
