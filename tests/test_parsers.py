"""Unit tests for janus_marimo.parsers across all 8 janus-core calculation modes."""

from __future__ import annotations

from pathlib import Path

import ase.build
import ase.io
import numpy as np
import yaml

from janus_marimo.parsers import (
    parse_descriptors_run,
    parse_elasticity_run,
    parse_eos_run,
    parse_geomopt_run,
    parse_md_run,
    parse_neb_run,
    parse_phonons_run,
    parse_singlepoint_run,
)


def test_parse_singlepoint_run(tmp_path: Path) -> None:
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.info["mace_mp_energy"] = -27.2
    atoms.info["mace_mp_stress"] = [0.01, 0.01, 0.01, 0.0, 0.0, 0.0]
    atoms.arrays["mace_mp_forces"] = np.full((len(atoms), 3), 0.03)
    ase.io.write(str(tmp_path / "job-results.extxyz"), atoms, format="extxyz")

    parsed = parse_singlepoint_run(tmp_path, "mace_mp")
    assert abs(parsed.energy_ev - (-27.2)) < 1e-6
    assert abs(parsed.energy_per_atom_ev - (-27.2 / 8.0)) < 1e-6
    assert parsed.stress_voigt_ev_ang3 is not None
    assert len(parsed.stress_voigt_ev_ang3) == 6
    assert len(parsed.atom_df) == 8
    assert list(parsed.atom_df.columns) == [
        "atom_index",
        "symbol",
        "x",
        "y",
        "z",
        "fx",
        "fy",
        "fz",
        "force_norm",
    ]


def test_parse_geomopt_run(tmp_path: Path) -> None:
    f0 = ase.build.bulk("Cu", "fcc", a=3.65, cubic=True)
    f0.info["mace_mp_energy"] = -14.0
    f0.arrays["mace_mp_forces"] = np.full((len(f0), 3), 0.05)

    f1 = ase.build.bulk("Cu", "fcc", a=3.61, cubic=True)
    f1.info["mace_mp_energy"] = -14.8
    f1.arrays["mace_mp_forces"] = np.full((len(f1), 3), 0.002)

    ase.io.write(str(tmp_path / "input.extxyz"), f0, format="extxyz")
    ase.io.write(str(tmp_path / "job-traj.extxyz"), [f0, f1], format="extxyz")
    ase.io.write(str(tmp_path / "job-opt.extxyz"), f1, format="extxyz")

    parsed = parse_geomopt_run(tmp_path, "mace_mp", fmax_target=0.01)
    assert parsed.converged is True
    assert len(parsed.trajectory) == 2
    assert len(parsed.traj_df) == 2
    assert parsed.traj_df["delta_energy_mev_atom"].iloc[-1] < 0.0


def test_parse_md_run(tmp_path: Path) -> None:
    atoms = ase.build.bulk("Si", "diamond", a=5.43, cubic=True)
    ase.io.write(str(tmp_path / "job-traj.extxyz"), [atoms, atoms], format="extxyz")
    ase.io.write(str(tmp_path / "job-final.extxyz"), atoms, format="extxyz")

    stats_text = (
        "# Step | Real_Time [s] | Time [fs] | Epot/N [eV] | EKin/N [eV] | T [K] | "
        "ETot/N [eV] | Density [g/cm^3] | Volume [Å^3] | P [GPa] | Pxx | Pyy | Pzz | Pyz | Pxz | Pxy\n"
        "0 0.01 0.0 -5.40 0.038 300.0 -5.362 2.33 160.1 0.1 0 0 0 0 0 0\n"
        "10 0.02 10.0 -5.39 0.039 305.0 -5.351 2.33 160.1 0.2 0 0 0 0 0 0\n"
    )
    (tmp_path / "job-stats.dat").write_text(stats_text)
    (tmp_path / "job-rdf.dat").write_text("# r g(r)\n2.35 3.1\n3.84 1.2\n")

    parsed = parse_md_run(tmp_path, "mace_mp")
    assert len(parsed.trajectory) == 2
    assert len(parsed.stats_df) == 2
    assert list(parsed.stats_df["step"]) == [0, 10]
    assert abs(float(parsed.stats_df["epot_ev"].iloc[0]) - (-5.40 * 8)) < 1e-5
    assert parsed.rdf_df is not None and len(parsed.rdf_df) == 2


def test_parse_phonons_run(tmp_path: Path) -> None:
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    ase.io.write(str(tmp_path / "input.extxyz"), atoms, format="extxyz")

    bands_data = {
        "phonon": [
            {
                "distance": 0.0,
                "label": "Gamma",
                "band": [{"frequency": -0.12}, {"frequency": 2.4}],
            },
            {
                "distance": 0.5,
                "label": "X",
                "band": [{"frequency": 1.1}, {"frequency": 4.2}],
            },
        ]
    }
    (tmp_path / "job-auto_bands.yml").write_text(yaml.safe_dump(bands_data))
    (tmp_path / "job-dos.dat").write_text("# freq dos\n0.0 0.0\n2.0 1.5\n")
    thermal_data = {
        "thermal_properties": [
            {
                "temperature": 300.0,
                "free_energy": -2.5,
                "entropy": 45.0,
                "heat_capacity": 48.2,
            }
        ]
    }
    (tmp_path / "job-thermal.yml").write_text(yaml.safe_dump(thermal_data))

    parsed = parse_phonons_run(tmp_path, "mace_mp")
    assert parsed.bands_df is not None and len(parsed.bands_df) == 4
    assert parsed.dos_df is not None and len(parsed.dos_df) == 2
    assert parsed.thermal_df is not None and len(parsed.thermal_df) == 1
    assert parsed.has_imaginary_modes is True
    assert parsed.min_frequency_thz == -0.12
    assert len(parsed.band_structures) == 2


def test_parse_eos_run(tmp_path: Path) -> None:
    atoms = ase.build.bulk("Si", "diamond", a=5.43, cubic=True)
    ase.io.write(str(tmp_path / "job-generated.extxyz"), [atoms], format="extxyz")

    raw_text = (
        "#Lattice Scalar | Energy [eV] | Volume [Å^3]\n"
        "0.98 -43.10 150.0\n"
        "0.99 -43.25 155.0\n"
        "1.00 -43.30 160.0\n"
        "1.01 -43.24 165.0\n"
        "1.02 -43.08 170.0\n"
    )
    (tmp_path / "job-eos-raw.dat").write_text(raw_text)
    (tmp_path / "job-eos-fit.dat").write_text(
        "#Bulk modulus [GPa] | Energy [eV] | Volume [Å^3]\n92.4 -43.30 160.0\n"
    )

    parsed = parse_eos_run(tmp_path, "mace_mp", eos_type="birchmurnaghan")
    assert len(parsed.raw_df) == 5
    assert len(parsed.fit_curve_df) == 100
    assert abs(parsed.bulk_modulus_gpa - 92.4) < 1e-6
    assert abs(parsed.v0_ang3 - 160.0) < 1e-6
    assert abs(parsed.e0_ev - (-43.30)) < 1e-6


def test_parse_elasticity_run(tmp_path: Path) -> None:
    atoms = ase.build.bulk("Cu", "fcc", a=3.61, cubic=True)
    ase.io.write(str(tmp_path / "input.extxyz"), atoms, format="extxyz")

    c_ij = np.diag([170.0, 170.0, 170.0, 75.0, 75.0, 75.0])
    c_ij[0, 1] = c_ij[1, 0] = c_ij[0, 2] = c_ij[2, 0] = c_ij[1, 2] = c_ij[2, 1] = 120.0
    nine_scalars = [136.0, 136.0, 136.0, 42.0, 55.0, 48.5, 130.0, 1.5, 0.34]
    row_45 = nine_scalars + c_ij.ravel().tolist()
    tensor_text = "# Elastic properties\n" + " ".join(f"{v:.4f}" for v in row_45) + "\n"
    (tmp_path / "job-elastic_tensor.dat").write_text(tensor_text)

    parsed = parse_elasticity_run(tmp_path, "mace_mp")
    assert len(parsed.c_ij_df) == 36
    assert len(parsed.moduli_df) == 7
    assert parsed.is_mechanically_stable is True
    assert abs(parsed.poisson_ratio - 0.34) < 1e-4


def test_parse_neb_run(tmp_path: Path) -> None:
    imgs = []
    energies = [-40.0, -39.4, -38.8, -39.5, -40.0]
    for idx, ene in enumerate(energies):
        a = ase.build.bulk("Li", "bcc", a=3.49, cubic=True)
        a.positions[0, 0] += 0.2 * idx
        a.info["mace_mp_energy"] = ene
        a.arrays["mace_mp_forces"] = np.full((len(a), 3), 0.01)
        imgs.append(a)

    ase.io.write(str(tmp_path / "job-neb-band.extxyz"), imgs, format="extxyz")
    (tmp_path / "job-neb-results.dat").write_text(
        "#Barrier [eV] | delta E [eV] | Max force [eV/Å]\n1.20 0.00 0.017\n"
    )

    parsed = parse_neb_run(tmp_path, "mace_mp")
    assert len(parsed.band_images) == 5
    assert parsed.saddle_image_index == 2
    assert abs(parsed.barrier_ev - 1.20) < 1e-6
    assert len(parsed.image_df) == 5

    # Verify MIC handling when an atom wraps across a periodic boundary
    img0 = ase.build.bulk("Li", "bcc", a=3.49, cubic=True)
    img1 = img0.copy()
    img1.positions[0, 0] += 3.49 - 0.1  # Wrapped across cell boundary (true MIC step is 0.1 Å)
    ase.io.write(str(tmp_path / "job-neb-band.extxyz"), [img0, img1], format="extxyz")
    parsed_mic = parse_neb_run(tmp_path, "mace_mp")
    assert abs(float(parsed_mic.image_df["rxn_coord_ang"].iloc[-1]) - 0.1) < 1e-4


def test_parse_descriptors_run(tmp_path: Path) -> None:
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.info["mace_mp_descriptor"] = -0.42
    atoms.info["mace_mp_Na_descriptor"] = -0.35
    atoms.info["mace_mp_Cl_descriptor"] = -0.49
    atoms.arrays["mace_mp_descriptors"] = np.linspace(-0.5, -0.3, len(atoms))
    ase.io.write(str(tmp_path / "job-descriptors.extxyz"), atoms, format="extxyz")

    parsed = parse_descriptors_run(tmp_path, "mace_mp")
    assert parsed.mean_descriptor is not None
    assert abs(parsed.mean_descriptor - (-0.42)) < 1e-6
    assert "Na" in parsed.element_descriptors
    assert "Cl" in parsed.element_descriptors
    assert len(parsed.atom_df) == 8

    # Element-only descriptors without structure-mean key
    atoms_el_only = ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms_el_only.info["mace_mp_Na_descriptor"] = -0.35
    atoms_el_only.info["mace_mp_Cl_descriptor"] = -0.49
    ase.io.write(str(tmp_path / "job-descriptors.extxyz"), atoms_el_only, format="extxyz")
    parsed_el = parse_descriptors_run(tmp_path, "mace_mp")
    assert parsed_el.mean_descriptor is None
    assert "Na" in parsed_el.element_descriptors
    assert "Cl" in parsed_el.element_descriptors
