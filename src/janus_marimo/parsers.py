"""Artifact parsers for all 8 janus-core calculation modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ase.geometry
import ase.io
import numpy as np
import pandas as pd
import yaml
from ase import Atoms
from ase.eos import EquationOfState


def _read_atoms_list(path: Path) -> list[Atoms]:
    """Read all frames from an ASE-compatible trajectory/structure file."""
    loaded = ase.io.read(str(path), index=":")
    if isinstance(loaded, Atoms):
        return [loaded]
    return list(loaded)


def _extract_energy(atoms: Atoms, arch: str) -> float:
    """Extract scalar potential energy (eV) from janus-core namespaced atoms.info or calc."""
    candidate_keys = [
        f"{arch}_d3_energy",
        f"{arch}_energy",
        "energy",
        "free_energy",
    ]
    for key in candidate_keys:
        if key in atoms.info and atoms.info[key] is not None:
            try:
                return float(atoms.info[key])
            except (TypeError, ValueError):
                pass

    for key, val in atoms.info.items():
        if key.endswith("_energy") and val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass

    if atoms.calc is not None and hasattr(atoms.calc, "results"):
        val = atoms.calc.results.get("energy")
        if val is not None:
            return float(val)

    return 0.0


def _extract_forces(atoms: Atoms, arch: str) -> np.ndarray:
    """Extract (N, 3) atomic forces (eV/Å) from janus-core namespaced atoms.arrays or calc."""
    n_atoms = len(atoms)
    if n_atoms == 0:
        return np.zeros((0, 3), dtype=float)

    candidate_keys = [
        f"{arch}_d3_forces",
        f"{arch}_forces",
        "forces",
    ]
    for key in candidate_keys:
        if key in atoms.arrays:
            arr = np.asarray(atoms.arrays[key], dtype=float)
            if arr.shape == (n_atoms, 3):
                return arr

    for key, val in atoms.arrays.items():
        if key.endswith("_forces"):
            arr = np.asarray(val, dtype=float)
            if arr.shape == (n_atoms, 3):
                return arr

    if atoms.calc is not None and hasattr(atoms.calc, "results"):
        val = atoms.calc.results.get("forces")
        if val is not None:
            arr = np.asarray(val, dtype=float)
            if arr.shape == (n_atoms, 3):
                return arr

    return np.zeros((n_atoms, 3), dtype=float)


def _extract_stress(atoms: Atoms, arch: str) -> list[float] | None:
    """Extract 6-component Voigt stress tensor (eV/Å^3) from atoms.info or calc."""
    candidate_keys = [
        f"{arch}_d3_stress",
        f"{arch}_stress",
        "stress",
    ]
    raw_stress: Any = None
    for key in candidate_keys:
        if key in atoms.info and atoms.info[key] is not None:
            raw_stress = atoms.info[key]
            break

    if raw_stress is None:
        for key, val in atoms.info.items():
            if key.endswith("_stress") and val is not None:
                raw_stress = val
                break

    if raw_stress is None and atoms.calc is not None and hasattr(atoms.calc, "results"):
        raw_stress = atoms.calc.results.get("stress")

    if raw_stress is None:
        return None

    arr = np.asarray(raw_stress, dtype=float)
    if arr.shape == (6,):
        return [float(x) for x in arr]
    if arr.shape == (3, 3):
        return [
            float(arr[0, 0]),
            float(arr[1, 1]),
            float(arr[2, 2]),
            float(arr[1, 2]),
            float(arr[0, 2]),
            float(arr[0, 1]),
        ]
    flat = arr.ravel()
    if flat.size == 6:
        return [float(x) for x in flat]
    if flat.size == 9:
        mat = flat.reshape((3, 3))
        return [
            float(mat[0, 0]),
            float(mat[1, 1]),
            float(mat[2, 2]),
            float(mat[1, 2]),
            float(mat[0, 2]),
            float(mat[0, 1]),
        ]
    return None


@dataclass(frozen=True)
class ParsedSinglePoint:
    """Parsed results from a `janus singlepoint` calculation."""

    structures: list[Atoms]
    energy_ev: float
    energy_per_atom_ev: float
    max_force_ev_ang: float
    stress_voigt_ev_ang3: list[float] | None
    atom_df: pd.DataFrame


@dataclass(frozen=True)
class ParsedGeomOpt:
    """Parsed results from a `janus geomopt` calculation."""

    initial_atoms: Atoms
    optimized_atoms: Atoms
    trajectory: list[Atoms]
    traj_df: pd.DataFrame
    converged: bool


@dataclass(frozen=True)
class ParsedMD:
    """Parsed results from a `janus md` simulation."""

    trajectory: list[Atoms]
    final_atoms: Atoms
    stats_df: pd.DataFrame
    rdf_df: pd.DataFrame | None


@dataclass(frozen=True)
class ParsedPhonons:
    """Parsed results from a `janus phonons` calculation."""

    bands_df: pd.DataFrame | None
    dos_df: pd.DataFrame | None
    thermal_df: pd.DataFrame | None
    has_imaginary_modes: bool
    min_frequency_thz: float | None
    band_structures: list[Atoms]


@dataclass(frozen=True)
class ParsedEOS:
    """Parsed results from a `janus eos` calculation."""

    raw_df: pd.DataFrame
    fit_curve_df: pd.DataFrame
    bulk_modulus_gpa: float
    v0_ang3: float
    e0_ev: float
    structures: list[Atoms]


@dataclass(frozen=True)
class ParsedElasticity:
    """Parsed results from a `janus elasticity` calculation."""

    c_ij_matrix: list[list[float]]
    c_ij_df: pd.DataFrame
    moduli_df: pd.DataFrame
    poisson_ratio: float
    universal_anisotropy: float
    is_mechanically_stable: bool
    structures: list[Atoms]


@dataclass(frozen=True)
class ParsedNEB:
    """Parsed results from a `janus neb` calculation."""

    band_images: list[Atoms]
    barrier_ev: float
    delta_e_ev: float
    max_force_ev_ang: float
    image_df: pd.DataFrame
    saddle_image_index: int


@dataclass(frozen=True)
class ParsedDescriptors:
    """Parsed results from a `janus descriptors` calculation."""

    structures: list[Atoms]
    mean_descriptor: float | None
    element_descriptors: dict[str, float]
    atom_df: pd.DataFrame


def parse_singlepoint_run(run_dir: Path, arch: str) -> ParsedSinglePoint:
    """Parse `job-results.extxyz` from a singlepoint run directory."""
    results_path = run_dir / "job-results.extxyz"
    if not results_path.exists():
        candidates = list(run_dir.glob("*-results.extxyz"))
        results_path = candidates[0] if candidates else (run_dir / "input.extxyz")

    structures = _read_atoms_list(results_path)
    atoms = structures[0]
    n_atoms = max(len(atoms), 1)
    energy_ev = _extract_energy(atoms, arch)
    energy_per_atom_ev = energy_ev / n_atoms
    forces = _extract_forces(atoms, arch)
    force_norms = np.linalg.norm(forces, axis=1) if len(atoms) > 0 else np.zeros(0)
    max_force_ev_ang = float(force_norms.max()) if force_norms.size > 0 else 0.0
    stress_voigt = _extract_stress(atoms, arch)

    pos = atoms.positions
    atom_df = pd.DataFrame(
        {
            "atom_index": list(range(len(atoms))),
            "symbol": atoms.get_chemical_symbols(),
            "x": pos[:, 0].astype(float),
            "y": pos[:, 1].astype(float),
            "z": pos[:, 2].astype(float),
            "fx": forces[:, 0].astype(float),
            "fy": forces[:, 1].astype(float),
            "fz": forces[:, 2].astype(float),
            "force_norm": force_norms.astype(float),
        }
    )

    return ParsedSinglePoint(
        structures=structures,
        energy_ev=energy_ev,
        energy_per_atom_ev=energy_per_atom_ev,
        max_force_ev_ang=max_force_ev_ang,
        stress_voigt_ev_ang3=stress_voigt,
        atom_df=atom_df,
    )


def parse_geomopt_run(run_dir: Path, arch: str, fmax_target: float = 0.01) -> ParsedGeomOpt:
    """Parse `job-opt.extxyz` and `job-traj.extxyz` from a geomopt run directory."""
    opt_path = run_dir / "job-opt.extxyz"
    if not opt_path.exists():
        candidates = list(run_dir.glob("*-opt.extxyz"))
        opt_path = candidates[0] if candidates else (run_dir / "input.extxyz")

    opt_list = _read_atoms_list(opt_path)
    optimized_atoms = opt_list[-1]

    traj_path = run_dir / "job-traj.extxyz"
    if not traj_path.exists():
        traj_candidates = list(run_dir.glob("*-traj.extxyz"))
        if traj_candidates:
            traj_path = traj_candidates[0]

    if traj_path.exists():
        trajectory = _read_atoms_list(traj_path)
    else:
        trajectory = [optimized_atoms]

    input_path = run_dir / "input.extxyz"
    if input_path.exists():
        initial_atoms = _read_atoms_list(input_path)[-1]
    else:
        initial_atoms = trajectory[0]

    e0 = _extract_energy(trajectory[0], arch)
    rows: list[dict[str, float | int]] = []
    for step_idx, frame in enumerate(trajectory):
        n_atoms = max(len(frame), 1)
        e = _extract_energy(frame, arch)
        forces = _extract_forces(frame, arch)
        fnorms = np.linalg.norm(forces, axis=1) if len(frame) > 0 else np.zeros(0)
        if fnorms.size > 0 and float(fnorms.max()) > 0.0:
            fmax = float(fnorms.max())
        elif "max_force" in frame.info and frame.info["max_force"] is not None:
            fmax = float(frame.info["max_force"])
        else:
            fmax = 0.0
        vol = (
            float(frame.get_volume())
            if any(frame.pbc) and abs(float(np.linalg.det(frame.cell.array))) > 1e-8
            else 0.0
        )
        rows.append(
            {
                "step": int(step_idx),
                "energy_ev": float(e),
                "delta_energy_mev_atom": float((e - e0) * 1000.0 / n_atoms),
                "max_force_ev_ang": fmax,
                "volume_ang3": vol,
            }
        )

    traj_df = pd.DataFrame(rows)
    if "converged" in optimized_atoms.info and isinstance(optimized_atoms.info["converged"], bool):
        converged = bool(optimized_atoms.info["converged"])
    else:
        converged = bool(float(traj_df["max_force_ev_ang"].iloc[-1]) <= fmax_target)

    return ParsedGeomOpt(
        initial_atoms=initial_atoms,
        optimized_atoms=optimized_atoms,
        trajectory=trajectory,
        traj_df=traj_df,
        converged=converged,
    )


def _parse_md_stats_file(stats_path: Path, n_atoms: int) -> pd.DataFrame:
    """Parse janus-core MD stats_file into standardized thermodynamic columns."""
    lines = [line.strip() for line in stats_path.read_text().splitlines() if line.strip()]
    header_line = ""
    data_rows: list[list[float]] = []
    for line in lines:
        if line.startswith("#"):
            header_line = line.lstrip("#").strip()
        else:
            parts = line.split()
            try:
                data_rows.append([float(x) for x in parts])
            except ValueError:
                continue

    raw_cols: list[str] = []
    if header_line:
        for token in header_line.split("|"):
            clean_name = token.strip().split("[")[0].strip()
            if clean_name:
                raw_cols.append(clean_name)

    if not data_rows:
        return pd.DataFrame(
            columns=[
                "step",
                "time_fs",
                "temp_k",
                "epot_ev",
                "ekin_ev",
                "etot_ev",
                "pressure_gpa",
                "volume_ang3",
            ]
        )

    arr = np.asarray(data_rows, dtype=float)
    col_map = {name: idx for idx, name in enumerate(raw_cols) if idx < arr.shape[1]}

    def _col_or_default(names: tuple[str, ...], default_idx: int, scale: float = 1.0) -> np.ndarray:
        for name in names:
            if name in col_map:
                return arr[:, col_map[name]] * scale
        if default_idx < arr.shape[1]:
            return arr[:, default_idx] * scale
        return np.zeros(arr.shape[0], dtype=float)

    scale_n = float(max(n_atoms, 1))
    step_col = _col_or_default(("Step", "step"), 0).astype(int)
    time_col = _col_or_default(("Time", "time_fs", "time"), 2)
    epot_col = (
        arr[:, col_map["Epot/N"]] * scale_n
        if "Epot/N" in col_map
        else _col_or_default(("Epot", "epot_ev"), 3)
    )
    ekin_col = (
        arr[:, col_map["EKin/N"]] * scale_n
        if "EKin/N" in col_map
        else _col_or_default(("EKin", "ekin_ev"), 4)
    )
    temp_col = _col_or_default(("T", "Temperature", "temp_k"), 5)
    etot_col = (
        arr[:, col_map["ETot/N"]] * scale_n
        if "ETot/N" in col_map
        else _col_or_default(("ETot", "etot_ev"), 6)
    )
    vol_col = _col_or_default(("Volume", "volume_ang3"), 8)
    press_col = _col_or_default(("P", "Pressure", "pressure_gpa"), 9)

    return pd.DataFrame(
        {
            "step": step_col,
            "time_fs": time_col,
            "temp_k": temp_col,
            "epot_ev": epot_col,
            "ekin_ev": ekin_col,
            "etot_ev": etot_col,
            "pressure_gpa": press_col,
            "volume_ang3": vol_col,
        }
    )


def parse_md_run(run_dir: Path, arch: str) -> ParsedMD:
    """Parse `job-traj.extxyz`, `job-final.extxyz`, `job-stats.dat`, and optional RDF files."""
    traj_path = run_dir / "job-traj.extxyz"
    if not traj_path.exists():
        candidates = list(run_dir.glob("*-traj.extxyz"))
        traj_path = candidates[0] if candidates else (run_dir / "input.extxyz")

    trajectory = _read_atoms_list(traj_path)

    final_path = run_dir / "job-final.extxyz"
    if final_path.exists():
        final_atoms = _read_atoms_list(final_path)[-1]
    else:
        final_atoms = trajectory[-1]

    stats_path = run_dir / "job-stats.dat"
    if not stats_path.exists():
        stats_candidates = list(run_dir.glob("*-stats.dat"))
        if stats_candidates:
            stats_path = stats_candidates[0]

    if stats_path.exists():
        stats_df = _parse_md_stats_file(stats_path, len(final_atoms))
    else:
        rows = []
        for idx, frame in enumerate(trajectory):
            e = _extract_energy(frame, arch)
            vol = float(frame.get_volume()) if any(frame.pbc) else 0.0
            rows.append(
                {
                    "step": idx,
                    "time_fs": float(idx),
                    "temp_k": float(frame.info.get("temperature", 300.0)),
                    "epot_ev": e,
                    "ekin_ev": 0.0,
                    "etot_ev": e,
                    "pressure_gpa": 0.0,
                    "volume_ang3": vol,
                }
            )
        stats_df = pd.DataFrame(rows)

    rdf_dfs: list[pd.DataFrame] = []
    for rdf_file in sorted(run_dir.glob("*rdf*.dat")):
        pair_label = (
            rdf_file.stem.replace("job-", "").replace("-rdf", "").replace("rdf", "").strip("-_")
        )
        if not pair_label:
            pair_label = "total"
        rdf_rows: list[dict[str, float | str]] = []
        for line in rdf_file.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    rdf_rows.append(
                        {
                            "r_ang": float(parts[0]),
                            "g_r": float(parts[1]),
                            "pair": pair_label,
                        }
                    )
                except ValueError:
                    continue
        if rdf_rows:
            rdf_dfs.append(pd.DataFrame(rdf_rows))

    rdf_df = pd.concat(rdf_dfs, ignore_index=True) if rdf_dfs else None

    return ParsedMD(
        trajectory=trajectory,
        final_atoms=final_atoms,
        stats_df=stats_df,
        rdf_df=rdf_df,
    )


def parse_phonons_run(run_dir: Path, arch: str) -> ParsedPhonons:
    """Parse `job-auto_bands.yml`, `job-dos.dat`, and `job-thermal.yml` from a phonons run."""
    bands_df: pd.DataFrame | None = None
    bands_path = run_dir / "job-auto_bands.yml"
    if not bands_path.exists():
        band_candidates = list(run_dir.glob("*bands*.yml")) + list(run_dir.glob("*bands*.yaml"))
        if band_candidates:
            bands_path = band_candidates[0]

    if bands_path.exists():
        bands_data = yaml.safe_load(bands_path.read_text()) or {}
        phonon_points = bands_data.get("phonon", [])
        band_rows: list[dict[str, Any]] = []
        for q_idx, qpt in enumerate(phonon_points):
            dist = float(qpt.get("distance", q_idx))
            q_label = str(qpt.get("label") or "")
            for b_idx, band in enumerate(qpt.get("band", [])):
                freq = float(band.get("frequency", 0.0))
                band_rows.append(
                    {
                        "q_index": int(q_idx),
                        "distance": dist,
                        "band_index": int(b_idx),
                        "frequency_thz": freq,
                        "q_label": q_label,
                    }
                )
        if band_rows:
            bands_df = pd.DataFrame(band_rows)

    dos_df: pd.DataFrame | None = None
    dos_path = run_dir / "job-dos.dat"
    if not dos_path.exists():
        dos_candidates = list(run_dir.glob("*dos.dat"))
        if dos_candidates:
            dos_path = dos_candidates[0]

    if dos_path.exists():
        dos_rows: list[dict[str, float]] = []
        for line in dos_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    dos_rows.append(
                        {
                            "frequency_thz": float(parts[0]),
                            "dos": float(parts[1]),
                        }
                    )
                except ValueError:
                    continue
        if dos_rows:
            dos_df = pd.DataFrame(dos_rows)

    thermal_df: pd.DataFrame | None = None
    thermal_path = run_dir / "job-thermal.yml"
    if not thermal_path.exists():
        thermal_candidates = list(run_dir.glob("*thermal*.yml")) + list(
            run_dir.glob("*thermal*.yaml")
        )
        if thermal_candidates:
            thermal_path = thermal_candidates[0]

    if thermal_path.exists():
        thermal_data = yaml.safe_load(thermal_path.read_text()) or {}
        tp_list = thermal_data.get("thermal_properties", [])
        thermal_rows: list[dict[str, float]] = []
        for entry in tp_list:
            thermal_rows.append(
                {
                    "temperature_k": float(entry.get("temperature", 0.0)),
                    "free_energy_kj_mol": float(entry.get("free_energy", 0.0)),
                    "entropy_j_k_mol": float(entry.get("entropy", 0.0)),
                    "heat_capacity_j_k_mol": float(entry.get("heat_capacity", 0.0)),
                }
            )
        if thermal_rows:
            thermal_df = pd.DataFrame(thermal_rows)

    min_frequency_thz: float | None = None
    if bands_df is not None and not bands_df.empty:
        min_frequency_thz = float(bands_df["frequency_thz"].min())
    elif dos_df is not None and not dos_df.empty:
        min_frequency_thz = float(dos_df["frequency_thz"].min())

    has_imaginary_modes = bool(min_frequency_thz is not None and min_frequency_thz < -0.05)

    band_structures: list[Atoms] = []
    input_path = run_dir / "input.extxyz"
    if input_path.exists():
        unit_atoms = _read_atoms_list(input_path)[-1]
        band_structures = [unit_atoms, unit_atoms * (2, 2, 2)]

    return ParsedPhonons(
        bands_df=bands_df,
        dos_df=dos_df,
        thermal_df=thermal_df,
        has_imaginary_modes=has_imaginary_modes,
        min_frequency_thz=min_frequency_thz,
        band_structures=band_structures,
    )


def parse_eos_run(run_dir: Path, arch: str, eos_type: str = "birchmurnaghan") -> ParsedEOS:
    """Parse `job-eos-raw.dat`, `job-eos-fit.dat`, and `job-generated.extxyz`."""
    raw_path = run_dir / "job-eos-raw.dat"
    if not raw_path.exists():
        candidates = list(run_dir.glob("*-eos-raw.dat"))
        if candidates:
            raw_path = candidates[0]

    raw_rows: list[dict[str, float]] = []
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    raw_rows.append(
                        {
                            "lattice_scalar": float(parts[0]),
                            "energy_ev": float(parts[1]),
                            "volume_ang3": float(parts[2]),
                        }
                    )
                except ValueError:
                    continue

    raw_df = pd.DataFrame(raw_rows, columns=["lattice_scalar", "volume_ang3", "energy_ev"])

    fit_path = run_dir / "job-eos-fit.dat"
    if not fit_path.exists():
        fit_candidates = list(run_dir.glob("*-eos-fit.dat"))
        if fit_candidates:
            fit_path = fit_candidates[0]

    bulk_modulus_gpa = 0.0
    e0_ev = float(raw_df["energy_ev"].min()) if not raw_df.empty else 0.0
    v0_ang3 = float(raw_df["volume_ang3"].mean()) if not raw_df.empty else 0.0

    if fit_path.exists():
        for line in fit_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    bulk_modulus_gpa = float(parts[0])
                    e0_ev = float(parts[1])
                    v0_ang3 = float(parts[2])
                    break
                except ValueError:
                    continue

    if len(raw_df) >= 3:
        vols = raw_df["volume_ang3"].to_numpy(dtype=float)
        enes = raw_df["energy_ev"].to_numpy(dtype=float)
        try:
            eos = EquationOfState(vols, enes, eos=eos_type)
            eos.fit()
            plot_data = eos.getplotdata()
            fit_v = np.asarray(plot_data[4], dtype=float)
            fit_e = np.asarray(plot_data[5], dtype=float)
            if fit_v.size != 100:
                v_grid = np.linspace(float(vols.min()), float(vols.max()), 100)
                fit_e = np.interp(v_grid, fit_v, fit_e)
                fit_v = v_grid
        except Exception:
            fit_v = np.linspace(float(vols.min()), float(vols.max()), 100)
            coeffs = np.polyfit(vols, enes, deg=min(2, len(vols) - 1))
            fit_e = np.polyval(coeffs, fit_v)
        fit_curve_df = pd.DataFrame({"volume_ang3": fit_v, "energy_ev": fit_e})
    else:
        fit_curve_df = raw_df[["volume_ang3", "energy_ev"]].copy()

    gen_path = run_dir / "job-generated.extxyz"
    if not gen_path.exists():
        gen_candidates = list(run_dir.glob("*generated.extxyz"))
        if gen_candidates:
            gen_path = gen_candidates[0]
        else:
            gen_path = run_dir / "input.extxyz"

    structures = _read_atoms_list(gen_path) if gen_path.exists() else []

    return ParsedEOS(
        raw_df=raw_df,
        fit_curve_df=fit_curve_df,
        bulk_modulus_gpa=bulk_modulus_gpa,
        v0_ang3=v0_ang3,
        e0_ev=e0_ev,
        structures=structures,
    )


def parse_elasticity_run(run_dir: Path, arch: str) -> ParsedElasticity:
    """Parse `job-elastic_tensor.dat` (45 space-separated floats) and generated structures."""
    tensor_path = run_dir / "job-elastic_tensor.dat"
    if not tensor_path.exists():
        candidates = list(run_dir.glob("*elastic_tensor.dat"))
        if candidates:
            tensor_path = candidates[0]

    values: list[float] = []
    if tensor_path.exists():
        for line in tensor_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values.extend(float(x) for x in stripped.split())

    if len(values) < 45:
        raise ValueError(
            f"Expected 45 floats in elastic_tensor.dat, found {len(values)} in {tensor_path}."
        )

    k_reuss, k_voigt, k_vrh, g_reuss, g_voigt, g_vrh = values[0:6]
    youngs_modulus_gpa = values[6]
    universal_anisotropy = values[7]
    poisson_ratio = values[8]
    c_ij_arr = np.asarray(values[9:45], dtype=float).reshape((6, 6))
    c_ij_matrix = [[float(c_ij_arr[i, j]) for j in range(6)] for i in range(6)]

    labels = ["C1", "C2", "C3", "C4", "C5", "C6"]
    c_ij_rows: list[dict[str, Any]] = []
    for i_idx, i_lbl in enumerate(labels):
        for j_idx, j_lbl in enumerate(labels):
            c_ij_rows.append(
                {
                    "i_label": i_lbl,
                    "j_label": j_lbl,
                    "c_ij_gpa": float(c_ij_arr[i_idx, j_idx]),
                }
            )
    c_ij_df = pd.DataFrame(c_ij_rows)

    moduli_df = pd.DataFrame(
        [
            {"property": "Bulk Modulus (K)", "method": "Reuss", "value_gpa": float(k_reuss)},
            {"property": "Bulk Modulus (K)", "method": "Voigt", "value_gpa": float(k_voigt)},
            {"property": "Bulk Modulus (K)", "method": "VRH", "value_gpa": float(k_vrh)},
            {"property": "Shear Modulus (G)", "method": "Reuss", "value_gpa": float(g_reuss)},
            {"property": "Shear Modulus (G)", "method": "Voigt", "value_gpa": float(g_voigt)},
            {"property": "Shear Modulus (G)", "method": "VRH", "value_gpa": float(g_vrh)},
            {
                "property": "Young's Modulus (E)",
                "method": "VRH",
                "value_gpa": float(youngs_modulus_gpa),
            },
        ]
    )

    sym_c = 0.5 * (c_ij_arr + c_ij_arr.T)
    eigvals = np.linalg.eigvalsh(sym_c)
    is_mechanically_stable = bool(np.all(eigvals > 0.0))

    gen_path = run_dir / "job-elasticity-generated.extxyz"
    if not gen_path.exists():
        gen_candidates = list(run_dir.glob("*generated.extxyz"))
        if gen_candidates:
            gen_path = gen_candidates[0]
        else:
            gen_path = run_dir / "input.extxyz"

    structures = _read_atoms_list(gen_path) if gen_path.exists() else []

    return ParsedElasticity(
        c_ij_matrix=c_ij_matrix,
        c_ij_df=c_ij_df,
        moduli_df=moduli_df,
        poisson_ratio=float(poisson_ratio),
        universal_anisotropy=float(universal_anisotropy),
        is_mechanically_stable=is_mechanically_stable,
        structures=structures,
    )


def parse_neb_run(run_dir: Path, arch: str) -> ParsedNEB:
    """Parse `job-neb-results.dat` and `job-neb-band.extxyz` from an NEB run."""
    res_path = run_dir / "job-neb-results.dat"
    if not res_path.exists():
        candidates = list(run_dir.glob("*-neb-results.dat"))
        if candidates:
            res_path = candidates[0]

    barrier_ev = 0.0
    delta_e_ev = 0.0
    max_force_ev_ang = 0.0
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                barrier_ev = float(parts[0])
                delta_e_ev = float(parts[1])
                max_force_ev_ang = float(parts[2])
                break

    band_path = run_dir / "job-neb-band.extxyz"
    if not band_path.exists():
        band_candidates = list(run_dir.glob("*-neb-band.extxyz"))
        if band_candidates:
            band_path = band_candidates[0]

    band_images = _read_atoms_list(band_path)
    e0 = _extract_energy(band_images[0], arch)
    cum_dist = 0.0
    rows: list[dict[str, float | int]] = []
    for idx, img in enumerate(band_images):
        if idx > 0:
            disp = img.get_positions() - band_images[idx - 1].get_positions()
            if any(img.pbc) and abs(float(np.linalg.det(img.cell.array))) > 1e-8:
                disp, _ = ase.geometry.find_mic(disp, img.cell, pbc=img.pbc)
            cum_dist += float(np.linalg.norm(disp))
        e = _extract_energy(img, arch)
        forces = _extract_forces(img, arch)
        fnorms = np.linalg.norm(forces, axis=1) if len(img) > 0 else np.zeros(0)
        fmax = float(fnorms.max()) if fnorms.size > 0 else 0.0
        rows.append(
            {
                "image_index": int(idx),
                "rxn_coord_ang": float(cum_dist),
                "energy_ev": float(e),
                "rel_energy_ev": float(e - e0),
                "max_force_ev_ang": fmax,
            }
        )

    image_df = pd.DataFrame(rows)
    saddle_image_index = int(image_df["rel_energy_ev"].idxmax())

    return ParsedNEB(
        band_images=band_images,
        barrier_ev=barrier_ev,
        delta_e_ev=delta_e_ev,
        max_force_ev_ang=max_force_ev_ang,
        image_df=image_df,
        saddle_image_index=saddle_image_index,
    )


def parse_descriptors_run(run_dir: Path, arch: str) -> ParsedDescriptors:
    """Parse `job-descriptors.extxyz` from a descriptors run."""
    desc_path = run_dir / "job-descriptors.extxyz"
    if not desc_path.exists():
        candidates = list(run_dir.glob("*-descriptors.extxyz"))
        desc_path = candidates[0] if candidates else (run_dir / "input.extxyz")

    structures = _read_atoms_list(desc_path)
    first = structures[0]
    unique_symbols = set(first.get_chemical_symbols())
    element_suffixes = tuple(f"_{el}_descriptor" for el in unique_symbols)

    mean_descriptor: float | None = None
    primary_key = f"{arch}_descriptor"
    if primary_key in first.info and first.info[primary_key] is not None:
        mean_descriptor = float(first.info[primary_key])
    else:
        for key, val in first.info.items():
            if (
                key.endswith("_descriptor")
                and not key.endswith(element_suffixes)
                and val is not None
            ):
                try:
                    mean_descriptor = float(val)
                    primary_key = key
                    break
                except (TypeError, ValueError):
                    continue

    element_descriptors: dict[str, float] = {}
    for key, val in first.info.items():
        if key.endswith("_descriptor") and key != primary_key and val is not None:
            el_prefix = key.replace(f"{arch}_", "").replace("_descriptor", "")
            if el_prefix in unique_symbols or el_prefix:
                try:
                    element_descriptors[el_prefix] = float(val)
                except (TypeError, ValueError):
                    continue

    atom_rows: list[dict[str, Any]] = []
    for s_idx, atoms in enumerate(structures):
        desc_arr: np.ndarray | None = None
        cand_array_keys = [f"{arch}_descriptors", "descriptors"] + [
            k for k in atoms.arrays if k.endswith("_descriptors")
        ]
        for akey in cand_array_keys:
            if akey in atoms.arrays:
                raw_arr = np.asarray(atoms.arrays[akey], dtype=float)
                if raw_arr.ndim == 1 and raw_arr.shape[0] == len(atoms):
                    desc_arr = raw_arr
                elif raw_arr.ndim >= 2 and raw_arr.shape[0] == len(atoms):
                    desc_arr = np.mean(raw_arr, axis=tuple(range(1, raw_arr.ndim)))
                break

        if desc_arr is None:
            fallback_val = float(mean_descriptor) if mean_descriptor is not None else 0.0
            desc_arr = np.full(len(atoms), fallback_val, dtype=float)

        forces = _extract_forces(atoms, arch)
        fnorms = np.linalg.norm(forces, axis=1) if len(atoms) > 0 else np.zeros(0)
        pos = atoms.positions
        symbols = atoms.get_chemical_symbols()
        for a_idx in range(len(atoms)):
            atom_rows.append(
                {
                    "structure_index": int(s_idx),
                    "atom_index": int(a_idx),
                    "symbol": symbols[a_idx],
                    "x": float(pos[a_idx, 0]),
                    "y": float(pos[a_idx, 1]),
                    "z": float(pos[a_idx, 2]),
                    "descriptor_value": float(desc_arr[a_idx]),
                    "force_norm": float(fnorms[a_idx]) if fnorms.size > a_idx else 0.0,
                }
            )

    atom_df = pd.DataFrame(atom_rows)
    return ParsedDescriptors(
        structures=structures,
        mean_descriptor=mean_descriptor,
        element_descriptors=element_descriptors,
        atom_df=atom_df,
    )
