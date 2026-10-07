"""Structure and periodic boundary condition (PBC) validator for janus-core simulations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import ase.io
import numpy as np
from ase import Atoms


@dataclass(frozen=True)
class StructureReport:
    """Validation report for an ASE Atoms structure prior to running janus-core."""

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
    """Inspect atomic coordinates, periodic boundary conditions, and interatomic distances."""
    warnings: list[str] = []
    errors: list[str] = []

    if isinstance(struct, (str, Path)):
        try:
            loaded = ase.io.read(str(struct), index=-1)
            atoms = loaded[-1] if isinstance(loaded, list) else loaded
        except Exception as exc:
            return StructureReport(
                formula="",
                n_atoms=0,
                pbc=(False, False, False),
                volume=None,
                cell_lengths=(0.0, 0.0, 0.0),
                cell_angles=(0.0, 0.0, 0.0),
                min_distance=None,
                has_overlapping_atoms=False,
                has_non_finite_coords=False,
                is_valid=False,
                warnings=[],
                errors=[f"Failed to read structure file '{struct}': {exc}"],
            )
    else:
        atoms = struct

    n_atoms = len(atoms)
    formula = atoms.get_chemical_formula() if n_atoms > 0 else ""
    pbc_arr = atoms.get_pbc()
    pbc = (bool(pbc_arr[0]), bool(pbc_arr[1]), bool(pbc_arr[2]))

    has_non_finite_coords = bool(n_atoms > 0 and not np.isfinite(atoms.positions).all())
    if n_atoms == 0:
        errors.append("Structure contains 0 atoms.")
    if has_non_finite_coords:
        errors.append("Structure contains NaN or infinite atomic coordinates.")

    cell_arr = np.asarray(atoms.cell.array, dtype=float)
    cell_finite = bool(np.isfinite(cell_arr).all())
    if not cell_finite:
        errors.append("Unit cell vectors contain NaN or infinite values.")

    if cell_finite and any(pbc) and abs(float(np.linalg.det(cell_arr))) > 1e-8:
        volume: float | None = float(atoms.get_volume())
    else:
        volume = None

    if cell_finite:
        lengths = atoms.cell.lengths()
        angles = atoms.cell.angles()
        cell_lengths = (float(lengths[0]), float(lengths[1]), float(lengths[2]))
        cell_angles = (
            float(angles[0]) if np.isfinite(angles[0]) else 0.0,
            float(angles[1]) if np.isfinite(angles[1]) else 0.0,
            float(angles[2]) if np.isfinite(angles[2]) else 0.0,
        )
    else:
        cell_lengths = (0.0, 0.0, 0.0)
        cell_angles = (0.0, 0.0, 0.0)

    min_distance: float | None = None
    if n_atoms > 1 and not has_non_finite_coords and cell_finite:
        use_mic = bool(any(pbc) and volume is not None and volume > 0.0)
        dist_matrix = np.asarray(atoms.get_all_distances(mic=use_mic), dtype=float)
        np.fill_diagonal(dist_matrix, np.inf)
        min_distance = float(np.min(dist_matrix))

    has_overlapping_atoms = bool(min_distance is not None and min_distance < min_dist_threshold)
    if has_overlapping_atoms and min_distance is not None:
        errors.append(
            f"Overlapping atoms detected: min distance {min_distance:.3f} Å < {min_dist_threshold:.3f} Å."
        )
    elif min_distance is not None and min_dist_threshold <= min_distance < 0.8:
        warnings.append(f"Short interatomic distance detected ({min_distance:.3f} Å < 0.800 Å).")

    if not all(pbc):
        warnings.append("Non-periodic boundary conditions detected in at least one direction.")

    if require_pbc and (not all(pbc) or volume is None or volume <= 0.0):
        errors.append(
            "3D periodic boundary conditions (PBC: True, True, True) and non-zero cell volume are required for this calculation."
        )

    return StructureReport(
        formula=formula,
        n_atoms=n_atoms,
        pbc=pbc,
        volume=volume,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
        min_distance=min_distance,
        has_overlapping_atoms=has_overlapping_atoms,
        has_non_finite_coords=has_non_finite_coords,
        is_valid=(len(errors) == 0),
        warnings=warnings,
        errors=errors,
    )


def validate_neb_endpoints(
    init_atoms: Atoms,
    final_atoms: Atoms | None,
    *,
    min_dist_threshold: float = 0.5,
) -> tuple[bool, list[str]]:
    """Validate initial and final NEB structures pre-flight."""
    if final_atoms is None:
        return (
            False,
            ["NEB requires a final structure endpoint (`final_struct_entry`)."],
        )

    errors: list[str] = []
    init_report = inspect_structure(init_atoms, min_dist_threshold=min_dist_threshold)
    for err in init_report.errors:
        errors.append(f"Initial structure: {err}")

    final_report = inspect_structure(final_atoms, min_dist_threshold=min_dist_threshold)
    for err in final_report.errors:
        errors.append(f"Final structure: {err}")

    if len(init_atoms) != len(final_atoms):
        errors.append(
            f"NEB endpoint atom count mismatch: initial has {len(init_atoms)} atoms, "
            f"final has {len(final_atoms)} atoms."
        )
    else:
        if list(init_atoms.numbers) != list(final_atoms.numbers):
            errors.append(
                "NEB endpoint chemical species/ordering mismatch between initial and final structures."
            )
        if (
            not init_report.has_non_finite_coords
            and not final_report.has_non_finite_coords
            and np.allclose(init_atoms.positions, final_atoms.positions, atol=1e-5)
        ):
            errors.append(
                "NEB initial and final atomic coordinates are identical (zero reaction pathway displacement)."
            )

    return (len(errors) == 0, errors)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for structure inspection."""
    parser = argparse.ArgumentParser(
        description="Inspect atomic coordinates and periodic boundary conditions before janus-core simulations."
    )
    parser.add_argument("structure_path", type=str, help="Path to structure file")
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.5,
        help="Minimum allowed interatomic distance in Å (default: 0.5)",
    )
    parser.add_argument(
        "--require-pbc",
        action="store_true",
        help="Require 3D periodic boundary conditions and non-zero cell volume",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit StructureReport as JSON",
    )

    args = parser.parse_args(argv)
    report = inspect_structure(
        args.structure_path,
        min_dist_threshold=args.min_dist,
        require_pbc=args.require_pbc,
    )

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        status_str = "PASS" if report.is_valid else "FAIL"
        vol_str = f"{report.volume:.3f} Å^3" if report.volume is not None else "N/A"
        dmin_str = f"{report.min_distance:.3f} Å" if report.min_distance is not None else "N/A"
        print(
            f"[{status_str}] Formula: {report.formula} | Atoms: {report.n_atoms} | "
            f"PBC: {report.pbc} | Volume: {vol_str} | d_min: {dmin_str}"
        )
        for warning in report.warnings:
            print(f"  WARNING: {warning}")
        for error in report.errors:
            print(f"  ERROR: {error}")

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
