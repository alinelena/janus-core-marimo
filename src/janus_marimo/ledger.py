"""Workspace ledger, structure relay, and content-addressed run hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import ase.build
import ase.io
import numpy as np
from ase import Atoms

from janus_marimo.inspector import StructureReport, inspect_structure

JanusMode = Literal[
    "singlepoint",
    "geomopt",
    "md",
    "phonons",
    "eos",
    "elasticity",
    "neb",
    "descriptors",
]


@dataclass(frozen=True)
class MLIPConfig:
    """Active MLIP calculator configuration."""

    arch: str = "mace_mp"
    model: str | None = "small"
    device: Literal["cpu", "cuda", "mps", "xpu"] = "cpu"
    dispersion: bool = False
    calc_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_env_path: str | None = None

    def to_calc_kwargs(self) -> dict[str, Any]:
        """Return a shallow copy of `calc_kwargs` merged with `{"dispersion": True}` when `self.dispersion` is True."""
        merged = dict(self.calc_kwargs)
        if self.dispersion:
            merged["dispersion"] = True
        return merged


@dataclass(frozen=True)
class StructureEntry:
    """Structure stored in the Cross-Tab Structure Relay Ledger."""

    entry_id: str
    label: str
    atoms: Atoms
    source_mode: str
    is_relaxed: bool
    report: StructureReport
    provenance_run_id: str | None = None


@dataclass(frozen=True)
class RunRecord:
    """Record of a completed, cached, or failed janus-core calculation."""

    run_id: str
    mode: JanusMode
    mlip_config: MLIPConfig
    struct_entry_id: str
    final_struct_entry_id: str | None
    mode_params: dict[str, Any]
    run_dir: Path
    config_yaml_path: Path
    summary_yaml_path: Path
    output_files: dict[str, Any]
    cli_command: list[str]
    status: Literal["cached", "succeeded", "failed"]
    duration_s: float
    stdout_stderr: str


def _serialize_atoms_for_hash(atoms: Atoms) -> dict[str, Any]:
    """Serialize an ASE Atoms object into a deterministic JSON-compatible dict."""
    return {
        "numbers": atoms.numbers.tolist(),
        "positions": np.round(atoms.positions, 6).tolist(),
        "cell": np.round(atoms.cell.array, 6).tolist(),
        "pbc": [bool(x) for x in atoms.pbc],
    }


def compute_run_id(
    mode: JanusMode,
    mlip_config: MLIPConfig,
    atoms: Atoms,
    mode_params: dict[str, Any],
    final_atoms: Atoms | None = None,
) -> str:
    """Compute a deterministic 16-hex-char SHA-256 digest over calculation inputs."""
    mlip_payload: dict[str, Any] = {
        "arch": mlip_config.arch,
        "model": mlip_config.model,
        "device": mlip_config.device,
        "dispersion": mlip_config.dispersion,
        "calc_kwargs": mlip_config.calc_kwargs,
    }
    if mlip_config.custom_env_path:
        mlip_payload["custom_env_path"] = str(mlip_config.custom_env_path)

    payload_dict = {
        "mode": str(mode),
        "mlip_config": mlip_payload,
        "atoms": _serialize_atoms_for_hash(atoms),
        "final_atoms": (
            _serialize_atoms_for_hash(final_atoms) if final_atoms is not None else None
        ),
        "mode_params": mode_params,
    }
    encoded = json.dumps(payload_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_preset_structures() -> dict[str, Atoms]:
    """Return 5 built-in ASE crystal presets so all 8 tabs (including NEB) work out-of-the-box."""
    nacl = ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    si = ase.build.bulk("Si", "diamond", a=5.43, cubic=True)
    cu = ase.build.bulk("Cu", "fcc", a=3.61, cubic=True)

    li_init = ase.build.bulk("Li", "bcc", a=3.49, cubic=True) * (2, 2, 2)
    del li_init[0]  # Vacancy at (0.0, 0.0, 0.0); atom 0 is now at (1.745, 1.745, 1.745)

    li_final = li_init.copy()
    li_final.positions[0] = np.array([0.0, 0.0, 0.0], dtype=float)

    return {
        "NaCl (rocksalt)": nacl,
        "Si (diamond)": si,
        "Cu (fcc)": cu,
        "Li (bcc vacancy hop - initial)": li_init,
        "Li (bcc vacancy hop - final)": li_final,
    }


@dataclass
class WorkspaceLedger:
    """Registry of structures and calculation runs across all 8 tabs."""

    structures: dict[str, StructureEntry] = field(default_factory=dict)
    runs: dict[str, RunRecord] = field(default_factory=dict)

    @classmethod
    def from_presets(cls) -> WorkspaceLedger:
        """Create a new WorkspaceLedger populated with all built-in preset structures."""
        ledger = cls()
        for name, atoms in build_preset_structures().items():
            ledger.register_structure(
                atoms,
                name=name,
                source_mode="seed",
                is_relaxed=False,
                provenance_run_id=None,
            )
        return ledger

    def register_structure(
        self,
        atoms: Atoms,
        *,
        name: str,
        source_mode: str,
        is_relaxed: bool = False,
        provenance_run_id: str | None = None,
    ) -> StructureEntry:
        """Inspect and register an ASE Atoms structure in the ledger."""
        atoms_copy = atoms.copy()
        report = inspect_structure(atoms_copy)
        idx = len(self.structures)
        entry_id = f"struct_{idx}"
        label = f"#{idx}: {name} ({atoms_copy.get_chemical_formula()}, {len(atoms_copy)} atoms)"
        entry = StructureEntry(
            entry_id=entry_id,
            label=label,
            atoms=atoms_copy,
            source_mode=source_mode,
            is_relaxed=is_relaxed,
            report=report,
            provenance_run_id=provenance_run_id,
        )
        self.structures[entry_id] = entry
        return entry

    def register_or_get_structure(
        self,
        atoms: Atoms,
        *,
        name: str,
        source_mode: str,
        is_relaxed: bool = False,
        provenance_run_id: str | None = None,
    ) -> tuple[StructureEntry, bool]:
        """Return `(existing_entry, False)` if already present, else register and return `(new_entry, True)`."""
        target_hash = _serialize_atoms_for_hash(atoms)
        name_token = f": {name} ("
        for entry in self.structures.values():
            if (
                entry.source_mode == source_mode
                and name_token in entry.label
                and _serialize_atoms_for_hash(entry.atoms) == target_hash
            ):
                return entry, False
        new_entry = self.register_structure(
            atoms,
            name=name,
            source_mode=source_mode,
            is_relaxed=is_relaxed,
            provenance_run_id=provenance_run_id,
        )
        return new_entry, True

    def register_run(self, record: RunRecord) -> None:
        """Store `record` in `self.runs[record.run_id]`."""
        self.runs[record.run_id] = record

    def get_structure(self, entry_id: str) -> StructureEntry:
        """Return `self.structures[entry_id]`, or fall back to the first structure if not found."""
        if entry_id in self.structures:
            return self.structures[entry_id]
        return next(iter(self.structures.values()))

    def structure_options(self) -> dict[str, str]:
        """Return `{entry.label: entry.entry_id}` suitable for `mo.ui.dropdown`."""
        return {entry.label: entry.entry_id for entry in self.structures.values()}

    def latest_run_for_mode(self, mode: JanusMode) -> RunRecord | None:
        """Return the most recently registered RunRecord for `mode`, or None."""
        matching = [r for r in self.runs.values() if r.mode == mode]
        return matching[-1] if matching else None


def load_uploaded_structure(
    filename: str,
    raw_bytes: bytes,
    *,
    upload_dir: Path = Path("runs") / "_uploads",
) -> Atoms:
    """Write uploaded bytes to disk and parse the structure via ASE across .xyz/.extxyz/.cif/.vasp/.poscar/.traj."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name or "uploaded.extxyz"
    target_path = upload_dir / safe_name
    target_path.write_bytes(raw_bytes)
    suffix = target_path.suffix.lower()
    if suffix == ".extxyz":
        fmt: str | None = "extxyz"
    elif suffix in {".vasp", ".poscar"}:
        fmt = "vasp"
    else:
        fmt = None
    loaded = ase.io.read(str(target_path), index=-1, format=fmt)
    return loaded[-1] if isinstance(loaded, list) else loaded
