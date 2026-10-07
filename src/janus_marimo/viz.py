"""Chemiscope 3D widget builders and Altair 2D interactive diagnostic charts."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal

import altair as alt
import ase.neighborlist
import chemiscope
import numpy as np
import pandas as pd
from ase import Atoms

warnings.filterwarnings(
    "ignore",
    message=".*deduplicated selection parameter.*",
    category=UserWarning,
)


def _extract_frame_energy(atoms: Atoms) -> float:
    """Extract scalar energy from an ASE Atoms frame for Chemiscope structure properties."""
    for key, val in atoms.info.items():
        if (key == "energy" or key.endswith("_energy")) and val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    if atoms.calc is not None and hasattr(atoms.calc, "results"):
        val = atoms.calc.results.get("energy")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


def _extract_frame_forces(atoms: Atoms) -> tuple[np.ndarray, bool]:
    """Extract (N, 3) forces array from an ASE Atoms frame if present."""
    n_atoms = len(atoms)
    if n_atoms == 0:
        return np.zeros((0, 3), dtype=float), False

    for key, val in atoms.arrays.items():
        if key == "forces" or key.endswith("_forces"):
            arr = np.asarray(val, dtype=float)
            if arr.shape == (n_atoms, 3):
                return arr, True

    if atoms.calc is not None and hasattr(atoms.calc, "results"):
        val = atoms.calc.results.get("forces")
        if val is not None:
            arr = np.asarray(val, dtype=float)
            if arr.shape == (n_atoms, 3):
                return arr, True

    return np.zeros((n_atoms, 3), dtype=float), False


def sanitize_atoms_for_chemiscope(
    structures: Sequence[Atoms],
    *,
    max_frames: int = 200,
) -> tuple[list[Atoms], dict[str, dict[str, Any]], str | None]:
    """Prepare a list of `ase.Atoms` for `chemiscope.show` without property warnings or bloat."""
    struct_list = list(structures)
    if not struct_list:
        return (
            [],
            {
                "step": {"target": "structure", "values": []},
                "energy_ev": {"target": "structure", "values": []},
            },
            None,
        )

    if len(struct_list) > max_frames and max_frames >= 2:
        indices = np.linspace(0, len(struct_list) - 1, max_frames, dtype=int)
        unique_indices = list(dict.fromkeys(int(i) for i in indices))
        if unique_indices[-1] != len(struct_list) - 1:
            unique_indices[-1] = len(struct_list) - 1
        struct_list = [struct_list[i] for i in unique_indices]

    cleaned_structures: list[Atoms] = []
    energies: list[float] = []
    has_any_forces = False

    for frame in struct_list:
        energy = _extract_frame_energy(frame)
        forces, found_forces = _extract_frame_forces(frame)
        if found_forces:
            has_any_forces = True

        cleaned = frame.copy()
        cleaned.calc = None
        cleaned.info = {}

        # Retain only numbers, positions, forces, force_norm, and 1D scalar descriptors
        keep_descriptors: dict[str, np.ndarray] = {}
        for k, arr in frame.arrays.items():
            if k.endswith("descriptors"):
                np_arr = np.asarray(arr, dtype=float)
                if np_arr.ndim == 1 and np_arr.shape[0] == len(frame):
                    keep_descriptors[k] = np_arr

        for k in list(cleaned.arrays.keys()):
            if k not in ("numbers", "positions"):
                del cleaned.arrays[k]

        cleaned.arrays["forces"] = np.asarray(forces, dtype=float)
        cleaned.arrays["force_norm"] = (
            np.linalg.norm(forces, axis=1).astype(float)
            if len(cleaned) > 0
            else np.zeros(0, dtype=float)
        )
        for k, desc_arr in keep_descriptors.items():
            cleaned.arrays[k] = desc_arr

        cleaned_structures.append(cleaned)
        energies.append(float(energy))

    atom_indices = np.concatenate(
        [np.arange(len(s), dtype=int) for s in cleaned_structures]
    ).tolist()
    force_norms = np.concatenate([s.arrays["force_norm"] for s in cleaned_structures]).tolist()

    properties: dict[str, dict[str, Any]] = {
        "step": {
            "target": "structure",
            "values": list(range(len(cleaned_structures))),
        },
        "energy_ev": {
            "target": "structure",
            "values": energies,
            "units": "eV",
        },
        "atom_index": {
            "target": "atom",
            "values": atom_indices,
        },
        "force_norm": {
            "target": "atom",
            "values": force_norms,
            "units": "eV/Å",
        },
    }

    return (
        cleaned_structures,
        properties,
        "forces" if has_any_forces else None,
    )


def build_chemiscope_widget(
    structures: Sequence[Atoms],
    *,
    mode: Literal["default", "structure", "map"] = "structure",
    show_force_arrows: bool = True,
    force_scale: float = 0.5,
    extra_properties: dict[str, dict[str, Any]] | None = None,
    environments_cutoff: float | None = None,
    max_frames: int = 200,
) -> Any:
    """Build and return a Chemiscope anywidget (`StructureWidget` or `ChemiscopeWidget`)."""
    cleaned_structures, properties, detected_force_key = sanitize_atoms_for_chemiscope(
        structures, max_frames=max_frames
    )
    if extra_properties:
        for key, prop_spec in extra_properties.items():
            properties[key] = prop_spec

    shapes: dict[str, Any] | None = None
    has_pbc = bool(cleaned_structures and any(cleaned_structures[0].pbc))
    struct_settings: dict[str, Any] = {
        "atoms": True,
        "unitCell": has_pbc,
    }

    if show_force_arrows and detected_force_key == "forces" and cleaned_structures:
        shapes = {
            "forces": chemiscope.ase_vectors_to_arrows(
                structures=cleaned_structures,
                key="forces",
                scale=force_scale,
                radius=0.08,
            )
        }
        struct_settings["shape"] = "forces"

    settings: dict[str, Any] = {"structure": [struct_settings]}

    environments = None
    if environments_cutoff is not None and cleaned_structures:
        environments = chemiscope.all_atomic_environments(
            cleaned_structures, cutoff=float(environments_cutoff)
        )

    # When environments is None and mode != "structure", Chemiscope map defaults to structure target;
    # keep atom-level properties only when environments are provided or mode == "structure".
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*chemiscope.show only displays a widget.*",
        )
        return chemiscope.show(
            structures=cleaned_structures,
            properties=properties,
            shapes=shapes,
            environments=environments,
            settings=settings,
            mode=mode,
        )


def compute_preflight_atom_properties(
    atoms: Atoms,
    *,
    supercell: tuple[int, int, int] = (1, 1, 1),
    cutoff: float = 4.0,
) -> tuple[Atoms, dict[str, dict[str, Any]]]:
    """Compute a non-destructively tiled copy of `atoms` and Chemiscope atom-target diagnostic properties."""
    base_atoms = atoms.copy()
    pbc_flags = [bool(x) for x in base_atoms.get_pbc()]
    cell_arr_init = np.asarray(base_atoms.cell.array, dtype=float)
    lengths_init = (
        base_atoms.cell.lengths() if np.isfinite(cell_arr_init).all() else [0.0, 0.0, 0.0]
    )

    rep: list[int] = []
    for k in range(3):
        req = max(1, int(supercell[k])) if k < len(supercell) else 1
        if pbc_flags[k] and float(lengths_init[k]) > 1e-6:
            rep.append(req)
        else:
            rep.append(1)

    display_atoms = (
        base_atoms * (rep[0], rep[1], rep[2])
        if (rep[0], rep[1], rep[2]) != (1, 1, 1) and len(base_atoms) > 0
        else base_atoms
    )

    n_atoms = len(display_atoms)
    coords_finite = bool(n_atoms > 0 and np.isfinite(display_atoms.positions).all())
    cell_arr = np.asarray(display_atoms.cell.array, dtype=float)
    cell_valid = bool(np.isfinite(cell_arr).all() and abs(float(np.linalg.det(cell_arr))) > 1e-8)

    d_min_ang: list[float] = [0.0] * n_atoms
    coord_num: list[int] = [0] * n_atoms

    if 0 < n_atoms <= 2000 and coords_finite:
        if any(display_atoms.pbc) and cell_valid:
            nl_cut = max(float(cutoff), 6.0)
            i_idx, _j_idx, d_ij = ase.neighborlist.neighbor_list(
                "ijd", display_atoms, cutoff=nl_cut
            )
            for a in range(n_atoms):
                mask = i_idx == a
                if np.any(mask):
                    d_a = d_ij[mask]
                    d_min_ang[a] = round(float(np.min(d_a)), 4)
                    coord_num[a] = int(np.sum(d_a <= float(cutoff)))
        elif n_atoms > 1:
            dist_mat = np.asarray(display_atoms.get_all_distances(mic=False), dtype=float)
            np.fill_diagonal(dist_mat, np.inf)
            for a in range(n_atoms):
                d_val = float(np.min(dist_mat[a]))
                d_min_ang[a] = round(d_val, 4) if np.isfinite(d_val) else 0.0
                coord_num[a] = int(np.sum(dist_mat[a] <= float(cutoff)))

    clash_flag = [1 if (0.0 < d < 0.8) else 0 for d in d_min_ang]

    if cell_valid and n_atoms > 0 and coords_finite:
        scaled = np.round(display_atoms.get_scaled_positions(wrap=False), 4)
    else:
        scaled = np.zeros((n_atoms, 3), dtype=float)

    extra_properties: dict[str, dict[str, Any]] = {
        "symbol": {
            "target": "atom",
            "values": display_atoms.get_chemical_symbols(),
        },
        "d_min_ang": {
            "target": "atom",
            "values": d_min_ang,
            "units": "Å",
        },
        "coordination_number": {
            "target": "atom",
            "values": coord_num,
        },
        "clash_flag": {
            "target": "atom",
            "values": clash_flag,
        },
        "frac_x": {
            "target": "atom",
            "values": scaled[:, 0].tolist() if n_atoms > 0 else [],
        },
        "frac_y": {
            "target": "atom",
            "values": scaled[:, 1].tolist() if n_atoms > 0 else [],
        },
        "frac_z": {
            "target": "atom",
            "values": scaled[:, 2].tolist() if n_atoms > 0 else [],
        },
    }
    return display_atoms, extra_properties


def build_preflight_chemiscope_widget(
    atoms: Atoms,
    *,
    supercell: tuple[int, int, int] = (1, 1, 1),
    cutoff: float = 4.0,
) -> Any:
    """Build a pre-flight Chemiscope 3D structure inspector widget enriched with per-atom diagnostics."""
    display_atoms, extra_props = compute_preflight_atom_properties(
        atoms,
        supercell=supercell,
        cutoff=cutoff,
    )
    return build_chemiscope_widget(
        [display_atoms],
        mode="structure",
        show_force_arrows=False,
        extra_properties=extra_props,
    )


def chart_singlepoint_forces(atom_df: pd.DataFrame) -> alt.Chart:
    """Bar chart of per-atom force norms colored by chemical symbol."""
    return (
        alt.Chart(atom_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("atom_index:O", title="Atom Index"),
            y=alt.Y("force_norm:Q", title="Force Magnitude |F| (eV/Å)"),
            color=alt.Color("symbol:N", title="Element"),
            tooltip=[
                alt.Tooltip("atom_index:Q", title="Atom"),
                alt.Tooltip("symbol:N", title="Symbol"),
                alt.Tooltip("force_norm:Q", title="|F| (eV/Å)", format=".4f"),
                alt.Tooltip("fx:Q", title="Fx (eV/Å)", format=".4f"),
                alt.Tooltip("fy:Q", title="Fy (eV/Å)", format=".4f"),
                alt.Tooltip("fz:Q", title="Fz (eV/Å)", format=".4f"),
            ],
        )
        .properties(
            title="Per-Atom Force Magnitude Distribution",
            width=420,
            height=260,
        )
    )


def chart_geomopt_convergence(
    traj_df: pd.DataFrame,
    selected_step: int | None = None,
) -> alt.HConcatChart:
    """Side-by-side Altair chart of potential energy and max force vs optimization step."""
    init_val = [{"step": int(selected_step)}] if selected_step is not None else alt.Undefined
    brush = alt.selection_point(
        fields=["step"],
        name="opt_step",
        on="click",
        clear="dblclick",
        empty=False,
        value=init_val,
    )

    tooltips = [
        alt.Tooltip("step:Q", title="Step"),
        alt.Tooltip("energy_ev:Q", title="Energy (eV)", format=".5f"),
        alt.Tooltip("delta_energy_mev_atom:Q", title="ΔE (meV/atom)", format=".3f"),
        alt.Tooltip("max_force_ev_ang:Q", title="fmax (eV/Å)", format=".5f"),
        alt.Tooltip("volume_ang3:Q", title="Volume (Å³)", format=".2f"),
    ]

    e_line = (
        alt.Chart()
        .mark_line(color="#2563eb")
        .encode(
            x=alt.X("step:Q", title="Optimization Step"),
            y=alt.Y("energy_ev:Q", title="Potential Energy E (eV)", scale=alt.Scale(zero=False)),
        )
    )
    e_pts = (
        alt.Chart()
        .mark_circle()
        .encode(
            x=alt.X("step:Q", title="Optimization Step"),
            y=alt.Y("energy_ev:Q", title="Potential Energy E (eV)", scale=alt.Scale(zero=False)),
            color=alt.condition(brush, alt.value("#1d4ed8"), alt.value("#93c5fd"), empty=False),
            size=alt.condition(brush, alt.value(110), alt.value(50), empty=False),
            tooltip=tooltips,
        )
    )
    e_chart = alt.layer(e_line, e_pts).properties(title="Energy Convergence", width=360, height=220)

    f_line = (
        alt.Chart()
        .mark_line(color="#dc2626")
        .encode(
            x=alt.X("step:Q", title="Optimization Step"),
            y=alt.Y("max_force_ev_ang:Q", title="Max Atomic Force fmax (eV/Å)", scale=alt.Scale(zero=False)),
        )
    )
    f_pts = (
        alt.Chart()
        .mark_circle()
        .encode(
            x=alt.X("step:Q", title="Optimization Step"),
            y=alt.Y("max_force_ev_ang:Q", title="Max Atomic Force fmax (eV/Å)", scale=alt.Scale(zero=False)),
            color=alt.condition(brush, alt.value("#b91c1c"), alt.value("#fca5a5"), empty=False),
            size=alt.condition(brush, alt.value(110), alt.value(50), empty=False),
            tooltip=tooltips,
        )
    )
    f_chart = alt.layer(f_line, f_pts).properties(title="Force Convergence", width=360, height=220)

    return alt.hconcat(e_chart, f_chart, data=traj_df).add_params(brush)


def extract_selected_step(
    selection_payload: Any,
    traj_df: pd.DataFrame | None = None,
) -> int | None:
    """Extract an integer optimization step from an Altair/Marimo chart selection payload.

    Supports DataFrame selections, nested Vega-Lite dicts, point value dicts, and integers.
    Safely handles empty dicts, empty lists, and invalid values (e.g. booleans).
    If traj_df is provided, validates step in set(traj_df["step"].dropna().astype(int)).
    Returns None if no selection, empty selection, or out-of-bounds.
    """
    if isinstance(selection_payload, bool):
        return None

    raw_val: Any = None

    if isinstance(selection_payload, pd.DataFrame):
        if not selection_payload.empty and "step" in selection_payload.columns:
            val = selection_payload["step"].iloc[0]
            if pd.notna(val):
                raw_val = val
    elif isinstance(selection_payload, dict):
        sub = selection_payload.get("opt_step", selection_payload)
        if isinstance(sub, dict):
            if "vlPoint" in sub and isinstance(sub["vlPoint"], dict):
                vl = sub["vlPoint"]
                if "or" in vl and isinstance(vl["or"], list) and len(vl["or"]) > 0:
                    first = vl["or"][0]
                    if isinstance(first, dict) and "step" in first:
                        raw_val = first["step"]
                elif "step" in vl:
                    raw_val = vl["step"]
            elif "step" in sub:
                val = sub["step"]
                if isinstance(val, (list, tuple)):
                    if len(val) > 0 and pd.notna(val[0]):
                        raw_val = val[0]
                elif pd.notna(val):
                    raw_val = val
        if raw_val is None and "step" in selection_payload:
            val = selection_payload["step"]
            if isinstance(val, (list, tuple)):
                if len(val) > 0 and pd.notna(val[0]):
                    raw_val = val[0]
            elif pd.notna(val):
                raw_val = val
    elif isinstance(selection_payload, (list, tuple)):
        if len(selection_payload) > 0:
            first = selection_payload[0]
            return extract_selected_step(first, traj_df=traj_df)
    elif isinstance(selection_payload, (int, float, np.integer)):
        if pd.notna(selection_payload):
            raw_val = selection_payload

    if raw_val is None:
        return None

    try:
        step = int(raw_val)
    except (TypeError, ValueError):
        return None

    if step < 0:
        return None

    if traj_df is not None and "step" in traj_df.columns:
        valid_steps = set(traj_df["step"].dropna().astype(int))
        if step not in valid_steps:
            return None

    return step


def chart_md_thermodynamics(
    stats_df: pd.DataFrame, rdf_df: pd.DataFrame | None = None
) -> alt.VConcatChart:
    """Vertical stack of MD temperature, energy components, and optional RDF g(r)."""
    temp_chart = (
        alt.Chart(stats_df)
        .mark_line(color="#ea580c")
        .encode(
            x=alt.X("time_fs:Q", title="Time (fs)"),
            y=alt.Y("temp_k:Q", title="Temperature (K)"),
            tooltip=[
                alt.Tooltip("step:Q", title="Step"),
                alt.Tooltip("time_fs:Q", title="Time (fs)", format=".1f"),
                alt.Tooltip("temp_k:Q", title="T (K)", format=".1f"),
                alt.Tooltip("pressure_gpa:Q", title="P (GPa)", format=".3f"),
            ],
        )
        .properties(title="MD Temperature Evolution", width=480, height=160)
    )

    energy_long = stats_df.melt(
        id_vars=["step", "time_fs"],
        value_vars=["epot_ev", "ekin_ev", "etot_ev"],
        var_name="component",
        value_name="energy_ev",
    )
    energy_chart = (
        alt.Chart(energy_long)
        .mark_line()
        .encode(
            x=alt.X("time_fs:Q", title="Time (fs)"),
            y=alt.Y("energy_ev:Q", title="Energy (eV)", scale=alt.Scale(zero=False)),
            color=alt.Color("component:N", title="Component"),
            tooltip=[
                alt.Tooltip("time_fs:Q", title="Time (fs)", format=".1f"),
                alt.Tooltip("component:N", title="Term"),
                alt.Tooltip("energy_ev:Q", title="Energy (eV)", format=".4f"),
            ],
        )
        .properties(title="MD Energy Conservation & Partitioning", width=480, height=160)
    )

    panels: list[alt.Chart] = [temp_chart, energy_chart]
    if rdf_df is not None and not rdf_df.empty:
        rdf_chart = (
            alt.Chart(rdf_df)
            .mark_line()
            .encode(
                x=alt.X("r_ang:Q", title="Pair Distance r (Å)"),
                y=alt.Y("g_r:Q", title="Radial Distribution g(r)"),
                color=alt.Color("pair:N", title="Pair"),
                tooltip=[
                    alt.Tooltip("r_ang:Q", title="r (Å)", format=".3f"),
                    alt.Tooltip("g_r:Q", title="g(r)", format=".3f"),
                    alt.Tooltip("pair:N", title="Pair"),
                ],
            )
            .properties(title="Radial Distribution Function g(r)", width=480, height=160)
        )
        panels.append(rdf_chart)

    return alt.vconcat(*panels)


def chart_phonons_spectrum(
    bands_df: pd.DataFrame | None,
    dos_df: pd.DataFrame | None,
    thermal_df: pd.DataFrame | None,
) -> alt.VConcatChart:
    """Stacked Altair charts for phonon dispersion, vibrational DOS, and thermal properties."""
    panels: list[alt.Chart | alt.LayerChart] = []

    if bands_df is not None and not bands_df.empty:
        zero_line = (
            alt.Chart(pd.DataFrame({"y": [0.0]}))
            .mark_rule(strokeDash=[4, 4], color="#dc2626")
            .encode(y="y:Q")
        )
        band_lines = (
            alt.Chart(bands_df)
            .mark_line(color="#1d4ed8", strokeWidth=1.5)
            .encode(
                x=alt.X("distance:Q", title="Wavevector Path Distance"),
                y=alt.Y("frequency_thz:Q", title="Frequency (THz)"),
                detail="band_index:N",
                tooltip=[
                    alt.Tooltip("band_index:Q", title="Branch"),
                    alt.Tooltip("distance:Q", title="q-distance", format=".4f"),
                    alt.Tooltip("frequency_thz:Q", title="Freq (THz)", format=".3f"),
                    alt.Tooltip("q_label:N", title="High-Symmetry Point"),
                ],
            )
        )
        panels.append(
            alt.layer(zero_line, band_lines).properties(
                title="Phonon Band Dispersion (0 THz stability threshold dashed)",
                width=460,
                height=200,
            )
        )

    if dos_df is not None and not dos_df.empty:
        dos_chart = (
            alt.Chart(dos_df)
            .mark_area(line={"color": "#0f766e"}, color="#99f6e4", opacity=0.5)
            .encode(
                x=alt.X("frequency_thz:Q", title="Frequency (THz)"),
                y=alt.Y("dos:Q", title="Vibrational DOS"),
                tooltip=[
                    alt.Tooltip("frequency_thz:Q", title="Freq (THz)", format=".3f"),
                    alt.Tooltip("dos:Q", title="g(ν)", format=".4f"),
                ],
            )
            .properties(title="Vibrational Density of States", width=460, height=150)
        )
        panels.append(dos_chart)

    if thermal_df is not None and not thermal_df.empty:
        thermal_long = thermal_df.melt(
            id_vars=["temperature_k"],
            value_vars=[
                "free_energy_kj_mol",
                "entropy_j_k_mol",
                "heat_capacity_j_k_mol",
            ],
            var_name="property",
            value_name="value",
        )
        thermal_chart = (
            alt.Chart(thermal_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("temperature_k:Q", title="Temperature (K)"),
                y=alt.Y("value:Q", title="Thermodynamic Value"),
                color=alt.Color("property:N", title="Observable"),
                tooltip=[
                    alt.Tooltip("temperature_k:Q", title="T (K)"),
                    alt.Tooltip("property:N", title="Property"),
                    alt.Tooltip("value:Q", title="Value", format=".3f"),
                ],
            )
            .properties(
                title="Harmonic Thermal Properties (F, S, Cv)",
                width=460,
                height=160,
            )
        )
        panels.append(thermal_chart)

    if not panels:
        empty_df = pd.DataFrame({"x": [0.0], "y": [0.0]})
        panels.append(
            alt.Chart(empty_df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .properties(title="No phonon spectrum data available", width=460, height=150)
        )

    return alt.vconcat(*panels)


def chart_eos_curve(
    raw_df: pd.DataFrame,
    fit_curve_df: pd.DataFrame,
    v0: float,
    e0: float,
    b0_gpa: float,
) -> alt.LayerChart:
    """Overlay of discrete E(V) points, continuous EOS fit curve, and equilibrium minimum V0."""
    fit_line = (
        alt.Chart(fit_curve_df)
        .mark_line(color="#2563eb", strokeWidth=2.5)
        .encode(
            x=alt.X(
                "volume_ang3:Q",
                title="Cell Volume V (Å³)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "energy_ev:Q",
                title="Potential Energy E (eV)",
                scale=alt.Scale(zero=False),
            ),
        )
    )

    raw_points = (
        alt.Chart(raw_df)
        .mark_circle(size=85, color="#dc2626")
        .encode(
            x=alt.X("volume_ang3:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("energy_ev:Q", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("lattice_scalar:Q", title="Scale Factor", format=".4f"),
                alt.Tooltip("volume_ang3:Q", title="Volume (Å³)", format=".3f"),
                alt.Tooltip("energy_ev:Q", title="Energy (eV)", format=".5f"),
            ],
        )
    )

    min_df = pd.DataFrame(
        [{"volume_ang3": float(v0), "energy_ev": float(e0), "b0_gpa": float(b0_gpa)}]
    )
    min_marker = (
        alt.Chart(min_df)
        .mark_point(shape="diamond", size=140, color="#16a34a", filled=True)
        .encode(
            x="volume_ang3:Q",
            y="energy_ev:Q",
            tooltip=[
                alt.Tooltip("volume_ang3:Q", title="Equilibrium V0 (Å³)", format=".3f"),
                alt.Tooltip("energy_ev:Q", title="Minimum E0 (eV)", format=".5f"),
                alt.Tooltip("b0_gpa:Q", title="Bulk Modulus B0 (GPa)", format=".2f"),
            ],
        )
    )

    return alt.layer(fit_line, raw_points, min_marker).properties(
        title=f"Equation of State Fit (V0 = {v0:.2f} Å³, B0 = {b0_gpa:.2f} GPa)",
        width=460,
        height=280,
    )


def chart_elasticity_tensor(c_ij_df: pd.DataFrame, moduli_df: pd.DataFrame) -> alt.HConcatChart:
    """Side-by-side 6x6 Voigt C_ij heatmap and polycrystalline moduli bar chart."""
    heatmap_base = alt.Chart(c_ij_df).encode(
        x=alt.X("j_label:O", title="Voigt Index j"),
        y=alt.Y("i_label:O", title="Voigt Index i"),
    )
    rects = heatmap_base.mark_rect().encode(
        color=alt.Color(
            "c_ij_gpa:Q",
            title="C_ij (GPa)",
            scale=alt.Scale(scheme="blues"),
        ),
        tooltip=[
            alt.Tooltip("i_label:N", title="Row"),
            alt.Tooltip("j_label:N", title="Column"),
            alt.Tooltip("c_ij_gpa:Q", title="C_ij (GPa)", format=".2f"),
        ],
    )
    text = heatmap_base.mark_text(baseline="middle", fontSize=11).encode(
        text=alt.Text("c_ij_gpa:Q", format=".1f"),
    )
    heatmap = alt.layer(rects, text).properties(
        title="6×6 Voigt Stiffness Tensor C_ij (GPa)",
        width=260,
        height=240,
    )

    moduli_chart = (
        alt.Chart(moduli_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("property:N", title="Mechanical Modulus"),
            y=alt.Y("value_gpa:Q", title="Modulus (GPa)"),
            color=alt.Color("method:N", title="Averaging Scheme"),
            xOffset="method:N",
            tooltip=[
                alt.Tooltip("property:N", title="Modulus"),
                alt.Tooltip("method:N", title="Scheme"),
                alt.Tooltip("value_gpa:Q", title="Value (GPa)", format=".2f"),
            ],
        )
        .properties(
            title="Voigt-Reuss-Hill Elastic Moduli",
            width=240,
            height=240,
        )
    )

    return alt.hconcat(heatmap, moduli_chart)


def chart_neb_barrier(image_df: pd.DataFrame) -> alt.LayerChart:
    """Minimum Energy Pathway (MEP) barrier curve across NEB band images."""
    line = (
        alt.Chart(image_df)
        .mark_line(interpolate="monotone", color="#7c3aed", strokeWidth=2.5)
        .encode(
            x=alt.X("rxn_coord_ang:Q", title="Cumulative Reaction Coordinate (Å)"),
            y=alt.Y("rel_energy_ev:Q", title="Relative Energy ΔE (eV)"),
        )
    )
    points = (
        alt.Chart(image_df)
        .mark_circle(size=90, color="#5b21b6")
        .encode(
            x="rxn_coord_ang:Q",
            y="rel_energy_ev:Q",
            tooltip=[
                alt.Tooltip("image_index:Q", title="Image"),
                alt.Tooltip("rxn_coord_ang:Q", title="Coord (Å)", format=".3f"),
                alt.Tooltip("rel_energy_ev:Q", title="ΔE (eV)", format=".4f"),
                alt.Tooltip("energy_ev:Q", title="Total E (eV)", format=".4f"),
                alt.Tooltip("max_force_ev_ang:Q", title="Max |F| (eV/Å)", format=".4f"),
            ],
        )
    )
    return alt.layer(line, points).properties(
        title="NEB Minimum Energy Pathway & Activation Barrier",
        width=460,
        height=260,
    )


def chart_descriptors_distribution(atom_df: pd.DataFrame) -> alt.Chart:
    """Scatter/distribution plot of per-atom MLIP invariant descriptors by element."""
    return (
        alt.Chart(atom_df)
        .mark_circle(size=80, opacity=0.85)
        .encode(
            x=alt.X("atom_index:Q", title="Atom Index"),
            y=alt.Y(
                "descriptor_value:Q",
                title="MLIP Invariant Descriptor",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("symbol:N", title="Element"),
            tooltip=[
                alt.Tooltip("structure_index:Q", title="Structure"),
                alt.Tooltip("atom_index:Q", title="Atom"),
                alt.Tooltip("symbol:N", title="Element"),
                alt.Tooltip("descriptor_value:Q", title="Descriptor", format=".5f"),
                alt.Tooltip("force_norm:Q", title="|F| (eV/Å)", format=".4f"),
            ],
        )
        .properties(
            title="Per-Atom MLIP Descriptor Distribution by Element",
            width=440,
            height=260,
        )
    )
