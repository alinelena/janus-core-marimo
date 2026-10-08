"""Unit tests for janus_marimo.viz (Chemiscope 3D widgets and Altair 2D charts)."""

from __future__ import annotations

from pathlib import Path

import ase.build
import numpy as np
import pandas as pd

from janus_marimo.viz import (
    build_chemiscope_widget,
    build_preflight_chemiscope_widget,
    chart_descriptors_distribution,
    chart_elasticity_tensor,
    chart_eos_curve,
    chart_geomopt_convergence,
    chart_md_thermodynamics,
    chart_neb_barrier,
    chart_phonons_spectrum,
    chart_singlepoint_forces,
    compute_preflight_atom_properties,
    extract_selected_step,
    sanitize_atoms_for_chemiscope,
)


def test_sanitize_atoms_for_chemiscope_strips_matrices_and_strides() -> None:
    frames = []
    for i in range(250):
        a = ase.build.bulk("Cu", "fcc", a=3.61, cubic=True)
        a.info["mace_mp_energy"] = -14.0 - 0.01 * i
        a.info["mace_mp_stress"] = np.eye(3)  # 3x3 matrix that would warn in Chemiscope
        a.arrays["mace_mp_forces"] = np.full((len(a), 3), 0.1)
        frames.append(a)

    cleaned, props, force_key = sanitize_atoms_for_chemiscope(frames, max_frames=50)
    assert len(cleaned) <= 50
    assert force_key == "forces"
    assert cleaned[0].info == {}
    assert "forces" in cleaned[0].arrays
    assert "force_norm" in cleaned[0].arrays
    assert len(props["step"]["values"]) == len(cleaned)
    assert len(props["energy_ev"]["values"]) == len(cleaned)
    assert len(props["atom_index"]["values"]) == len(cleaned) * 4
    assert len(props["force_norm"]["values"]) == len(cleaned) * 4


def test_build_chemiscope_widget_structure_and_default_modes() -> None:
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.info["mace_mp_energy"] = -27.2
    atoms.arrays["mace_mp_forces"] = np.full((len(atoms), 3), 0.05)

    w_struct = build_chemiscope_widget(
        [atoms],
        mode="structure",
        show_force_arrows=True,
    )
    assert w_struct.mode == "structure"

    w_default = build_chemiscope_widget(
        [atoms],
        mode="default",
        show_force_arrows=True,
        environments_cutoff=4.0,
        extra_properties={
            "descriptor": {
                "target": "atom",
                "values": np.linspace(-0.5, -0.2, len(atoms)).tolist(),
            }
        },
    )
    assert w_default.mode == "default"


def test_build_chemiscope_widget_with_initial_structure_index() -> None:
    import json

    frames = [ase.build.bulk("Cu", "fcc", a=3.6 + 0.05 * i) for i in range(5)]
    w = build_chemiscope_widget(frames, mode="structure", initial_structure_index=3)
    val = json.loads(w.value)
    assert len(val["structures"]) == 5
    assert val["settings"]["pinned"] == [3]
    assert w.selected_ids == {"structure": 3}

    # Test out-of-bounds clamping
    w_clamp = build_chemiscope_widget(frames, mode="structure", initial_structure_index=10)
    val_clamp = json.loads(w_clamp.value)
    assert val_clamp["settings"]["pinned"] == [4]


def test_all_8_altair_chart_builders_produce_valid_specs() -> None:
    # 1. Singlepoint
    atom_df = pd.DataFrame(
        {
            "atom_index": [0, 1],
            "symbol": ["Na", "Cl"],
            "x": [0.0, 2.8],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "fx": [0.01, -0.01],
            "fy": [0.0, 0.0],
            "fz": [0.0, 0.0],
            "force_norm": [0.01, 0.01],
            "descriptor_value": [-0.3, -0.5],
            "structure_index": [0, 0],
        }
    )
    c1 = chart_singlepoint_forces(atom_df)
    assert "mark" in c1.to_dict()

    # 2. Geomopt
    traj_df = pd.DataFrame(
        {
            "step": [0, 1],
            "energy_ev": [-10.0, -10.2],
            "delta_energy_mev_atom": [0.0, -25.0],
            "max_force_ev_ang": [0.1, 0.005],
            "volume_ang3": [150.0, 149.2],
        }
    )
    c2 = chart_geomopt_convergence(traj_df)
    assert "hconcat" in c2.to_dict()

    # 3. MD
    stats_df = pd.DataFrame(
        {
            "step": [0, 10],
            "time_fs": [0.0, 10.0],
            "temp_k": [300.0, 305.0],
            "epot_ev": [-40.0, -39.9],
            "ekin_ev": [0.3, 0.31],
            "etot_ev": [-39.7, -39.59],
            "pressure_gpa": [0.1, 0.2],
            "volume_ang3": [160.0, 160.0],
        }
    )
    rdf_df = pd.DataFrame({"r_ang": [2.0, 3.0], "g_r": [0.5, 2.2], "pair": ["Na_Cl", "Na_Cl"]})
    c3 = chart_md_thermodynamics(stats_df, rdf_df)
    assert "vconcat" in c3.to_dict()

    # 4. Phonons
    bands_df = pd.DataFrame(
        {
            "q_index": [0, 1],
            "distance": [0.0, 0.5],
            "band_index": [0, 0],
            "frequency_thz": [0.0, 3.2],
            "q_label": ["G", "X"],
        }
    )
    dos_df = pd.DataFrame({"frequency_thz": [0.0, 3.2], "dos": [0.0, 1.4]})
    thermal_df = pd.DataFrame(
        {
            "temperature_k": [100.0, 300.0],
            "free_energy_kj_mol": [5.0, -2.0],
            "entropy_j_k_mol": [15.0, 45.0],
            "heat_capacity_j_k_mol": [25.0, 48.0],
        }
    )
    c4 = chart_phonons_spectrum(bands_df, dos_df, thermal_df)
    assert "vconcat" in c4.to_dict()

    # 5. EOS
    raw_df = pd.DataFrame(
        {
            "lattice_scalar": [0.99, 1.0, 1.01],
            "volume_ang3": [155.0, 160.0, 165.0],
            "energy_ev": [-43.1, -43.3, -43.1],
        }
    )
    fit_df = pd.DataFrame(
        {"volume_ang3": [155.0, 160.0, 165.0], "energy_ev": [-43.1, -43.3, -43.1]}
    )
    c5 = chart_eos_curve(raw_df, fit_df, 160.0, -43.3, 92.0)
    assert "layer" in c5.to_dict()

    # 6. Elasticity
    c_ij_df = pd.DataFrame(
        {"i_label": ["C1", "C2"], "j_label": ["C1", "C2"], "c_ij_gpa": [170.0, 120.0]}
    )
    moduli_df = pd.DataFrame(
        {
            "property": ["Bulk Modulus (K)", "Shear Modulus (G)"],
            "method": ["VRH", "VRH"],
            "value_gpa": [136.0, 48.5],
        }
    )
    c6 = chart_elasticity_tensor(c_ij_df, moduli_df)
    assert "hconcat" in c6.to_dict()

    # 7. NEB
    image_df = pd.DataFrame(
        {
            "image_index": [0, 1, 2],
            "rxn_coord_ang": [0.0, 1.2, 2.4],
            "energy_ev": [-40.0, -39.2, -40.0],
            "rel_energy_ev": [0.0, 0.8, 0.0],
            "max_force_ev_ang": [0.0, 0.02, 0.0],
        }
    )
    c7 = chart_neb_barrier(image_df)
    assert "layer" in c7.to_dict()

    # 8. Descriptors
    c8 = chart_descriptors_distribution(atom_df)
    assert "mark" in c8.to_dict()


def test_janus_workbench_app_run_executes_tabs_cell() -> None:
    import janus_workbench

    outputs, defs = janus_workbench.app.run()
    assert "workbench_layout" in defs
    assert "mode_tabs" in defs
    assert defs["mode_tabs"].value == "1. SinglePoint"


def test_preflight_chemiscope_properties_and_widget() -> None:
    cu = ase.build.bulk("Cu", "fcc", a=3.61, cubic=True)
    tiled_cu, props_cu = compute_preflight_atom_properties(
        cu,
        supercell=(2, 2, 1),
        cutoff=4.0,
    )
    assert len(tiled_cu) == 16
    expected_keys = {
        "symbol",
        "d_min_ang",
        "coordination_number",
        "clash_flag",
        "frac_x",
        "frac_y",
        "frac_z",
    }
    assert set(props_cu.keys()) == expected_keys
    # FCC Cu (a=3.61 Å): 12 first-shell neighbors at 2.553 Å + 6 second-shell at 3.610 Å = 18 within 4.0 Å
    assert all(cn == 18 for cn in props_cu["coordination_number"]["values"])
    assert all(cf == 0 for cf in props_cu["clash_flag"]["values"])

    _, props_cu_1st = compute_preflight_atom_properties(cu, supercell=(1, 1, 1), cutoff=3.0)
    assert all(cn == 12 for cn in props_cu_1st["coordination_number"]["values"])

    widget = build_preflight_chemiscope_widget(cu, supercell=(2, 2, 1), cutoff=4.0)
    assert widget.mode == "structure"

    # Non-periodic molecule should not stack duplicates when supercell=(2, 2, 2)
    h2o = ase.build.molecule("H2O")
    tiled_h2o, props_h2o = compute_preflight_atom_properties(
        h2o,
        supercell=(2, 2, 2),
        cutoff=4.0,
    )
    assert len(tiled_h2o) == 3
    assert all(d > 0.9 for d in props_h2o["d_min_ang"]["values"])
    assert props_h2o["frac_x"]["values"] == [0.0, 0.0, 0.0]


def test_janus_workbench_lazy_preview_load_succeeds() -> None:
    import asyncio
    import gc
    import types

    from marimo._plugins.stateless.lazy import lazy as LazyElement
    from marimo._runtime.functions import EmptyArgs

    import janus_workbench

    _, defs = janus_workbench.app.run()
    assert "structure_header_panel" in defs

    # Locate the lazy elements created during app.run() and invoke their `load` RPC
    # after Kernel temporary cleanup (_invalidate_cell_state deletes _cell_* temporaries).
    lazy_elems = [obj for obj in gc.get_objects() if isinstance(obj, LazyElement)]
    assert len(lazy_elems) >= 1
    for elem in lazy_elems:
        if hasattr(elem._element, "__globals__"):
            for key in [k for k in elem._element.__globals__ if k.startswith("_cell_")]:
                elem._element.__globals__.pop(key, None)
        res = asyncio.run(elem._load(EmptyArgs()))
        assert res.html

    # Verify that no nested function/lambda in any cell of janus_workbench.py
    # references a cell-local temporary (_cell_*) via LOAD_GLOBAL (co_names).
    def _collect_nested_code_objects(code: types.CodeType) -> list[types.CodeType]:
        nested: list[types.CodeType] = []
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                nested.append(const)
                nested.extend(_collect_nested_code_objects(const))
        return nested

    for cell_data in janus_workbench.app._cell_manager.cell_data():
        cell = cell_data.cell
        assert cell is not None
        for root_code in (cell._cell.body, cell._cell.last_expr):
            if root_code is None:
                continue
            for nested_code in _collect_nested_code_objects(root_code):
                leaked_temps = [n for n in nested_code.co_names if n.startswith("_cell_")]
                assert not leaked_temps, (
                    f"Cell {cell_data.name} nested {nested_code.co_name} references "
                    f"mangled cell temporaries {leaked_temps} via LOAD_GLOBAL"
                )


def test_chart_geomopt_convergence_spec_and_tooltips() -> None:
    traj_df = pd.DataFrame(
        {
            "step": [0, 1, 2],
            "energy_ev": [-10.0, -10.2, -10.25],
            "delta_energy_mev_atom": [0.0, -25.0, -31.25],
            "max_force_ev_ang": [0.5, 0.05, 0.008],
            "volume_ang3": [150.0, 149.2, 149.1],
        }
    )
    # 1. Unseeded (selected_step is None)
    c_none = chart_geomopt_convergence(traj_df, selected_step=None)
    assert c_none.data is traj_df
    d_none = c_none.to_dict()
    assert "hconcat" in d_none
    params_none = d_none.get("params", [])
    assert len(params_none) == 1
    p_none = params_none[0]
    assert p_none["name"] == "opt_step"
    assert p_none["select"]["type"] == "point"
    assert p_none["select"]["fields"] == ["step"]
    assert p_none["select"]["on"] == "click"
    assert p_none["select"]["clear"] == "dblclick"
    assert "value" not in p_none

    # Check y-axis scale zero=False
    for sub in d_none["hconcat"]:
        layer = sub["layer"]
        for lyr in layer:
            y_enc = lyr.get("encoding", {}).get("y", {})
            if "scale" in y_enc:
                assert y_enc["scale"]["zero"] is False

    # Check tooltips on circle layers
    for sub in d_none["hconcat"]:
        circle_layer = next(lyr for lyr in sub["layer"] if lyr.get("mark", {}).get("type") == "circle")
        tooltips = [t["field"] for t in circle_layer["encoding"]["tooltip"]]
        assert "step" in tooltips
        assert "energy_ev" in tooltips
        assert "delta_energy_mev_atom" in tooltips
        assert "max_force_ev_ang" in tooltips
        assert "volume_ang3" in tooltips
        color_cond = circle_layer["encoding"]["color"]["condition"]
        size_cond = circle_layer["encoding"]["size"]["condition"]
        assert color_cond.get("empty") is False
        assert size_cond.get("empty") is False

    # 2. Seeded (selected_step=2)
    c_seeded = chart_geomopt_convergence(traj_df, selected_step=2)
    d_seeded = c_seeded.to_dict()
    p_seeded = d_seeded["params"][0]
    assert p_seeded["value"] == [{"step": 2}]


def test_extract_selected_step_all_payload_variants() -> None:
    traj_df = pd.DataFrame({"step": [0, 1, 2, 3]})

    # DataFrame variants
    assert extract_selected_step(pd.DataFrame({"step": [2]}), traj_df) == 2
    assert extract_selected_step(pd.DataFrame({"step": [10]}), traj_df) is None
    assert extract_selected_step(pd.DataFrame(), traj_df) is None
    assert extract_selected_step(pd.DataFrame({"other": [1]}), traj_df) is None

    # Nested Vega-Lite dict variants
    assert extract_selected_step({"opt_step": {"vlPoint": {"or": [{"step": 3}]}}}, traj_df) == 3
    assert extract_selected_step({"opt_step": {"vlPoint": {"step": 1}}}, traj_df) == 1
    assert extract_selected_step({"opt_step": {"step": [2]}}, traj_df) == 2
    assert extract_selected_step({"opt_step": {"step": 2}}, traj_df) == 2
    assert extract_selected_step({"opt_step": {"step": []}}, traj_df) is None
    assert extract_selected_step({"opt_step": {}}, traj_df) is None
    assert extract_selected_step({"step": 0}, traj_df) == 0
    assert extract_selected_step({}, traj_df) is None

    # List / tuple / numeric variants
    assert extract_selected_step([{"step": 1}], traj_df) == 1
    assert extract_selected_step(2, traj_df) == 2
    assert extract_selected_step(2.0, traj_df) == 2
    assert extract_selected_step(-1, traj_df) is None
    assert extract_selected_step(5, traj_df) is None

    # Boolean and None rejection
    assert extract_selected_step(False, traj_df) is None
    assert extract_selected_step(True, traj_df) is None
    assert extract_selected_step(None, traj_df) is None


def test_janus_workbench_tabs_cell_with_geomopt_view(tmp_path: Path) -> None:
    import ase.build
    import marimo as mo

    import janus_workbench
    from janus_marimo.ledger import MLIPConfig, RunRecord, WorkspaceLedger
    from janus_marimo.parsers import ParsedGeomOpt

    # 1. Run the app and check initial tab
    _, defs = janus_workbench.app.run()
    mode_tabs = defs["mode_tabs"]
    assert mode_tabs.value == "1. SinglePoint"

    # 2. Test opt_view_cell across different branches
    views_state: dict = {}

    def mock_set_mode_views(fn):
        nonlocal views_state
        views_state = fn(views_state)

    # Initial boot: rec=None, parsed=None
    janus_workbench.opt_view_cell.run(
        WorkspaceLedger=WorkspaceLedger,
        build_chemiscope_widget=build_chemiscope_widget,
        chart_geomopt_convergence=chart_geomopt_convergence,
        extract_selected_step=extract_selected_step,
        get_opt_parsed=lambda: None,
        get_opt_run_record=lambda: None,
        get_opt_selected_step=lambda: None,
        mo=mo,
        set_ledger=lambda fn: None,
        set_mode_views=mock_set_mode_views,
        set_opt_selected_step=lambda val: None,
    )
    assert "geomopt" not in views_state

    # Failed run
    cfg = MLIPConfig(arch="mace_mp", model="small", device="cpu")
    failed_rec = RunRecord(
        run_id="run-fail",
        mode="geomopt",
        mlip_config=cfg,
        struct_entry_id="s1",
        final_struct_entry_id=None,
        mode_params={},
        run_dir=tmp_path,
        config_yaml_path=tmp_path / "cfg.yml",
        summary_yaml_path=tmp_path / "summary.yml",
        output_files={},
        cli_command=["janus", "geomopt"],
        status="failed",
        duration_s=1.2,
        stdout_stderr="Error message",
    )
    janus_workbench.opt_view_cell.run(
        WorkspaceLedger=WorkspaceLedger,
        build_chemiscope_widget=build_chemiscope_widget,
        chart_geomopt_convergence=chart_geomopt_convergence,
        extract_selected_step=extract_selected_step,
        get_opt_parsed=lambda: None,
        get_opt_run_record=lambda: failed_rec,
        get_opt_selected_step=lambda: None,
        mo=mo,
        set_ledger=lambda fn: None,
        set_mode_views=mock_set_mode_views,
        set_opt_selected_step=lambda val: None,
    )
    assert "geomopt" in views_state

    # Converged run with trajectory
    cfg_file = tmp_path / "cfg.yml"
    cfg_file.write_text("struct: test.extxyz\n")
    success_rec = RunRecord(
        run_id="run-succ",
        mode="geomopt",
        mlip_config=cfg,
        struct_entry_id="s1",
        final_struct_entry_id=None,
        mode_params={},
        run_dir=tmp_path,
        config_yaml_path=cfg_file,
        summary_yaml_path=tmp_path / "summary.yml",
        output_files={},
        cli_command=["janus", "geomopt"],
        status="succeeded",
        duration_s=2.5,
        stdout_stderr="",
    )
    atoms0 = ase.build.bulk("Cu", "fcc", a=3.61)
    atoms1 = ase.build.bulk("Cu", "fcc", a=3.62)
    atoms2 = ase.build.bulk("Cu", "fcc", a=3.63)
    traj_df = pd.DataFrame(
        {
            "step": [0, 1, 2],
            "energy_ev": [-10.0, -10.1, -10.12],
            "delta_energy_mev_atom": [0.0, -10.0, -12.0],
            "max_force_ev_ang": [0.2, 0.05, 0.005],
            "volume_ang3": [47.0, 47.4, 47.8],
        }
    )
    parsed = ParsedGeomOpt(
        initial_atoms=atoms0,
        optimized_atoms=atoms2,
        trajectory=[atoms0, atoms1, atoms2],
        traj_df=traj_df,
        converged=True,
    )

    # Full trajectory view (selected_step=None)
    janus_workbench.opt_view_cell.run(
        WorkspaceLedger=WorkspaceLedger,
        build_chemiscope_widget=build_chemiscope_widget,
        chart_geomopt_convergence=chart_geomopt_convergence,
        extract_selected_step=extract_selected_step,
        get_opt_parsed=lambda: parsed,
        get_opt_run_record=lambda: success_rec,
        get_opt_selected_step=lambda: None,
        mo=mo,
        set_ledger=lambda fn: None,
        set_mode_views=mock_set_mode_views,
        set_opt_selected_step=lambda val: None,
    )
    assert "geomopt" in views_state

    # Step inspection view (selected_step=1)
    janus_workbench.opt_view_cell.run(
        WorkspaceLedger=WorkspaceLedger,
        build_chemiscope_widget=build_chemiscope_widget,
        chart_geomopt_convergence=chart_geomopt_convergence,
        extract_selected_step=extract_selected_step,
        get_opt_parsed=lambda: parsed,
        get_opt_run_record=lambda: success_rec,
        get_opt_selected_step=lambda: 1,
        mo=mo,
        set_ledger=lambda fn: None,
        set_mode_views=mock_set_mode_views,
        set_opt_selected_step=lambda val: None,
    )
    assert "geomopt" in views_state

    # Empty trajectory
    empty_parsed = ParsedGeomOpt(
        initial_atoms=atoms0,
        optimized_atoms=atoms0,
        trajectory=[],
        traj_df=pd.DataFrame(),
        converged=False,
    )
    janus_workbench.opt_view_cell.run(
        WorkspaceLedger=WorkspaceLedger,
        build_chemiscope_widget=build_chemiscope_widget,
        chart_geomopt_convergence=chart_geomopt_convergence,
        extract_selected_step=extract_selected_step,
        get_opt_parsed=lambda: empty_parsed,
        get_opt_run_record=lambda: success_rec,
        get_opt_selected_step=lambda: None,
        mo=mo,
        set_ledger=lambda fn: None,
        set_mode_views=mock_set_mode_views,
        set_opt_selected_step=lambda val: None,
    )
    assert "geomopt" in views_state
