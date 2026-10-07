import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full", app_title="Janus-Core 8-Mode Atomistic MLIP Workbench")


@app.cell
def bootstrap_imports():
    import hashlib
    import importlib
    import io
    import json
    import sys
    from pathlib import Path

    import ase.io
    import marimo as mo
    import pandas as pd
    import yaml

    _repo_root = Path(__file__).resolve().parent
    for _p in (str(_repo_root / "src"), str(_repo_root)):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    for _mod_name in (
        "janus_marimo.envs",
        "janus_marimo.inspector",
        "janus_marimo.ledger",
        "janus_marimo.runner",
        "janus_marimo.parsers",
        "janus_marimo.viz",
    ):
        if _mod_name in sys.modules:
            importlib.reload(sys.modules[_mod_name])

    from janus_marimo.envs import (
        ARCH_REQUIRES_EXPLICIT_MODEL,
        ARCH_TO_ENV_GROUP,
        DEFAULT_JANUS_PYTHON,
        DEFAULT_MODEL_BY_ARCH,
        DEFAULT_VENVS_ROOT,
        MLIP_ENV_CATALOG,
        build_provision_commands,
        discover_local_environments,
        normalize_python_executable,
        provision_mlip_env,
        resolve_env_status,
    )
    from janus_marimo.ledger import (
        MLIPConfig,
        WorkspaceLedger,
        load_uploaded_structure,
    )
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
    from janus_marimo.runner import execute_janus_mode
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
    )
    from tools.structure_inspector import inspect_structure

    return (
        ARCH_REQUIRES_EXPLICIT_MODEL,
        ARCH_TO_ENV_GROUP,
        DEFAULT_JANUS_PYTHON,
        DEFAULT_MODEL_BY_ARCH,
        DEFAULT_VENVS_ROOT,
        MLIPConfig,
        MLIP_ENV_CATALOG,
        Path,
        WorkspaceLedger,
        ase,
        build_chemiscope_widget,
        build_preflight_chemiscope_widget,
        build_provision_commands,
        chart_descriptors_distribution,
        chart_elasticity_tensor,
        chart_eos_curve,
        chart_geomopt_convergence,
        chart_md_thermodynamics,
        chart_neb_barrier,
        chart_phonons_spectrum,
        chart_singlepoint_forces,
        compute_preflight_atom_properties,
        discover_local_environments,
        execute_janus_mode,
        extract_selected_step,
        hashlib,
        inspect_structure,
        io,
        json,
        load_uploaded_structure,
        mo,
        normalize_python_executable,
        parse_descriptors_run,
        parse_elasticity_run,
        parse_eos_run,
        parse_geomopt_run,
        parse_md_run,
        parse_neb_run,
        parse_phonons_run,
        parse_singlepoint_run,
        pd,
        provision_mlip_env,
        resolve_env_status,
        yaml,
    )


@app.cell
def state_init_cell(WorkspaceLedger, mo):
    get_ledger, set_ledger = mo.state(WorkspaceLedger.from_presets())
    get_env_refresh, set_env_refresh = mo.state(0)
    get_mode_views, set_mode_views = mo.state({})
    get_provision_view, set_provision_view = mo.state(None)
    get_last_upload_sig, set_last_upload_sig = mo.state(None)
    get_active_tab, set_active_tab = mo.state("1. SinglePoint", allow_self_loops=True)
    get_opt_run_record, set_opt_run_record = mo.state(None)
    get_opt_parsed, set_opt_parsed = mo.state(None)
    get_opt_selected_step, set_opt_selected_step = mo.state(None, allow_self_loops=True)
    return (
        get_active_tab,
        get_env_refresh,
        get_last_upload_sig,
        get_ledger,
        get_mode_views,
        get_opt_parsed,
        get_opt_run_record,
        get_opt_selected_step,
        get_provision_view,
        set_active_tab,
        set_env_refresh,
        set_last_upload_sig,
        set_ledger,
        set_mode_views,
        set_opt_parsed,
        set_opt_run_record,
        set_opt_selected_step,
        set_provision_view,
    )


@app.cell
def mlip_config_controls_cell(
    ARCH_TO_ENV_GROUP,
    MLIP_ENV_CATALOG,
    mo,
):
    arch_ui = mo.ui.dropdown(
        options=list(ARCH_TO_ENV_GROUP.keys()),
        value="mace_mp",
        label="MLIP Architecture (`arch`)",
    )
    device_ui = mo.ui.dropdown(
        options=["cpu", "cuda", "mps", "xpu"],
        value="cpu",
        label="Compute Device (`device`)",
    )
    dispersion_ui = mo.ui.checkbox(
        value=False,
        label="Enable D3 Dispersion (`dispersion=True`)",
    )
    calc_kwargs_ui = mo.ui.text(
        value="{}",
        label="Extra `calc_kwargs` (JSON dict)",
    )
    force_rerun_ui = mo.ui.checkbox(
        value=False,
        label="Force Re-run (bypass SHA-256 cache)",
    )

    env_groups_select_ui = mo.ui.multiselect(
        options=list(MLIP_ENV_CATALOG.keys()),
        value=["mace"],
        label="Select Conflicting MLIP Environment Groups to Provision (`.venvs/janus-<group>`)",
    )
    env_include_d3_ui = mo.ui.checkbox(
        value=True,
        label="Include `janus-core[d3]` (`torch-dftd`) in provisioned venvs",
    )
    provision_venvs_btn = mo.ui.run_button(
        label="Provision Selected MLIP Venvs via uv",
        kind="warn",
    )
    return (
        arch_ui,
        calc_kwargs_ui,
        device_ui,
        dispersion_ui,
        env_groups_select_ui,
        env_include_d3_ui,
        force_rerun_ui,
        provision_venvs_btn,
    )


@app.cell
def mlip_env_controls_cell(
    DEFAULT_JANUS_PYTHON,
    DEFAULT_VENVS_ROOT,
    discover_local_environments,
    get_env_refresh,
    mo,
):
    _ = get_env_refresh()
    _env_options = discover_local_environments(
        venvs_root=DEFAULT_VENVS_ROOT,
        base_python=DEFAULT_JANUS_PYTHON,
    )
    discovered_env_ui = mo.ui.dropdown(
        options=_env_options,
        value="Auto (.venvs/janus-<group> or base janus)",
        label="Existing Environment",
    )
    custom_env_path_ui = mo.ui.text(
        value="",
        placeholder="e.g. /opt/micromamba/envs/my-env or /path/to/bin/python",
        label="Custom Env Root or Python Path (override)",
    )
    return custom_env_path_ui, discovered_env_ui


@app.cell
def mlip_model_control_cell(
    ARCH_REQUIRES_EXPLICIT_MODEL,
    DEFAULT_MODEL_BY_ARCH,
    arch_ui,
    mo,
):
    _default_model = DEFAULT_MODEL_BY_ARCH.get(arch_ui.value) or ""
    _placeholder = (
        "Required checkpoint path (e.g. model.model)"
        if arch_ui.value in ARCH_REQUIRES_EXPLICIT_MODEL
        else "Leave blank for architecture default"
    )
    model_ui = mo.ui.text(
        value=_default_model,
        placeholder=_placeholder,
        label=f"Model / Checkpoint (`{arch_ui.value}`)",
    )
    return (model_ui,)


@app.cell
def mlip_resolved_config_cell(
    MLIPConfig,
    MLIP_ENV_CATALOG,
    arch_ui,
    build_provision_commands,
    calc_kwargs_ui,
    custom_env_path_ui,
    device_ui,
    discovered_env_ui,
    dispersion_ui,
    env_groups_select_ui,
    env_include_d3_ui,
    force_rerun_ui,
    get_env_refresh,
    json,
    mo,
    model_ui,
    provision_venvs_btn,
    resolve_env_status,
):
    _ = get_env_refresh()
    try:
        _parsed_kwargs = json.loads(calc_kwargs_ui.value or "{}")
        if not isinstance(_parsed_kwargs, dict):
            _parsed_kwargs = {}
        _kwargs_err = None
    except Exception as _exc:
        _parsed_kwargs = {}
        _kwargs_err = str(_exc)

    _effective_custom_env = (
        custom_env_path_ui.value.strip() or (discovered_env_ui.value or "").strip() or None
    )
    _clean_model = model_ui.value.strip() if model_ui.value else None
    active_mlip_config = MLIPConfig(
        arch=arch_ui.value,
        model=_clean_model,
        device=device_ui.value,
        dispersion=bool(dispersion_ui.value),
        calc_kwargs=_parsed_kwargs,
        custom_env_path=_effective_custom_env,
    )

    active_env_status = resolve_env_status(
        arch_ui.value,  # type: ignore[arg-type]
        custom_env_path=_effective_custom_env,
    )
    _badge_kind = (
        "success"
        if active_env_status.is_ready
        else ("danger" if active_env_status.source == "custom_env" else "warn")
    )

    _preview_cmds = []
    for _grp in env_groups_select_ui.value:
        for _cmd in build_provision_commands(_grp, include_d3=bool(env_include_d3_ui.value)):
            _preview_cmds.append(" ".join(_cmd))
    _cmd_preview_md = (
        "```bash\n" + "\n".join(_preview_cmds) + "\n```"
        if _preview_cmds
        else "_Select one or more MLIP groups above to preview `uv` provisioning commands._"
    )

    _catalog_rows = [
        {
            "Env Group": _spec.group,
            "Architectures": ", ".join(_spec.architectures),
            "Extra": _spec.janus_extra,
            "Default Model": _spec.default_model or "(user path / built-in)",
            "Conflict Rationale": _spec.conflict_note,
        }
        for _spec in MLIP_ENV_CATALOG.values()
    ]

    mlip_header_panel = mo.vstack(
        [
            mo.md("### 1. Shared MLIP Configuration (Single Active Calculator)"),
            mo.hstack(
                [arch_ui, model_ui, device_ui, dispersion_ui, force_rerun_ui],
                justify="start",
                wrap=True,
            ),
            mo.hstack(
                [discovered_env_ui, custom_env_path_ui, calc_kwargs_ui],
                justify="start",
                wrap=True,
            ),
            (
                mo.callout(
                    mo.md(f"Invalid `calc_kwargs` JSON: `{_kwargs_err}`"),
                    kind="danger",
                )
                if _kwargs_err
                else mo.callout(
                    mo.md(
                        f"**Environment Status (`{active_env_status.group}`)**: `{active_env_status.source}` — "
                        f"{active_env_status.detail}"
                    ),
                    kind=_badge_kind,
                )
            ),
            mo.accordion(
                {
                    "Provision Conflicting MLIP Virtual Environments (`.venvs/janus-<group>`)": mo.vstack(
                        [
                            mo.md(
                                "Because MLIP packages (`mace`, `sevennet`, `chgnet`, `orb`, `mattersim`, "
                                "`fairchem`, `nequip`, `dpa3`, `grace`, `upet`) pin conflicting `torch`/`e3nn`/`dgl` "
                                "versions, select which isolated environments you want to provision on demand via `uv`:"
                            ),
                            mo.hstack(
                                [
                                    env_groups_select_ui,
                                    env_include_d3_ui,
                                    provision_venvs_btn,
                                ],
                                justify="start",
                                wrap=True,
                            ),
                            mo.md(_cmd_preview_md),
                            mo.ui.table(_catalog_rows, selection=None),
                        ]
                    )
                }
            ),
        ]
    )
    return active_env_status, active_mlip_config, mlip_header_panel


@app.cell
def env_provision_exec_cell(
    env_groups_select_ui,
    env_include_d3_ui,
    mo,
    provision_mlip_env,
    provision_venvs_btn,
    set_env_refresh,
    set_provision_view,
):
    mo.stop(not provision_venvs_btn.value, None)
    _logs = []
    _all_ok = True
    for _grp in env_groups_select_ui.value:
        _ok, _log = provision_mlip_env(_grp, include_d3=bool(env_include_d3_ui.value))
        _all_ok = _all_ok and _ok
        _logs.append(f"=== janus-{_grp} ({'OK' if _ok else 'FAILED'}) ===\n{_log}")
    set_env_refresh(lambda v: v + 1)
    set_provision_view(
        mo.callout(
            mo.md("```text\n" + "\n\n".join(_logs) + "\n```"),
            kind="success" if _all_ok else "danger",
        )
    )
    return ()


@app.cell
def structure_upload_control_cell(mo):
    upload_struct_ui = mo.ui.file(
        filetypes=[".xyz", ".extxyz", ".cif", ".vasp", ".poscar", ".traj"],
        multiple=False,
        label="Upload Custom Structure (.cif / .extxyz / .xyz / POSCAR)",
    )
    show_3d_preview_ui = mo.ui.checkbox(
        value=True,
        label="Show 3D Structure Preview (Chemiscope)",
    )
    preview_supercell_ui = mo.ui.dropdown(
        options={
            "1×1×1 (Unit Cell)": (1, 1, 1),
            "2×2×1 (Surface / Slab Tiling)": (2, 2, 1),
            "2×2×2 (Bulk Supercell Tiling)": (2, 2, 2),
        },
        value="1×1×1 (Unit Cell)",
        label="3D Preview Tiling",
    )
    return preview_supercell_ui, show_3d_preview_ui, upload_struct_ui


@app.cell
def structure_ledger_controls_cell(
    WorkspaceLedger,
    get_last_upload_sig,
    get_ledger,
    hashlib,
    load_uploaded_structure,
    mo,
    set_last_upload_sig,
    set_ledger,
    upload_struct_ui,
):
    _ledger = get_ledger()
    upload_error_msg = None
    _target_label = None

    if upload_struct_ui.value:
        _uploaded_file = upload_struct_ui.value[0]
        _fname = _uploaded_file.name
        _raw_bytes = _uploaded_file.contents
        _upload_sig = (_fname, hashlib.sha256(_raw_bytes).hexdigest()[:16])
        if _upload_sig != get_last_upload_sig():
            try:
                _up_atoms = load_uploaded_structure(_fname, _raw_bytes)
                _new_ledger = WorkspaceLedger(
                    structures=dict(_ledger.structures),
                    runs=dict(_ledger.runs),
                )
                _entry, _ = _new_ledger.register_or_get_structure(
                    _up_atoms,
                    name=f"Uploaded {_fname}",
                    source_mode="upload",
                    is_relaxed=False,
                )
                _ledger = _new_ledger
                set_ledger(_new_ledger)
                set_last_upload_sig(_upload_sig)
                _target_label = _entry.label
            except Exception as _u_err:
                upload_error_msg = f"Failed to parse uploaded file `{_fname}`: `{_u_err}`"
                set_last_upload_sig(_upload_sig)

    _opts = _ledger.structure_options()
    _labels = list(_opts.keys())
    _default_label = (
        _target_label
        if (_target_label and _target_label in _opts)
        else (_labels[-1] if len(_labels) > 5 else _labels[0])
    )
    active_struct_dropdown = mo.ui.dropdown(
        options=_opts,
        value=_default_label,
        label="Active Structure from Relay Ledger",
    )
    return active_struct_dropdown, upload_error_msg


@app.cell
def structure_ledger_panel_cell(
    active_struct_dropdown,
    build_preflight_chemiscope_widget,
    get_ledger,
    mo,
    preview_supercell_ui,
    show_3d_preview_ui,
    upload_error_msg,
    upload_struct_ui,
):
    _ledger = get_ledger()
    active_struct_entry = _ledger.get_structure(active_struct_dropdown.value or "struct_0")
    _rep = active_struct_entry.report

    _ledger_rows = [
        {
            "ID": _e.entry_id,
            "Label": _e.label,
            "Formula": _e.report.formula,
            "Atoms": _e.report.n_atoms,
            "PBC": str(_e.report.pbc),
            "Volume (Å³)": (f"{_e.report.volume:.2f}" if _e.report.volume is not None else "N/A"),
            "d_min (Å)": (
                f"{_e.report.min_distance:.3f}" if _e.report.min_distance is not None else "N/A"
            ),
            "Source": _e.source_mode,
            "Relaxed": "Yes" if _e.is_relaxed else "No",
        }
        for _e in _ledger.structures.values()
    ]

    _vol_txt = f"{_rep.volume:.2f} Å³" if _rep.volume is not None else "Non-periodic"
    _dmin_txt = f"{_rep.min_distance:.3f} Å" if _rep.min_distance is not None else "N/A"
    _status_kind = "success" if _rep.is_valid else "danger"
    _supercell_tuple = preview_supercell_ui.value or (1, 1, 1)

    _panel_items = [
        mo.md("### 2. Structure Relay Ledger & Pre-Flight Inspector"),
        mo.hstack(
            [
                active_struct_dropdown,
                upload_struct_ui,
                show_3d_preview_ui,
                preview_supercell_ui,
            ],
            justify="start",
            wrap=True,
        ),
    ]
    if upload_error_msg:
        _panel_items.append(
            mo.callout(
                mo.md(upload_error_msg),
                kind="danger",
            )
        )
    _panel_items.append(
        mo.callout(
            mo.md(
                f"**Inspector (`tools/structure_inspector.py`)**: "
                f"`{'PASS' if _rep.is_valid else 'FAIL'}` | "
                f"**Formula**: `{_rep.formula}` ({_rep.n_atoms} atoms) | "
                f"**PBC**: `{_rep.pbc}` | **Volume**: `{_vol_txt}` | **d_min**: `{_dmin_txt}`"
            ),
            kind=_status_kind,
        )
    )
    if show_3d_preview_ui.value:
        _panel_items.append(
            mo.lazy(
                lambda _atoms=active_struct_entry.atoms, _sc=_supercell_tuple: (
                    build_preflight_chemiscope_widget(
                        _atoms,
                        supercell=_sc,
                    )
                )
            )
        )
    _panel_items.append(
        mo.accordion(
            {
                f"Structure Relay Ledger ({len(_ledger_rows)} structures available across all 8 tabs)": mo.ui.table(
                    _ledger_rows, selection=None
                )
            }
        )
    )

    structure_header_panel = mo.vstack(_panel_items)
    return active_struct_entry, structure_header_panel


@app.cell
def sp_controls_cell(mo):
    sp_props_ui = mo.ui.multiselect(
        options=["energy", "forces", "stress", "hessian"],
        value=["energy", "forces", "stress"],
        label="Properties (`--properties`)",
    )
    sp_arrows_ui = mo.ui.checkbox(
        value=True,
        label="Show 3D Force Vectors in Chemiscope",
    )
    sp_run_btn = mo.ui.run_button(
        label="Run Singlepoint Calculation",
        kind="success",
    )
    return sp_arrows_ui, sp_props_ui, sp_run_btn


@app.cell
def sp_exec_cell(
    active_env_status,
    active_mlip_config,
    active_struct_entry,
    build_chemiscope_widget,
    chart_singlepoint_forces,
    execute_janus_mode,
    force_rerun_ui,
    mo,
    parse_singlepoint_run,
    set_mode_views,
    sp_arrows_ui,
    sp_props_ui,
    sp_run_btn,
):
    mo.stop(not sp_run_btn.value, None)
    if not active_env_status.is_ready:
        _sp_view = mo.callout(
            mo.md(
                f"Cannot run `singlepoint`: {active_env_status.detail}\n\n"
                "Open **Provision Conflicting MLIP Virtual Environments** above to install it."
            ),
            kind="warn",
        )
    else:
        _params = {
            "properties": list(sp_props_ui.value)
            if sp_props_ui.value
            else ["energy", "forces", "stress"]
        }
        _rec = execute_janus_mode(
            "singlepoint",
            active_mlip_config,
            active_struct_entry,
            _params,
            python_executable=active_env_status.python_executable,
            force_rerun=bool(force_rerun_ui.value),
        )
        if _rec.status == "failed":
            _sp_view = mo.callout(
                mo.md(
                    f"**Singlepoint Run Failed** (`{_rec.run_id}`)\n\n"
                    f"Command: `{' '.join(_rec.cli_command)}`\n\n"
                    f"```text\n{_rec.stdout_stderr}\n```"
                ),
                kind="danger",
            )
        else:
            _parsed = parse_singlepoint_run(_rec.run_dir, active_mlip_config.arch)
            _chart = mo.ui.altair_chart(chart_singlepoint_forces(_parsed.atom_df))
            _cs_view = mo.lazy(
                lambda _structs=_parsed.structures, _arrows=bool(sp_arrows_ui.value): (
                    build_chemiscope_widget(
                        _structs,
                        mode="structure",
                        show_force_arrows=_arrows,
                    )
                )
            )
            _stress_str = (
                ", ".join(f"{v:.4f}" for v in _parsed.stress_voigt_ev_ang3)
                if _parsed.stress_voigt_ev_ang3
                else "N/A"
            )
            _sp_view = mo.vstack(
                [
                    mo.callout(
                        mo.md(
                            f"**Singlepoint {_rec.status.upper()}** (`{_rec.run_id}`, `{_rec.duration_s:.2f}s`) | "
                            f"**Energy**: `{_parsed.energy_ev:.5f} eV` (`{_parsed.energy_per_atom_ev:.5f} eV/atom`) | "
                            f"**Max Force**: `{_parsed.max_force_ev_ang:.5f} eV/Å` | "
                            f"**Voigt Stress (eV/Å³)**: `[{_stress_str}]`"
                        ),
                        kind="success",
                    ),
                    mo.hstack([_chart, _cs_view], widths="equal", wrap=True),
                    mo.ui.table(_parsed.atom_df, selection=None),
                    mo.accordion(
                        {
                            "Reproducible YAML Config & CLI Command": mo.md(
                                f"**CLI**:\n```bash\n{' '.join(_rec.cli_command)}\n```\n\n"
                                f"**`config.yml`**:\n```yaml\n{_rec.config_yaml_path.read_text()}\n```"
                            )
                        }
                    ),
                ]
            )
    set_mode_views(lambda views, _v=_sp_view: {**views, "singlepoint": _v})
    return ()


@app.cell
def opt_controls_cell(mo):
    opt_optimizer_ui = mo.ui.dropdown(
        options=["LBFGS", "BFGS", "FIRE"],
        value="LBFGS",
        label="Optimizer (`--optimizer`)",
    )
    opt_fmax_ui = mo.ui.number(
        start=0.0001,
        stop=1.0,
        step=0.005,
        value=0.01,
        label="Force Convergence `fmax` (eV/Å)",
    )
    opt_steps_ui = mo.ui.number(
        start=1,
        stop=5000,
        step=50,
        value=200,
        label="Max Steps (`--steps`)",
    )
    opt_cell_mode_ui = mo.ui.dropdown(
        options={
            "Full Cell + Angles (FrechetCellFilter)": "full",
            "Cell Lengths Only (FrechetCellFilter)": "lengths",
            "Fixed Cell (Atomic Positions Only)": "fixed",
        },
        value="Full Cell + Angles (FrechetCellFilter)",
        label="Cell Relaxation Mode",
    )
    opt_pressure_ui = mo.ui.number(
        start=-100.0,
        stop=100.0,
        step=0.5,
        value=0.0,
        label="Target Pressure (GPa)",
    )
    opt_symmetrize_ui = mo.ui.checkbox(
        value=False,
        label="Symmetrize (`--symmetrize`)",
    )
    opt_run_btn = mo.ui.run_button(
        label="Run Geometry Optimization",
        kind="success",
    )
    return (
        opt_cell_mode_ui,
        opt_fmax_ui,
        opt_optimizer_ui,
        opt_pressure_ui,
        opt_run_btn,
        opt_steps_ui,
        opt_symmetrize_ui,
    )


@app.cell
def opt_exec_cell(
    active_env_status,
    active_mlip_config,
    active_struct_entry,
    execute_janus_mode,
    force_rerun_ui,
    mo,
    opt_cell_mode_ui,
    opt_fmax_ui,
    opt_optimizer_ui,
    opt_pressure_ui,
    opt_run_btn,
    opt_steps_ui,
    opt_symmetrize_ui,
    parse_geomopt_run,
    set_mode_views,
    set_opt_parsed,
    set_opt_run_record,
    set_opt_selected_step,
):
    mo.stop(not opt_run_btn.value, None)
    if not active_env_status.is_ready:
        set_opt_run_record(None)
        set_opt_parsed(None)
        set_opt_selected_step(None)
        _warn = mo.callout(
            mo.md(f"Cannot run `geomopt`: {active_env_status.detail}"),
            kind="warn",
        )
        set_mode_views(lambda v, _w=_warn: {**v, "geomopt": _w})
    else:
        _cell_mode = opt_cell_mode_ui.value
        _params = {
            "optimizer": opt_optimizer_ui.value,
            "fmax": float(opt_fmax_ui.value),
            "steps": int(opt_steps_ui.value),
            "opt_cell_fully": _cell_mode == "full",
            "opt_cell_lengths": _cell_mode == "lengths",
            "filter_class": "FrechetCellFilter",
            "pressure": float(opt_pressure_ui.value),
            "symmetrize": bool(opt_symmetrize_ui.value),
        }
        _rec = execute_janus_mode(
            "geomopt",
            active_mlip_config,
            active_struct_entry,
            _params,
            python_executable=active_env_status.python_executable,
            force_rerun=bool(force_rerun_ui.value),
        )
        if _rec.status == "failed":
            set_opt_run_record(_rec)
            set_opt_parsed(None)
            set_opt_selected_step(None)
        else:
            _parsed = parse_geomopt_run(
                _rec.run_dir,
                active_mlip_config.arch,
                fmax_target=float(opt_fmax_ui.value),
            )
            set_opt_run_record(_rec)
            set_opt_parsed(_parsed)
            set_opt_selected_step(None)
    return ()


@app.cell
def opt_view_cell(
    WorkspaceLedger,
    build_chemiscope_widget,
    chart_geomopt_convergence,
    extract_selected_step,
    get_opt_parsed,
    get_opt_run_record,
    get_opt_selected_step,
    mo,
    set_ledger,
    set_mode_views,
    set_opt_selected_step,
):
    _rec = get_opt_run_record()
    _parsed = get_opt_parsed()
    _raw_step = get_opt_selected_step()

    if _rec is None and _parsed is None:
        pass
    elif _rec is not None and _rec.status == "failed":
        _failed_view = mo.callout(
            mo.md(
                f"**Geometry Optimization Failed** (`{_rec.run_id}`)\n\n"
                f"```text\n{_rec.stdout_stderr}\n```"
            ),
            kind="danger",
        )
        set_mode_views(lambda v, _f=_failed_view: {**v, "geomopt": _f})
    elif _parsed is not None:
        if _parsed.traj_df.empty or len(_parsed.trajectory) == 0:
            _empty_view = mo.vstack(
                [mo.callout(mo.md("Geometry optimization finished with 0 trajectory frames recorded."), kind="warn")]
            )
            set_mode_views(lambda v, _e=_empty_view: {**v, "geomopt": _e})
        else:
            _selected_step = (
                _raw_step
                if (_raw_step is not None and 0 <= _raw_step < len(_parsed.trajectory))
                else None
            )

            _chart = chart_geomopt_convergence(_parsed.traj_df, selected_step=_selected_step)
            _chart_ui = mo.ui.altair_chart(
                _chart,
                on_change=lambda df, _ext=extract_selected_step, _setter=set_opt_selected_step, _p=_parsed: _setter(
                    _ext(df, _p.traj_df)
                ),
            )

            def _make_updater(_atoms_val, _name_val, _mode_val, _rel_val, _rec_val, _led_cls):
                def _updater(led):
                    new_led = _led_cls(structures=dict(led.structures), runs=dict(led.runs))
                    new_led.register_run(_rec_val)
                    new_led.register_structure(
                        _atoms_val,
                        name=_name_val,
                        source_mode=_mode_val,
                        is_relaxed=_rel_val,
                        provenance_run_id=_rec_val.run_id,
                    )
                    return new_led

                return _updater

            _n_frames = len(_parsed.trajectory)

            if _selected_step is None:
                _cs_view = mo.lazy(
                    lambda _t=_parsed.trajectory, _build=build_chemiscope_widget: _build(
                        _t, mode="structure", show_force_arrows=True
                    )
                )
                _final_e = _parsed.traj_df["energy_ev"].iloc[-1]
                _final_f = _parsed.traj_df["max_force_ev_ang"].iloc[-1]
                _banner = mo.callout(
                    mo.md(
                        f"**GeomOpt {_rec.status.upper()}** (`{_rec.run_id}`, `{_rec.duration_s:.2f}s`) | "
                        f"**Converged**: `{_parsed.converged}` | "
                        f"**Frames**: `{len(_parsed.trajectory)}` | "
                        f"**Final Energy**: `{_final_e:.5f} eV` | "
                        f"**Final fmax**: `{_final_f:.5f} eV/Å`"
                    ),
                    kind=("success" if _parsed.converged else "warn"),
                )
                _nav_row_or_empty = mo.md("")
                if _parsed.converged:
                    _target_label = "Promote Relaxed Structure to Relay Ledger"
                    _target_name = f"Relaxed ({_rec.mlip_config.arch})"
                    _target_mode = "geomopt"
                    _target_is_relaxed = True
                else:
                    _target_label = "Promote Final Structure (Unconverged) to Relay Ledger"
                    _target_name = f"Final Unconverged ({_rec.mlip_config.arch})"
                    _target_mode = "geomopt"
                    _target_is_relaxed = False
                _target_atoms = _parsed.optimized_atoms
            else:
                _step_atoms = _parsed.trajectory[_selected_step]
                _cs_view = mo.lazy(
                    lambda _a=_step_atoms, _build=build_chemiscope_widget: _build(
                        [_a], mode="structure", show_force_arrows=True
                    )
                )
                _step_e = _parsed.traj_df["energy_ev"].iloc[_selected_step]
                _step_de = _parsed.traj_df["delta_energy_mev_atom"].iloc[_selected_step]
                _step_f = _parsed.traj_df["max_force_ev_ang"].iloc[_selected_step]
                _step_vol = _parsed.traj_df["volume_ang3"].iloc[_selected_step]
                _banner = mo.callout(
                    mo.md(
                        f"**Inspecting Step {_selected_step} of {_n_frames - 1}** | "
                        f"**Energy**: `{_step_e:.5f} eV` (ΔE: `{_step_de:.3f} meV/atom`) | "
                        f"**Max Force**: `{_step_f:.5f} eV/Å` | "
                        f"**Volume**: `{_step_vol:.2f} Å³`"
                    ),
                    kind="info",
                )
                _prev_btn = mo.ui.button(
                    label="◀ Prev",
                    on_change=lambda _, _s=_selected_step, _setter=set_opt_selected_step: _setter(max(0, _s - 1)),
                    disabled=(_selected_step == 0),
                )
                _next_btn = mo.ui.button(
                    label="Next ▶",
                    on_change=lambda _, _s=_selected_step, _n=_n_frames, _setter=set_opt_selected_step: _setter(
                        min(_n - 1, _s + 1)
                    ),
                    disabled=(_selected_step >= _n_frames - 1),
                )
                _reset_btn = mo.ui.button(
                    label="Reset to Full Trajectory",
                    on_change=lambda _, _setter=set_opt_selected_step: _setter(None),
                )
                _nav_row_or_empty = mo.hstack([_prev_btn, _next_btn, _reset_btn], justify="start")

                if _selected_step == _n_frames - 1 and _parsed.converged is True:
                    _target_label = "Promote Relaxed Structure to Relay Ledger"
                    _target_name = f"Relaxed ({_rec.mlip_config.arch})"
                    _target_mode = "geomopt"
                    _target_is_relaxed = True
                else:
                    _target_label = f"Promote Step {_selected_step} Structure to Relay Ledger"
                    _target_name = f"GeomOpt Step {_selected_step} ({_rec.mlip_config.arch})"
                    _target_mode = "geomopt_intermediate"
                    _target_is_relaxed = False
                _target_atoms = _step_atoms

            _updater = _make_updater(
                _target_atoms,
                _target_name,
                _target_mode,
                _target_is_relaxed,
                _rec,
                WorkspaceLedger,
            )
            _promote_btn = mo.ui.button(
                label=_target_label,
                on_change=lambda _, _u=_updater, _setter=set_ledger: _setter(_u),
                kind="success",
                disabled=bool(len(_parsed.trajectory) == 0 or _parsed.traj_df.empty),
            )

            _accordion = mo.accordion(
                {
                    "Reproducible YAML Config & CLI Command": mo.md(
                        f"**CLI**:\n```bash\n{' '.join(_rec.cli_command)}\n```\n\n"
                        f"**`config.yml`**:\n```yaml\n{_rec.config_yaml_path.read_text()}\n```"
                    )
                }
            )

            _opt_view = mo.vstack(
                [
                    _banner,
                    _nav_row_or_empty,
                    _promote_btn,
                    mo.hstack([_chart_ui, _cs_view], widths="equal", wrap=True),
                    _accordion,
                ]
            )
            set_mode_views(lambda views, _v=_opt_view: {**views, "geomopt": _v})

    return ()


@app.cell
def md_controls_cell(mo):
    md_ensemble_ui = mo.ui.dropdown(
        options=["nvt", "npt", "nve", "nph", "nvt-nh", "nvt-csvr", "npt-mtk"],
        value="nvt",
        label="Ensemble (`--ensemble`)",
    )
    md_temp_ui = mo.ui.number(
        start=1.0,
        stop=5000.0,
        step=25.0,
        value=300.0,
        label="Temperature (K)",
    )
    md_steps_ui = mo.ui.number(
        start=10,
        stop=100000,
        step=50,
        value=100,
        label="MD Steps (`--steps`)",
    )
    md_timestep_ui = mo.ui.number(
        start=0.1,
        stop=5.0,
        step=0.25,
        value=1.0,
        label="Timestep (fs)",
    )
    md_stats_every_ui = mo.ui.number(
        start=1,
        stop=1000,
        step=5,
        value=10,
        label="Stats/Traj Output Interval",
    )
    md_rdf_ui = mo.ui.checkbox(
        value=True,
        label="Compute On-the-Fly RDF g(r) (`post_process_kwargs`)",
    )
    md_run_btn = mo.ui.run_button(
        label="Run Molecular Dynamics",
        kind="success",
    )
    return (
        md_ensemble_ui,
        md_rdf_ui,
        md_run_btn,
        md_stats_every_ui,
        md_steps_ui,
        md_temp_ui,
        md_timestep_ui,
    )


@app.cell
def md_exec_cell(
    WorkspaceLedger,
    active_env_status,
    active_mlip_config,
    active_struct_entry,
    build_chemiscope_widget,
    chart_md_thermodynamics,
    execute_janus_mode,
    force_rerun_ui,
    md_ensemble_ui,
    md_rdf_ui,
    md_run_btn,
    md_stats_every_ui,
    md_steps_ui,
    md_temp_ui,
    md_timestep_ui,
    mo,
    parse_md_run,
    set_ledger,
    set_mode_views,
):
    mo.stop(not md_run_btn.value, None)
    if not active_env_status.is_ready:
        _md_view = mo.callout(
            mo.md(f"Cannot run `md`: {active_env_status.detail}"),
            kind="warn",
        )
    else:
        _params = {
            "ensemble": md_ensemble_ui.value,
            "temp": float(md_temp_ui.value),
            "steps": int(md_steps_ui.value),
            "timestep": float(md_timestep_ui.value),
            "stats_every": int(md_stats_every_ui.value),
            "traj_every": int(md_stats_every_ui.value),
            "rdf_compute": bool(md_rdf_ui.value),
        }
        _rec = execute_janus_mode(
            "md",
            active_mlip_config,
            active_struct_entry,
            _params,
            python_executable=active_env_status.python_executable,
            force_rerun=bool(force_rerun_ui.value),
        )
        if _rec.status == "failed":
            _md_view = mo.callout(
                mo.md(
                    f"**MD Simulation Failed** (`{_rec.run_id}`)\n\n"
                    f"```text\n{_rec.stdout_stderr}\n```"
                ),
                kind="danger",
            )
        else:
            _parsed = parse_md_run(_rec.run_dir, active_mlip_config.arch)
            _final_atoms = _parsed.final_atoms

            def _promote_md_final(
                _,
                _rec_val=_rec,
                _atoms_val=_final_atoms,
                _ens_val=md_ensemble_ui.value,
                _temp_val=md_temp_ui.value,
            ):
                def _updater(led):
                    new_led = WorkspaceLedger(
                        structures=dict(led.structures),
                        runs=dict(led.runs),
                    )
                    new_led.register_run(_rec_val)
                    new_led.register_structure(
                        _atoms_val,
                        name=f"MD {_ens_val} {_temp_val:.0f}K final",
                        source_mode="md",
                        is_relaxed=False,
                        provenance_run_id=_rec_val.run_id,
                    )
                    return new_led

                set_ledger(_updater)

            _promote_btn = mo.ui.button(
                label="Promote Final MD Frame to Relay Ledger",
                on_change=_promote_md_final,
                kind="neutral",
            )
            _chart = mo.ui.altair_chart(chart_md_thermodynamics(_parsed.stats_df, _parsed.rdf_df))
            _cs_view = mo.lazy(
                lambda _traj=_parsed.trajectory: build_chemiscope_widget(
                    _traj,
                    mode="structure",
                    show_force_arrows=True,
                )
            )
            _md_view = mo.vstack(
                [
                    mo.callout(
                        mo.md(
                            f"**MD {_rec.status.upper()}** (`{_rec.run_id}`, `{_rec.duration_s:.2f}s`) | "
                            f"**Ensemble**: `{md_ensemble_ui.value}` | "
                            f"**Saved Trajectory Frames**: `{len(_parsed.trajectory)}`"
                        ),
                        kind="success",
                    ),
                    _promote_btn,
                    mo.hstack([_chart, _cs_view], widths="equal", wrap=True),
                    mo.accordion(
                        {
                            "Reproducible YAML Config & CLI Command": mo.md(
                                f"**CLI**:\n```bash\n{' '.join(_rec.cli_command)}\n```\n\n"
                                f"**`config.yml`**:\n```yaml\n{_rec.config_yaml_path.read_text()}\n```"
                            )
                        }
                    ),
                ]
            )
    set_mode_views(lambda views, _v=_md_view: {**views, "md": _v})
    return ()


@app.cell
def ph_controls_cell(mo):
    ph_supercell_ui = mo.ui.text(
        value="2 2 2",
        label="Supercell (`--supercell`)",
    )
    ph_disp_ui = mo.ui.number(
        start=0.001,
        stop=0.1,
        step=0.005,
        value=0.01,
        label="Atomic Displacement (Å)",
    )
    ph_mesh_ui = mo.ui.text(
        value="10, 10, 10",
        label="q-Mesh (`--mesh`)",
    )
    ph_minimize_ui = mo.ui.checkbox(
        value=False,
        label="Pre-relax Structure (`--minimize`)",
    )
    ph_symmetrize_ui = mo.ui.checkbox(
        value=True,
        label="Symmetrize Force Constants (`--symmetrize`)",
    )
    ph_run_btn = mo.ui.run_button(
        label="Run Phonon Calculation",
        kind="success",
    )
    return (
        ph_disp_ui,
        ph_mesh_ui,
        ph_minimize_ui,
        ph_run_btn,
        ph_supercell_ui,
        ph_symmetrize_ui,
    )


@app.cell
def ph_exec_cell(
    active_env_status,
    active_mlip_config,
    active_struct_entry,
    build_chemiscope_widget,
    chart_phonons_spectrum,
    execute_janus_mode,
    force_rerun_ui,
    mo,
    parse_phonons_run,
    ph_disp_ui,
    ph_mesh_ui,
    ph_minimize_ui,
    ph_run_btn,
    ph_supercell_ui,
    ph_symmetrize_ui,
    set_mode_views,
):
    mo.stop(not ph_run_btn.value, None)
    if not active_env_status.is_ready:
        _ph_view = mo.callout(
            mo.md(f"Cannot run `phonons`: {active_env_status.detail}"),
            kind="warn",
        )
    else:
        try:
            _mesh_list = [int(x.strip()) for x in ph_mesh_ui.value.split(",") if x.strip()]
            if len(_mesh_list) != 3:
                _mesh_list = [10, 10, 10]
        except ValueError:
            _mesh_list = [10, 10, 10]

        _params = {
            "supercell": ph_supercell_ui.value.strip() or "2 2 2",
            "displacement": float(ph_disp_ui.value),
            "mesh": _mesh_list,
            "minimize": bool(ph_minimize_ui.value),
            "symmetrize": bool(ph_symmetrize_ui.value),
            "bands": True,
            "dos": True,
            "thermal": True,
        }
        _rec = execute_janus_mode(
            "phonons",
            active_mlip_config,
            active_struct_entry,
            _params,
            python_executable=active_env_status.python_executable,
            force_rerun=bool(force_rerun_ui.value),
        )
        if _rec.status == "failed":
            _ph_view = mo.callout(
                mo.md(
                    f"**Phonon Calculation Failed** (`{_rec.run_id}`)\n\n"
                    f"```text\n{_rec.stdout_stderr}\n```"
                ),
                kind="danger",
            )
        else:
            _parsed = parse_phonons_run(_rec.run_dir, active_mlip_config.arch)
            _chart = mo.ui.altair_chart(
                chart_phonons_spectrum(_parsed.bands_df, _parsed.dos_df, _parsed.thermal_df)
            )
            _cs_view = mo.lazy(
                lambda _structs=(_parsed.band_structures or [active_struct_entry.atoms]): (
                    build_chemiscope_widget(
                        _structs,
                        mode="structure",
                        show_force_arrows=False,
                    )
                )
            )
            _min_f_str = (
                f"{_parsed.min_frequency_thz:.3f} THz"
                if _parsed.min_frequency_thz is not None
                else "N/A"
            )
            _ph_view = mo.vstack(
                [
                    mo.callout(
                        mo.md(
                            f"**Phonons {_rec.status.upper()}** (`{_rec.run_id}`, `{_rec.duration_s:.2f}s`) | "
                            f"**Min Frequency**: `{_min_f_str}` | "
                            f"**Dynamically Stable**: `{'No (Imaginary Modes < -0.05 THz)' if _parsed.has_imaginary_modes else 'Yes'}`"
                        ),
                        kind="warn" if _parsed.has_imaginary_modes else "success",
                    ),
                    mo.hstack([_chart, _cs_view], widths="equal", wrap=True),
                    mo.accordion(
                        {
                            "Reproducible YAML Config & CLI Command": mo.md(
                                f"**CLI**:\n```bash\n{' '.join(_rec.cli_command)}\n```\n\n"
                                f"**`config.yml`**:\n```yaml\n{_rec.config_yaml_path.read_text()}\n```"
                            )
                        }
                    ),
                ]
            )
    set_mode_views(lambda views, _v=_ph_view: {**views, "phonons": _v})
    return ()


@app.cell
def eos_controls_cell(mo):
    eos_type_ui = mo.ui.dropdown(
        options=[
            "birchmurnaghan",
            "murnaghan",
            "vinet",
            "pouriertarantola",
            "sjeos",
            "taylor",
            "antonschmidt",
        ],
        value="birchmurnaghan",
        label="EOS Analytic Form (`--eos-type`)",
    )
    eos_min_vol_ui = mo.ui.number(
        start=0.70,
        stop=0.99,
        step=0.01,
        value=0.95,
        label="Min Volume Scale (`--min-volume`)",
    )
    eos_max_vol_ui = mo.ui.number(
        start=1.01,
        stop=1.30,
        step=0.01,
        value=1.05,
        label="Max Volume Scale (`--max-volume`)",
    )
    eos_n_vols_ui = mo.ui.number(
        start=5,
        stop=25,
        step=2,
        value=7,
        label="Number of Volumes (`--n-volumes`)",
    )
    eos_minimize_all_ui = mo.ui.checkbox(
        value=False,
        label="Relax Atomic Coordinates at Each Volume (`--minimize-all`)",
    )
    eos_run_btn = mo.ui.run_button(
        label="Run Equation of State (EOS)",
        kind="success",
    )
    return (
        eos_max_vol_ui,
        eos_min_vol_ui,
        eos_minimize_all_ui,
        eos_n_vols_ui,
        eos_run_btn,
        eos_type_ui,
    )


@app.cell
def eos_exec_cell(
    active_env_status,
    active_mlip_config,
    active_struct_entry,
    build_chemiscope_widget,
    chart_eos_curve,
    eos_max_vol_ui,
    eos_min_vol_ui,
    eos_minimize_all_ui,
    eos_n_vols_ui,
    eos_run_btn,
    eos_type_ui,
    execute_janus_mode,
    force_rerun_ui,
    mo,
    parse_eos_run,
    set_mode_views,
):
    mo.stop(not eos_run_btn.value, None)
    if not active_env_status.is_ready:
        _eos_view = mo.callout(
            mo.md(f"Cannot run `eos`: {active_env_status.detail}"),
            kind="warn",
        )
    else:
        _params = {
            "eos_type": eos_type_ui.value,
            "min_volume": float(eos_min_vol_ui.value),
            "max_volume": float(eos_max_vol_ui.value),
            "n_volumes": int(eos_n_vols_ui.value),
            "minimize_all": bool(eos_minimize_all_ui.value),
        }
        _rec = execute_janus_mode(
            "eos",
            active_mlip_config,
            active_struct_entry,
            _params,
            python_executable=active_env_status.python_executable,
            force_rerun=bool(force_rerun_ui.value),
        )
        if _rec.status == "failed":
            _eos_view = mo.callout(
                mo.md(
                    f"**EOS Calculation Failed** (`{_rec.run_id}`)\n\n"
                    f"```text\n{_rec.stdout_stderr}\n```"
                ),
                kind="danger",
            )
        else:
            _parsed = parse_eos_run(
                _rec.run_dir,
                active_mlip_config.arch,
                eos_type=eos_type_ui.value,
            )
            _chart = mo.ui.altair_chart(
                chart_eos_curve(
                    _parsed.raw_df,
                    _parsed.fit_curve_df,
                    _parsed.v0_ang3,
                    _parsed.e0_ev,
                    _parsed.bulk_modulus_gpa,
                )
            )
            _cs_view = mo.lazy(
                lambda _structs=(_parsed.structures or [active_struct_entry.atoms]): (
                    build_chemiscope_widget(
                        _structs,
                        mode="structure",
                        show_force_arrows=True,
                    )
                )
            )
            _eos_view = mo.vstack(
                [
                    mo.callout(
                        mo.md(
                            f"**EOS {_rec.status.upper()}** (`{_rec.run_id}`, `{_rec.duration_s:.2f}s`) | "
                            f"**Bulk Modulus $B_0$**: `{_parsed.bulk_modulus_gpa:.2f} GPa` | "
                            f"**Equilibrium Volume $V_0$**: `{_parsed.v0_ang3:.3f} Å³` | "
                            f"**Minimum Energy $E_0$**: `{_parsed.e0_ev:.5f} eV`"
                        ),
                        kind="success",
                    ),
                    mo.hstack([_chart, _cs_view], widths="equal", wrap=True),
                    mo.ui.table(_parsed.raw_df, selection=None),
                    mo.accordion(
                        {
                            "Reproducible YAML Config & CLI Command": mo.md(
                                f"**CLI**:\n```bash\n{' '.join(_rec.cli_command)}\n```\n\n"
                                f"**`config.yml`**:\n```yaml\n{_rec.config_yaml_path.read_text()}\n```"
                            )
                        }
                    ),
                ]
            )
    set_mode_views(lambda views, _v=_eos_view: {**views, "eos": _v})
    return ()


@app.cell
def el_controls_cell(mo):
    el_norm_mag_ui = mo.ui.number(
        start=0.001,
        stop=0.05,
        step=0.005,
        value=0.01,
        label="Normal Strain Magnitude (`--normal-magnitude`)",
    )
    el_shear_mag_ui = mo.ui.number(
        start=0.005,
        stop=0.15,
        step=0.01,
        value=0.06,
        label="Shear Strain Magnitude (`--shear-magnitude`)",
    )
    el_n_strains_ui = mo.ui.number(
        start=3,
        stop=9,
        step=1,
        value=4,
        label="Strains per Deformation (`--n-strains`)",
    )
    el_minimize_all_ui = mo.ui.checkbox(
        value=False,
        label="Relax Internal Coordinates of Strained Cells (`--minimize-all`)",
    )
    el_run_btn = mo.ui.run_button(
        label="Run Elasticity Stiffness Tensor",
        kind="success",
    )
    return (
        el_minimize_all_ui,
        el_n_strains_ui,
        el_norm_mag_ui,
        el_run_btn,
        el_shear_mag_ui,
    )


@app.cell
def el_exec_cell(
    active_env_status,
    active_mlip_config,
    active_struct_entry,
    build_chemiscope_widget,
    chart_elasticity_tensor,
    el_minimize_all_ui,
    el_n_strains_ui,
    el_norm_mag_ui,
    el_run_btn,
    el_shear_mag_ui,
    execute_janus_mode,
    force_rerun_ui,
    mo,
    parse_elasticity_run,
    set_mode_views,
):
    mo.stop(not el_run_btn.value, None)
    if not active_env_status.is_ready:
        _el_view = mo.callout(
            mo.md(f"Cannot run `elasticity`: {active_env_status.detail}"),
            kind="warn",
        )
    else:
        _params = {
            "normal_magnitude": float(el_norm_mag_ui.value),
            "shear_magnitude": float(el_shear_mag_ui.value),
            "n_strains": int(el_n_strains_ui.value),
            "minimize_all": bool(el_minimize_all_ui.value),
        }
        _rec = execute_janus_mode(
            "elasticity",
            active_mlip_config,
            active_struct_entry,
            _params,
            python_executable=active_env_status.python_executable,
            force_rerun=bool(force_rerun_ui.value),
        )
        if _rec.status == "failed":
            _el_view = mo.callout(
                mo.md(
                    f"**Elasticity Calculation Failed** (`{_rec.run_id}`)\n\n"
                    f"```text\n{_rec.stdout_stderr}\n```"
                ),
                kind="danger",
            )
        else:
            _parsed = parse_elasticity_run(_rec.run_dir, active_mlip_config.arch)
            _chart = mo.ui.altair_chart(chart_elasticity_tensor(_parsed.c_ij_df, _parsed.moduli_df))
            _cs_view = mo.lazy(
                lambda _structs=(_parsed.structures or [active_struct_entry.atoms]): (
                    build_chemiscope_widget(
                        _structs,
                        mode="structure",
                        show_force_arrows=True,
                    )
                )
            )
            _el_view = mo.vstack(
                [
                    mo.callout(
                        mo.md(
                            f"**Elasticity {_rec.status.upper()}** (`{_rec.run_id}`, `{_rec.duration_s:.2f}s`) | "
                            f"**Born Mechanically Stable**: `{'Yes' if _parsed.is_mechanically_stable else 'No (Non-Positive Eigenvalue)'}` | "
                            f"**Poisson Ratio $\\nu$**: `{_parsed.poisson_ratio:.3f}` | "
                            f"**Universal Anisotropy $A^U$**: `{_parsed.universal_anisotropy:.3f}`"
                        ),
                        kind="success" if _parsed.is_mechanically_stable else "warn",
                    ),
                    mo.hstack([_chart, _cs_view], widths="equal", wrap=True),
                    mo.ui.table(_parsed.moduli_df, selection=None),
                    mo.accordion(
                        {
                            "Reproducible YAML Config & CLI Command": mo.md(
                                f"**CLI**:\n```bash\n{' '.join(_rec.cli_command)}\n```\n\n"
                                f"**`config.yml`**:\n```yaml\n{_rec.config_yaml_path.read_text()}\n```"
                            )
                        }
                    ),
                ]
            )
    set_mode_views(lambda views, _v=_el_view: {**views, "elasticity": _v})
    return ()


@app.cell
def neb_endpoint_controls_cell(get_ledger, mo):
    _ledger = get_ledger()
    _opts = _ledger.structure_options()
    _labels = list(_opts.keys())
    _init_default = _labels[3] if len(_labels) > 3 else _labels[0]
    _final_default = _labels[4] if len(_labels) > 4 else _labels[-1]

    neb_init_struct_ui = mo.ui.dropdown(
        options=_opts,
        value=_init_default,
        label="Initial Endpoint (`--init-struct`)",
    )
    neb_final_struct_ui = mo.ui.dropdown(
        options=_opts,
        value=_final_default,
        label="Final Endpoint (`--final-struct`)",
    )
    return neb_final_struct_ui, neb_init_struct_ui


@app.cell
def neb_controls_cell(mo):
    neb_n_images_ui = mo.ui.number(
        start=3,
        stop=21,
        step=2,
        value=5,
        label="Intermediate Images (`--n-images`)",
    )
    neb_interpolator_ui = mo.ui.dropdown(
        options=["ase", "pymatgen"],
        value="ase",
        label="Interpolator (`--interpolator`)",
    )
    neb_climb_ui = mo.ui.checkbox(
        value=True,
        label="Climbing Image CI-NEB (`climb=True`)",
    )
    neb_k_ui = mo.ui.number(
        start=0.01,
        stop=2.0,
        step=0.05,
        value=0.1,
        label="Spring Constant k (eV/Å²)",
    )
    neb_fmax_ui = mo.ui.number(
        start=0.005,
        stop=0.5,
        step=0.01,
        value=0.05,
        label="Force Threshold `fmax` (eV/Å)",
    )
    neb_steps_ui = mo.ui.number(
        start=5,
        stop=1000,
        step=10,
        value=50,
        label="Max Steps (`--steps`)",
    )
    neb_run_btn = mo.ui.run_button(
        label="Run Nudged Elastic Band (NEB)",
        kind="success",
    )
    return (
        neb_climb_ui,
        neb_fmax_ui,
        neb_interpolator_ui,
        neb_k_ui,
        neb_n_images_ui,
        neb_run_btn,
        neb_steps_ui,
    )


@app.cell
def neb_exec_cell(
    WorkspaceLedger,
    active_env_status,
    active_mlip_config,
    build_chemiscope_widget,
    chart_neb_barrier,
    execute_janus_mode,
    force_rerun_ui,
    get_ledger,
    mo,
    neb_climb_ui,
    neb_final_struct_ui,
    neb_fmax_ui,
    neb_init_struct_ui,
    neb_interpolator_ui,
    neb_k_ui,
    neb_n_images_ui,
    neb_run_btn,
    neb_steps_ui,
    parse_neb_run,
    set_ledger,
    set_mode_views,
):
    mo.stop(not neb_run_btn.value, None)
    if not active_env_status.is_ready:
        _neb_view = mo.callout(
            mo.md(f"Cannot run `neb`: {active_env_status.detail}"),
            kind="warn",
        )
    else:
        _ledger = get_ledger()
        _init_entry = _ledger.get_structure(neb_init_struct_ui.value or "struct_3")
        _final_entry = _ledger.get_structure(neb_final_struct_ui.value or "struct_4")
        _params = {
            "n_images": int(neb_n_images_ui.value),
            "interpolator": neb_interpolator_ui.value,
            "climb": bool(neb_climb_ui.value),
            "k": float(neb_k_ui.value),
            "fmax": float(neb_fmax_ui.value),
            "steps": int(neb_steps_ui.value),
        }
        _rec = execute_janus_mode(
            "neb",
            active_mlip_config,
            _init_entry,
            _params,
            python_executable=active_env_status.python_executable,
            final_struct_entry=_final_entry,
            force_rerun=bool(force_rerun_ui.value),
        )
        if _rec.status == "failed":
            _neb_view = mo.callout(
                mo.md(
                    f"**NEB Calculation Failed** (`{_rec.run_id}`)\n\n"
                    f"```text\n{_rec.stdout_stderr}\n```"
                ),
                kind="danger",
            )
        else:
            _parsed = parse_neb_run(_rec.run_dir, active_mlip_config.arch)
            _saddle_atoms = _parsed.band_images[_parsed.saddle_image_index]

            def _promote_saddle(
                _,
                _rec_val=_rec,
                _atoms_val=_saddle_atoms,
                _saddle_idx=_parsed.saddle_image_index,
            ):
                def _updater(led):
                    new_led = WorkspaceLedger(
                        structures=dict(led.structures),
                        runs=dict(led.runs),
                    )
                    new_led.register_run(_rec_val)
                    new_led.register_structure(
                        _atoms_val,
                        name=f"NEB Saddle Image #{_saddle_idx}",
                        source_mode="neb",
                        is_relaxed=False,
                        provenance_run_id=_rec_val.run_id,
                    )
                    return new_led

                set_ledger(_updater)

            _promote_btn = mo.ui.button(
                label=f"Promote Transition-State Image #{_parsed.saddle_image_index} to Ledger",
                on_change=_promote_saddle,
                kind="neutral",
            )
            _chart = mo.ui.altair_chart(chart_neb_barrier(_parsed.image_df))
            _cs_view = mo.lazy(
                lambda _imgs=_parsed.band_images: build_chemiscope_widget(
                    _imgs,
                    mode="structure",
                    show_force_arrows=True,
                )
            )
            _neb_view = mo.vstack(
                [
                    mo.callout(
                        mo.md(
                            f"**NEB {_rec.status.upper()}** (`{_rec.run_id}`, `{_rec.duration_s:.2f}s`) | "
                            f"**Activation Barrier $\\Delta E^\\ddagger$**: `{_parsed.barrier_ev:.4f} eV` | "
                            f"**Reaction $\\Delta E$**: `{_parsed.delta_e_ev:.4f} eV` | "
                            f"**Max Band Force**: `{_parsed.max_force_ev_ang:.4f} eV/Å` | "
                            f"**Saddle Image**: `#{_parsed.saddle_image_index}`"
                        ),
                        kind="success",
                    ),
                    _promote_btn,
                    mo.hstack([_chart, _cs_view], widths="equal", wrap=True),
                    mo.ui.table(_parsed.image_df, selection=None),
                    mo.accordion(
                        {
                            "Reproducible YAML Config & CLI Command": mo.md(
                                f"**CLI**:\n```bash\n{' '.join(_rec.cli_command)}\n```\n\n"
                                f"**`config.yml`**:\n```yaml\n{_rec.config_yaml_path.read_text()}\n```"
                            )
                        }
                    ),
                ]
            )
    set_mode_views(lambda views, _v=_neb_view: {**views, "neb": _v})
    return ()


@app.cell
def desc_controls_cell(mo):
    desc_invariants_ui = mo.ui.checkbox(
        value=True,
        label="Invariants Only (`--invariants-only`)",
    )
    desc_per_element_ui = mo.ui.checkbox(
        value=True,
        label="Calculate Per-Element Mean (`--calc-per-element`)",
    )
    desc_per_atom_ui = mo.ui.checkbox(
        value=True,
        label="Calculate Per-Atom Descriptors (`--calc-per-atom`)",
    )
    desc_cutoff_ui = mo.ui.number(
        start=2.0,
        stop=8.0,
        step=0.5,
        value=4.0,
        label="Chemiscope Atomic Environment Cutoff (Å)",
    )
    desc_run_btn = mo.ui.run_button(
        label="Run MLIP Descriptors",
        kind="success",
    )
    return (
        desc_cutoff_ui,
        desc_invariants_ui,
        desc_per_atom_ui,
        desc_per_element_ui,
        desc_run_btn,
    )


@app.cell
def desc_exec_cell(
    active_env_status,
    active_mlip_config,
    active_struct_entry,
    build_chemiscope_widget,
    chart_descriptors_distribution,
    desc_cutoff_ui,
    desc_invariants_ui,
    desc_per_atom_ui,
    desc_per_element_ui,
    desc_run_btn,
    execute_janus_mode,
    force_rerun_ui,
    mo,
    parse_descriptors_run,
    set_mode_views,
):
    mo.stop(not desc_run_btn.value, None)
    if not active_env_status.is_ready:
        _desc_view = mo.callout(
            mo.md(f"Cannot run `descriptors`: {active_env_status.detail}"),
            kind="warn",
        )
    else:
        _params = {
            "invariants_only": bool(desc_invariants_ui.value),
            "calc_per_element": bool(desc_per_element_ui.value),
            "calc_per_atom": bool(desc_per_atom_ui.value),
        }
        _rec = execute_janus_mode(
            "descriptors",
            active_mlip_config,
            active_struct_entry,
            _params,
            python_executable=active_env_status.python_executable,
            force_rerun=bool(force_rerun_ui.value),
        )
        if _rec.status == "failed":
            _desc_view = mo.callout(
                mo.md(
                    f"**Descriptors Calculation Failed** (`{_rec.run_id}`)\n\n"
                    f"```text\n{_rec.stdout_stderr}\n```"
                ),
                kind="danger",
            )
        else:
            _parsed = parse_descriptors_run(_rec.run_dir, active_mlip_config.arch)
            _chart = mo.ui.altair_chart(chart_descriptors_distribution(_parsed.atom_df))
            _extra_props = {
                "descriptor_value": {
                    "target": "atom",
                    "values": _parsed.atom_df["descriptor_value"].tolist(),
                }
            }
            _cs_view = mo.lazy(
                lambda _structs=_parsed.structures,
                _props=_extra_props,
                _cut=float(desc_cutoff_ui.value): build_chemiscope_widget(
                    _structs,
                    mode="default",
                    show_force_arrows=False,
                    extra_properties=_props,
                    environments_cutoff=_cut,
                )
            )
            _mean_str = (
                f"{_parsed.mean_descriptor:.5f}" if _parsed.mean_descriptor is not None else "N/A"
            )
            _el_str = (
                ", ".join(f"{k}: {v:.5f}" for k, v in _parsed.element_descriptors.items())
                if _parsed.element_descriptors
                else "N/A"
            )
            _desc_view = mo.vstack(
                [
                    mo.callout(
                        mo.md(
                            f"**Descriptors {_rec.status.upper()}** (`{_rec.run_id}`, `{_rec.duration_s:.2f}s`) | "
                            f"**Structure Mean Descriptor**: `{_mean_str}` | "
                            f"**Per-Element Descriptors**: `{_el_str}`"
                        ),
                        kind="success",
                    ),
                    mo.hstack([_chart, _cs_view], widths="equal", wrap=True),
                    mo.ui.table(_parsed.atom_df, selection=None),
                    mo.accordion(
                        {
                            "Reproducible YAML Config & CLI Command": mo.md(
                                f"**CLI**:\n```bash\n{' '.join(_rec.cli_command)}\n```\n\n"
                                f"**`config.yml`**:\n```yaml\n{_rec.config_yaml_path.read_text()}\n```"
                            )
                        }
                    ),
                ]
            )
    set_mode_views(lambda views, _v=_desc_view: {**views, "descriptors": _v})
    return ()


@app.cell
def workbench_tabs_cell(
    desc_cutoff_ui,
    desc_invariants_ui,
    desc_per_atom_ui,
    desc_per_element_ui,
    desc_run_btn,
    el_minimize_all_ui,
    el_n_strains_ui,
    el_norm_mag_ui,
    el_run_btn,
    el_shear_mag_ui,
    eos_max_vol_ui,
    eos_min_vol_ui,
    eos_minimize_all_ui,
    eos_n_vols_ui,
    eos_run_btn,
    eos_type_ui,
    get_active_tab,
    get_mode_views,
    get_provision_view,
    md_ensemble_ui,
    md_rdf_ui,
    md_run_btn,
    md_stats_every_ui,
    md_steps_ui,
    md_temp_ui,
    md_timestep_ui,
    mlip_header_panel,
    mo,
    neb_climb_ui,
    neb_final_struct_ui,
    neb_fmax_ui,
    neb_init_struct_ui,
    neb_interpolator_ui,
    neb_k_ui,
    neb_n_images_ui,
    neb_run_btn,
    neb_steps_ui,
    opt_cell_mode_ui,
    opt_fmax_ui,
    opt_optimizer_ui,
    opt_pressure_ui,
    opt_run_btn,
    opt_steps_ui,
    opt_symmetrize_ui,
    ph_disp_ui,
    ph_mesh_ui,
    ph_minimize_ui,
    ph_run_btn,
    ph_supercell_ui,
    ph_symmetrize_ui,
    set_active_tab,
    sp_arrows_ui,
    sp_props_ui,
    sp_run_btn,
    structure_header_panel,
):
    _views = get_mode_views()
    _prov_view = get_provision_view()

    _sp_tab = mo.vstack(
        [
            mo.md("#### Singlepoint Energy, Forces, Stress & Hessian (`janus singlepoint`)"),
            mo.hstack([sp_props_ui, sp_arrows_ui, sp_run_btn], justify="start", wrap=True),
            _views.get(
                "singlepoint",
                mo.callout(
                    mo.md(
                        "Configure singlepoint properties above and click **Run Singlepoint Calculation**."
                    ),
                    kind="info",
                ),
            ),
        ]
    )
    _opt_tab = mo.vstack(
        [
            mo.md(
                "#### Geometry & Unit-Cell Optimization (`janus geomopt` with `FrechetCellFilter`)"
            ),
            mo.hstack(
                [
                    opt_optimizer_ui,
                    opt_cell_mode_ui,
                    opt_fmax_ui,
                    opt_steps_ui,
                    opt_pressure_ui,
                    opt_symmetrize_ui,
                    opt_run_btn,
                ],
                justify="start",
                wrap=True,
            ),
            _views.get(
                "geomopt",
                mo.callout(
                    mo.md(
                        "Configure relaxation settings (`FrechetCellFilter` enabled by default for variable cell) "
                        "and click **Run Geometry Optimization**."
                    ),
                    kind="info",
                ),
            ),
        ]
    )
    _md_tab = mo.vstack(
        [
            mo.md("#### Molecular Dynamics Simulation (`janus md`)"),
            mo.hstack(
                [
                    md_ensemble_ui,
                    md_temp_ui,
                    md_steps_ui,
                    md_timestep_ui,
                    md_stats_every_ui,
                    md_rdf_ui,
                    md_run_btn,
                ],
                justify="start",
                wrap=True,
            ),
            _views.get(
                "md",
                mo.callout(
                    mo.md(
                        "Configure ensemble and temperature above and click **Run Molecular Dynamics**."
                    ),
                    kind="info",
                ),
            ),
        ]
    )
    _ph_tab = mo.vstack(
        [
            mo.md("#### Harmonic Phonons, Bands, DOS & Thermal Properties (`janus phonons`)"),
            mo.hstack(
                [
                    ph_supercell_ui,
                    ph_disp_ui,
                    ph_mesh_ui,
                    ph_minimize_ui,
                    ph_symmetrize_ui,
                    ph_run_btn,
                ],
                justify="start",
                wrap=True,
            ),
            _views.get(
                "phonons",
                mo.callout(
                    mo.md(
                        "Configure supercell and mesh above and click **Run Phonon Calculation**. "
                        "(Tip: use a relaxed crystal from **GeomOpt** to avoid imaginary modes.)"
                    ),
                    kind="info",
                ),
            ),
        ]
    )
    _eos_tab = mo.vstack(
        [
            mo.md("#### Equation of State & Bulk Modulus (`janus eos`)"),
            mo.hstack(
                [
                    eos_type_ui,
                    eos_min_vol_ui,
                    eos_max_vol_ui,
                    eos_n_vols_ui,
                    eos_minimize_all_ui,
                    eos_run_btn,
                ],
                justify="start",
                wrap=True,
            ),
            _views.get(
                "eos",
                mo.callout(
                    mo.md(
                        "Configure volume range and EOS fit type above and click **Run Equation of State (EOS)**."
                    ),
                    kind="info",
                ),
            ),
        ]
    )
    _el_tab = mo.vstack(
        [
            mo.md("#### 6×6 Voigt Elasticity Stiffness Tensor (`janus elasticity`)"),
            mo.hstack(
                [
                    el_norm_mag_ui,
                    el_shear_mag_ui,
                    el_n_strains_ui,
                    el_minimize_all_ui,
                    el_run_btn,
                ],
                justify="start",
                wrap=True,
            ),
            _views.get(
                "elasticity",
                mo.callout(
                    mo.md(
                        "Configure strain magnitudes above and click **Run Elasticity Stiffness Tensor**."
                    ),
                    kind="info",
                ),
            ),
        ]
    )
    _neb_tab = mo.vstack(
        [
            mo.md("#### Nudged Elastic Band Minimum Energy Pathway (`janus neb`)"),
            mo.hstack(
                [
                    neb_init_struct_ui,
                    neb_final_struct_ui,
                    neb_n_images_ui,
                    neb_interpolator_ui,
                    neb_climb_ui,
                    neb_k_ui,
                    neb_fmax_ui,
                    neb_steps_ui,
                    neb_run_btn,
                ],
                justify="start",
                wrap=True,
            ),
            _views.get(
                "neb",
                mo.callout(
                    mo.md(
                        "Select compatible initial and final endpoints (e.g. `#3: Li (bcc vacancy hop - initial)` "
                        "and `#4: Li (bcc vacancy hop - final)`) and click **Run Nudged Elastic Band (NEB)**."
                    ),
                    kind="info",
                ),
            ),
        ]
    )
    _desc_tab = mo.vstack(
        [
            mo.md("#### MLIP Atomic & Structure Descriptors (`janus descriptors`)"),
            mo.hstack(
                [
                    desc_invariants_ui,
                    desc_per_element_ui,
                    desc_per_atom_ui,
                    desc_cutoff_ui,
                    desc_run_btn,
                ],
                justify="start",
                wrap=True,
            ),
            _views.get(
                "descriptors",
                mo.callout(
                    mo.md("Configure descriptor flags above and click **Run MLIP Descriptors**."),
                    kind="info",
                ),
            ),
        ]
    )

    mode_tabs = mo.ui.tabs(
        {
            "1. SinglePoint": _sp_tab,
            "2. GeomOpt": _opt_tab,
            "3. MD": _md_tab,
            "4. Phonons": _ph_tab,
            "5. EOS": _eos_tab,
            "6. Elasticity": _el_tab,
            "7. NEB": _neb_tab,
            "8. Descriptors": _desc_tab,
        },
        value=get_active_tab(),
        on_change=set_active_tab,
        lazy=True,
    )

    workbench_layout = mo.vstack(
        [
            mo.md(
                "# STFC `janus-core` 8-Mode Atomistic MLIP Workbench\n"
                "_Stateless YAML-Artifact Subprocess Bus + Structure Relay Ledger • "
                "3D Visualization by **Chemiscope** • 2D Diagnostics by **Altair**_"
            ),
            mlip_header_panel,
            _prov_view if _prov_view is not None else mo.md(""),
            structure_header_panel,
            mo.md("### 3. `janus-core` Calculation Modes"),
            mode_tabs,
        ]
    )
    workbench_layout
    return mode_tabs, workbench_layout


if __name__ == "__main__":
    app.run()
