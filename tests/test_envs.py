"""Unit tests for janus_marimo.envs."""

from __future__ import annotations

from pathlib import Path
from typing import get_args
from unittest.mock import patch

from janus_core.helpers.janus_types import Architectures

from janus_marimo.envs import (
    ARCH_REQUIRES_EXPLICIT_MODEL,
    ARCH_TO_ENV_GROUP,
    DEFAULT_MODEL_BY_ARCH,
    MLIP_ENV_CATALOG,
    build_provision_commands,
    discover_local_environments,
    normalize_python_executable,
    provision_mlip_env,
    resolve_env_status,
)


def test_arch_catalog_covers_all_janus_core_architectures() -> None:
    janus_archs = set(get_args(Architectures))
    assert set(ARCH_TO_ENV_GROUP.keys()) == janus_archs
    assert set(DEFAULT_MODEL_BY_ARCH.keys()) == janus_archs
    assert len(MLIP_ENV_CATALOG) == 10

    covered_by_specs = {arch for spec in MLIP_ENV_CATALOG.values() for arch in spec.architectures}
    assert covered_by_specs == janus_archs
    assert ARCH_REQUIRES_EXPLICIT_MODEL == frozenset({"mace", "nequip", "dpa3"})


def test_build_provision_commands(tmp_path: Path) -> None:
    cmds = build_provision_commands(
        "sevennet",
        venvs_root=tmp_path,
        python_version="3.12",
        include_d3=True,
    )
    assert len(cmds) == 2
    assert cmds[0] == [
        "uv",
        "venv",
        str(tmp_path / "janus-sevennet"),
        "--python",
        "3.12",
    ]
    assert cmds[1] == [
        "uv",
        "pip",
        "install",
        "--python",
        str(tmp_path / "janus-sevennet" / "bin" / "python"),
        "janus-core[sevennet]",
        "janus-core[d3]",
    ]


def test_resolve_env_status_sources(tmp_path: Path) -> None:
    fake_base = tmp_path / "base_python"
    fake_base.write_text("#!/bin/sh\n")

    # 1. Missing when neither isolated venv nor base_python has probe module
    with patch("janus_marimo.envs._probe_python_imports", return_value=False):
        status_missing = resolve_env_status(
            "orb",
            venvs_root=tmp_path / "venvs",
            base_python=fake_base,
        )
        assert status_missing.is_ready is False
        assert status_missing.source == "missing"
        assert status_missing.group == "orb"

    # 2. Base janus ready
    with patch("janus_marimo.envs._probe_python_imports", return_value=True):
        status_base = resolve_env_status(
            "mace_mp",
            venvs_root=tmp_path / "venvs",
            base_python=fake_base,
        )
        assert status_base.is_ready is True
        assert status_base.source == "base_janus"
        assert status_base.python_executable == fake_base

    # 3. Isolated venv takes precedence when present and ready
    iso_python = tmp_path / "venvs" / "janus-sevennet" / "bin" / "python"
    iso_python.parent.mkdir(parents=True, exist_ok=True)
    iso_python.write_text("#!/bin/sh\n")
    with patch("janus_marimo.envs._probe_python_imports", return_value=True):
        status_iso = resolve_env_status(
            "sevennet",
            venvs_root=tmp_path / "venvs",
            base_python=fake_base,
        )
        assert status_iso.is_ready is True
        assert status_iso.source == "isolated_venv"
        assert status_iso.python_executable == iso_python


def test_provision_mlip_env_mocked(tmp_path: Path) -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Installed packages"
        mock_run.return_value.stderr = ""
        ok, log = provision_mlip_env("chgnet", venvs_root=tmp_path)
        assert ok is True
        assert "Successfully provisioned environment janus-chgnet" in log
        assert mock_run.call_count == 2


def test_normalize_python_executable_dir_binary_and_symlink(tmp_path: Path) -> None:
    real_bin = tmp_path / "real_python"
    real_bin.write_text("#!/bin/sh\n")

    env_dir = tmp_path / "my_env"
    bin_dir = env_dir / "bin"
    bin_dir.mkdir(parents=True)
    symlink_py = bin_dir / "python"
    symlink_py.symlink_to(real_bin)

    # 1. Directory input resolves to <dir>/bin/python WITHOUT dereferencing symlink
    norm_from_dir = normalize_python_executable(env_dir)
    assert norm_from_dir == symlink_py
    assert norm_from_dir.is_symlink()

    # 2. Direct binary/symlink input preserves the symlink path
    norm_from_bin = normalize_python_executable(symlink_py)
    assert norm_from_bin == symlink_py


def test_discover_local_environments(tmp_path: Path) -> None:
    venvs_root = tmp_path / "venvs"
    iso_py = venvs_root / "janus-mace" / "bin" / "python"
    iso_py.parent.mkdir(parents=True)
    iso_py.write_text("#!/bin/sh\n")

    extra_root = tmp_path / "custom_envs"
    custom_py = extra_root / "my_sevennet" / "bin" / "python"
    custom_py.parent.mkdir(parents=True)
    custom_py.write_text("#!/bin/sh\n")

    # Broken symlink should be excluded
    broken_py = extra_root / "broken_env" / "bin" / "python"
    broken_py.parent.mkdir(parents=True)
    broken_py.symlink_to(tmp_path / "nonexistent_target")

    discovered = discover_local_environments(
        venvs_root=venvs_root,
        base_python=tmp_path / "missing_base",
        extra_env_roots=[extra_root],
    )
    assert discovered["Auto (.venvs/janus-<group> or base janus)"] == ""
    assert str(iso_py) in discovered.values()
    assert str(custom_py) in discovered.values()
    assert str(broken_py) not in discovered.values()


def test_resolve_env_status_custom_env_path(tmp_path: Path) -> None:
    fake_base = tmp_path / "base_python"
    fake_base.write_text("#!/bin/sh\n")

    custom_env_dir = tmp_path / "custom_mace_env"
    custom_py = custom_env_dir / "bin" / "python"
    custom_py.parent.mkdir(parents=True)
    custom_py.write_text("#!/bin/sh\n")

    # 1. Custom env ready (passing directory root)
    with patch("janus_marimo.envs._probe_python_imports", return_value=True):
        st_ready = resolve_env_status(
            "mace_mp",
            venvs_root=tmp_path / "venvs",
            base_python=fake_base,
            custom_env_path=custom_env_dir,
        )
        assert st_ready.is_ready is True
        assert st_ready.source == "custom_env"
        assert st_ready.python_executable == custom_py

    # 2. Custom env missing required package does NOT silently fall back to base_janus
    with patch("janus_marimo.envs._probe_python_imports", return_value=False):
        st_missing_pkg = resolve_env_status(
            "sevennet",
            venvs_root=tmp_path / "venvs",
            base_python=fake_base,
            custom_env_path=custom_py,
        )
        assert st_missing_pkg.is_ready is False
        assert st_missing_pkg.source == "custom_env"
        assert "sevenn" in st_missing_pkg.detail

    # 3. Non-existent custom path reports missing executable with source == "custom_env"
    st_not_found = resolve_env_status(
        "mace_mp",
        venvs_root=tmp_path / "venvs",
        base_python=fake_base,
        custom_env_path=tmp_path / "no_such_env",
    )
    assert st_not_found.is_ready is False
    assert st_not_found.source == "custom_env"
    assert "not found" in st_not_found.detail
