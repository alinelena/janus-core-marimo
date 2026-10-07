# Container Deployment (`containers/`)

This directory contains the Docker build and orchestration files for running the **`janus-core-marimo`** 8-tab workbench in an isolated container environment.

---

## Contents

| File | Description |
|---|---|
| [`Dockerfile`](Dockerfile) | Container image built on `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` with `/opt/micromamba/envs/janus` pre-provisioned with `janus-core[mace,d3]`, `marimo`, `chemiscope`, `altair`, `pytest`, `ruff`, and `janus-core-marimo` (built via `uv_build`). |
| [`Dockerfile.dockerignore`](Dockerfile.dockerignore) | BuildKit ignore rules associated with `containers/Dockerfile` to exclude local `.venvs/`, `runs/`, `dist/`, and `.git/` from the Docker build context. |
| [`docker-compose.yml`](docker-compose.yml) | Docker Compose service exposing port `2718` and mounting persistent host directories (`../runs` and `../.venvs`) into `/workspace`. |

---

## 1. Quick Start with Docker Compose

From the **repository root**, run:

```bash
docker compose -f containers/docker-compose.yml up --build
```

Then open **`http://localhost:2718`** in your browser.

To stop the container:
```bash
docker compose -f containers/docker-compose.yml down
```

---

## 2. Building & Running with `docker` CLI

All `docker build` commands should use the **repository root** (`.`) as the build context and pass `-f containers/Dockerfile`:

### Build the Image
```bash
docker build -f containers/Dockerfile -t janus-core-marimo:latest .
```

### Run Interactive Workbench (Edit Mode, CPU)
```bash
docker run --rm -it \
  -p 2718:2718 \
  -v "$(pwd)/runs:/workspace/runs" \
  -v "$(pwd)/.venvs:/workspace/.venvs" \
  janus-core-marimo:latest
```

### Run with NVIDIA GPU Passthrough
Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host machine:
```bash
docker run --rm -it \
  --gpus all \
  -p 2718:2718 \
  -v "$(pwd)/runs:/workspace/runs" \
  -v "$(pwd)/.venvs:/workspace/.venvs" \
  janus-core-marimo:latest
```

### Run as a Read-Only Web App (`marimo run`)
```bash
docker run --rm -it \
  -p 2718:2718 \
  -v "$(pwd)/runs:/workspace/runs" \
  -v "$(pwd)/.venvs:/workspace/.venvs" \
  janus-core-marimo:latest \
  uv run --no-project --python /opt/micromamba/envs/janus/bin/python \
  marimo run janus_workbench.py --host 0.0.0.0 --port 2718
```

---

## 3. Persistent Volumes & Isolated MLIP Environments

Mounting `./runs` and `./.venvs` into `/workspace/runs` and `/workspace/.venvs` ensures that:
1. **Simulation Cache (`/workspace/runs`)**: Content-addressed calculation outputs (`runs/<mode>/<run_id>/`) persist across container restarts.
2. **Isolated MLIP Virtual Environments (`/workspace/.venvs`)**: When you click **"Provision `janus-<group>` Environment via `uv`"** in the workbench header (for example, to install `sevennet`, `chgnet`, `orb`, `mattersim`, `fairchem`, `nequip`, `dpa3`, `grace`, or `upet`), `uv` creates `.venvs/janus-<group>` inside the mounted volume so you only provision each backend once.

---

## 4. Running Tests & Linters Inside the Container

```bash
# Run pytest suite
docker run --rm janus-core-marimo:latest \
  uv run --no-project --python /opt/micromamba/envs/janus/bin/python pytest

# Run Ruff linter & formatter checks
docker run --rm janus-core-marimo:latest \
  uv run --no-project --python /opt/micromamba/envs/janus/bin/python ruff check .

# Validate Marimo reactive DAG
docker run --rm janus-core-marimo:latest \
  uv run --no-project --python /opt/micromamba/envs/janus/bin/python marimo check janus_workbench.py
```
