# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    JANUS_PYTHON=/opt/micromamba/envs/janus/bin/python \
    PATH="/opt/micromamba/envs/janus/bin:${PATH}"

# Install minimal system dependencies required by scientific Python / ASE / Phonopy / Git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create the canonical base janus environment expected by janus-core-marimo
RUN mkdir -p /opt/micromamba/envs && \
    uv venv /opt/micromamba/envs/janus --python 3.12

WORKDIR /workspace

# Copy project metadata and source files from the repository root build context
COPY pyproject.toml README.md AGENTS.md janus_workbench.py ./
COPY src/ ./src/
COPY tools/ ./tools/
COPY tests/ ./tests/

# Install janus-core (with default MACE + D3 extras), dev tools, and janus-core-marimo (via uv_build)
RUN uv pip install --python /opt/micromamba/envs/janus/bin/python \
    "janus-core[mace,d3]" \
    pytest \
    ruff \
    .

# Create persistent directories for cached simulation runs and on-demand isolated MLIP venvs
RUN mkdir -p /workspace/runs /workspace/.venvs

EXPOSE 2718

CMD ["uv", "run", "--no-project", "--python", "/opt/micromamba/envs/janus/bin/python", "marimo", "edit", "janus_workbench.py", "--host", "0.0.0.0", "--port", "2718", "--no-token"]
