# Brainstorm Brief: Custom Environment Selector & Immediate Chemiscope Upload Preview

- **Date**: 2026-09-26
- **Seed**: Extend `janus-core-marimo` so users can (1) specify or pick any existing Python/micromamba/conda/uv environment on disk to run calculations (with live package/readiness probing), and (2) immediately auto-register, activate, and inspect uploaded and ledger structures in an interactive 3D Chemiscope viewer before launching any simulation.
- **Stage**: Feature ideation for existing product
- **Breadth**: ~10 concepts, balanced (5 Goldfish spawned: Technical Architect, UX Designer, Contrarian Pre-Mortem, Prior Art Researcher, and Outsider Contrarian Sweep)
- **Chosen Direction**: **Pick #1 — Auto-Discovered + Custom Path Environment Selector (`normalize_python_executable` + Static-First Probe) paired with Immediate Auto-Active Upload & Lazy Chemiscope X-Ray Preview in the Structure Ledger**

---

## Concepts Clusters

### Cluster 1: Custom Environment Discovery, Normalization & Zero-Lag Probing
- **Zero-Subprocess PEP-376 Static Inspector (`normalize_python_executable` + `.dist-info` scan)**: Accept either an environment root directory (`/opt/micromamba/envs/my-env`, `~/.venv`) or a binary path (`.../bin/python`), normalize without `Path.resolve()` symlink escape (`pyvenv.cfg` preservation), and inspect `.dist-info` / `conda-meta` in <5 ms so typing or switching environments never freezes the Marimo reactive loop; reserve subprocess `import` verification for an explicit probe button or cached lookup.
- **VS Code "PET" Auto-Locator & Capability Matrix**: Scan standard local environment roots (`/opt/micromamba/envs/*`, `~/.micromamba/envs/*`, `~/.conda/environments.txt`, `.venvs/*`) to populate a `"Discovered Local Envs"` dropdown alongside a free-text `"Custom Environment Path"` override, displaying compatibility badges (`janus_core ✓`, `<arch> ✓`, `torch-dftd ✓`).
- **Deep Runtime JSON Probe & Provenance Hashing**: Provide an explicit `"Probe Runtime"` action and fold custom environment paths into `compute_run_id` so benchmarking across two different environments never collides in the SHA-256 run cache.

### Cluster 2: Upload-to-Chemiscope Workflow & WebGL Context Guardrails
- **Immediate Auto-Registration & Auto-Activation + Single-Slot Lazy Chemiscope Viewport**: When a structure file is uploaded via `mo.ui.file`, automatically register it (deduplicated by content hash), set it as the active structure in the Ledger, and immediately render it inside a single `mo.lazy`-gated Chemiscope 3D viewer in the Structure Relay Ledger panel—preventing browser `MAX_ACTIVE_WEBGL_CONTEXTS` exhaustion.
- **Visual Supercell Repeat Preview**: Provide a non-destructive visual supercell tiling toggle ($1\times 1\times 1$ vs. $2\times 2\times 1$ / $2\times 2\times 2$) inside the Chemiscope preview to verify periodic boundary continuity across cell walls.

### Cluster 3: Forensic 3D Pre-Flight Diagnostics in Chemiscope
- **Chemiscope Pre-Flight Structural X-Ray**: Fuse `inspector.py` and `viz.py` so the pre-calculation Chemiscope viewer colors atoms by per-atom nearest-neighbor distance ($d_{\min, i}$ with minimum-image convention), coordination number, and fractional coordinates, and highlights overlapping atom pairs ($d_{ij} < 0.8\text{ \AA}$) in 3D so users immediately see *which* atoms triggered a `StructureReport` warning or error.
- **Model-Structure Receptive-Field Handshake in 3D**: Overlay local neighbor counts within the selected MLIP cutoff radius ($r_{\text{cut}}$) in the pre-flight Chemiscope viewer.

### Cluster 4: Multi-Structure Cohort Exploration
- **Ledger-Wide Chemiscope Cohort Atlas**: Optionally view all structures in the Structure Relay Ledger together in Chemiscope (comparing density $\rho$, volume/atom, and $d_{\min}$ across presets, uploads, and relaxed structures).

---

## Ranked Picks

1. **(Selected) Auto-Discovered + Custom Path Env Selector paired with Auto-Active Upload & Lazy Chemiscope X-Ray Preview in the Structure Ledger**
   - Users can pick from auto-discovered local environments (`/opt/micromamba/envs/*`, `~/.conda/envs/*`, `.venvs/*`) or enter any custom environment root or `bin/python` path, with symlink-safe normalization (`os.path.abspath` without `.resolve()` breaking `pyvenv.cfg`), fast metadata checks, and environment-aware `compute_run_id` cache keys.
   - Uploading a file immediately registers it into `WorkspaceLedger`, sets it as the active structure, and renders it in a `mo.lazy`-wrapped Chemiscope 3D viewer right inside Section 2 (Structure Relay Ledger), enriched with per-atom nearest-neighbor distance ($d_{\min, i}$), coordination numbers, unit-cell box display, and optional visual supercell tiling ($1\times 1\times 1$ vs. $2\times 2\times 2$).
2. **Upload Dock + Deep JSON Subprocess Runtime Probe**
3. **Split-Screen Ledger Cohort Explorer & Per-Architecture Env Pinning**
