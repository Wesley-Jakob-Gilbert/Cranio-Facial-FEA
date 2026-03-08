# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Simplified cranio-maxillary FEA pipeline comparing bone response trends under low/medium/high tongue pressure on the palate. Uses CalculiX (linear static solver) on a procedurally generated hexahedral mesh. **Results are comparative/trend analysis only — not clinical.**

## Commands

**Full pipeline (one command):**
```bash
./scripts/reproduce_all.sh
```

**Step-by-step:**
```bash
python3 mesh/build_mesh.py            # generates mesh/mesh_data.json
python3 solver/make_inp.py            # generates solver/*.inp decks for all configured cases
./solver/run_case.sh <case_name>      # runs ccx solver; outputs go to results/<case>/
python3 post/extract_metrics.py       # reads .dat files → results/metrics_summary.csv
python3 post/extract_fields.py        # reads .dat files → results/<case>/node_u.csv + elem_vm.csv
python3 post/make_plots.py
python3 post/make_3d_visuals.py
python3 post/make_case_grid_visuals.py
python3 post/make_comparison_artifacts.py
python3 post/simulate_adaptation.py   # heuristic adaptation proxy on medium_kpa results
python3 scripts/write_results_manifest.py  # SHA256 hashes of key outputs
```

**Run a single case:**
```bash
./solver/run_case.sh low_kpa
./solver/run_case.sh medium_kpa
./solver/run_case.sh high_kpa
./solver/run_case.sh medium_kpa_plus_muscle
```

**Install Python deps:**
```bash
pip install matplotlib pyyaml
```

**External solver dependency:** `ccx` (CalculiX) must be installed and in `PATH`.

## Architecture

### Data Flow

```
configs/geometry.json
       ↓
mesh/build_mesh.py → mesh/mesh_data.json
       ↓
configs/loads.yaml + configs/material.yaml + configs/bc.yaml
       ↓
solver/make_inp.py → solver/<case>.inp  (one deck per load case)
       ↓
solver/run_case.sh <case> → results/<case>/<case>.frd + <case>.dat
       ↓
post/extract_metrics.py  → results/metrics_summary.csv
post/extract_fields.py   → results/<case>/node_u.csv, elem_vm.csv
       ↓
post/make_plots.py, make_3d_visuals.py, etc. → results/*.png
post/simulate_adaptation.py → results/adaptation_timeseries.csv + adaptation_trend.png
scripts/write_results_manifest.py → results/RESULTS_MANIFEST.md
```

### Load Cases

All cases are defined in [configs/loads.yaml](configs/loads.yaml). Each entry has `tongue_kpa` and `muscle_force_n`. The 9 configured cases are:
- **Tongue-only:** `low_kpa` (0.5 kPa), `medium_kpa` (2.0 kPa), `high_kpa` (5.0 kPa)
- **Tongue + muscle (6 N):** `low_kpa_plus_muscle`, `medium_kpa_plus_muscle`, `high_kpa_plus_muscle`
- **Muscle sensitivity sweep (medium tongue):** `medium_kpa_plus_muscle_2n/4n/8n`

### Mesh

[mesh/build_mesh.py](mesh/build_mesh.py) generates a structured C3D8 (8-node hexahedral) brick mesh with mild palate curvature. Profile `template_maxilla_v1` adds anterior taper, alveolar ridge lift, central palate groove, and posterior downward skew. The mesh JSON encodes nodes, elements, and named node sets:
- `NSET_FIXED` — posterior face (x=0), fully constrained
- `NSET_PALATE` — top face (z=Lz), tongue load applied here
- `NSET_MUSCLE_LEFT/RIGHT/ALL` — anterior-lateral top strips, jaw muscle proxy

Geometry parameters (dimensions, divisions, curvature) are in [configs/geometry.json](configs/geometry.json).

### Solver Input Deck Generation

[solver/make_inp.py](solver/make_inp.py) converts `mesh_data.json` + YAML configs into CalculiX `.inp` files. Tongue pressure is converted to nodal point forces using tributary area (`Lx * Ly / num_palate_nodes`). Muscle resultant forces are split left/right and decomposed into a fixed unit direction vector.

### Post-Processing Parsers

[post/extract_metrics.py](post/extract_metrics.py) and [post/extract_fields.py](post/extract_fields.py) parse CalculiX `.dat` text output using regex. Von Mises stress is computed from the 6 stress tensor components. `extract_fields.py` requires section headers (`displacements`, `stresses`) in the `.dat` file to switch parsing state.

### Adaptation Module

[post/simulate_adaptation.py](post/simulate_adaptation.py) is a heuristic post-processing layer — not a coupled FEA/growth model. It applies a simple stimulus-driven density update rule over discrete steps using the `medium_kpa` element von Mises field as input.

## Key Configuration Files

| File | Purpose |
|------|---------|
| [configs/geometry.json](configs/geometry.json) | Mesh dimensions, divisions, curvature amplitude |
| [configs/material.yaml](configs/material.yaml) | Young's modulus (1 GPa), Poisson's ratio (0.3) |
| [configs/loads.yaml](configs/loads.yaml) | All load case definitions |
| [configs/bc.yaml](configs/bc.yaml) | Boundary condition reference |

## MVP Constraints (Do Not Change Without Deliberate Intent)

- Geometry is a toy analytic approximation — not CT-derived
- Material is isotropic linear elastic (no cortical/cancellous split)
- Solver is linear static (no nonlinearities, no contact)
- Results are for relative trend comparison only
