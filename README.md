# Cranio FEA MVP

This project runs a simplified cranio-maxillary FEA MVP with tongue-pressure load cases.

## Run (current MVP)
From this folder:

```bash
python3 mesh/build_mesh.py
python3 solver/make_inp.py
./solver/run_case.sh low_kpa
./solver/run_case.sh medium_kpa
./solver/run_case.sh high_kpa
python3 post/extract_metrics.py
python3 post/extract_fields.py
python3 post/make_plots.py
python3 post/make_3d_visuals.py
python3 post/make_case_grid_visuals.py
python3 post/make_comparison_artifacts.py
python3 post/simulate_adaptation.py
python3 scripts/write_results_manifest.py
```

One-command full reproduction:

```bash
./scripts/reproduce_all.sh
```

## Outputs
- Per-case solver outputs: `results/low_kpa/`, `results/medium_kpa/`, `results/high_kpa/`
- Summary metrics CSV: `results/metrics_summary.csv`
- Baseline comparison plot: `results/mvp_load_response.png`
- Additional comparison artifacts:
  - `results/case_comparison_table.csv`
  - `results/case_pairwise_deltas.csv`
  - `results/linearity_check.csv`
  - `results/normalized_response.png`
  - `results/linearity_check.png`
- 3D visuals:
  - `results/geometry_wireframe.png`
  - `results/<case>/u_mag_3d_scatter.png`
  - `results/<case>/vm_3d_scatter.png`
  - `results/u_mag_case_grid.png`
  - `results/vm_case_grid.png`
- Toy long-term adaptation proxy:
  - `results/adaptation_timeseries.csv`
  - `results/adaptation_trend.png`
- Reproducibility/integrity:
  - `results/RESULTS_MANIFEST.md`

## MVP assumptions
- Structured simplified **template maxilla** mesh profile (C3D8), configured via `configs/geometry.json`
- Isotropic linear-elastic material
- Posterior face fixed (proxy boundary condition)
- Tongue load modeled as distributed nodal force over palate node set
- Comparative trend analysis only (non-clinical)

## Reproducibility notes
- Solver dependency: `ccx` must be installed and available in `PATH` for solve steps.
- Python dependencies used by plotting scripts: `matplotlib` + stdlib.
- `scripts/write_results_manifest.py` records SHA256 hashes for key output artifacts.

## Notes
- Keep MVP assumptions fixed until baseline is complete.
- Next realism steps should be introduced one axis at a time.
- The adaptation module is a heuristic visualization layer (not a validated growth/remodeling law).
# Cranio-Facial-FEA
