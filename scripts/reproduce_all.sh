#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[1/10] Build mesh"
python3 mesh/build_mesh.py

echo "[2/10] Build solver decks"
python3 solver/make_inp.py

echo "[3/10] Run all configured cases"
python3 - <<'PY'
import yaml, subprocess, pathlib
root=pathlib.Path('.').resolve()
loads=yaml.safe_load((root/'configs'/'loads.yaml').read_text())['load_cases']
for i,case in enumerate(loads.keys(), start=1):
    print(f"  - running case {i}/{len(loads)}: {case}")
    subprocess.run([str(root/'solver'/'run_case.sh'), case], check=True)
PY

echo "[4/10] Extract scalar metrics"
python3 post/extract_metrics.py

echo "[5/10] Extract field CSVs"
python3 post/extract_fields.py

echo "[6/10] Build plots + 3D visuals"
python3 post/make_plots.py
python3 post/make_3d_visuals.py
python3 post/make_case_grid_visuals.py
python3 post/make_comparison_artifacts.py

echo "[7/10] Adaptation proxy"
python3 post/simulate_adaptation.py

echo "[8/10] Results manifest"
python3 scripts/write_results_manifest.py

echo "Done. Key outputs in results/"
