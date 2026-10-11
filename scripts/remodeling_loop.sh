#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v ccx >/dev/null 2>&1; then
  echo "Blocker: ccx not found in PATH"
  exit 2
fi

SCENARIOS=(mewing mouth_breathing)
CYCLES="$(python3 - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load((pathlib.Path('configs/remodeling.yaml')).read_text())['remodeling']
print(int(cfg.get('cycles',12)))
PY
)"

# Load tongue_kpa for each scenario from loads.yaml
get_tongue_kpa() {
  local scenario="$1"
  python3 - <<PY
import yaml, pathlib
loads = yaml.safe_load((pathlib.Path('configs/loads.yaml')).read_text())
case = loads.get('load_cases', {}).get('$scenario', {})
if isinstance(case, dict):
    print(float(case.get('tongue_kpa', 2.0)))
else:
    print(2.0)
PY
}

# initialize fresh summary for this run
rm -f results/remodeling_summary.csv

for scenario in "${SCENARIOS[@]}"; do
  echo "=== Scenario: $scenario ==="
  mkdir -p "results/$scenario/snapshots"

  TONGUE_KPA="$(get_tongue_kpa "$scenario")"
  echo "  tongue_kpa=$TONGUE_KPA (used for cyclic-load SED correction)"

  # reset per-scenario evolving state
  rm -f "results/$scenario/elem_density.json"
  rm -f "results/$scenario/suture_modulus.json"
  # Also clear the solver-side sidecars: otherwise cycle 1 of this scenario
  # silently inherits the final state of the previous scenario (or a prior run).
  rm -f solver/elem_density.json solver/suture_modulus.json

  for cycle in $(seq 1 "$CYCLES"); do
    echo "  cycle $cycle/$CYCLES"

    # feedback hook: scenario density -> solver input sidecar expected by make_inp.py
    cp "results/$scenario/elem_density.json" solver/elem_density.json 2>/dev/null || true
    cp "results/$scenario/suture_modulus.json" solver/suture_modulus.json 2>/dev/null || true

    python3 solver/make_inp.py "$scenario"
    ./solver/run_case.sh "$scenario"
    python3 post/extract_fields.py
    # Pass tongue_kpa for cyclic SED weighting
    python3 post/bone_remodeling.py "$scenario" "$cycle" "$TONGUE_KPA"
  done
done

echo "Remodeling loop complete. See results/remodeling_summary.csv"
