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

# initialize fresh summary for this run
rm -f results/remodeling_summary.csv

for scenario in "${SCENARIOS[@]}"; do
  echo "=== Scenario: $scenario ==="
  mkdir -p "results/$scenario/snapshots"

  # reset per-scenario evolving state
  rm -f "results/$scenario/elem_density.json"

  for cycle in $(seq 1 "$CYCLES"); do
    echo "  cycle $cycle/$CYCLES"

    python3 solver/make_inp.py
    ./solver/run_case.sh "$scenario"
    python3 post/extract_fields.py
    python3 post/bone_remodeling.py "$scenario" "$cycle"
  done
done

echo "Remodeling loop complete. See results/remodeling_summary.csv"
