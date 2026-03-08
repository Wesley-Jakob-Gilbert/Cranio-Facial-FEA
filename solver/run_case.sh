#!/usr/bin/env bash
set -euo pipefail

CASE="${1:-low_kpa}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INP="$ROOT/solver/${CASE}.inp"

if [[ ! -f "$INP" ]]; then
  echo "Input not found: $INP"
  echo "Generate decks first: python3 solver/make_inp.py"
  exit 1
fi

if ! command -v ccx >/dev/null 2>&1; then
  echo "ccx not found in PATH. Install calculix-ccx first."
  exit 2
fi

WORKDIR="$ROOT/results/$CASE"
mkdir -p "$WORKDIR"
cp "$INP" "$WORKDIR/${CASE}.inp"

pushd "$WORKDIR" >/dev/null
ccx "$CASE" >/tmp/ccx_${CASE}.log 2>&1 || {
  echo "ccx failed for case: $CASE"
  echo "--- tail ccx log ---"
  tail -n 60 /tmp/ccx_${CASE}.log || true
  exit 3
}
popd >/dev/null

echo "Solve complete: $CASE"
echo "Outputs in: $WORKDIR"
