#!/usr/bin/env python3
"""Suture stiffness sensitivity sweep — Neo Day 7 task (N-7).

Runs the mewing remodeling scenario three times with different suture Young's
modulus values to show how sutural compliance (an adolescent age proxy) affects
the magnitude and distribution of predicted bone response.

Suture E values tested:
  1e7 Pa  — very compliant (young adolescent, open suture, high cell activity)
  5e7 Pa  — moderate (mid-adolescent, default model value)
  1e8 Pa  — stiffer (late adolescent, approaching closure)

For each value:
  1. Temporarily updates configs/material.yaml suture modulus
  2. Regenerates solver/mewing.inp via make_inp.py
  3. Runs the solver (requires ccx in PATH)
  4. Extracts fields via extract_fields.py
  5. Runs one remodeling cycle via bone_remodeling.py
  6. Records peak von Mises, max displacement, and mean density shift

Output: results/suture_sensitivity.csv

Usage:
  python3 scripts/suture_sensitivity.py

Requires:
  ccx installed and in PATH
  mesh/mesh_data.json already generated (run mesh/build_mesh.py first)
  configs/remodeling.yaml present (Morpheus's file)
"""
from __future__ import annotations
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAT_PATH = ROOT / "configs" / "material.yaml"
RESULTS = ROOT / "results"

SUTURE_E_VALUES = [1e7, 5e7, 1e8]
SCENARIO = "mewing"

SENSITIVITY_CASE_TEMPLATE = "suture_sens_{label}"


def label_for(e_pa: float) -> str:
    """Short label: 1e7 → '10MPa', 5e7 → '50MPa', 1e8 → '100MPa'."""
    mpa = e_pa / 1e6
    if mpa == int(mpa):
        return f"{int(mpa)}MPa"
    return f"{mpa:.0f}MPa"


def run(cmd: list[str], cwd: Path = ROOT) -> int:
    """Run a subprocess, stream output, return exit code."""
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def check_ccx() -> bool:
    return shutil.which("ccx") is not None


def patch_suture_e(e_pa: float) -> None:
    """Write suture youngs_modulus_pa into material.yaml."""
    mat = yaml.safe_load(MAT_PATH.read_text())
    mat["materials"]["suture"]["youngs_modulus_pa"] = e_pa
    MAT_PATH.write_text(yaml.dump(mat, default_flow_style=False))


def restore_suture_e(original_e: float) -> None:
    patch_suture_e(original_e)


def read_peak_vm(case: str) -> float:
    csv_path = RESULTS / case / "elem_vm.csv"
    if not csv_path.exists():
        return float("nan")
    peak = 0.0
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vm = float(row["vm_avg_pa"])
                if vm > peak:
                    peak = vm
            except (ValueError, KeyError):
                pass
    return peak


def read_max_u(case: str) -> float:
    csv_path = RESULTS / case / "node_u.csv"
    if not csv_path.exists():
        return float("nan")
    max_u = 0.0
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                u = float(row["u_mag"])
                if u > max_u:
                    max_u = u
            except (ValueError, KeyError):
                pass
    return max_u


def read_mean_density(case: str) -> float:
    density_path = RESULTS / case / "elem_density.json"
    if not density_path.exists():
        return float("nan")
    data = json.loads(density_path.read_text())
    vals = list(data.values())
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    if not check_ccx():
        print("ERROR: ccx not found in PATH. Install CalculiX and retry.")
        sys.exit(1)

    # Read original suture E to restore after sweep
    mat_orig = yaml.safe_load(MAT_PATH.read_text())
    original_e = float(mat_orig["materials"]["suture"]["youngs_modulus_pa"])

    rows: list[dict] = []

    try:
        for e_pa in SUTURE_E_VALUES:
            lbl = label_for(e_pa)
            case = SENSITIVITY_CASE_TEMPLATE.format(label=lbl)
            case_dir = RESULTS / case
            case_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== Suture E = {e_pa:.1e} Pa ({lbl}) ===")

            # 1. Patch material.yaml with this suture E
            patch_suture_e(e_pa)

            # 2. Regenerate mewing.inp (make_inp.py generates all cases; solver
            #    only picks up mewing.inp, but we rename it to our sensitivity case)
            rc = run([sys.executable, "solver/make_inp.py"])
            if rc != 0:
                print(f"  make_inp.py failed (rc={rc}), skipping {lbl}")
                continue

            mewing_inp = ROOT / "solver" / "mewing.inp"
            sens_inp = ROOT / "solver" / f"{case}.inp"
            shutil.copy(mewing_inp, sens_inp)

            # 3. Run solver
            rc = run(["./solver/run_case.sh", case])
            if rc != 0:
                print(f"  Solver failed (rc={rc}) for {lbl}")
                continue

            # 4. Extract fields (runs for all cases; our new case will be processed
            #    if it appears in loads.yaml — but it doesn't, so we call extract
            #    logic directly for this case)
            #    Simpler: call extract_fields logic via subprocess with env override
            #    Actually: call extract_fields.py after temporarily patching loads.yaml
            #    is messy. Instead, copy the .dat to case_dir and run the parser inline.
            dat_src = ROOT / "results" / case / f"{case}.dat"
            if not dat_src.exists():
                print(f"  No .dat file found for {lbl}, skipping metrics")
                continue

            # Use extract_fields logic inline for this single case
            rc = run([sys.executable, "-c", f"""
import sys, math, re, csv
from pathlib import Path
ROOT = Path('{ROOT}')
case = '{case}'
dat = ROOT / 'results' / case / f'{{case}}.dat'
node_re = re.compile(r'^\\s*(\\d+)\\s+([+-]?\\d\\.\\d+E[+-]\\d+)\\s+([+-]?\\d\\.\\d+E[+-]\\d+)\\s+([+-]?\\d\\.\\d+E[+-]\\d+)')
stress_re = re.compile(r'^\\s*(\\d+)\\s+\\d+\\s+([+-]?\\d\\.\\d+E[+-]\\d+)\\s+([+-]?\\d\\.\\d+E[+-]\\d+)\\s+([+-]?\\d\\.\\d+E[+-]\\d+)\\s+([+-]?\\d\\.\\d+E[+-]\\d+)\\s+([+-]?\\d\\.\\d+E[+-]\\d+)\\s+([+-]?\\d\\.\\d+E[+-]\\d+)')
mises = lambda s,sy,sz,xy,xz,yz: math.sqrt(0.5*((s-sy)**2+(sy-sz)**2+(sz-s)**2)+3*(xy**2+xz**2+yz**2))
nodes, elem_vm = {{}}, {{}}
in_disp = in_stress = False
for line in dat.read_text(errors='ignore').splitlines():
    ls = line.strip().lower()
    if ls.startswith('displacements'): in_disp=True; in_stress=False; continue
    if ls.startswith('stresses'): in_stress=True; in_disp=False; continue
    if not ls: continue
    if in_disp:
        m = node_re.match(line)
        if m:
            nid,ux,uy,uz = int(m.group(1)),*map(float,m.groups()[1:])
            nodes[nid]=(ux,uy,uz,math.sqrt(ux**2+uy**2+uz**2))
    elif in_stress:
        s = stress_re.match(line)
        if s:
            eid=int(s.group(1)); vals=list(map(float,s.groups()[1:]))
            acc=elem_vm.setdefault(eid,[0.0,0]); acc[0]+=mises(*vals); acc[1]+=1
out_u = ROOT/'results'/case/'node_u.csv'
with out_u.open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['node_id','ux','uy','uz','u_mag'])
    for nid in sorted(nodes): w.writerow([nid,*nodes[nid]])
out_e = ROOT/'results'/case/'elem_vm.csv'
with out_e.open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['elem_id','vm_avg_pa'])
    for eid in sorted(elem_vm): s,c=elem_vm[eid]; w.writerow([eid,s/c if c else 0])
print(f'Extracted {{len(nodes)}} nodes, {{len(elem_vm)}} elements for {case}')
"""])

            # 5. One remodeling cycle
            rc = run([sys.executable, "post/bone_remodeling.py", case, "1"])
            if rc != 0:
                print(f"  bone_remodeling.py failed (rc={rc}) for {lbl} — density metrics unavailable")

            # 6. Collect metrics
            peak_vm = read_peak_vm(case)
            max_u = read_max_u(case)
            mean_rho = read_mean_density(case)

            print(f"  peak_vm={peak_vm:.3e} Pa  max_u={max_u:.3e} m  mean_rho={mean_rho:.4f}")

            rows.append({
                "suture_e_pa": f"{e_pa:.1e}",
                "suture_label": lbl,
                "peak_von_mises_pa": f"{peak_vm:.6e}",
                "max_displacement_m": f"{max_u:.6e}",
                "mean_density_after_1cycle": f"{mean_rho:.6f}",
            })

    finally:
        # Always restore original material.yaml
        restore_suture_e(original_e)
        print(f"\nRestored material.yaml suture E to {original_e:.1e} Pa")

    # Write summary CSV
    out_csv = RESULTS / "suture_sensitivity.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {out_csv}")
        for r in rows:
            print(r)
    else:
        print("\nNo results collected — check ccx and solver outputs.")


if __name__ == "__main__":
    main()
