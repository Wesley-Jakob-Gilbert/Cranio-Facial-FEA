#!/usr/bin/env python3
"""Extract simple metrics from CalculiX .dat files for all configured cases."""
from __future__ import annotations
from pathlib import Path
import csv
import math
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOADS = yaml.safe_load((ROOT / "configs" / "loads.yaml").read_text())["load_cases"]
CASES = list(LOADS.keys())

num_line_4 = re.compile(r"^\s*\d+\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)")
stress_line = re.compile(
    r"^\s*\d+\s+\d+\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)"
)


def case_meta(case_name: str):
    spec = LOADS[case_name]
    if isinstance(spec, (int, float)):
        return float(spec), 0.0
    return float(spec.get("tongue_kpa", 0.0)), float(spec.get("muscle_force_n", 0.0))


def mises(sxx, syy, szz, sxy, sxz, syz):
    return math.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + sxz**2 + syz**2)
    )


rows = [["case", "tongue_pressure_kpa", "muscle_force_n", "max_displacement_m", "peak_von_mises_pa", "status"]]

for case in CASES:
    tongue_kpa, muscle_n = case_meta(case)
    dat = RESULTS / case / f"{case}.dat"
    if not dat.exists() or dat.stat().st_size == 0:
        rows.append([case, f"{tongue_kpa:.3f}", f"{muscle_n:.3f}", "NA", "NA", "missing_dat"])
        continue

    max_u = 0.0
    max_vm = 0.0

    with dat.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = num_line_4.match(line)
            if m:
                ux, uy, uz = map(float, m.groups())
                u = math.sqrt(ux * ux + uy * uy + uz * uz)
                if u > max_u:
                    max_u = u
                continue

            s = stress_line.match(line)
            if s:
                sxx, syy, szz, sxy, sxz, syz = map(float, s.groups())
                vm = mises(sxx, syy, szz, sxy, sxz, syz)
                if vm > max_vm:
                    max_vm = vm

    status = "ok" if (max_u > 0 and max_vm > 0) else "parse_warning"
    rows.append([case, f"{tongue_kpa:.3f}", f"{muscle_n:.3f}", f"{max_u:.6e}", f"{max_vm:.6e}", status])

out = RESULTS / "metrics_summary.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)

print(f"Wrote {out}")
for r in rows[1:]:
    print(r)
