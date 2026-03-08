#!/usr/bin/env python3
"""Toy bone adaptation simulation (post-processing layer, not coupled FEA).

Purpose:
- Explore long-term directional effects of repeated loading using a simple,
  explicit remodeling proxy on element density-like state.

This is intentionally heuristic for MVP+1 and not a validated biological model.
"""
from __future__ import annotations
from pathlib import Path
import csv

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
VM_CSV = ROOT / "results" / "medium_kpa" / "elem_vm.csv"
OUT_CSV = ROOT / "results" / "adaptation_timeseries.csv"
OUT_PLOT = ROOT / "results" / "adaptation_trend.png"

# parameters (heuristic)
steps = 40
alpha = 0.01          # adaptation rate per step
rho0 = 1.0
rho_min, rho_max = 0.7, 1.3

# load baseline vm field
elem_vm = {}
with VM_CSV.open("r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        elem_vm[int(row["elem_id"])] = float(row["vm_avg_pa"])

if not elem_vm:
    raise SystemExit("No vm data found. Run extract_fields.py first.")

vref = sum(elem_vm.values()) / len(elem_vm)

rho = {eid: rho0 for eid in elem_vm}
records = []

for t in range(steps + 1):
    vals = list(rho.values())
    records.append(
        {
            "step": t,
            "rho_mean": sum(vals) / len(vals),
            "rho_min": min(vals),
            "rho_max": max(vals),
        }
    )

    if t == steps:
        break

    new_rho = {}
    for eid, vm in elem_vm.items():
        stimulus = (vm / vref) - 1.0
        r_next = rho[eid] * (1.0 + alpha * stimulus)
        r_next = max(rho_min, min(rho_max, r_next))
        new_rho[eid] = r_next
    rho = new_rho

# write timeseries
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["step", "rho_mean", "rho_min", "rho_max"])
    w.writeheader()
    w.writerows(records)

# plot
x = [r["step"] for r in records]
mean = [r["rho_mean"] for r in records]
rmin = [r["rho_min"] for r in records]
rmax = [r["rho_max"] for r in records]

plt.figure(figsize=(7, 4))
plt.plot(x, mean, label="mean density-like state")
plt.plot(x, rmin, "--", label="min")
plt.plot(x, rmax, "--", label="max")
plt.xlabel("adaptation step")
plt.ylabel("density-like state (a.u.)")
plt.title("Toy adaptation trend under repeated medium load")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=170)

print(f"Wrote {OUT_CSV}")
print(f"Wrote {OUT_PLOT}")
