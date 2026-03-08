#!/usr/bin/env python3
"""Create cross-case comparison figures with shared color scales.

Outputs:
- results/u_mag_case_grid.png
- results/vm_case_grid.png
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import yaml

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOADS = yaml.safe_load((ROOT / "configs" / "loads.yaml").read_text())["load_cases"]
CASES = list(LOADS.keys())

mesh = json.loads((ROOT / "mesh" / "mesh_data.json").read_text())
nodes = {int(k): v for k, v in mesh["nodes"].items()}
elems = {int(k): v for k, v in mesh["elements"].items()}

# element centers
centers = {}
for eid, conn in elems.items():
    pts = [nodes[n] for n in conn]
    centers[eid] = (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
        sum(p[2] for p in pts) / len(pts),
    )

u_data = {}
vm_data = {}

for case in CASES:
    node_csv = ROOT / "results" / case / "node_u.csv"
    elem_csv = ROOT / "results" / case / "elem_vm.csv"
    if not node_csv.exists() or not elem_csv.exists():
        print(f"Skipping {case}: missing field csv")
        continue

    nx, ny, nz, nu = [], [], [], []
    with node_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            nid = int(row["node_id"])
            x, y, z = nodes[nid]
            nx.append(x)
            ny.append(y)
            nz.append(z)
            nu.append(float(row["u_mag"]))
    u_data[case] = (nx, ny, nz, nu)

    ex, ey, ez, ev = [], [], [], []
    with elem_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = int(row["elem_id"])
            if eid not in centers:
                continue
            cx, cy, cz = centers[eid]
            ex.append(cx)
            ey.append(cy)
            ez.append(cz)
            ev.append(float(row["vm_avg_pa"]))
    vm_data[case] = (ex, ey, ez, ev)

valid_cases = [c for c in CASES if c in u_data and c in vm_data]
if len(valid_cases) < 1:
    raise SystemExit("No case data available for grid visuals")

# shared scales
u_vals = [v for case in valid_cases for v in u_data[case][3]]
vm_vals = [v for case in valid_cases for v in vm_data[case][3]]

u_min, u_max = min(u_vals), max(u_vals)
vm_min, vm_max = min(vm_vals), max(vm_vals)

# consistent camera
ELEV, AZIM = 20, -65

n = len(valid_cases)
ncols = 3
nrows = math.ceil(n / ncols)

# displacement grid
fig = plt.figure(figsize=(4.8 * ncols, 4.2 * nrows), constrained_layout=True)
for i, case in enumerate(valid_cases, start=1):
    ax = fig.add_subplot(nrows, ncols, i, projection="3d")
    x, y, z, u = u_data[case]
    sc = ax.scatter(x, y, z, c=u, cmap="viridis", s=8, vmin=u_min, vmax=u_max)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_title(case)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.75, pad=0.03)
cbar.set_label("u_mag (m)")
fig.suptitle("Nodal displacement magnitude across configured cases")
fig.savefig(ROOT / "results" / "u_mag_case_grid.png", dpi=180)
plt.close(fig)

# stress grid
fig = plt.figure(figsize=(4.8 * ncols, 4.2 * nrows), constrained_layout=True)
for i, case in enumerate(valid_cases, start=1):
    ax = fig.add_subplot(nrows, ncols, i, projection="3d")
    x, y, z, vm = vm_data[case]
    sc = ax.scatter(x, y, z, c=vm, cmap="inferno", s=12, vmin=vm_min, vmax=vm_max)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_title(case)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.75, pad=0.03)
cbar.set_label("von Mises (Pa)")
fig.suptitle("Element von Mises averages across configured cases")
fig.savefig(ROOT / "results" / "vm_case_grid.png", dpi=180)
plt.close(fig)

print("Wrote case grid visuals")
