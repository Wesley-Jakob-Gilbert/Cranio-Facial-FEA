#!/usr/bin/env python3
"""Create 3D visualizations of mesh and FE fields using matplotlib.

Outputs:
- results/geometry_wireframe.png
- results/<case>/u_mag_3d_scatter.png
- results/<case>/vm_3d_scatter.png
"""
from __future__ import annotations
from pathlib import Path
import csv
import json
import yaml

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
mesh = json.loads((ROOT / "mesh" / "mesh_data.json").read_text())
LOADS = yaml.safe_load((ROOT / "configs" / "loads.yaml").read_text())["load_cases"]

nodes = {int(k): v for k, v in mesh["nodes"].items()}
elems = {int(k): v for k, v in mesh["elements"].items()}

# 1) geometry wireframe-ish scatter
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")
xs = [v[0] for _, v in sorted(nodes.items())]
ys = [v[1] for _, v in sorted(nodes.items())]
zs = [v[2] for _, v in sorted(nodes.items())]
ax.scatter(xs, ys, zs, s=8, alpha=0.7)
ax.set_title("Toy curved maxilla mesh nodes")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
fig.tight_layout()
out_geo = ROOT / "results" / "geometry_wireframe.png"
fig.savefig(out_geo, dpi=170)
plt.close(fig)

# precompute element centers
centers = {}
for eid, conn in elems.items():
    pts = [nodes[n] for n in conn]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    cz = sum(p[2] for p in pts) / len(pts)
    centers[eid] = (cx, cy, cz)

for case in LOADS.keys():
    case_dir = ROOT / "results" / case
    node_csv = case_dir / "node_u.csv"
    elem_csv = case_dir / "elem_vm.csv"
    if not node_csv.exists() or not elem_csv.exists():
        print(f"Skip {case}: missing field CSVs")
        continue

    # nodal displacement magnitude
    n_x, n_y, n_z, n_u = [], [], [], []
    with node_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            nid = int(row["node_id"])
            x, y, z = nodes[nid]
            n_x.append(x)
            n_y.append(y)
            n_z.append(z)
            n_u.append(float(row["u_mag"]))

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(n_x, n_y, n_z, c=n_u, cmap="viridis", s=14)
    ax.set_title(f"{case}: nodal displacement magnitude")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    fig.colorbar(sc, ax=ax, shrink=0.75, label="u_mag (m)")
    fig.tight_layout()
    fig.savefig(case_dir / "u_mag_3d_scatter.png", dpi=170)
    plt.close(fig)

    # element vm on centers
    e_x, e_y, e_z, e_vm = [], [], [], []
    with elem_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = int(row["elem_id"])
            if eid not in centers:
                continue
            cx, cy, cz = centers[eid]
            e_x.append(cx)
            e_y.append(cy)
            e_z.append(cz)
            e_vm.append(float(row["vm_avg_pa"]))

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(e_x, e_y, e_z, c=e_vm, cmap="inferno", s=20)
    ax.set_title(f"{case}: element von Mises (avg)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    fig.colorbar(sc, ax=ax, shrink=0.75, label="von Mises (Pa)")
    fig.tight_layout()
    fig.savefig(case_dir / "vm_3d_scatter.png", dpi=170)
    plt.close(fig)

print(f"Wrote {out_geo} and case 3D field figures")
