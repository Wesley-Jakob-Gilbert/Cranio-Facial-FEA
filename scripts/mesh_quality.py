#!/usr/bin/env python3
"""Mesh quality metrics for C3D8 hexahedral mesh (N-16).

Computes per-element:
  - Aspect ratio (max edge length / min edge length)
  - Jacobian determinant at element center

Reports summary statistics and flags any inverted (negative Jacobian) elements.
Saves results/mesh_quality_report.txt.

Usage:
    python3 scripts/mesh_quality.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MESH_PATH = ROOT / "mesh" / "mesh_data.json"
OUT = ROOT / "results" / "mesh_quality_report.txt"

mesh = json.loads(MESH_PATH.read_text())
nodes = {int(k): v for k, v in mesh["nodes"].items()}
elems = {int(k): v for k, v in mesh["elements"].items()}
profile = mesh.get("profile", "unknown")


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


# C3D8 edges: 12 edges connecting the 8 nodes
# Node ordering: 0-1-2-3 (bottom face), 4-5-6-7 (top face)
HEX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]


def aspect_ratio(conn):
    pts = [nodes[n] for n in conn]
    lengths = [dist(pts[e[0]], pts[e[1]]) for e in HEX_EDGES]
    lo, hi = min(lengths), max(lengths)
    return hi / lo if lo > 1e-15 else float("inf")


def jacobian_det_center(conn):
    """Jacobian determinant at the center (xi=eta=zeta=0) of a trilinear hex."""
    pts = [nodes[n] for n in conn]
    # Shape function derivatives at center of C3D8 (all xi=eta=zeta=0)
    # dN/dxi, dN/deta, dN/dzeta for each of 8 nodes
    signs = [
        (-1, -1, -1), (+1, -1, -1), (+1, +1, -1), (-1, +1, -1),
        (-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1),
    ]
    # At center, dN_i/dxi = si * (1+ej*0) * (1+zk*0) / 8 = si/8
    # etc.
    J = [[0.0, 0.0, 0.0] for _ in range(3)]
    for idx, (si, ei, zi) in enumerate(signs):
        dNdxi = si / 8.0
        dNdeta = ei / 8.0
        dNdzeta = zi / 8.0
        for c in range(3):
            J[0][c] += dNdxi * pts[idx][c]
            J[1][c] += dNdeta * pts[idx][c]
            J[2][c] += dNdzeta * pts[idx][c]

    # det(J) = J[0] . (J[1] x J[2])
    cross = [
        J[1][1] * J[2][2] - J[1][2] * J[2][1],
        J[1][2] * J[2][0] - J[1][0] * J[2][2],
        J[1][0] * J[2][1] - J[1][1] * J[2][0],
    ]
    return J[0][0] * cross[0] + J[0][1] * cross[1] + J[0][2] * cross[2]


# Compute metrics
aspect_ratios = {}
jacobians = {}
for eid, conn in elems.items():
    aspect_ratios[eid] = aspect_ratio(conn)
    jacobians[eid] = jacobian_det_center(conn)

ar_vals = list(aspect_ratios.values())
jac_vals = list(jacobians.values())
neg_jac = [eid for eid, j in jacobians.items() if j <= 0]

lines = [
    f"Mesh Quality Report — {profile}",
    f"{'=' * 50}",
    f"Elements: {len(elems)}",
    f"",
    f"Aspect Ratio (max edge / min edge):",
    f"  Min:  {min(ar_vals):.3f}",
    f"  Max:  {max(ar_vals):.3f}",
    f"  Mean: {sum(ar_vals)/len(ar_vals):.3f}",
    f"",
    f"Jacobian Determinant (at element center):",
    f"  Min:  {min(jac_vals):.6e}",
    f"  Max:  {max(jac_vals):.6e}",
    f"  Mean: {sum(jac_vals)/len(jac_vals):.6e}",
    f"",
    f"Inverted elements (J <= 0): {len(neg_jac)}",
]

if neg_jac:
    lines.append(f"  IDs: {neg_jac[:20]}{'...' if len(neg_jac) > 20 else ''}")

report = "\n".join(lines) + "\n"

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(report)
print(report)
print(f"Wrote {OUT}")
