#!/usr/bin/env python3
"""Comprehensive mesh quality metrics for C3D8 hexahedral mesh.

Computes per-element:
  - Aspect ratio (longest edge / shortest edge)
  - Jacobian determinant at all 8 Gauss integration points
  - Scaled Jacobian (min |J| / max |J| per element)
  - Element volume (via 8-point Gauss quadrature)
  - Skewness (deviation from ideal hex shape)

Outputs:
  results/mesh_quality.csv         — per-element metrics
  results/mesh_quality_summary.json — aggregate statistics

Usage:
    python3 scripts/mesh_quality_report.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MESH_PATH = ROOT / "mesh" / "mesh_data.json"
CSV_OUT = ROOT / "results" / "mesh_quality.csv"
JSON_OUT = ROOT / "results" / "mesh_quality_summary.json"

# ---------------------------------------------------------------------------
# Load mesh
# ---------------------------------------------------------------------------
mesh = json.loads(MESH_PATH.read_text())
nodes_raw = mesh["nodes"]
elems_raw = mesh["elements"]
profile = mesh.get("profile", "unknown")

# Build numpy arrays for fast access
# node_coords[node_id] = [x, y, z]  (using max node id + 1 for direct indexing)
max_nid = max(int(k) for k in nodes_raw)
node_coords = np.zeros((max_nid + 1, 3))
for nid_str, xyz in nodes_raw.items():
    node_coords[int(nid_str)] = xyz

# elem_conn: list of (elem_id, conn_array) sorted by elem_id
elem_ids = sorted(int(k) for k in elems_raw)
elem_conn = np.array([elems_raw[str(eid)] for eid in elem_ids], dtype=int)  # (N, 8)
n_elems = len(elem_ids)

# ---------------------------------------------------------------------------
# C3D8 reference geometry
# ---------------------------------------------------------------------------
# 12 edges of a hex: pairs of local node indices
HEX_EDGES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],  # top face
    [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
], dtype=int)

# Corner signs for trilinear hex: (xi, eta, zeta) per node
# Node ordering: 0-1-2-3 bottom, 4-5-6-7 top (CalculiX C3D8 convention)
CORNER_SIGNS = np.array([
    [-1, -1, -1],
    [+1, -1, -1],
    [+1, +1, -1],
    [-1, +1, -1],
    [-1, -1, +1],
    [+1, -1, +1],
    [+1, +1, +1],
    [-1, +1, +1],
], dtype=float)  # (8, 3)

# 2x2x2 Gauss points: +/- 1/sqrt(3)
_g = 1.0 / np.sqrt(3.0)
GAUSS_PTS = np.array([
    [-_g, -_g, -_g],
    [+_g, -_g, -_g],
    [+_g, +_g, -_g],
    [-_g, +_g, -_g],
    [-_g, -_g, +_g],
    [+_g, -_g, +_g],
    [+_g, +_g, +_g],
    [-_g, +_g, +_g],
], dtype=float)  # (8, 3)

# Gauss weights are all 1.0 for 2x2x2 scheme
GAUSS_WEIGHTS = np.ones(8, dtype=float)

# Ideal hex unit cube node positions (for skewness): [-1,1]^3
IDEAL_CORNERS = CORNER_SIGNS.copy()  # (8, 3) in [-1, +1]


def shape_function_derivs(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Compute shape function derivatives dN/d(xi,eta,zeta) at a parametric point.

    Returns array of shape (8, 3):
        dN[i, 0] = dN_i / d_xi
        dN[i, 1] = dN_i / d_eta
        dN[i, 2] = dN_i / d_zeta
    """
    dN = np.empty((8, 3))
    for i in range(8):
        si, ei, zi = CORNER_SIGNS[i]
        dN[i, 0] = si * (1.0 + ei * eta) * (1.0 + zi * zeta) / 8.0
        dN[i, 1] = ei * (1.0 + si * xi) * (1.0 + zi * zeta) / 8.0
        dN[i, 2] = zi * (1.0 + si * xi) * (1.0 + ei * eta) / 8.0
    return dN


def jacobian_matrix(dN: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Compute 3x3 Jacobian matrix J = dN^T @ coords.

    dN: (8, 3) shape function derivatives
    coords: (8, 3) physical node coordinates

    Returns (3, 3) Jacobian.
    """
    return dN.T @ coords  # (3, 3)


# ---------------------------------------------------------------------------
# Precompute shape function derivatives at all 8 Gauss points
# ---------------------------------------------------------------------------
dN_at_gp = np.array([shape_function_derivs(*gp) for gp in GAUSS_PTS])  # (8, 8, 3)

# Also precompute shape functions at Gauss points (for skewness mapping)
def shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Trilinear shape functions N_i at parametric point. Returns (8,)."""
    N = np.empty(8)
    for i in range(8):
        si, ei, zi = CORNER_SIGNS[i]
        N[i] = (1.0 + si * xi) * (1.0 + ei * eta) * (1.0 + zi * zeta) / 8.0
    return N


N_at_gp = np.array([shape_functions(*gp) for gp in GAUSS_PTS])  # (8, 8)

# ---------------------------------------------------------------------------
# Metric computation (vectorized over Gauss points per element)
# ---------------------------------------------------------------------------

# Pre-gather all element coordinates: (n_elems, 8, 3)
all_coords = node_coords[elem_conn]  # fancy indexing

# --- Aspect ratio -----------------------------------------------------------
# Edge vectors for all elements at once
edge_start = all_coords[:, HEX_EDGES[:, 0], :]  # (n_elems, 12, 3)
edge_end = all_coords[:, HEX_EDGES[:, 1], :]    # (n_elems, 12, 3)
edge_vecs = edge_end - edge_start                # (n_elems, 12, 3)
edge_lengths = np.linalg.norm(edge_vecs, axis=2)  # (n_elems, 12)

min_edge = edge_lengths.min(axis=1)  # (n_elems,)
max_edge = edge_lengths.max(axis=1)  # (n_elems,)
aspect_ratios = np.where(min_edge > 1e-15, max_edge / min_edge, np.inf)

# --- Jacobian at all 8 Gauss points per element -----------------------------
# For each Gauss point g, J_g = dN_at_gp[g].T @ coords  -> (3, 3)
# We need the determinant at each.
# Vectorized: for each GP, compute all elements at once.

jac_dets = np.empty((n_elems, 8))  # (n_elems, 8 gauss points)
for g in range(8):
    # dN_at_gp[g] is (8, 3)
    # all_coords is (n_elems, 8, 3)
    # J = dN^T @ coords => for each element: (3, 8) @ (8, 3) = (3, 3)
    # Vectorized: J_all = einsum('ji,nji->njk'... or just matmul
    # J[e] = dN_at_gp[g].T @ all_coords[e]
    # = (3, 8) @ (N, 8, 3)^elem -> need einsum
    dN_g = dN_at_gp[g]  # (8, 3)
    # J_all[e, i, j] = sum_k dN_g[k, i] * all_coords[e, k, j]
    J_all = np.einsum("ki,eki->ei", dN_g, all_coords).reshape(n_elems, 1, 3)
    # That's not right. Let me do it properly.
    # J[i,j] = sum_k dN[k,i] * x[k,j]   where i indexes (xi,eta,zeta), j indexes (x,y,z)
    # For all elements: J_all[e,i,j] = sum_k dN_g[k,i] * all_coords[e,k,j]
    J_all = np.einsum("ki,ekj->eij", dN_g, all_coords)  # (n_elems, 3, 3)
    jac_dets[:, g] = np.linalg.det(J_all)

min_jac = jac_dets.min(axis=1)  # (n_elems,)
max_jac = jac_dets.max(axis=1)  # (n_elems,)

# --- Scaled Jacobian --------------------------------------------------------
# Scaled Jacobian = min(J) / max(J) per element, considering absolute values
# Convention: if all Jacobians are positive, scaled_jac = min/max in (0, 1].
# If any negative, it signals inverted element.
abs_jac = np.abs(jac_dets)
abs_min_jac = abs_jac.min(axis=1)
abs_max_jac = abs_jac.max(axis=1)
scaled_jac = np.where(abs_max_jac > 1e-30, abs_min_jac / abs_max_jac, 0.0)
# Flip sign for elements with any negative Jacobian
has_neg = (jac_dets < 0).any(axis=1)
scaled_jac = np.where(has_neg, -scaled_jac, scaled_jac)

# --- Volume (8-point Gauss quadrature) --------------------------------------
# V = sum_g w_g * det(J_g)  where w_g = 1.0 for 2x2x2 Gauss
volumes = jac_dets.sum(axis=1)  # weights are all 1.0

# --- Skewness ---------------------------------------------------------------
# Skewness measures deviation from the ideal (regular) hex shape.
# We use the approach: for each Gauss point, map the ideal hex nodes through
# the element's shape functions to get "ideal" Jacobian, then compare.
#
# A simpler and more standard approach for hex skewness:
# Compute the diagonal vectors of each face and measure the angle between them.
# Skewness = max over all faces of |90 - theta| / 90
# where theta is the angle between face diagonals.
#
# Face definitions (4 faces with their diagonal node pairs):
# Bottom: 0-1-2-3, diags: 0-2, 1-3
# Top:    4-5-6-7, diags: 4-6, 5-7
# Front:  0-1-5-4, diags: 0-5, 1-4
# Back:   3-2-6-7, diags: 3-6, 2-7
# Left:   0-3-7-4, diags: 0-7, 3-4
# Right:  1-2-6-5, diags: 1-6, 2-5

FACE_DIAG_PAIRS = [
    # (node_a1, node_a2, node_b1, node_b2) for each face's two diagonals
    (0, 2, 1, 3),  # bottom
    (4, 6, 5, 7),  # top
    (0, 5, 1, 4),  # front
    (3, 6, 2, 7),  # back
    (0, 7, 3, 4),  # left
    (1, 6, 2, 5),  # right
]

skewness = np.zeros(n_elems)
for a1, a2, b1, b2 in FACE_DIAG_PAIRS:
    diag_a = all_coords[:, a2, :] - all_coords[:, a1, :]  # (n_elems, 3)
    diag_b = all_coords[:, b2, :] - all_coords[:, b1, :]  # (n_elems, 3)
    # Angle between diagonals
    dot = np.einsum("ij,ij->i", diag_a, diag_b)
    mag_a = np.linalg.norm(diag_a, axis=1)
    mag_b = np.linalg.norm(diag_b, axis=1)
    denom = mag_a * mag_b
    cos_theta = np.where(denom > 1e-30, dot / denom, 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(np.abs(cos_theta)))  # angle in [0, 90]
    # Skewness contribution: |90 - theta| / 90
    face_skew = np.abs(90.0 - theta_deg) / 90.0
    skewness = np.maximum(skewness, face_skew)

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
CSV_OUT.parent.mkdir(parents=True, exist_ok=True)

with open(CSV_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["elem_id", "aspect_ratio", "min_jacobian", "scaled_jacobian",
                     "volume", "skewness"])
    for i, eid in enumerate(elem_ids):
        writer.writerow([
            eid,
            f"{aspect_ratios[i]:.6f}",
            f"{min_jac[i]:.6e}",
            f"{scaled_jac[i]:.6f}",
            f"{volumes[i]:.6e}",
            f"{skewness[i]:.6f}",
        ])

# ---------------------------------------------------------------------------
# Compute summary statistics
# ---------------------------------------------------------------------------
ar_p95 = float(np.percentile(aspect_ratios, 95))
num_negative_jac = int((min_jac < 0).sum())
quality_pass = bool(num_negative_jac == 0 and ar_p95 < 10.0)

summary = {
    "profile": profile,
    "total_elements": n_elems,
    "aspect_ratio": {
        "min": round(float(aspect_ratios.min()), 6),
        "max": round(float(aspect_ratios.max()), 6),
        "mean": round(float(aspect_ratios.mean()), 6),
        "p95": round(ar_p95, 6),
    },
    "min_jacobian": {
        "min": float(f"{min_jac.min():.6e}"),
        "max": float(f"{min_jac.max():.6e}"),
        "mean": float(f"{min_jac.mean():.6e}"),
        "num_negative": num_negative_jac,
    },
    "scaled_jacobian": {
        "min": round(float(scaled_jac.min()), 6),
        "max": round(float(scaled_jac.max()), 6),
        "mean": round(float(scaled_jac.mean()), 6),
    },
    "volume": {
        "min": float(f"{volumes.min():.6e}"),
        "max": float(f"{volumes.max():.6e}"),
        "total": float(f"{volumes.sum():.6e}"),
    },
    "quality_pass": quality_pass,
}

JSON_OUT.write_text(json.dumps(summary, indent=2) + "\n")

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print(f"Mesh Quality Report -- {profile}")
print(f"{'=' * 55}")
print(f"Elements: {n_elems}")
print()
print(f"Aspect Ratio (max edge / min edge):")
print(f"  Min:  {aspect_ratios.min():.4f}")
print(f"  Max:  {aspect_ratios.max():.4f}")
print(f"  Mean: {aspect_ratios.mean():.4f}")
print(f"  P95:  {ar_p95:.4f}")
print()
print(f"Min Jacobian Determinant (over 8 Gauss points):")
print(f"  Min:  {min_jac.min():.6e}")
print(f"  Max:  {min_jac.max():.6e}")
print(f"  Mean: {min_jac.mean():.6e}")
print(f"  Negative Jacobian elements: {num_negative_jac}")
print()
print(f"Scaled Jacobian (|J_min| / |J_max| per element):")
print(f"  Min:  {scaled_jac.min():.4f}")
print(f"  Max:  {scaled_jac.max():.4f}")
print(f"  Mean: {scaled_jac.mean():.4f}")
print()
print(f"Volume:")
print(f"  Min:    {volumes.min():.6e} m^3")
print(f"  Max:    {volumes.max():.6e} m^3")
print(f"  Total:  {volumes.sum():.6e} m^3")
print()
qp = "PASS" if quality_pass else "FAIL"
print(f"Quality Check: {qp}")
if not quality_pass:
    reasons = []
    if num_negative_jac > 0:
        reasons.append(f"{num_negative_jac} elements with negative Jacobian")
    if ar_p95 >= 10.0:
        reasons.append(f"aspect ratio P95 = {ar_p95:.2f} >= 10.0")
    print(f"  Reason: {'; '.join(reasons)}")
print()
print(f"Wrote {CSV_OUT}")
print(f"Wrote {JSON_OUT}")
