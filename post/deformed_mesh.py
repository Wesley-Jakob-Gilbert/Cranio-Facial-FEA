#!/usr/bin/env python3
"""Compute deformed node positions from FEA displacement results.

Used by the animation pipeline to visualise mesh deformation under load.
Displacements are typically tiny relative to mesh dimensions, so an
auto-scaling utility is provided to make deformation visible.

Inputs:
- mesh/mesh_data.json  (nodes dict: {str(id): [x, y, z]})
- results/<scenario>/node_u.csv  (node_id, ux, uy, uz, u_mag)

Functions:
    load_displacements  — parse node_u.csv into {int_id: (ux, uy, uz)}
    deformed_nodes      — original + scale_factor * displacement
    auto_scale_factor   — pick scale so max disp = fraction of bbox diagonal
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_displacements(csv_path: Path) -> dict[int, tuple[float, float, float]]:
    """Load node displacements from a node_u.csv file.

    Returns {node_id: (ux, uy, uz)} — displacement components in metres.
    """
    displacements: dict[int, tuple[float, float, float]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = int(row["node_id"])
            ux = float(row["ux"])
            uy = float(row["uy"])
            uz = float(row["uz"])
            displacements[nid] = (ux, uy, uz)
    return displacements


def deformed_nodes(
    nodes: dict,
    displacements: dict[int, tuple[float, float, float]],
    scale_factor: float = 1.0,
) -> dict:
    """Return new node dict with positions = original + scale_factor * displacement.

    Parameters
    ----------
    nodes : dict
        {node_id (str or int): [x, y, z]}  — original mesh positions.
    displacements : dict
        {node_id (int): (ux, uy, uz)}  — from load_displacements().
    scale_factor : float
        Multiplier applied to displacement before adding to position.

    Returns
    -------
    dict
        Same structure as *nodes* (string keys, [x, y, z] lists).
        Nodes without a matching displacement entry are copied unchanged.
    """
    result: dict[str, list[float]] = {}
    for nid_key, xyz in nodes.items():
        nid_int = int(nid_key)
        if nid_int in displacements:
            ux, uy, uz = displacements[nid_int]
            result[str(nid_key)] = [
                xyz[0] + scale_factor * ux,
                xyz[1] + scale_factor * uy,
                xyz[2] + scale_factor * uz,
            ]
        else:
            result[str(nid_key)] = list(xyz)
    return result


def auto_scale_factor(
    nodes: dict,
    displacements: dict[int, tuple[float, float, float]],
    target_fraction: float = 0.05,
) -> float:
    """Compute a scale factor so max displacement equals *target_fraction* of bbox diagonal.

    This ensures deformation is visible but not cartoonish.
    Default: deformed shape shows ~5 % of bounding-box diagonal as max
    displacement magnitude.

    Returns 1.0 if max displacement is zero (no deformation).
    """
    if not nodes or not displacements:
        return 1.0

    # Bounding-box diagonal of original mesh
    xs = [xyz[0] for xyz in nodes.values()]
    ys = [xyz[1] for xyz in nodes.values()]
    zs = [xyz[2] for xyz in nodes.values()]
    diag = math.sqrt(
        (max(xs) - min(xs)) ** 2
        + (max(ys) - min(ys)) ** 2
        + (max(zs) - min(zs)) ** 2
    )

    # Maximum displacement magnitude
    max_disp = 0.0
    for ux, uy, uz in displacements.values():
        mag = math.sqrt(ux * ux + uy * uy + uz * uz)
        if mag > max_disp:
            max_disp = mag

    if max_disp == 0.0:
        return 1.0

    return (target_fraction * diag) / max_disp


# ---------------------------------------------------------------------------
# Bounding-box helper (used by __main__ and potentially by callers)
# ---------------------------------------------------------------------------

def _bbox_str(nodes: dict) -> str:
    """Return a human-readable bounding-box summary string."""
    xs = [xyz[0] for xyz in nodes.values()]
    ys = [xyz[1] for xyz in nodes.values()]
    zs = [xyz[2] for xyz in nodes.values()]
    diag = math.sqrt(
        (max(xs) - min(xs)) ** 2
        + (max(ys) - min(ys)) ** 2
        + (max(zs) - min(zs)) ** 2
    )
    return (
        f"  x: [{min(xs):.6f}, {max(xs):.6f}]\n"
        f"  y: [{min(ys):.6f}, {max(ys):.6f}]\n"
        f"  z: [{min(zs):.6f}, {max(zs):.6f}]\n"
        f"  diagonal: {diag:.6f} m"
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mesh_path = ROOT / "mesh" / "mesh_data.json"
    disp_path = ROOT / "results" / "mewing" / "node_u.csv"

    print(f"Loading mesh from {mesh_path}")
    with open(mesh_path, encoding="utf-8") as f:
        mesh_data = json.load(f)
    nodes_orig = mesh_data["nodes"]
    print(f"  {len(nodes_orig)} nodes\n")

    print(f"Loading displacements from {disp_path}")
    disps = load_displacements(disp_path)
    print(f"  {len(disps)} displacement entries\n")

    # Max displacement magnitude
    max_mag = 0.0
    max_nid = -1
    for nid, (ux, uy, uz) in disps.items():
        mag = math.sqrt(ux * ux + uy * uy + uz * uz)
        if mag > max_mag:
            max_mag = mag
            max_nid = nid
    print(f"Max displacement: {max_mag:.6e} m  (node {max_nid})\n")

    # Auto scale
    sf = auto_scale_factor(nodes_orig, disps)
    print(f"Auto scale factor (5% of bbox diagonal): {sf:.1f}\n")

    # Deform
    nodes_def = deformed_nodes(nodes_orig, disps, scale_factor=sf)

    print("Original mesh bounding box:")
    print(_bbox_str(nodes_orig))
    print()
    print("Deformed mesh bounding box (scaled):")
    print(_bbox_str(nodes_def))
