#!/usr/bin/env python3
"""Generate CalculiX .inp decks from structured mesh + configs.

Loads:
- tongue pressure proxy as distributed nodal force on palate nodes
- optional simplified bilateral jaw-muscle resultant nodal forces
"""
from __future__ import annotations
from pathlib import Path
import json
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]
MESH_PATH = ROOT / "mesh" / "mesh_data.json"
LOADS_PATH = ROOT / "configs" / "loads.yaml"
MAT_PATH = ROOT / "configs" / "material.yaml"
OUTDIR = ROOT / "solver"

mesh = json.loads(MESH_PATH.read_text())
loads = yaml.safe_load(LOADS_PATH.read_text())["load_cases"]
mat = yaml.safe_load(MAT_PATH.read_text())["material"]

nodes = {int(k): v for k, v in mesh["nodes"].items()}
elems = {int(k): v for k, v in mesh["elements"].items()}
sets = mesh["sets"]

fixed = sets["NSET_FIXED"]
palate = sets["NSET_PALATE"]
muscle_left = sets.get("NSET_MUSCLE_LEFT", [])
muscle_right = sets.get("NSET_MUSCLE_RIGHT", [])
muscle_all = sets.get("NSET_MUSCLE_ALL", [])
all_nodes = sorted(nodes.keys())

# approximate tributary area per palate node from top face area
Lx = mesh["dimensions"]["Lx"]
Ly = mesh["dimensions"]["Ly"]
num_palate = len(palate)
area_per_node = (Lx * Ly) / num_palate

E = float(mat["youngs_modulus_pa"])
nu = float(mat["poisson_ratio"])


def write_id_list(fh, ids, chunk=16):
    ids = list(ids)
    for i in range(0, len(ids), chunk):
        fh.write(", ".join(map(str, ids[i:i + chunk])) + "\n")


def parse_case(spec):
    # backward-compatible: scalar = tongue_kpa only
    if isinstance(spec, (int, float)):
        return float(spec), 0.0
    if isinstance(spec, dict):
        return float(spec.get("tongue_kpa", 0.0)), float(spec.get("muscle_force_n", 0.0))
    raise ValueError(f"Unsupported load case format: {spec}")


# muscle resultant direction (posterior, inward, downward), normalized
vx, vy, vz = -0.40, 0.20, -0.90
norm = math.sqrt(vx * vx + vy * vy + vz * vz)
ux, uy, uz = vx / norm, vy / norm, vz / norm

for case, spec in loads.items():
    tongue_kpa, muscle_force_n = parse_case(spec)

    p_pa = tongue_kpa * 1000.0
    # tongue nodal force in z- (downward): F = p * A
    f_tongue_node = -p_pa * area_per_node

    # muscle resultant split L/R and per node
    f_muscle_each_side = muscle_force_n / 2.0 if muscle_force_n > 0 else 0.0
    nL = max(1, len(muscle_left))
    nR = max(1, len(muscle_right))

    fLx = ux * (f_muscle_each_side / nL)
    fLy = +uy * (f_muscle_each_side / nL)   # left side inward (+y)
    fLz = uz * (f_muscle_each_side / nL)

    fRx = ux * (f_muscle_each_side / nR)
    fRy = -uy * (f_muscle_each_side / nR)   # right side inward (-y)
    fRz = uz * (f_muscle_each_side / nR)

    out = OUTDIR / f"{case}.inp"
    with out.open("w", encoding="utf-8") as f:
        f.write("*HEADING\n")
        f.write(f"Template maxilla MVP - {case}\n")
        f.write("*NODE\n")
        for nid, (x, y, z) in nodes.items():
            f.write(f"{nid}, {x:.8f}, {y:.8f}, {z:.8f}\n")

        f.write("*ELEMENT, TYPE=C3D8, ELSET=EALL\n")
        for eid, conn in elems.items():
            f.write(f"{eid}, {', '.join(str(n) for n in conn)}\n")

        f.write("*NSET, NSET=NALL\n")
        write_id_list(f, all_nodes)

        f.write("*NSET, NSET=NSET_FIXED\n")
        write_id_list(f, fixed)

        f.write("*NSET, NSET=NSET_PALATE\n")
        write_id_list(f, palate)

        if muscle_all:
            f.write("*NSET, NSET=NSET_MUSCLE_ALL\n")
            write_id_list(f, muscle_all)

        f.write("*MATERIAL, NAME=BONE\n")
        f.write("*ELASTIC\n")
        f.write(f"{E:.6e}, {nu:.6f}\n")

        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=BONE\n")
        f.write(",\n")

        f.write("*STEP\n")
        f.write("*STATIC\n")
        f.write("1., 1.\n")

        # Fix all translational DOFs on posterior face proxy
        f.write("*BOUNDARY\n")
        f.write("NSET_FIXED, 1, 3, 0.0\n")

        f.write("*CLOAD\n")
        # Tongue pressure proxy
        for nid in palate:
            f.write(f"{nid}, 3, {f_tongue_node:.6e}\n")

        # Jaw-muscle resultant proxy
        if muscle_force_n > 0:
            for nid in muscle_left:
                f.write(f"{nid}, 1, {fLx:.6e}\n")
                f.write(f"{nid}, 2, {fLy:.6e}\n")
                f.write(f"{nid}, 3, {fLz:.6e}\n")
            for nid in muscle_right:
                f.write(f"{nid}, 1, {fRx:.6e}\n")
                f.write(f"{nid}, 2, {fRy:.6e}\n")
                f.write(f"{nid}, 3, {fRz:.6e}\n")

        f.write("*NODE FILE\n")
        f.write("U\n")
        f.write("*EL FILE\n")
        f.write("S\n")

        # Also print to .dat for easy metric extraction
        f.write("*NODE PRINT, NSET=NALL\n")
        f.write("U\n")
        f.write("*EL PRINT, ELSET=EALL\n")
        f.write("S\n")
        f.write("*END STEP\n")

    print(
        f"Wrote {out} "
        f"(tongue_kPa={tongue_kpa}, Ftongue_node={f_tongue_node:.3e} N, Fmuscle_total={muscle_force_n:.3e} N)"
    )
