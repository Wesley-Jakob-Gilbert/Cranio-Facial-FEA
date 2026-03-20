#!/usr/bin/env python3
"""Generate CalculiX .inp decks from structured mesh + configs.

Multi-material support:
- Bone elements: BONE material at baseline, or density-binned materials when
  solver/elem_density.json is present (written by bone_remodeling.py each cycle).
  E = E_bone * rho^n_power  (power-law, n_power from remodeling.yaml)
- Sutural elements: SUTURE material, fixed modulus, never remodeled.

Density feedback loop:
  Morpheus writes solver/elem_density.json  →  this script reads it  →
  bins bone elements by density  →  writes one *MATERIAL per bin  →
  CalculiX sees spatially heterogeneous stiffness.

Loads:
- tongue pressure proxy as distributed nodal force on palate nodes
- optional simplified bilateral jaw-muscle resultant nodal forces
- mouth-breathing case (tongue_kpa=0.0) produces a zero-load step: correct
  mechanically (all displacements zero) and valid for CalculiX
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
REMODEL_PATH = ROOT / "configs" / "remodeling.yaml"
DENSITY_PATH = ROOT / "solver" / "elem_density.json"
OUTDIR = ROOT / "solver"

mesh = json.loads(MESH_PATH.read_text())
loads = yaml.safe_load(LOADS_PATH.read_text())["load_cases"]
mat_cfg = yaml.safe_load(MAT_PATH.read_text())

# Multi-material config: prefer 'materials' key, fall back to legacy 'material'
if "materials" in mat_cfg:
    mats = mat_cfg["materials"]
    E_bone = float(mats["bone"]["youngs_modulus_pa"])
    nu_bone = float(mats["bone"]["poisson_ratio"])
    E_suture = float(mats["suture"]["youngs_modulus_pa"])
    nu_suture = float(mats["suture"]["poisson_ratio"])
else:
    leg = mat_cfg["material"]
    E_bone = float(leg["youngs_modulus_pa"])
    nu_bone = float(leg["poisson_ratio"])
    E_suture = E_bone * 0.05
    nu_suture = nu_bone

# Remodeling parameters (defaults; overridden by remodeling.yaml when present)
n_power = 2      # power-law exponent E(rho) = E_bone * rho^n
n_bins = 10      # number of density bins for per-element material grouping
if REMODEL_PATH.exists():
    rm = yaml.safe_load(REMODEL_PATH.read_text()).get("remodeling", {})
    n_power = float(rm.get("n_power", n_power))
    n_bins = int(rm.get("n_bins", n_bins))

# Per-element density override written by bone_remodeling.py each remodeling cycle.
# Keys are string element IDs; values are relative density (rho, dimensionless).
density_override: dict[int, float] = {}
if DENSITY_PATH.exists():
    raw = json.loads(DENSITY_PATH.read_text())
    density_override = {int(k): float(v) for k, v in raw.items()}

nodes = {int(k): v for k, v in mesh["nodes"].items()}
elems = {int(k): v for k, v in mesh["elements"].items()}
sets = mesh["sets"]

fixed = sets["NSET_FIXED"]
palate = sets["NSET_PALATE"]
muscle_left = sets.get("NSET_MUSCLE_LEFT", [])
muscle_right = sets.get("NSET_MUSCLE_RIGHT", [])
muscle_all = sets.get("NSET_MUSCLE_ALL", [])
eset_bone = sets.get("ESET_BONE", [])
eset_suture_mid = sets.get("ESET_SUTURE_MID", [])
eset_suture_lat = sets.get("ESET_SUTURE_LAT", [])
all_nodes = sorted(nodes.keys())

# approximate tributary area per palate node from actual oral-surface face areas
# For v2 (U-shaped arch), Lx*Ly overestimates the curved surface by ~14%.
# Instead, sum the actual k=0 (or k=nz for v1) element face areas.
def _oral_surface_area(nodes, elems, eset_all, nz_div):
    """Sum quad face areas on the oral surface of the mesh."""
    import math as _m
    total = 0.0
    for eid in eset_all:
        conn = elems[eid]
        # C3D8 connectivity: first 4 nodes are k-low face, last 4 are k-high face
        # For v2, oral surface is k=0 (nodes 0-3); for v1, it's k=nz (nodes 4-7)
        # We use the face that palate nodes sit on — just use k-low face (indices 0-3)
        face = conn[:4]
        p = [nodes[n] for n in face]
        # quad area via two triangle diagonals
        d1 = [p[2][c] - p[0][c] for c in range(3)]
        d2 = [p[3][c] - p[1][c] for c in range(3)]
        cross = [d1[1]*d2[2]-d1[2]*d2[1], d1[2]*d2[0]-d1[0]*d2[2], d1[0]*d2[1]-d1[1]*d2[0]]
        total += 0.5 * _m.sqrt(sum(c*c for c in cross))
    return total

_all_elems = sorted(set(eset_bone) | set(eset_suture_mid) | set(eset_suture_lat))
_oral_area = _oral_surface_area(nodes, elems, _all_elems, 0)
area_per_node = _oral_area / max(1, len(palate))


def write_id_list(fh, ids, chunk=16):
    ids = list(ids)
    for i in range(0, len(ids), chunk):
        fh.write(", ".join(map(str, ids[i:i + chunk])) + "\n")


def parse_case(spec):
    if isinstance(spec, (int, float)):
        return float(spec), 0.0
    if isinstance(spec, dict):
        return float(spec.get("tongue_kpa", 0.0)), float(spec.get("muscle_force_n", 0.0))
    raise ValueError(f"Unsupported load case format: {spec}")


def write_material_blocks(fh):
    """Write *ELSET, *MATERIAL, and *SOLID SECTION blocks.

    Baseline (no elem_density.json): two materials — BONE for bulk, SUTURE for
    sutural zones.

    Remodeling iteration (elem_density.json present): bone elements are binned by
    density into n_bins groups; each bin gets its own *MATERIAL with E computed
    from the power law. Suture elements always use SUTURE material.
    """
    if density_override:
        # --- Density-binned mode ---
        rho_vals = [density_override.get(e, 1.0) for e in eset_bone]
        rho_lo = min(rho_vals) if rho_vals else 0.4
        rho_hi = max(rho_vals) if rho_vals else 1.5
        span = rho_hi - rho_lo
        bin_width = span / n_bins if span > 0 else 1.0

        bins: dict[int, list[int]] = {i: [] for i in range(n_bins)}
        for e in eset_bone:
            rho = density_override.get(e, 1.0)
            idx = int((rho - rho_lo) / bin_width)
            idx = max(0, min(n_bins - 1, idx))
            bins[idx].append(e)
        active = {i: v for i, v in bins.items() if v}

        for idx, bin_elems in sorted(active.items()):
            fh.write(f"*ELSET, ELSET=EBONE_BIN_{idx:02d}\n")
            write_id_list(fh, bin_elems)
        if eset_suture_mid:
            fh.write("*ELSET, ELSET=ESET_SUTURE_MID\n")
            write_id_list(fh, eset_suture_mid)
        if eset_suture_lat:
            fh.write("*ELSET, ELSET=ESET_SUTURE_LAT\n")
            write_id_list(fh, eset_suture_lat)

        for idx in sorted(active):
            rho_center = rho_lo + (idx + 0.5) * bin_width
            E_bin = E_bone * (rho_center ** n_power)
            fh.write(f"*MATERIAL, NAME=BONE_BIN_{idx:02d}\n")
            fh.write("*ELASTIC\n")
            fh.write(f"{E_bin:.6e}, {nu_bone:.6f}\n")

        fh.write("*MATERIAL, NAME=SUTURE\n")
        fh.write("*ELASTIC\n")
        fh.write(f"{E_suture:.6e}, {nu_suture:.6f}\n")

        for idx in sorted(active):
            fh.write(f"*SOLID SECTION, ELSET=EBONE_BIN_{idx:02d}, MATERIAL=BONE_BIN_{idx:02d}\n")
            fh.write(",\n")
        if eset_suture_mid:
            fh.write("*SOLID SECTION, ELSET=ESET_SUTURE_MID, MATERIAL=SUTURE\n")
            fh.write(",\n")
        if eset_suture_lat:
            fh.write("*SOLID SECTION, ELSET=ESET_SUTURE_LAT, MATERIAL=SUTURE\n")
            fh.write(",\n")

    else:
        # --- Baseline two-material mode ---
        if eset_bone:
            fh.write("*ELSET, ELSET=ESET_BONE\n")
            write_id_list(fh, eset_bone)
        if eset_suture_mid:
            fh.write("*ELSET, ELSET=ESET_SUTURE_MID\n")
            write_id_list(fh, eset_suture_mid)
        if eset_suture_lat:
            fh.write("*ELSET, ELSET=ESET_SUTURE_LAT\n")
            write_id_list(fh, eset_suture_lat)

        fh.write("*MATERIAL, NAME=BONE\n")
        fh.write("*ELASTIC\n")
        fh.write(f"{E_bone:.6e}, {nu_bone:.6f}\n")

        fh.write("*MATERIAL, NAME=SUTURE\n")
        fh.write("*ELASTIC\n")
        fh.write(f"{E_suture:.6e}, {nu_suture:.6f}\n")

        if eset_bone:
            fh.write("*SOLID SECTION, ELSET=ESET_BONE, MATERIAL=BONE\n")
            fh.write(",\n")
        if eset_suture_mid:
            fh.write("*SOLID SECTION, ELSET=ESET_SUTURE_MID, MATERIAL=SUTURE\n")
            fh.write(",\n")
        if eset_suture_lat:
            fh.write("*SOLID SECTION, ELSET=ESET_SUTURE_LAT, MATERIAL=SUTURE\n")
            fh.write(",\n")


# muscle resultant direction (posterior, inward, downward), normalized
vx, vy, vz = -0.40, 0.20, -0.90
norm = math.sqrt(vx * vx + vy * vy + vz * vz)
ux, uy, uz = vx / norm, vy / norm, vz / norm

mode_label = "density-binned" if density_override else "baseline"

for case, spec in loads.items():
    tongue_kpa, muscle_force_n = parse_case(spec)

    p_pa = tongue_kpa * 1000.0
    f_tongue_node = -p_pa * area_per_node

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

        write_material_blocks(f)

        f.write("*STEP\n")
        f.write("*STATIC\n")
        f.write("1., 1.\n")

        f.write("*BOUNDARY\n")
        f.write("NSET_FIXED, 1, 3, 0.0\n")

        # Tongue pressure — omit entirely for mouth-breathing (tongue_kpa=0.0)
        if f_tongue_node != 0.0:
            f.write("*CLOAD\n")
            for nid in palate:
                f.write(f"{nid}, 3, {f_tongue_node:.6e}\n")

        # Jaw-muscle resultant proxy
        if muscle_force_n > 0:
            f.write("*CLOAD\n")
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

        f.write("*NODE PRINT, NSET=NALL\n")
        f.write("U\n")
        f.write("*EL PRINT, ELSET=EALL\n")
        f.write("S\n")
        f.write("*END STEP\n")

    print(
        f"Wrote {out} [{mode_label}] "
        f"(tongue_kPa={tongue_kpa}, Ftongue_node={f_tongue_node:.3e} N, "
        f"Fmuscle_total={muscle_force_n:.3e} N)"
    )
