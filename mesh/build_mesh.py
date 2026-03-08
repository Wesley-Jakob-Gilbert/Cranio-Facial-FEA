#!/usr/bin/env python3
"""Generate structured hexahedral mesh for cranio_FEA.

Profiles:
- toy: rectangular block with mild palate curvature
- template_maxilla_v1: maxilla-like warped template (still simplified)

Output: mesh/mesh_data.json
"""
from __future__ import annotations
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mesh" / "mesh_data.json"
CFG = ROOT / "configs" / "geometry.json"

# defaults
cfg = {
    "profile": "template_maxilla_v1",
    "dimensions_m": {"Lx": 0.08, "Ly": 0.05, "Lz": 0.03},
    "divisions": {"nx": 14, "ny": 10, "nz": 5},
    "curvature_amplitude_m": 0.004,
}
if CFG.exists():
    cfg.update(json.loads(CFG.read_text()))

Lx = float(cfg["dimensions_m"]["Lx"])
Ly = float(cfg["dimensions_m"]["Ly"])
Lz = float(cfg["dimensions_m"]["Lz"])
nx = int(cfg["divisions"]["nx"])
ny = int(cfg["divisions"]["ny"])
nz = int(cfg["divisions"]["nz"])
curv_amp = float(cfg.get("curvature_amplitude_m", 0.004))
profile = cfg.get("profile", "template_maxilla_v1")

nodes: dict[int, list[float]] = {}
node_id = 1

for k in range(nz + 1):
    z_frac = k / nz
    z0 = Lz * z_frac

    for j in range(ny + 1):
        v = j / ny
        y = Ly * v
        y_mid = y - Ly / 2.0

        for i in range(nx + 1):
            u = i / nx
            x = Lx * u

            # base mild palate dome
            dome = math.sin(math.pi * u) * math.sin(math.pi * v)
            z = z0 + curv_amp * dome * (z_frac**2)

            if profile == "template_maxilla_v1":
                # 1) anterior taper in width (maxilla narrows toward incisors)
                #    posterior at x~0, anterior at x~Lx
                width_scale = 1.0 - 0.18 * u
                y_tapered = (Ly / 2.0) + y_mid * width_scale

                # 2) slight alveolar ridge lift near anterior-lateral top
                ridge = 0.0016 * (z_frac**1.8) * (u**1.4) * (abs(y_mid) / (Ly / 2.0))

                # 3) shallow central palate groove near midline on top surface
                groove = -0.0011 * (z_frac**2.0) * math.exp(-((y_mid / (0.22 * Ly)) ** 2)) * (0.5 + 0.5 * u)

                # 4) posterior downward skew to mimic cranial base transition
                posterior_bias = -0.0009 * (1.0 - u) * (z_frac**1.2)

                y = y_tapered
                z = z + ridge + groove + posterior_bias

            nodes[node_id] = [x, y, z]
            node_id += 1

nnx = nx + 1
nny = ny + 1

def nid(i: int, j: int, k: int) -> int:
    return k * (nny * nnx) + j * nnx + i + 1

# C3D8 elements
elements: dict[int, list[int]] = {}
eid = 1
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            elements[eid] = [
                nid(i, j, k),
                nid(i + 1, j, k),
                nid(i + 1, j + 1, k),
                nid(i, j + 1, k),
                nid(i, j, k + 1),
                nid(i + 1, j, k + 1),
                nid(i + 1, j + 1, k + 1),
                nid(i, j + 1, k + 1),
            ]
            eid += 1

# ---------------------------------------------------------------------------
# Element sets for multi-material remodeling
# ---------------------------------------------------------------------------
suture_mid_thick = int(cfg.get("suture_mid_thickness_elems", 1))
suture_lat_width = int(cfg.get("suture_lat_width_elems", 2))
suture_lat_min_i = int(cfg.get("suture_lat_min_i_frac", 0.5) * nx)


def eid(i: int, j: int, k: int) -> int:
    """Element ID for 0-based grid position (i, j, k)."""
    return k * (ny * nx) + j * nx + i + 1


# Midpalatal suture: 1-element-thick sagittal band at j = ny//2 (full i, full k)
mid_center = ny // 2
mid_lo = max(0, mid_center - suture_mid_thick // 2)
mid_hi = min(ny - 1, mid_lo + suture_mid_thick - 1)
suture_mid_set = {
    eid(i, j, k)
    for k in range(nz)
    for j in range(mid_lo, mid_hi + 1)
    for i in range(nx)
}

# Lateral sutures: anterior-lateral bilateral bands (high i, low/high j, all k)
suture_lat_set = {
    eid(i, j, k)
    for k in range(nz)
    for j in range(ny)
    for i in range(suture_lat_min_i, nx)
    if (j < suture_lat_width or j >= ny - suture_lat_width)
} - suture_mid_set  # exclude any overlap with midpalatal band

suture_all_set = suture_mid_set | suture_lat_set
bone_elems = sorted(e for e in elements if e not in suture_all_set)
suture_mid_elems = sorted(suture_mid_set)
suture_lat_elems = sorted(suture_lat_set)

# ---------------------------------------------------------------------------
# Node sets
# ---------------------------------------------------------------------------
fixed_nodes = [nid(0, j, k) for k in range(nz + 1) for j in range(ny + 1)]
palate_nodes = [nid(i, j, nz) for j in range(ny + 1) for i in range(nx + 1)]

# simplified bilateral jaw-muscle resultant attachment proxy nodes
# anterior-lateral top strips (left/right)
muscle_left_nodes = [
    nid(i, j, nz)
    for j in range(ny + 1)
    for i in range(nx + 1)
    if (i >= int(0.65 * nx) and j <= int(0.2 * ny))
]
muscle_right_nodes = [
    nid(i, j, nz)
    for j in range(ny + 1)
    for i in range(nx + 1)
    if (i >= int(0.65 * nx) and j >= int(0.8 * ny))
]
muscle_all_nodes = sorted(set(muscle_left_nodes + muscle_right_nodes))

mesh = {
    "units": "m",
    "profile": profile,
    "element_type": "C3D8",
    "dimensions": {"Lx": Lx, "Ly": Ly, "Lz": Lz},
    "divisions": {"nx": nx, "ny": ny, "nz": nz},
    "nodes": nodes,
    "elements": elements,
    "sets": {
        "NSET_FIXED": fixed_nodes,
        "NSET_PALATE": palate_nodes,
        "NSET_MUSCLE_LEFT": muscle_left_nodes,
        "NSET_MUSCLE_RIGHT": muscle_right_nodes,
        "NSET_MUSCLE_ALL": muscle_all_nodes,
        "ESET_BONE": bone_elems,
        "ESET_SUTURE_MID": suture_mid_elems,
        "ESET_SUTURE_LAT": suture_lat_elems,
    },
}

OUT.write_text(json.dumps(mesh))
print(f"Wrote {OUT}")
print(f"Profile: {profile}")
print(f"Nodes: {len(nodes)}, Elements: {len(elements)}")
print(f"ESET_BONE: {len(bone_elems)}, ESET_SUTURE_MID: {len(suture_mid_elems)}, ESET_SUTURE_LAT: {len(suture_lat_elems)}")
