#!/usr/bin/env python3
"""Generate structured hexahedral mesh for cranio_FEA.

Profiles
--------
template_maxilla_v1
    Rectangular brick with mild palate curvature, alveolar ridge lift,
    central palate groove, and posterior downward skew. Fast, simple.

template_maxilla_v2  (Week 2 — anatomical arch-sweep)
    U-shaped dental arch mesh swept from a parabolic centreline.
    Produces a recognisable horseshoe palate outline in occlusal view.

    Parameterisation:
      i  (0 .. nx)  along arch: posterior-left → anterior → posterior-right
      j  (0 .. ny)  mediolateral: j=0 alveolar crest (outer), j=ny midline (inner)
      k  (0 .. nz)  through thickness: k=0 oral/inferior surface, k=nz nasal/superior

    mesh_data.json schema is identical to v1 — all set keys preserved.

Output: mesh/mesh_data.json
"""
from __future__ import annotations
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mesh" / "mesh_data.json"
CFG = ROOT / "configs" / "geometry.json"

# ── config ────────────────────────────────────────────────────────────────────
cfg: dict = {
    "profile": "template_maxilla_v1",
    "dimensions_m": {"Lx": 0.08, "Ly": 0.05, "Lz": 0.03},
    "divisions": {"nx": 14, "ny": 10, "nz": 5},
    "curvature_amplitude_m": 0.004,
    "suture_mid_thickness_elems": 1,
    "suture_lat_width_elems": 2,
    "suture_lat_min_i_frac": 0.5,
    "arch_width_m": 0.038,
    "arch_depth_m": 0.048,
    "vault_depth_m": 0.013,
    "ridge_height_m": 0.017,
    "bone_thickness_m": 0.003,
    "divisions_v2": {"nx": 44, "ny": 12, "nz": 4},
}
if CFG.exists():
    cfg.update(json.loads(CFG.read_text()))

profile = cfg.get("profile", "template_maxilla_v1")


# ── profile: template_maxilla_v1 ──────────────────────────────────────────────
def generate_v1(cfg: dict):
    Lx = float(cfg["dimensions_m"]["Lx"])
    Ly = float(cfg["dimensions_m"]["Ly"])
    Lz = float(cfg["dimensions_m"]["Lz"])
    nx = int(cfg["divisions"]["nx"])
    ny = int(cfg["divisions"]["ny"])
    nz = int(cfg["divisions"]["nz"])
    curv_amp = float(cfg.get("curvature_amplitude_m", 0.004))
    suture_mid_thick = int(cfg.get("suture_mid_thickness_elems", 1))
    suture_lat_width = int(cfg.get("suture_lat_width_elems", 2))
    suture_lat_min_i = int(cfg.get("suture_lat_min_i_frac", 0.5) * nx)

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
                dome = math.sin(math.pi * u) * math.sin(math.pi * v)
                z = z0 + curv_amp * dome * (z_frac ** 2)

                width_scale = 1.0 - 0.18 * u
                y_tapered = (Ly / 2.0) + y_mid * width_scale
                ridge = 0.0016 * (z_frac ** 1.8) * (u ** 1.4) * (abs(y_mid) / (Ly / 2.0))
                groove = -0.0011 * (z_frac ** 2.0) * math.exp(-((y_mid / (0.22 * Ly)) ** 2)) * (0.5 + 0.5 * u)
                posterior_bias = -0.0009 * (1.0 - u) * (z_frac ** 1.2)

                nodes[node_id] = [x, y_tapered, z + ridge + groove + posterior_bias]
                node_id += 1

    nnx, nny = nx + 1, ny + 1

    def nid(i, j, k):
        return k * (nny * nnx) + j * nnx + i + 1

    def eid_fn(i, j, k):
        return k * (ny * nx) + j * nx + i + 1

    elements: dict[int, list[int]] = {}
    eid_counter = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elements[eid_counter] = [
                    nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1),
                ]
                eid_counter += 1

    # node sets
    fixed_nodes = [nid(0, j, k) for k in range(nz + 1) for j in range(ny + 1)]
    palate_nodes = [nid(i, j, nz) for j in range(ny + 1) for i in range(nx + 1)]
    muscle_left = [
        nid(i, j, nz)
        for j in range(ny + 1) for i in range(nx + 1)
        if i >= int(0.65 * nx) and j <= int(0.2 * ny)
    ]
    muscle_right = [
        nid(i, j, nz)
        for j in range(ny + 1) for i in range(nx + 1)
        if i >= int(0.65 * nx) and j >= int(0.8 * ny)
    ]

    # element sets (v1 topology: midpalatal suture at j=ny//2, lateral at high-i edges)
    mid_center = ny // 2
    mid_lo = max(0, mid_center - suture_mid_thick // 2)
    mid_hi = min(ny - 1, mid_lo + suture_mid_thick - 1)
    suture_mid_set = {
        eid_fn(i, j, k)
        for k in range(nz) for j in range(mid_lo, mid_hi + 1) for i in range(nx)
    }
    suture_lat_set = {
        eid_fn(i, j, k)
        for k in range(nz) for j in range(ny) for i in range(suture_lat_min_i, nx)
        if (j < suture_lat_width or j >= ny - suture_lat_width)
    } - suture_mid_set
    suture_all = suture_mid_set | suture_lat_set
    bone_elems = sorted(e for e in elements if e not in suture_all)

    sets = {
        "NSET_FIXED": fixed_nodes,
        "NSET_PALATE": palate_nodes,
        "NSET_MUSCLE_LEFT": muscle_left,
        "NSET_MUSCLE_RIGHT": muscle_right,
        "NSET_MUSCLE_ALL": sorted(set(muscle_left + muscle_right)),
        "ESET_BONE": bone_elems,
        "ESET_SUTURE_MID": sorted(suture_mid_set),
        "ESET_SUTURE_LAT": sorted(suture_lat_set),
    }
    dims = {"Lx": Lx, "Ly": Ly, "Lz": Lz}
    return nodes, elements, sets, nx, ny, nz, dims


# ── profile: template_maxilla_v2 ──────────────────────────────────────────────
def generate_v2(cfg: dict):
    """
    Anatomical arch-sweep mesh.

    Arch centreline (alveolar crest, j=0, k=0):
        x_arch(s) = arch_depth * sin(π·s)
        y_arch(s) = (arch_width/2) · cos(π·s)
    s = i/nx ∈ [0,1]:
        s=0 → posterior-left  (x=0,            y=+arch_width/2)
        s=0.5 → anterior      (x=arch_depth,   y=0            )
        s=1 → posterior-right (x=0,            y=-arch_width/2)

    Inward normal at each arch position N = (Ty, -Tx)/|T| points from
    the alveolar crest toward the palate midline.  An 80 % inward reach
    avoids degenerate pinching at the two posterior arm terminations.

    Oral-surface z-profile across the palate (t = j/ny):
        z_oral(t) = ridge_height·(1-t)² + vault_depth·t²  +  small_sine_bump
      → t=0 (alveolus): ridge_height (tall ridge)
      → t=0.5 (mid-palate): dips to palatal shelf
      → t=1 (midline):  vault_depth  (vault dome)
    """
    arch_width  = float(cfg.get("arch_width_m",      0.038))
    arch_depth  = float(cfg.get("arch_depth_m",      0.048))
    vault_depth = float(cfg.get("vault_depth_m",     0.013))
    ridge_h     = float(cfg.get("ridge_height_m",    0.017))
    bone_thick  = float(cfg.get("bone_thickness_m",  0.003))
    suture_mid_thick = int(cfg.get("suture_mid_thickness_elems", 1))
    suture_lat_width = int(cfg.get("suture_lat_width_elems",     2))

    # Anatomical refinement parameters
    arch_flatness   = float(cfg.get("arch_flatness",   0.25))   # flattens anterior curve
    post_flare_deg  = float(cfg.get("post_flare_deg",  4.0))    # posterior arm outward angle
    vault_ant_scale = float(cfg.get("vault_ant_scale", 1.35))   # anterior vault is deeper
    thick_lat_scale = float(cfg.get("thick_lat_scale", 1.4))    # lateral bone thicker

    divs = cfg.get("divisions_v2", cfg.get("divisions", {}))
    nx = int(divs.get("nx", 44))
    ny = int(divs.get("ny", 12))
    nz = int(divs.get("nz",  4))

    nodes: dict[int, list[float]] = {}
    node_id = 1

    for k in range(nz + 1):
        depth_frac = k / nz          # 0=oral, 1=nasal
        for j in range(ny + 1):
            t = j / ny               # 0=alveolar crest, 1=midline
            for i in range(nx + 1):
                s = i / nx           # arch position

                # Anatomical arch centreline
                # Base elliptical arch + flatness correction at anterior
                # arch_flatness > 0 makes the anterior curve broader/flatter
                # by mixing in a higher-power sine (sin^(1+f) stays near 1
                # at the apex but rises more steeply from the posterior arms)
                sin_ps = math.sin(math.pi * s)
                cos_ps = math.cos(math.pi * s)
                flat_sin = sin_ps * (1.0 - arch_flatness
                                     + arch_flatness * sin_ps)
                x_arch = arch_depth * flat_sin

                # Posterior arm flare: tuberosities angle slightly outward
                # Smooth sign avoids discontinuity at s=0.5 (apex)
                flare_rad = math.radians(post_flare_deg)
                post_factor = (1.0 - sin_ps) ** 2  # quadratic: strong at arms, zero at apex
                smooth_sign = cos_ps / (abs(cos_ps) + 0.15)
                y_flare = post_factor * math.tan(flare_rad) * x_arch * smooth_sign
                y_arch = (arch_width / 2.0) * cos_ps + y_flare

                # Tangent vector (centered finite difference)
                ds = 1e-5
                s_lo = max(0.0, s - ds / 2.0)
                s_hi = min(1.0, s + ds / 2.0)
                actual_ds = s_hi - s_lo
                def _arch_pos(sv):
                    _sin = math.sin(math.pi * sv)
                    _cos = math.cos(math.pi * sv)
                    _flat = _sin * (1.0 - arch_flatness + arch_flatness * _sin)
                    _x = arch_depth * _flat
                    _pf = (1.0 - _sin) ** 2
                    _ss = _cos / (abs(_cos) + 0.15)
                    _yf = _pf * math.tan(flare_rad) * _x * _ss
                    _y = (arch_width / 2.0) * _cos + _yf
                    return _x, _y
                x_lo, y_lo = _arch_pos(s_lo)
                x_hi, y_hi = _arch_pos(s_hi)
                Tx = (x_hi - x_lo) / actual_ds
                Ty = (y_hi - y_lo) / actual_ds
                Tmag = math.sqrt(Tx * Tx + Ty * Ty) + 1e-12
                Nx_in = Ty / Tmag
                Ny_in = -Tx / Tmag

                # inward reach: taper aggressively at the anterior apex
                # (s≈0.5) where arch curvature is highest and adjacent
                # i-positions converge, causing near-degenerate elements.
                inner_reach = 0.80 * (1.0 - 0.62 * sin_ps)
                half_width  = (arch_width / 2.0) * inner_reach

                # At s→0 and s→1 the arch collapses, so add a minimum
                # cross-section width to prevent degenerate elements at
                # the posterior arm terminations.
                min_hw = 0.002  # 2mm minimum mediolateral half-width
                half_width = max(half_width, min_hw)

                # 2-D position
                x = x_arch + t * half_width * Nx_in
                y = y_arch + t * half_width * Ny_in

                # Variable vault depth: deeper anteriorly, shallower posteriorly
                vault_local = vault_depth * (1.0 + (vault_ant_scale - 1.0) * sin_ps)

                # oral-surface z profile with anatomical vault variation
                z_oral = (ridge_h * (1.0 - t) ** 2
                          + vault_local * t ** 2
                          + 0.12 * (ridge_h + vault_local) * 0.5
                            * math.sin(math.pi * t))

                # subtle posterior lowering (anatomical)
                posterior_bias = -0.0006 * (1.0 - sin_ps)

                # Variable bone thickness: thicker laterally, thinner at
                # midpalatal suture (anatomically, palatal bone thins medially)
                thick_factor = 1.0 + (thick_lat_scale - 1.0) * (1.0 - t)
                local_thick = bone_thick * thick_factor

                z = z_oral + depth_frac * local_thick + posterior_bias

                nodes[node_id] = [x, y, z]
                node_id += 1

    nnx, nny = nx + 1, ny + 1

    def nid(i, j, k):
        return k * (nny * nnx) + j * nnx + i + 1

    def eid_fn(i, j, k):
        return k * (ny * nx) + j * nx + i + 1

    # Element connectivity: reverse j-winding so that the local coordinate
    # system (xi along arch, eta crest-to-midline, zeta oral-to-nasal) is
    # right-handed.  At s=0 the inward normal points in −y, so mapping
    # eta from j+1→j (midline→crest) gives eta ≈ +y → det(J) > 0.
    elements: dict[int, list[int]] = {}
    eid_counter = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elements[eid_counter] = [
                    nid(i, j + 1, k), nid(i + 1, j + 1, k),
                    nid(i + 1, j, k), nid(i, j, k),
                    nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1),
                    nid(i + 1, j, k + 1), nid(i, j, k + 1),
                ]
                eid_counter += 1

    # node sets
    # NSET_FIXED: both posterior arm faces (i=0 AND i=nx)
    fixed_nodes = (
        [nid(0,  j, k) for k in range(nz + 1) for j in range(ny + 1)]
        + [nid(nx, j, k) for k in range(nz + 1) for j in range(ny + 1)]
    )
    # NSET_PALATE: oral surface (k=0) — tongue load
    palate_nodes = [nid(i, j, 0) for j in range(ny + 1) for i in range(nx + 1)]

    # NSET_MUSCLE_*: posterior-lateral, nasal face (masseter/pterygoid proxy)
    muscle_left = [
        nid(i, j, nz)
        for i in range(nx + 1) for j in range(ny + 1)
        if i <= int(0.20 * nx) and j <= int(0.30 * ny)
    ]
    muscle_right = [
        nid(i, j, nz)
        for i in range(nx + 1) for j in range(ny + 1)
        if i >= int(0.80 * nx) and j <= int(0.30 * ny)
    ]

    # element sets
    # ESET_SUTURE_MID: medial-most j layer (midpalatal suture)
    mid_j_lo = max(0, ny - suture_mid_thick)
    suture_mid_set = {
        eid_fn(i, j, k)
        for k in range(nz) for j in range(mid_j_lo, ny) for i in range(nx)
    }

    # ESET_SUTURE_LAT: anterior arch zone, lateral j (premaxillary/incisive region)
    ant_lo = max(0, nx // 2 - int(0.15 * nx))
    ant_hi = min(nx - 1, nx // 2 + int(0.15 * nx))
    suture_lat_set = {
        eid_fn(i, j, k)
        for k in range(nz) for j in range(suture_lat_width)
        for i in range(ant_lo, ant_hi + 1)
    } - suture_mid_set

    suture_all = suture_mid_set | suture_lat_set
    bone_elems = sorted(e for e in elements if e not in suture_all)

    sets = {
        "NSET_FIXED": fixed_nodes,
        "NSET_PALATE": palate_nodes,
        "NSET_MUSCLE_LEFT": muscle_left,
        "NSET_MUSCLE_RIGHT": muscle_right,
        "NSET_MUSCLE_ALL": sorted(set(muscle_left + muscle_right)),
        "ESET_BONE": bone_elems,
        "ESET_SUTURE_MID": sorted(suture_mid_set),
        "ESET_SUTURE_LAT": sorted(suture_lat_set),
    }
    dims = {"Lx": arch_depth, "Ly": arch_width, "Lz": bone_thick,
            "arch_flatness": arch_flatness, "post_flare_deg": post_flare_deg}
    return nodes, elements, sets, nx, ny, nz, dims


# ── dispatch ──────────────────────────────────────────────────────────────────
if profile == "template_maxilla_v2":
    nodes, elements, sets, nx, ny, nz, dims = generate_v2(cfg)
else:
    nodes, elements, sets, nx, ny, nz, dims = generate_v1(cfg)


# ── write output ──────────────────────────────────────────────────────────────
mesh = {
    "units": "m",
    "profile": profile,
    "element_type": "C3D8",
    "dimensions": dims,
    "divisions": {"nx": nx, "ny": ny, "nz": nz},
    "nodes": nodes,
    "elements": elements,
    "sets": sets,
}

OUT.write_text(json.dumps(mesh))
print(f"Wrote {OUT}")
print(f"Profile: {profile}")
print(f"Nodes: {len(nodes)}, Elements: {len(elements)}")
bone = sets["ESET_BONE"]
smid = sets["ESET_SUTURE_MID"]
slat = sets["ESET_SUTURE_LAT"]
print(f"ESET_BONE: {len(bone)}, ESET_SUTURE_MID: {len(smid)}, ESET_SUTURE_LAT: {len(slat)}")
assert len(bone) + len(smid) + len(slat) == len(elements), "Element sets not exhaustive!"
print("Element set partition: OK")
