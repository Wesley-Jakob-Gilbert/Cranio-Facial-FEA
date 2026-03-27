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
SUTURE_MOD_PATH = ROOT / "solver" / "suture_modulus.json"
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
    # New layered materials (with fallback)
    E_cortical = float(mats["cortical"]["youngs_modulus_pa"]) if "cortical" in mats else E_bone
    nu_cortical = float(mats["cortical"]["poisson_ratio"]) if "cortical" in mats else nu_bone
    E_cancellous = float(mats["cancellous"]["youngs_modulus_pa"]) if "cancellous" in mats else E_bone
    nu_cancellous = float(mats["cancellous"]["poisson_ratio"]) if "cancellous" in mats else nu_bone
    E_tooth_root = float(mats["tooth_root"]["youngs_modulus_pa"]) if "tooth_root" in mats else 2.0e10
    nu_tooth_root = float(mats["tooth_root"]["poisson_ratio"]) if "tooth_root" in mats else 0.31
else:
    leg = mat_cfg["material"]
    E_bone = float(leg["youngs_modulus_pa"])
    nu_bone = float(leg["poisson_ratio"])
    E_suture = E_bone * 0.05
    nu_suture = nu_bone
    E_cortical = E_bone
    nu_cortical = nu_bone
    E_cancellous = E_bone
    nu_cancellous = nu_bone
    E_tooth_root = 2.0e10
    nu_tooth_root = 0.31

# Physical mass densities for gravity loading (kg/m³)
RHO_CORTICAL = 1900.0
RHO_CANCELLOUS = 800.0
RHO_BONE = 1800.0       # legacy single-material
RHO_SUTURE = 1200.0
RHO_TOOTH = 2100.0       # dentin

# Remodeling parameters (defaults; overridden by remodeling.yaml when present)
n_power = 2      # power-law exponent E(rho) = E_bone * rho^n
n_power_cortical = 3.0   # cortical bone: Carter-Hayes n~3
n_power_cancellous = 2.0 # cancellous bone: n~2
n_bins = 10      # number of density bins for per-element material grouping
E_cortical_remodel = E_cortical    # base modulus for density-binned cortical
E_cancellous_remodel = E_cancellous  # base modulus for density-binned cancellous
if REMODEL_PATH.exists():
    rm = yaml.safe_load(REMODEL_PATH.read_text()).get("remodeling", {})
    n_power = float(rm.get("n_power", n_power))
    n_power_cortical = float(rm.get("n_power_cortical", n_power))
    n_power_cancellous = float(rm.get("n_power_cancellous", n_power))
    n_bins = int(rm.get("n_bins", n_bins))
    E_cortical_remodel = float(rm.get("E_cortical_pa", E_cortical))
    E_cancellous_remodel = float(rm.get("E_cancellous_pa", E_cancellous))

# Per-element density override written by bone_remodeling.py each remodeling cycle.
# Keys are string element IDs; values are relative density (rho, dimensionless).
density_override: dict[int, float] = {}
if DENSITY_PATH.exists():
    raw = json.loads(DENSITY_PATH.read_text())
    density_override = {int(k): float(v) for k, v in raw.items()}

# Per-element suture modulus override written by bone_remodeling.py each cycle.
suture_mod: dict[int, float] = {}
if SUTURE_MOD_PATH.exists():
    raw_sm = json.loads(SUTURE_MOD_PATH.read_text())
    suture_mod = {int(k): float(v) for k, v in raw_sm.items()}

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
eset_cortical = sets.get("ESET_CORTICAL", [])
eset_cancellous = sets.get("ESET_CANCELLOUS", [])
eset_tooth_root = sets.get("ESET_TOOTH_ROOT", [])
# Detect whether the mesh has the fine-grained layer split
_has_layers = bool(eset_cortical) or bool(eset_cancellous) or bool(eset_tooth_root)
all_nodes = sorted(nodes.keys())

# ---------------------------------------------------------------------------
# Oral-surface geometry: face areas and per-node surface normals
# ---------------------------------------------------------------------------
# For the v2 U-shaped arch, the palatal surface is curved, so tongue pressure
# should act normal to the surface at each node rather than uniformly in -Z.
# We find k=0 face quads (oral surface) by checking which element faces have
# all four nodes in NSET_PALATE, then compute area-weighted normals per node.

def _quad_cross(p):
    """Return cross product of quad diagonals (p[0]->p[2] x p[1]->p[3]).

    The magnitude of the result equals twice the quad area. The direction
    is the face outward normal (sign depends on winding).
    """
    d1 = [p[2][c] - p[0][c] for c in range(3)]
    d2 = [p[3][c] - p[1][c] for c in range(3)]
    return [
        d1[1] * d2[2] - d1[2] * d2[1],
        d1[2] * d2[0] - d1[0] * d2[2],
        d1[0] * d2[1] - d1[1] * d2[0],
    ]


def _oral_face(conn, palate_set):
    """Return the oral-surface face of a C3D8 element, or None.

    For v2 (palate at k=0), the oral face is conn[:4] (k-low).
    For v1 (palate at k=nz), the oral face is conn[4:8] (k-high).
    Returns the face whose 4 nodes are all in palate_set, or None if
    neither face qualifies (element is not on the oral surface).
    """
    lo = conn[:4]
    if all(n in palate_set for n in lo):
        return lo
    hi = conn[4:8]
    if all(n in palate_set for n in hi):
        return hi
    return None


def _oral_surface_area(nodes, elems, eset_all, palate_set):
    """Sum quad face areas on the oral surface only.

    Only element faces whose four nodes are all in *palate_set* are
    counted. This works for both v1 (palate at k=nz, face is conn[4:8])
    and v2 (palate at k=0, face is conn[:4]).
    """
    total = 0.0
    for eid in eset_all:
        face = _oral_face(elems[eid], palate_set)
        if face is None:
            continue
        p = [nodes[n] for n in face]
        cx = _quad_cross(p)
        total += 0.5 * math.sqrt(sum(c * c for c in cx))
    return total


def _palate_node_normals(nodes, elems, eset_all, palate_set):
    """Compute area-weighted unit surface normals for each palate node.

    For every k=0 face quad (element face whose 4 nodes all lie in
    palate_set), the face normal is obtained via cross product of the
    quad diagonals. The raw cross product is proportional to face area,
    so accumulating it at each node gives an area-weighted average.

    After accumulation each node's normal is normalised to unit length.

    Convention: the raw cross product from _quad_cross points in the
    direction determined by the element winding. For v2, the reversed-j
    winding makes the k-low face normal point downward (toward -z /
    toward the tongue, i.e. outward from the bone). For v1, the standard
    winding gives a normal pointing inward (+z into the bone on the k=nz
    face). We enforce a consistent convention: the returned normal points
    INTO the bone (away from the oral cavity). If the raw cross product's
    z-component is positive (pointing up / into bone), we keep it;
    otherwise we flip it. This works for both v1 and v2 because the
    nasal surface is always above (higher z) the oral surface.

    Returns dict mapping node_id -> [nx, ny, nz] unit normal (into bone).
    """
    # Accumulate raw (area-weighted) normals per node
    accum: dict[int, list[float]] = {}
    for eid in eset_all:
        face = _oral_face(elems[eid], palate_set)
        if face is None:
            continue
        p = [nodes[n] for n in face]
        cx = _quad_cross(p)

        # Ensure normal points into bone (positive z direction on average).
        # The palatal vault means z isn't purely vertical, but across the
        # whole surface the "into bone" direction has a positive z component.
        if cx[2] < 0:
            cx = [-cx[0], -cx[1], -cx[2]]

        for nid in face:
            if nid not in accum:
                accum[nid] = [0.0, 0.0, 0.0]
            accum[nid][0] += cx[0]
            accum[nid][1] += cx[1]
            accum[nid][2] += cx[2]

    # Normalise to unit vectors
    normals: dict[int, list[float]] = {}
    for nid, raw in accum.items():
        mag = math.sqrt(raw[0] ** 2 + raw[1] ** 2 + raw[2] ** 2)
        if mag < 1e-30:
            # Degenerate — fall back to pure +z (into bone)
            normals[nid] = [0.0, 0.0, 1.0]
        else:
            normals[nid] = [raw[0] / mag, raw[1] / mag, raw[2] / mag]
    return normals


_palate_set = set(palate)
_all_elems = sorted(set(eset_bone) | set(eset_suture_mid) | set(eset_suture_lat))
_oral_area = _oral_surface_area(nodes, elems, _all_elems, _palate_set)
area_per_node = _oral_area / max(1, len(palate))
_node_normals = _palate_node_normals(nodes, elems, _all_elems, _palate_set)

# ---------------------------------------------------------------------------
# Nonuniform pressure distribution (anterior-posterior gradient)
# ---------------------------------------------------------------------------
_loads_cfg = yaml.safe_load(LOADS_PATH.read_text())
_pdist = _loads_cfg.get("pressure_distribution", {})
_ant_post_ratio = float(_pdist.get("ant_post_ratio", 1.0))


def _pressure_weights(palate_nids, nds, ant_post_ratio):
    """Per-node pressure weight based on x-position along the arch.

    Returns dict nid -> weight. Mean weight = 1.0 (total force preserved).
    With ant_post_ratio = 1.0, all weights = 1.0.
    """
    if ant_post_ratio == 1.0 or len(palate_nids) == 0:
        return {nid: 1.0 for nid in palate_nids}

    xs = {nid: nds[nid][0] for nid in palate_nids}
    x_min, x_max = min(xs.values()), max(xs.values())
    x_range = x_max - x_min
    if x_range < 1e-12:
        return {nid: 1.0 for nid in palate_nids}

    # Linear: w_post at x_min, w_ant at x_max, w_ant/w_post = R, mean = 1.0
    R = ant_post_ratio
    w_post = 2.0 / (1.0 + R)
    w_ant = 2.0 * R / (1.0 + R)

    weights = {}
    for nid in palate_nids:
        t = (xs[nid] - x_min) / x_range
        weights[nid] = w_post + (w_ant - w_post) * t
    return weights


_pressure_wt = _pressure_weights(palate, nodes, _ant_post_ratio)


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


def _bin_elements(elem_list, density_map, rho_lo, rho_hi, bin_width, prefix):
    """Bin elements by density, return dict[bin_idx] -> list[eid]."""
    bins: dict[int, list[int]] = {i: [] for i in range(n_bins)}
    for e in elem_list:
        rho = density_map.get(e, 1.0)
        idx = int((rho - rho_lo) / bin_width)
        idx = max(0, min(n_bins - 1, idx))
        bins[idx].append(e)
    return {i: v for i, v in bins.items() if v}


def _write_elastic_density(fh, E, nu, rho_mass):
    """Write *ELASTIC and *DENSITY cards for a material."""
    fh.write("*ELASTIC\n")
    fh.write(f"{E:.6e}, {nu:.6f}\n")
    fh.write("*DENSITY\n")
    fh.write(f"{rho_mass:.1f}\n")


def _write_suture_blocks(fh):
    """Write suture ELSET, MATERIAL, SOLID SECTION blocks.

    If suture_mod is populated and spans a range, bins suture elements
    by modulus into up to 5 groups. Otherwise writes a single SUTURE material.
    """
    suture_eids = list(eset_suture_mid) + list(eset_suture_lat)
    if not suture_eids:
        return

    suture_E_vals = [float(suture_mod.get(eid, E_suture)) for eid in suture_eids]
    s_E_lo = min(suture_E_vals)
    s_E_hi = max(suture_E_vals)

    if suture_mod and (s_E_hi - s_E_lo) > 1.0:
        # Binned suture materials
        n_suture_bins = 5
        s_bin_width = (s_E_hi - s_E_lo) / n_suture_bins
        s_bins: dict[int, list[int]] = {b: [] for b in range(n_suture_bins)}
        for eid, E_val in zip(suture_eids, suture_E_vals):
            bidx = int((E_val - s_E_lo) / s_bin_width)
            bidx = max(0, min(n_suture_bins - 1, bidx))
            s_bins[bidx].append(eid)
        active = {b: eids for b, eids in s_bins.items() if eids}

        for bidx, bin_eids in sorted(active.items()):
            fh.write(f"*ELSET, ELSET=ESUTURE_BIN_{bidx:02d}\n")
            write_id_list(fh, bin_eids)
        for bidx in sorted(active):
            E_center = s_E_lo + (bidx + 0.5) * s_bin_width
            fh.write(f"*MATERIAL, NAME=SUTURE_BIN_{bidx:02d}\n")
            _write_elastic_density(fh, E_center, nu_suture, RHO_SUTURE)
        for bidx in sorted(active):
            fh.write(f"*SOLID SECTION, ELSET=ESUTURE_BIN_{bidx:02d}, MATERIAL=SUTURE_BIN_{bidx:02d}\n")
            fh.write(",\n")
    else:
        # Single SUTURE material
        if eset_suture_mid:
            fh.write("*ELSET, ELSET=ESUTURE_MID\n")
            write_id_list(fh, eset_suture_mid)
        if eset_suture_lat:
            fh.write("*ELSET, ELSET=ESUTURE_LAT\n")
            write_id_list(fh, eset_suture_lat)
        fh.write("*MATERIAL, NAME=SUTURE\n")
        _write_elastic_density(fh, E_suture, nu_suture, RHO_SUTURE)
        if eset_suture_mid:
            fh.write("*SOLID SECTION, ELSET=ESUTURE_MID, MATERIAL=SUTURE\n")
            fh.write(",\n")
        if eset_suture_lat:
            fh.write("*SOLID SECTION, ELSET=ESUTURE_LAT, MATERIAL=SUTURE\n")
            fh.write(",\n")


def write_material_blocks(fh):
    """Write *ELSET, *MATERIAL, and *SOLID SECTION blocks.

    Supports two mesh configurations:
    - Legacy 2-material: ESET_BONE + SUTURE (when ESET_CORTICAL absent)
    - Layered 5-material: CORTICAL + CANCELLOUS + TOOTH_ROOT + SUTURE_MID + SUTURE_LAT

    And two solver modes:
    - Baseline (no elem_density.json): fixed material properties
    - Density-binned (elem_density.json present): cortical and cancellous elements
      binned by density; tooth root held fixed; suture held fixed
    """
    if density_override and _has_layers:
        # --- Density-binned mode with layered materials ---
        # Compute global rho range across all remodeling-eligible elements
        remodel_elems = list(eset_cortical) + list(eset_cancellous)
        rho_vals = [density_override.get(e, 1.0) for e in remodel_elems]
        rho_lo = min(rho_vals) if rho_vals else 0.4
        rho_hi = max(rho_vals) if rho_vals else 1.5
        span = rho_hi - rho_lo
        bin_width = span / n_bins if span > 0 else 1.0

        # Bin cortical elements
        cort_bins = _bin_elements(eset_cortical, density_override, rho_lo, rho_hi, bin_width, "CORT")
        # Bin cancellous elements
        canc_bins = _bin_elements(eset_cancellous, density_override, rho_lo, rho_hi, bin_width, "CANC")

        # Write element sets
        for idx, bin_elems in sorted(cort_bins.items()):
            fh.write(f"*ELSET, ELSET=ECORT_BIN_{idx:02d}\n")
            write_id_list(fh, bin_elems)
        for idx, bin_elems in sorted(canc_bins.items()):
            fh.write(f"*ELSET, ELSET=ECANC_BIN_{idx:02d}\n")
            write_id_list(fh, bin_elems)
        if eset_tooth_root:
            fh.write("*ELSET, ELSET=ETOOTHROOT\n")
            write_id_list(fh, eset_tooth_root)
        # Write materials: cortical bins (n=3 per Carter-Hayes)
        for idx in sorted(cort_bins):
            rho_center = rho_lo + (idx + 0.5) * bin_width
            E_bin = E_cortical_remodel * (rho_center ** n_power_cortical)
            fh.write(f"*MATERIAL, NAME=CORT_BIN_{idx:02d}\n")
            _write_elastic_density(fh, E_bin, nu_cortical, RHO_CORTICAL)

        # Write materials: cancellous bins (n=2)
        for idx in sorted(canc_bins):
            rho_center = rho_lo + (idx + 0.5) * bin_width
            E_bin = E_cancellous_remodel * (rho_center ** n_power_cancellous)
            fh.write(f"*MATERIAL, NAME=CANC_BIN_{idx:02d}\n")
            _write_elastic_density(fh, E_bin, nu_cancellous, RHO_CANCELLOUS)

        # Tooth root: fixed material (no density update)
        fh.write("*MATERIAL, NAME=TOOTH_ROOT\n")
        _write_elastic_density(fh, E_tooth_root, nu_tooth_root, RHO_TOOTH)

        # Solid sections: cortical + cancellous bins
        for idx in sorted(cort_bins):
            fh.write(f"*SOLID SECTION, ELSET=ECORT_BIN_{idx:02d}, MATERIAL=CORT_BIN_{idx:02d}\n")
            fh.write(",\n")
        for idx in sorted(canc_bins):
            fh.write(f"*SOLID SECTION, ELSET=ECANC_BIN_{idx:02d}, MATERIAL=CANC_BIN_{idx:02d}\n")
            fh.write(",\n")
        if eset_tooth_root:
            fh.write("*SOLID SECTION, ELSET=ETOOTHROOT, MATERIAL=TOOTH_ROOT\n")
            fh.write(",\n")

        # Suture blocks (possibly modulus-binned)
        _write_suture_blocks(fh)

    elif density_override:
        # --- Density-binned mode (legacy 2-material mesh) ---
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

        for idx in sorted(active):
            rho_center = rho_lo + (idx + 0.5) * bin_width
            E_bin = E_bone * (rho_center ** n_power)
            fh.write(f"*MATERIAL, NAME=BONE_BIN_{idx:02d}\n")
            _write_elastic_density(fh, E_bin, nu_bone, RHO_BONE)

        for idx in sorted(active):
            fh.write(f"*SOLID SECTION, ELSET=EBONE_BIN_{idx:02d}, MATERIAL=BONE_BIN_{idx:02d}\n")
            fh.write(",\n")

        _write_suture_blocks(fh)

    elif _has_layers:
        # --- Baseline 5-material mode (layered mesh, no density override) ---
        if eset_cortical:
            fh.write("*ELSET, ELSET=ECORTICAL\n")
            write_id_list(fh, eset_cortical)
        if eset_cancellous:
            fh.write("*ELSET, ELSET=ECANCELLOUS\n")
            write_id_list(fh, eset_cancellous)
        if eset_tooth_root:
            fh.write("*ELSET, ELSET=ETOOTHROOT\n")
            write_id_list(fh, eset_tooth_root)

        fh.write("*MATERIAL, NAME=CORTICAL\n")
        _write_elastic_density(fh, E_cortical, nu_cortical, RHO_CORTICAL)

        fh.write("*MATERIAL, NAME=CANCELLOUS\n")
        _write_elastic_density(fh, E_cancellous, nu_cancellous, RHO_CANCELLOUS)

        fh.write("*MATERIAL, NAME=TOOTH_ROOT\n")
        _write_elastic_density(fh, E_tooth_root, nu_tooth_root, RHO_TOOTH)

        if eset_cortical:
            fh.write("*SOLID SECTION, ELSET=ECORTICAL, MATERIAL=CORTICAL\n")
            fh.write(",\n")
        if eset_cancellous:
            fh.write("*SOLID SECTION, ELSET=ECANCELLOUS, MATERIAL=CANCELLOUS\n")
            fh.write(",\n")
        if eset_tooth_root:
            fh.write("*SOLID SECTION, ELSET=ETOOTHROOT, MATERIAL=TOOTH_ROOT\n")
            fh.write(",\n")

        _write_suture_blocks(fh)

    else:
        # --- Baseline two-material mode (legacy mesh) ---
        if eset_bone:
            fh.write("*ELSET, ELSET=ESET_BONE\n")
            write_id_list(fh, eset_bone)
        fh.write("*MATERIAL, NAME=BONE\n")
        _write_elastic_density(fh, E_bone, nu_bone, RHO_BONE)

        if eset_bone:
            fh.write("*SOLID SECTION, ELSET=ESET_BONE, MATERIAL=BONE\n")
            fh.write(",\n")

        _write_suture_blocks(fh)


# muscle resultant direction (posterior, inward, downward), normalized
vx, vy, vz = -0.40, 0.20, -0.90
norm = math.sqrt(vx * vx + vy * vy + vz * vz)
ux, uy, uz = vx / norm, vy / norm, vz / norm

if density_override and _has_layers:
    mode_label = "density-binned-layered"
elif density_override:
    mode_label = "density-binned"
elif _has_layers:
    mode_label = "baseline-layered"
else:
    mode_label = "baseline"

for case, spec in loads.items():
    tongue_kpa, muscle_force_n = parse_case(spec)
    gravity_enabled = spec.get("gravity", False) if isinstance(spec, dict) else False

    p_pa = tongue_kpa * 1000.0
    # Force magnitude per palate node (positive value; direction comes from normals)
    f_tongue_mag = p_pa * area_per_node

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

        # Tongue pressure as surface-normal loads on palate nodes.
        # Force per node = weight * magnitude * (-normal), where the normal
        # points into bone and weight encodes anterior-posterior gradient.
        if f_tongue_mag != 0.0:
            f.write("*CLOAD\n")
            for nid in palate:
                nn = _node_normals.get(nid, [0.0, 0.0, 1.0])
                w = _pressure_wt.get(nid, 1.0)
                fx = -f_tongue_mag * w * nn[0]
                fy = -f_tongue_mag * w * nn[1]
                fz = -f_tongue_mag * w * nn[2]
                f.write(f"{nid}, 1, {fx:.6e}\n")
                f.write(f"{nid}, 2, {fy:.6e}\n")
                f.write(f"{nid}, 3, {fz:.6e}\n")

        # Gravity body force (rho_bone ~ 1800 kg/m³, g = -9.81 m/s² in Z)
        if gravity_enabled:
            f.write("*DLOAD\n")
            f.write("EALL, GRAV, 9.81, 0.0, 0.0, -1.0\n")
            # Density card for gravity: CalculiX uses *DENSITY on material
            # We add *DENSITY to each material block separately — see below

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
        f.write("S, ENER\n")

        f.write("*NODE PRINT, NSET=NALL\n")
        f.write("U\n")
        f.write("*EL PRINT, ELSET=EALL\n")
        f.write("S\n")
        f.write("*EL PRINT, ELSET=EALL\n")
        f.write("ENER\n")
        f.write("*END STEP\n")

    print(
        f"Wrote {out} [{mode_label}] "
        f"(tongue_kPa={tongue_kpa}, Ftongue_node_mag={f_tongue_mag:.3e} N, "
        f"Fmuscle_total={muscle_force_n:.3e} N)"
    )
