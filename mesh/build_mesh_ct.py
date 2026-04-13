"""Load Open-Full-Jaw CT maxilla mesh and label node/element sets via spatial heuristics.

Reads a Gmsh .msh file (tetrahedral), converts to the mesh_data.json schema
used by the rest of the cranio_FEA pipeline.  All coordinates are converted
from the dataset's mm to metres.

Spatial heuristic labeling:
  NSET_PALATE    — nodes on inferior-facing boundary (oral surface)
  NSET_FIXED     — nodes on posterior cut plane (boundary conditions)
  ESET_CORTICAL  — elements owning >=1 boundary face (surface layer)
  ESET_CANCELLOUS— interior bone elements (no boundary face)
  ESET_SUTURE_MID— elements near the sagittal midplane
  ESET_SUTURE_LAT— elements in anterior-lateral zone
  ESET_BONE      — union of cortical + cancellous (backward compat)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    import meshio
except ImportError:
    sys.exit("meshio is required: pip install meshio")

# 4 triangular faces of a tet, as local index triples
_TET_FACE_LOCAL = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


# ---------------------------------------------------------------------------
# Vectorised boundary extraction
# ---------------------------------------------------------------------------

def _extract_boundary_np(tets: np.ndarray):
    """Vectorised boundary-face extraction for tet meshes.

    Parameters
    ----------
    tets : (M, 4) int array — 0-based node indices

    Returns
    -------
    bfaces : (B, 3) int array — boundary face node triples (sorted per row)
    bface_tet : (B,) int array — owning tet index (0-based) for each boundary face
    """
    n_tets = len(tets)
    # Build all faces: (M*4, 3)
    all_faces = tets[:, _TET_FACE_LOCAL].reshape(-1, 3)
    # Owning tet index for each face row
    tet_ids = np.repeat(np.arange(n_tets), 4)
    # Sort each face's nodes so identical faces have the same triple
    all_faces_sorted = np.sort(all_faces, axis=1)
    # Encode each face as a single int for fast hashing:
    #   face_key = n0 * N^2 + n1 * N + n2   (N > max_node_id)
    N = int(all_faces_sorted.max()) + 1
    keys = (all_faces_sorted[:, 0].astype(np.int64) * N * N +
            all_faces_sorted[:, 1].astype(np.int64) * N +
            all_faces_sorted[:, 2].astype(np.int64))

    # Find unique keys; faces appearing exactly once are boundary faces
    unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    boundary_mask = counts[inverse] == 1

    bfaces = all_faces_sorted[boundary_mask]
    bface_tet = tet_ids[boundary_mask]
    return bfaces, bface_tet


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_ct_mesh(ct_cfg: dict, repo_root: Path | None = None) -> dict:
    """Load .msh, label sets, return mesh_data dict."""
    msh_path = ct_cfg["msh_path"]
    if repo_root is not None:
        msh_path = str(repo_root / msh_path)

    print(f"[ct] Loading {msh_path} ...")
    mesh = meshio.read(msh_path)

    # --- Extract points and tets ------------------------------------------
    pts_mm = mesh.points.astype(np.float64)   # (N, 3) in mm
    if "tetra" not in mesh.cells_dict:
        sys.exit(f"No tetrahedral cells found in {msh_path}")
    tets_0 = mesh.cells_dict["tetra"]  # (M, 4), 0-based

    n_nodes = len(pts_mm)
    n_tets_raw = len(tets_0)
    print(f"[ct] Raw mesh: {n_nodes} nodes, {n_tets_raw} tets")

    # --- Convert mm -> m --------------------------------------------------
    scale = ct_cfg.get("scale_to_m", 0.001)
    pts = pts_mm * scale  # (N, 3) in metres

    # --- Filter degenerate tets (near-zero volume) ------------------------
    # Tet volume = (1/6) * det([p1-p0, p2-p0, p3-p0])
    v0 = pts[tets_0[:, 0]]
    e1 = pts[tets_0[:, 1]] - v0
    e2 = pts[tets_0[:, 2]] - v0
    e3 = pts[tets_0[:, 3]] - v0
    # Scalar triple product (vectorised determinant)
    det = (e1[:, 0] * (e2[:, 1] * e3[:, 2] - e2[:, 2] * e3[:, 1]) +
           e1[:, 1] * (e2[:, 2] * e3[:, 0] - e2[:, 0] * e3[:, 2]) +
           e1[:, 2] * (e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0]))
    vol = np.abs(det) / 6.0
    min_vol = ct_cfg.get("min_tet_volume_m3", 1e-16)
    good = vol > min_vol
    n_degenerate = (~good).sum()
    if n_degenerate > 0:
        print(f"[ct] Filtered {n_degenerate} degenerate tets (vol < {min_vol:.0e} m³)")
        tets_0 = tets_0[good]

    n_tets = len(tets_0)
    print(f"[ct] Working mesh: {n_nodes} nodes, {n_tets} tets")

    # --- Boundary faces (vectorised) --------------------------------------
    print("[ct] Extracting boundary faces ...")
    bfaces, bface_tet = _extract_boundary_np(tets_0)
    print(f"[ct] Boundary faces: {len(bfaces)}")

    # --- Boundary face normals and centroids (vectorised) -----------------
    p0 = pts[bfaces[:, 0]]
    p1 = pts[bfaces[:, 1]]
    p2 = pts[bfaces[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)  # (B, 3)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    normals /= norms
    centroids_bf = (p0 + p1 + p2) / 3.0  # (B, 3)

    # --- NSET_PALATE: palatal vault (downward-facing boundary, mid-Z band) -
    # The palatal vault sits ABOVE the tooth tips (low Z) but BELOW the nasal
    # floor (high Z).  We select downward-facing boundary faces whose centroid
    # Z falls within a percentile band of *downward-facing* face centroids.
    # This excludes tooth occlusal surfaces (low Z) and nasal-floor edges (high Z).
    nz_thresh = ct_cfg.get("palate_normal_z_threshold", -0.3)
    down_mask = normals[:, 2] < nz_thresh
    down_z = centroids_bf[down_mask, 2]

    z_lo_pct = ct_cfg.get("palate_z_lo_percentile", 35)
    z_hi_pct = ct_cfg.get("palate_z_hi_percentile", 85)
    z_lo = np.percentile(down_z, z_lo_pct)
    z_hi = np.percentile(down_z, z_hi_pct)

    palate_mask = down_mask & (centroids_bf[:, 2] >= z_lo) & (centroids_bf[:, 2] <= z_hi)
    palate_nodes_0 = np.unique(bfaces[palate_mask])  # 0-based
    palate_nodes_1 = (palate_nodes_0 + 1).tolist()
    print(f"[ct] NSET_PALATE: {len(palate_nodes_1)} nodes ({palate_mask.sum()} faces)")
    print(f"[ct]   Z band: [{z_lo:.6f}, {z_hi:.6f}] m  "
          f"(p{z_lo_pct}-p{z_hi_pct} of downward-facing faces)")
    print(f"[ct]   Node Z range: [{pts[palate_nodes_0, 2].min():.6f}, "
          f"{pts[palate_nodes_0, 2].max():.6f}] m")

    # --- NSET_FIXED: nodes at extreme boundaries --------------------------
    fix_tol = ct_cfg.get("fixed_posterior_tolerance_m", 0.002)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    fixed_mask = ((pts[:, 0] < x_min + fix_tol) |
                  (pts[:, 0] > x_max - fix_tol) |
                  (pts[:, 1] < y_min + fix_tol) |
                  (pts[:, 1] > y_max - fix_tol))
    fixed_nodes_1 = (np.where(fixed_mask)[0] + 1).tolist()
    print(f"[ct] NSET_FIXED: {len(fixed_nodes_1)} nodes")

    # --- Element centroids (vectorised) -----------------------------------
    centroids_el = pts[tets_0].mean(axis=1)  # (M, 3)

    # --- Suture labeling --------------------------------------------------
    mid_half_w = ct_cfg.get("suture_mid_half_width_m", 0.001)
    lat_y_thresh = ct_cfg.get("suture_lat_y_threshold_m", 0.015)
    lat_x_frac = ct_cfg.get("suture_lat_x_threshold_frac", 0.6)

    cx = centroids_el[:, 0]
    cy = centroids_el[:, 1]
    x_med = np.median(cx)
    y_med = np.median(cy)
    cx_min, cx_max = cx.min(), cx.max()
    cx_range = max(cx_max - cx_min, 1e-12)
    x_frac_arr = (cx - cx_min) / cx_range

    suture_mid_mask = np.abs(cx - x_med) < mid_half_w
    suture_lat_mask = (x_frac_arr > lat_x_frac) & (np.abs(cy - y_med) > lat_y_thresh)

    # 1-based element IDs
    suture_mid_1 = (np.where(suture_mid_mask)[0] + 1).tolist()
    suture_lat_1 = (np.where(suture_lat_mask)[0] + 1).tolist()
    suture_all_mask = suture_mid_mask | suture_lat_mask
    print(f"[ct] ESET_SUTURE_MID: {len(suture_mid_1)} elems")
    print(f"[ct] ESET_SUTURE_LAT: {len(suture_lat_1)} elems")

    # --- Cortical / Cancellous split --------------------------------------
    # Elements owning >= 1 boundary face = cortical (surface layer)
    boundary_tet_set = set(bface_tet.tolist())   # 0-based tet indices
    is_cortical = np.zeros(n_tets, dtype=bool)
    for ti in boundary_tet_set:
        is_cortical[ti] = True
    is_bone = ~suture_all_mask
    cortical_mask = is_bone & is_cortical
    cancellous_mask = is_bone & ~is_cortical

    cortical_1 = (np.where(cortical_mask)[0] + 1).tolist()
    cancellous_1 = (np.where(cancellous_mask)[0] + 1).tolist()
    bone_1 = (np.where(is_bone)[0] + 1).tolist()
    print(f"[ct] ESET_CORTICAL: {len(cortical_1)} elems")
    print(f"[ct] ESET_CANCELLOUS: {len(cancellous_1)} elems")

    # --- Assemble mesh_data -----------------------------------------------
    nodes_dict = {str(i + 1): pts[i].tolist() for i in range(n_nodes)}
    elems_dict = {str(i + 1): (tets_0[i] + 1).tolist() for i in range(n_tets)}

    mesh_data = {
        "elem_type": "C3D4",
        "nodes": nodes_dict,
        "elements": elems_dict,
        "sets": {
            "NSET_FIXED": fixed_nodes_1,
            "NSET_PALATE": palate_nodes_1,
            "NSET_MUSCLE_LEFT": [],
            "NSET_MUSCLE_RIGHT": [],
            "NSET_MUSCLE_ALL": [],
            "ESET_BONE": bone_1,
            "ESET_CORTICAL": cortical_1,
            "ESET_CANCELLOUS": cancellous_1,
            "ESET_SUTURE_MID": suture_mid_1,
            "ESET_SUTURE_LAT": suture_lat_1,
        },
    }

    total = len(cortical_1) + len(cancellous_1) + len(suture_mid_1) + len(suture_lat_1)
    print(f"[ct] Total labeled: {total} / {n_tets} "
          f"(cortical={len(cortical_1)}, cancellous={len(cancellous_1)}, "
          f"suture_mid={len(suture_mid_1)}, suture_lat={len(suture_lat_1)})")

    return mesh_data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    repo = Path(__file__).resolve().parent.parent
    geo_path = repo / "configs" / "geometry.json"
    with open(geo_path) as f:
        geo = json.load(f)

    if "ct_mesh" not in geo:
        sys.exit("No ct_mesh block in geometry.json")

    mesh_data = build_ct_mesh(geo["ct_mesh"], repo_root=repo)

    out_path = repo / "mesh" / "mesh_data.json"
    with open(out_path, "w") as f:
        json.dump(mesh_data, f)
    print(f"[ct] Wrote {out_path} "
          f"({len(mesh_data['nodes'])} nodes, {len(mesh_data['elements'])} elems)")
