"""Import of the Open-Full-Jaw volumetric maxilla mesh.

Open-Full-Jaw (Gholamalizadeh et al. 2022, *Comput. Methods Programs Biomed.*
224:107009) provides, per patient, one conforming linear-tetrahedral gmsh
mesh of bone, teeth and periodontal ligament (PDL) with shared interface
nodes. The volumes are distinguished by the gmsh *geometrical* tag; for the
maxilla of patient 3 these are (verified from the volumes and shapes):

==== ====== ============ ======================================================
tag  tissue  volume      evidence
==== ====== ============ ======================================================
1    teeth   9103 mm^3   14 closed bodies reaching below the bone (crowns)
2    PDL      511 mm^3   shell of 0.08-0.39 mm thickness (pipeline log: mean
                         0.20 mm) between tag 1 and tag 3
3    bone   15877 mm^3   one connected body, cut flat at its superior extent
==== ====== ============ ======================================================

For the bone-only model the PDL and teeth are removed; the bone faces that
were shared with the PDL are kept as the surface set ``socket`` (the
alveolar socket walls / lamina dura through which occlusal forces reach the
bone). Coordinates are DICOM patient (LPS) millimetres: +x patient left,
+y posterior, +z superior.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from craniofea.mesh.primitives import (
    TET_FACES,
    boundary_faces,
    face_keys,
    orient_tets,
    tet_signed_volumes,
)
from craniofea.model import Mesh

log = logging.getLogger(__name__)

#: gmsh geometrical tags of the Open-Full-Jaw maxilla meshes.
OPEN_FULL_JAW_TAGS = {"teeth": 1, "pdl": 2, "bone": 3}


def read_gmsh_tets(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a gmsh tetrahedral mesh.

    Returns (points (N,3) float64 in file units, tets (M,4) int64,
    geometrical tag (M,) int32; zeros if the file carries none).
    """
    import meshio

    m = meshio.read(str(path))
    blocks = [i for i, c in enumerate(m.cells) if c.type == "tetra"]
    if not blocks:
        raise ValueError(f"no tetrahedra in {path}")
    tets = np.concatenate([m.cells[i].data for i in blocks]).astype(np.int64)
    geo = m.cell_data.get("gmsh:geometrical")
    tags = (np.concatenate([geo[i] for i in blocks]).astype(np.int32) if geo is not None
            else np.zeros(len(tets), dtype=np.int32))
    return np.asarray(m.points, dtype=np.float64), tets, tags


def normalized_volume(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Signed volume divided by that of the regular tet with the same RMS edge length.

    Equals 1 for a regular tetrahedron and -> 0 for degenerate (flat) ones;
    scale invariant, so a single threshold works in mm or m.
    """
    x = nodes[tets]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    l2 = sum(np.einsum("ij,ij->i", x[:, i] - x[:, j], x[:, i] - x[:, j]) for i, j in edges) / 6
    return tet_signed_volumes(nodes, tets) / np.maximum(l2 ** 1.5 / (6 * np.sqrt(2)), 1e-300)


def compact(nodes: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop unreferenced nodes. Returns (nodes, tets, old ids of the kept nodes)."""
    used = np.unique(tets)
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return nodes[used], remap[tets], used


def interface_faces(tets_a: np.ndarray, tets_b: np.ndarray) -> np.ndarray:
    """Boolean mask over the 4*len(tets_a) local faces of A that are also faces of B."""
    ka = face_keys(tets_a[:, TET_FACES].reshape(-1, 3))
    kb = np.unique(face_keys(tets_b[:, TET_FACES].reshape(-1, 3)))
    return np.isin(ka, kb)


def load_open_full_jaw(path: str | Path, scale_to_m: float = 1e-3,
                       keep: tuple[str, ...] = ("bone",),
                       tags: dict[str, int] | None = None,
                       min_normalized_volume: float = 1e-6) -> Mesh:
    """Load an Open-Full-Jaw volumetric mesh as a `Mesh` in metres.

    Parameters
    ----------
    path : gmsh .msh file (e.g. ``.../maxilla/volumetric_meshes/maxilla_volumetric_mesh.msh``)
    scale_to_m : file units -> metres (the dataset is in millimetres)
    keep : tissues to retain (keys of `tags`); the default bone-only model
        exposes the tooth sockets as free surfaces
    tags : tissue -> gmsh geometrical tag (default `OPEN_FULL_JAW_TAGS`)
    min_normalized_volume : tets below this `normalized_volume` are removed
        as degenerate

    The returned mesh has positively oriented tets, no unused nodes, surface
    sets ``all`` and (when the PDL is not kept) ``socket``, and, when more
    than one tissue is kept, the elem field ``tissue_tag`` (gmsh tag).
    """
    tags = dict(OPEN_FULL_JAW_TAGS if tags is None else tags)
    pts, tets, geo = read_gmsh_tets(path)
    n_raw_nodes, n_raw = len(pts), len(tets)
    kept_tags = [tags[k] for k in keep]
    sel = np.isin(geo, kept_tags)
    if not sel.any():
        raise ValueError(f"no tets with tags {kept_tags} in {path} (found {np.unique(geo)})")

    # Socket walls: faces of the kept tissue shared with the PDL (as vertex
    # triples in file numbering, so they survive reorientation/compaction).
    socket_tri = None
    if "pdl" in tags and "pdl" not in keep and np.any(geo == tags["pdl"]):
        shared = interface_faces(tets[sel], tets[geo == tags["pdl"]])
        socket_tri = tets[sel][:, TET_FACES].reshape(-1, 3)[shared]

    tets, geo = tets[sel], geo[sel]
    nodes = pts * scale_to_m
    del pts

    q = normalized_volume(nodes, tets)
    good = np.abs(q) >= min_normalized_volume
    n_degenerate = int((~good).sum())
    if n_degenerate:
        log.warning("removing %d degenerate tets (|normalized volume| < %g)",
                    n_degenerate, min_normalized_volume)
        tets, geo = tets[good], geo[good]
    n_flipped = int((q[good] < 0).sum())
    tets = orient_tets(nodes, tets)
    nodes, tets, used = compact(nodes, tets)

    faces, _ = boundary_faces(nodes, tets)
    surface_sets = {"all": faces}
    if socket_tri is not None:
        # Every socket vertex belongs to a kept tet unless that tet was degenerate.
        socket_tri = socket_tri[np.isin(socket_tri, used).all(axis=1)]
        socket_keys = np.unique(face_keys(np.searchsorted(used, socket_tri)))
        surface_sets["socket"] = faces[np.isin(face_keys(faces), socket_keys)]

    elem_fields = {}
    if len(kept_tags) > 1:
        elem_fields["tissue_tag"] = geo.astype(np.int32)

    meta = {
        "source": "Open-Full-Jaw",
        "source_file": str(Path(path).name),
        "units_in_file": "mm",
        "scale_to_m": scale_to_m,
        "tissues_kept": list(keep),
        "gmsh_tags": tags,
        "raw_nodes": n_raw_nodes,
        "raw_tets": n_raw,
        "degenerate_removed": n_degenerate,
        "reoriented": n_flipped,
    }
    log.info("Open-Full-Jaw import: %d/%d tets kept (%s), %d nodes, %d reoriented, "
             "%d boundary faces%s", len(tets), n_raw, "+".join(keep), len(nodes), n_flipped,
             len(faces), f", {len(surface_sets['socket'])} socket faces"
             if "socket" in surface_sets else "")
    return Mesh(nodes=nodes, tets=tets, surface_sets=surface_sets, elem_fields=elem_fields,
                meta=meta)
