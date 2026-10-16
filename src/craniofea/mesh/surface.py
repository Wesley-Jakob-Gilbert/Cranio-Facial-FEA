"""Vectorised geometric queries on triangulated boundary surfaces.

Used by the labeling code for mesh-size-independent quantities such as the
depth of an element below the bone surface (cortical shell) and the
nearest-surface normal.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def closest_point_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray,
                               c: np.ndarray) -> np.ndarray:
    """Closest point to p on triangle (a, b, c), row-wise for (..., 3) arrays.

    Voronoi-region algorithm of Ericson, *Real-Time Collision Detection*
    (2005), sec. 5.1.5, evaluated for all rows at once.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("...i,...i", ab, ap)
    d2 = np.einsum("...i,...i", ac, ap)
    bp = p - b
    d3 = np.einsum("...i,...i", ab, bp)
    d4 = np.einsum("...i,...i", ac, bp)
    cp = p - c
    d5 = np.einsum("...i,...i", ab, cp)
    d6 = np.einsum("...i,...i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    tiny = 1e-300
    denom = 1.0 / np.maximum(va + vb + vc, tiny)
    out = a + ab * (vb * denom)[..., None] + ac * (vc * denom)[..., None]   # face interior

    def put(mask, value):
        np.copyto(out, value, where=mask[..., None])

    # Ericson tests the regions in the order a, b, ab, c, ac, bc, interior;
    # writing them in reverse order lets earlier tests take precedence.
    m_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), tiny)
    put(m_bc, b + (c - b) * w[..., None])
    m_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w = d2 / np.maximum(d2 - d6, tiny)
    put(m_ac, a + ac * w[..., None])
    put((d6 >= 0) & (d5 <= d6), c)
    m_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    w = d1 / np.maximum(d1 - d3, tiny)
    put(m_ab, a + ab * w[..., None])
    put((d3 >= 0) & (d4 <= d3), b)
    put((d1 <= 0) & (d2 <= 0), a)
    return out


class SurfaceQuery:
    """Nearest-triangle queries against a fixed triangle set.

    Candidate triangles are the `k` nearest by centroid (cKDTree); the exact
    point-triangle distance is then evaluated for each candidate. With
    k >= 8 on shape-regular surface meshes the result equals the exact
    nearest triangle except in pathological configurations, and the error is
    in any case bounded by the triangle diameter.
    """

    def __init__(self, nodes: np.ndarray, faces: np.ndarray):
        if len(faces) == 0:
            raise ValueError("SurfaceQuery needs at least one triangle")
        self.nodes = np.asarray(nodes, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.tree = cKDTree(self.nodes[self.faces].mean(axis=1))

    def query(self, points: np.ndarray, k: int = 8,
              chunk: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
        """Return (distance (P,), index of nearest triangle (P,))."""
        points = np.asarray(points, dtype=float)
        k = int(min(k, len(self.faces)))
        dist = np.empty(len(points))
        idx = np.empty(len(points), dtype=np.int64)
        for s in range(0, len(points), chunk):
            p = points[s:s + chunk]
            _, cand = self.tree.query(p, k=k)
            cand = cand.reshape(len(p), k)
            tri = self.nodes[self.faces[cand]]                       # (P, k, 3, 3)
            q = closest_point_on_triangles(p[:, None, :], tri[:, :, 0], tri[:, :, 1],
                                           tri[:, :, 2])
            d2 = np.einsum("pki,pki->pk", q - p[:, None, :], q - p[:, None, :])
            j = np.argmin(d2, axis=1)
            dist[s:s + chunk] = np.sqrt(d2[np.arange(len(p)), j])
            idx[s:s + chunk] = cand[np.arange(len(p)), j]
        return dist, idx


def tet_centroids(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    return nodes[tets].mean(axis=1)


def face_components(faces: np.ndarray) -> np.ndarray:
    """Connected-component label per triangle (triangles sharing an edge are connected)."""
    faces = np.asarray(faces, dtype=np.int64)
    nf = len(faces)
    if nf == 0:
        return np.zeros(0, dtype=np.int64)
    e = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    owner = np.repeat(np.arange(nf), 3)
    key = e[:, 0] * (int(faces.max()) + 1) + e[:, 1]
    order = np.argsort(key, kind="stable")
    key, owner = key[order], owner[order]
    same = key[1:] == key[:-1]
    i, j = owner[:-1][same], owner[1:][same]
    adj = coo_matrix((np.ones(len(i)), (i, j)), shape=(nf, nf))
    _, lab = connected_components(adj, directed=False)
    return lab


def keep_large_components(faces: np.ndarray, areas: np.ndarray,
                          min_fraction: float) -> np.ndarray:
    """Boolean mask of triangles in components holding >= min_fraction of the total area."""
    if len(faces) == 0:
        return np.zeros(0, dtype=bool)
    lab = face_components(faces)
    comp_area = np.bincount(lab, weights=areas)
    return comp_area[lab] >= min_fraction * comp_area.sum()
