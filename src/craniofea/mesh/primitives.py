"""Geometric primitives shared by tests, verification cases and mesh builders."""
from __future__ import annotations

from itertools import permutations

import numpy as np

from craniofea.model import Mesh

# Local (i, j, k) offsets of the 8 hex corners, indexed by bit pattern i + 2j + 4k.
_CORNER = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)])


def tet_signed_volumes(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    x = nodes[tets]
    return np.einsum("ij,ij->i", x[:, 1] - x[:, 0],
                     np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0])) / 6.0


def orient_tets(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return a copy of `tets` with every element positively oriented."""
    tets = np.array(tets, dtype=np.int64, copy=True)
    neg = tet_signed_volumes(nodes, tets) < 0
    tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()
    return tets


def box_mesh(lx: float, ly: float, lz: float, nx: int, ny: int, nz: int,
             origin=(0.0, 0.0, 0.0)) -> Mesh:
    """Structured box [0,lx]x[0,ly]x[0,lz] split into 6 conforming tets per hex (Kuhn).

    Node sets: xmin, xmax, ymin, ymax, zmin, zmax (nodes on each face).
    Surface sets: the same six names (outward boundary triangles) plus 'all'.
    """
    xs = np.linspace(0, lx, nx + 1) + origin[0]
    ys = np.linspace(0, ly, ny + 1) + origin[1]
    zs = np.linspace(0, lz, nz + 1) + origin[2]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def nid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    ii, jj, kk = (g.ravel() for g in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                                                 indexing="ij"))
    corners = np.stack([nid(ii + c[0], jj + c[1], kk + c[2]) for c in _CORNER], axis=1)

    # Kuhn: walk from corner 0 to corner 7 adding one axis bit at a time.
    tets = []
    for perm in permutations((1, 2, 4)):
        path = [0, perm[0], perm[0] + perm[1], 7]
        tets.append(corners[:, path])
    tets = orient_tets(nodes, np.concatenate(tets, axis=0))

    faces, _ = boundary_faces(nodes, tets)
    fc = nodes[faces].mean(axis=1)
    tol = 1e-9 * max(lx, ly, lz)
    lo = np.array(origin, dtype=float)
    hi = lo + np.array([lx, ly, lz])
    node_sets, surface_sets = {}, {"all": faces}
    for ax, name in enumerate("xyz"):
        for side, val in (("min", lo[ax]), ("max", hi[ax])):
            key = f"{name}{side}"
            node_sets[key] = np.flatnonzero(np.abs(nodes[:, ax] - val) < tol)
            surface_sets[key] = faces[np.abs(fc[:, ax] - val) < tol]
    return Mesh(nodes=nodes, tets=tets, node_sets=node_sets, surface_sets=surface_sets,
                meta={"source": "box_mesh", "dims": [lx, ly, lz], "divisions": [nx, ny, nz]})


_TET_FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
_OPPOSITE = np.array([3, 2, 1, 0])
#: Local vertex triples of the four faces of a positively oriented tet (outward winding).
TET_FACES = _TET_FACES


def face_keys(faces: np.ndarray) -> np.ndarray:
    """Orientation-independent integer key per triangle (equal keys <=> same vertex set).

    Packs the sorted vertex ids into one int64 (21 bits each) when ids < 2**21,
    otherwise falls back to a structured row view. Either way keys sort
    lexicographically by sorted vertex ids, so they can be used with
    np.unique / np.searchsorted / np.isin.
    """
    s = np.sort(np.asarray(faces), axis=1).astype(np.int64)
    if s.size == 0 or s.max() < (1 << 21):
        return (s[:, 0] << 42) | (s[:, 1] << 21) | s[:, 2]
    return np.ascontiguousarray(s).view([("a", np.int64), ("b", np.int64), ("c", np.int64)])[:, 0]


def face_multiplicity(faces: np.ndarray) -> np.ndarray:
    """(F,) number of times each triangle's vertex set occurs in `faces`."""
    _, inv, cnt = np.unique(face_keys(faces), return_inverse=True, return_counts=True)
    return cnt[inv.ravel()]


def boundary_faces(nodes: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boundary triangles of a tet mesh with outward orientation.

    Returns (faces (F, 3), owner element index (F,)).
    """
    tets = np.asarray(tets)
    all_faces = tets[:, _TET_FACES].reshape(-1, 3)
    owner = np.repeat(np.arange(len(tets)), 4)
    opposite = tets[:, _OPPOSITE].reshape(-1)
    is_bnd = face_multiplicity(all_faces) == 1
    f, own, opp = all_faces[is_bnd], owner[is_bnd], opposite[is_bnd]
    x = nodes[f]
    n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    inward = np.einsum("ij,ij->i", n, nodes[opp] - x[:, 0]) > 0
    f = f.copy()
    f[inward, 1], f[inward, 2] = f[inward, 2].copy(), f[inward, 1].copy()
    return f, own


def face_normals_areas(nodes: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit outward normals (F, 3) and areas (F,) of triangles."""
    x = nodes[faces]
    n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-300)[:, None], 0.5 * a
