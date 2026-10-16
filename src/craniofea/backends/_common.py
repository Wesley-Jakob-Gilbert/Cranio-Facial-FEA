"""Helpers shared by the external-solver backends.

Everything here is pure NumPy so deck writers stay fast on large meshes and the
helpers can be unit-tested without any solver executable.
"""
from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import numpy as np

# C3D4 / tet4 faces in CalculiX numbering (0-based local node ids):
#   face 1 = nodes 1-2-3, face 2 = 1-4-2, face 3 = 2-4-3, face 4 = 3-4-1.
TET_FACE_NODES = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]], dtype=np.int64)
# Local node opposite each face above.
TET_FACE_OPPOSITE = np.array([3, 2, 0, 1], dtype=np.int64)

# Voigt index pairs in the craniofea order [xx, yy, zz, yz, xz, xy].
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


class BackendError(RuntimeError):
    """Raised when an external solver fails or produces unreadable output."""


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def write_rows(fh: TextIO, row_fmt: str, table: np.ndarray, chunk: int = 100_000) -> None:
    """Write `table` (n, k) with one printf-style `row_fmt` (k fields, ends in newline).

    Rows are formatted in chunks with a single ``%`` operation each, which is an
    order of magnitude faster than per-row ``savetxt`` for million-row tables.
    Integer fields (``%d``) may be passed as exactly representable floats.
    """
    table = np.asarray(table)
    if table.ndim == 1:
        table = table[:, None]
    for start in range(0, table.shape[0], chunk):
        block = table[start:start + chunk]
        fh.write((row_fmt * block.shape[0]) % tuple(block.ravel().tolist()))


def write_id_list(fh: TextIO, ids: np.ndarray, per_line: int = 10, sep: str = ",") -> None:
    """Write one comma-separated list of integer ids, `per_line` ids per line.

    Every line but the last ends with the separator, so the list stays a single
    comma-separated sequence when the reader ignores line breaks (FEBio).
    """
    ids = np.asarray(ids, dtype=np.int64).ravel()
    if ids.size == 0:
        return
    n_full = ((ids.size - 1) // per_line) * per_line     # last line always written apart
    if n_full:
        write_rows(fh, sep.join(["%d"] * per_line) + sep + "\n",
                   ids[:n_full].reshape(-1, per_line))
    rest = ids[n_full:]
    write_rows(fh, sep.join(["%d"] * rest.size) + "\n", rest[None, :])


# --------------------------------------------------------------------------- #
# Surface faces -> owner element faces
# --------------------------------------------------------------------------- #
def _face_codes(keys_a: np.ndarray, keys_b: np.ndarray, n_nodes: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """Integer codes for sorted node triplets so that equal faces get equal codes."""
    if n_nodes < 2**21:
        def enc(k: np.ndarray) -> np.ndarray:
            return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
        return enc(keys_a), enc(keys_b)
    _, inv = np.unique(np.concatenate([keys_a, keys_b]), axis=0, return_inverse=True)
    inv = inv.ravel()
    return inv[:len(keys_a)], inv[len(keys_a):]


def face_owners(nodes: np.ndarray, tets: np.ndarray, faces: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map surface triangles to an owning element face.

    Parameters
    ----------
    nodes, tets : mesh arrays (0-based).
    faces : (F, 3) triangles given as node ids.

    Returns
    -------
    elem : (F,) owner element index (0-based).
    local_face : (F,) face index 0..3 in CalculiX C3D4 numbering (P1..P4 = 0..3).
    sign : (F,) +1 if the triangle's right-hand normal points out of the owner
        element, -1 otherwise. A pressure ``p`` acting along ``-n_triangle``
        (the craniofea convention) equals an element-face pressure ``sign * p``
        in the solver convention (positive = pushes into the element).

    For a face shared by two elements (an internal surface) the element for which
    the triangle normal is outward is chosen, so ``sign`` is +1.
    """
    tets = np.asarray(tets, dtype=np.int64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    all_faces = tets[:, TET_FACE_NODES].reshape(-1, 3)
    code_all, code_q = _face_codes(np.sort(all_faces, axis=1), np.sort(faces, axis=1),
                                   len(nodes))
    order = np.argsort(code_all, kind="stable")
    sorted_codes = code_all[order]
    pos = np.searchsorted(sorted_codes, code_q)
    pos_c = np.minimum(pos, len(sorted_codes) - 1)
    found = sorted_codes[pos_c] == code_q
    if not np.all(found):
        bad = np.flatnonzero(~found)
        raise ValueError(f"{bad.size} surface triangles are not faces of any element "
                         f"(first: nodes {faces[bad[0]].tolist()})")

    def candidate_sign(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flat = order[p]
        e, lf = flat // 4, flat % 4
        x = nodes[faces]
        n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
        opp = nodes[tets[e, TET_FACE_OPPOSITE[lf]]]
        s = np.where(np.einsum("ij,ij->i", n, opp - x[:, 0]) < 0, 1, -1)
        return e, lf, s

    elem, lface, sign = candidate_sign(pos_c)
    # Internal faces: a second element shares the triangle; prefer the outward one.
    nxt = np.minimum(pos_c + 1, len(sorted_codes) - 1)
    shared = (sorted_codes[nxt] == code_q) & (nxt != pos_c) & (sign < 0)
    if np.any(shared):
        e2, l2, s2 = candidate_sign(nxt)
        elem = np.where(shared, e2, elem)
        lface = np.where(shared, l2, lface)
        sign = np.where(shared, s2, sign)
    return elem, lface, sign


def collect_face_pressures(mesh, pressures: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine all pressure sets of a load case into unique element-face loads.

    Returns (elem (K,), local_face (K,), pressure (K,)) where pressure follows the
    solver convention (positive pushes into the owning element). Contributions of
    several sets on the same face are summed.
    """
    elems, lfaces, vals = [], [], []
    for set_name, p in pressures.items():
        if set_name not in mesh.surface_sets:
            raise KeyError(f"pressure on unknown surface set {set_name!r}")
        faces = np.asarray(mesh.surface_sets[set_name])
        if len(faces) == 0:
            continue
        pv = np.broadcast_to(np.asarray(p, dtype=float), (len(faces),))
        e, lf, s = face_owners(mesh.nodes, mesh.tets, faces)
        elems.append(e)
        lfaces.append(lf)
        vals.append(s * pv)
    if not elems:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0)
    e, lf, v = np.concatenate(elems), np.concatenate(lfaces), np.concatenate(vals)
    key, inv = np.unique(4 * e + lf, return_inverse=True)
    total = np.zeros(len(key))
    np.add.at(total, inv.ravel(), v)
    return key // 4, key % 4, total


# --------------------------------------------------------------------------- #
# Constitutive helpers
# --------------------------------------------------------------------------- #
def voigt_stress_rotation(R: np.ndarray) -> np.ndarray:
    """(..., 6, 6) matrix T with sigma'_voigt = T @ sigma_voigt for sigma' = R sigma R^T.

    With engineering-shear strain the stiffness transforms as C' = T C T^T.
    Rows of `R` are the new basis vectors expressed in the old frame.
    """
    R = np.asarray(R, dtype=float)
    T = np.zeros(R.shape[:-2] + (6, 6))
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        for b, (k, q) in enumerate(VOIGT_PAIRS):
            if k == q:
                T[..., a, b] = R[..., i, k] * R[..., j, q]
            else:
                T[..., a, b] = R[..., i, k] * R[..., j, q] + R[..., i, q] * R[..., j, k]
    return T


def rotate_stiffness(C: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Express global Voigt stiffness `C` (..., 6, 6) in the frame whose axes are rows of R."""
    T = voigt_stress_rotation(R)
    return T @ C @ np.swapaxes(T, -1, -2)


def isotropic_parameters(C: np.ndarray, rtol: float = 1e-10
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect isotropic stiffness matrices and recover (E, nu).

    Returns (is_isotropic (M,), E (M,), nu (M,)); E and nu are NaN where the
    matrix is not isotropic to relative tolerance `rtol` (w.r.t. max |C_ij|).
    """
    C = np.asarray(C, dtype=float).reshape(-1, 6, 6)
    lam = C[:, 0, 1]
    mu = C[:, 3, 3]
    ref = np.zeros_like(C)
    ref[:, :3, :3] = lam[:, None, None]
    for i in range(3):
        ref[:, i, i] += 2 * mu
        ref[:, 3 + i, 3 + i] = mu
    scale = np.abs(C).max(axis=(1, 2))
    iso = np.abs(C - ref).max(axis=(1, 2)) <= rtol * np.maximum(scale, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        E = mu * (3 * lam + 2 * mu) / (lam + mu)
        nu = lam / (2 * (lam + mu))
    E = np.where(iso, E, np.nan)
    nu = np.where(iso, nu, np.nan)
    return iso, E, nu


def orthotropic_parameters(C_local: np.ndarray, rtol: float = 1e-8
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Engineering constants of stiffness matrices given in their symmetry frame.

    Parameters
    ----------
    C_local : (M, 6, 6) stiffness expressed in the material axes.

    Returns
    -------
    is_orthotropic : (M,) bool, True where normal-shear and shear-shear couplings
        vanish to `rtol` relative to max |C_ij|.
    ortho : (M, 9) [E1, E2, E3, G12, G13, G23, nu12, nu13, nu23] (the
        ``Material.ortho`` layout); nu_ij is the contraction in j for stress in i.
    """
    C = np.asarray(C_local, dtype=float).reshape(-1, 6, 6)
    mask = np.ones((6, 6), dtype=bool)
    mask[:3, :3] = False
    mask[[3, 4, 5], [3, 4, 5]] = False
    scale = np.abs(C).max(axis=(1, 2))
    ok = np.abs(C[:, mask]).max(axis=1) <= rtol * np.maximum(scale, 1e-300)
    S = np.linalg.inv(C[:, :3, :3])
    E1, E2, E3 = 1 / S[:, 0, 0], 1 / S[:, 1, 1], 1 / S[:, 2, 2]
    ortho = np.column_stack([
        E1, E2, E3,
        C[:, 5, 5], C[:, 4, 4], C[:, 3, 3],          # G12 (xy), G13 (xz), G23 (yz)
        -S[:, 0, 1] * E1, -S[:, 0, 2] * E1, -S[:, 1, 2] * E2,
    ])
    return ok, ortho


def orthotropic_C(ortho: np.ndarray) -> np.ndarray:
    """(M, 6, 6) local-frame stiffness from (M, 9) [E1,E2,E3,G12,G13,G23,nu12,nu13,nu23]."""
    o = np.atleast_2d(np.asarray(ortho, dtype=float))
    E1, E2, E3, G12, G13, G23, n12, n13, n23 = o.T
    S = np.zeros((len(o), 6, 6))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = 1 / E1, 1 / E2, 1 / E3
    S[:, 0, 1] = S[:, 1, 0] = -n12 / E1
    S[:, 0, 2] = S[:, 2, 0] = -n13 / E1
    S[:, 1, 2] = S[:, 2, 1] = -n23 / E2
    S[:, 3, 3], S[:, 4, 4], S[:, 5, 5] = 1 / G23, 1 / G13, 1 / G12
    return np.linalg.inv(S)


# --------------------------------------------------------------------------- #
# Result verification
# --------------------------------------------------------------------------- #
def voigt_to_tensor(v: np.ndarray) -> np.ndarray:
    """(M, 6) Voigt stress [xx, yy, zz, yz, xz, xy] -> (M, 3, 3) symmetric tensors."""
    v = np.asarray(v, dtype=float)
    T = np.empty(v.shape[:-1] + (3, 3))
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        T[..., i, j] = v[..., a]
        T[..., j, i] = v[..., a]
    return T


def equilibrium_residual(problem, case, stress: np.ndarray) -> float:
    """Relative nodal force imbalance of an external solution.

    Internal forces are rebuilt from the solver's element stresses,
    ``f_int[a] = V * sigma @ grad(N_a)``, and compared with the external loads of
    `case` (assembled exactly as the reference solver does) at the free dofs.
    The result is ``max |f_int - f_ext| / max(|f_ext|, sum_e |f_int,e|)``; for a
    correct linear solution it is at the level of the output rounding (~1e-6
    for CalculiX's 7-digit ``.dat`` values). Large values reveal a corrupted
    solve, mismatched load mapping or a wrong material assignment.
    """
    from craniofea.fem.reference import assemble_rhs, tet4_gradients

    mesh = problem.mesh
    grads, vol = tet4_gradients(mesh.nodes, mesh.tets)
    fe = vol[:, None, None] * np.einsum("mij,maj->mai", voigt_to_tensor(stress), grads)
    f_int = np.zeros((mesh.n_nodes, 3))
    f_abs = np.zeros((mesh.n_nodes, 3))
    idx = mesh.tets.ravel()
    np.add.at(f_int, idx, fe.reshape(-1, 3))
    np.add.at(f_abs, idx, np.abs(fe).reshape(-1, 3))
    f_ext = assemble_rhs(problem, case, vol).reshape(-1, 3)
    free = ~problem.constrained_mask()
    if not np.any(free):
        return 0.0
    scale = max(float(np.abs(f_ext).max()), float(f_abs[free].max()), 1e-300)
    return float(np.abs(f_int - f_ext)[free].max() / scale)


# --------------------------------------------------------------------------- #
# Process helpers
# --------------------------------------------------------------------------- #
def find_executable(candidates: Sequence[str | os.PathLike]) -> str | None:
    """First candidate that is an existing file or resolvable on PATH."""
    for c in candidates:
        if not c:
            continue
        c = os.path.expanduser(str(c))
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
        hit = shutil.which(c)
        if hit:
            return hit
    return None


def tail(path: Path | str, n_lines: int = 30) -> str:
    """Last `n_lines` of a text file (empty string if it does not exist)."""
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n_lines:])
