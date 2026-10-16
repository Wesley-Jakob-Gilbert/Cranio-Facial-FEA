"""Production linear-elastic tet4 solver with implicit-adjoint gradients.

Problem
-------
Find ``u`` with ``u = 0`` on the constrained dofs such that

    K u = f,        K = sum_e V_e B_e^T C_e B_e,

where ``C_e`` is the global-frame Voigt stiffness of element ``e`` and ``f``
collects consistent surface-pressure, gravity and nodal loads (identical to
:mod:`craniofea.fem.reference`):

* pressure ``p`` on a boundary triangle of area ``A`` and outward normal
  ``n``: ``-p A n / 3`` on each of its nodes (positive ``p`` pushes inward);
* gravity ``g``: ``rho_e V_e g / 4`` on each node of element ``e``;
* nodal forces added as given.

Discretisation
--------------
Element kernels (gradients, ``B``, ``K_e``, strain / stress / energy
recovery) are vectorised, jit-compiled JAX functions
(:mod:`craniofea.fem.kernels`), evaluated on fixed-size element chunks so
memory stays bounded on large meshes. Homogeneous Dirichlet conditions are
imposed by decoupling the constrained dofs: their rows and columns are
zeroed and the diagonal entry kept. This is algebraically identical to
eliminating them, keeps the 3x3 nodal block structure intact and lets
nodes with only some dofs constrained (rollers, symmetry planes) share one
code path.

Linear solvers
--------------
``method="amg"`` (default, CPU)
    ``K`` is assembled as a SciPy block-sparse (BSR, 3x3) matrix from the
    JAX element blocks and solved by preconditioned CG. The preconditioner
    is PyAMG smoothed aggregation with the six rigid-body modes as
    near-nullspace (Vanek, Mandel & Brezina 1996), combined with
    *rigid-body-mode deflation* when the model contains compliant layers.
    Elements softer than ``soft_ratio`` times the stiffest material (e.g.
    a 50 MPa suture between 17 GPa cortical plates) separate the mesh into
    stiff components whose relative rigid motions are nearly
    energy-free; these modes are removed with the A-DEF2 two-level
    preconditioner of Tang, Nabben, Vuik & Erlangga (J. Sci. Comput. 39,
    2009), using the rigid-body modes of every stiff component as deflation
    space (Jonsthovel, van Gijzen, Vuik et al., CMES 47, 2009). One
    hierarchy serves all load cases.

``method="jax-cg"`` (GPU when available)
    Matrix-free ``K u`` (gather, constitutive update, scatter-add) inside a
    ``lax.while_loop`` CG with a 3x3 nodal block-Jacobi preconditioner. No
    global matrix is formed.

Both stop on the relative residual ``||f - K u|| / ||f|| <= rtol``; the
reported residual is recomputed from the final iterate.

Gradients
---------
:class:`DifferentiableSolver` exposes ``(s, l) -> u, strain, stress, sed``
as a JAX-differentiable function of per-element modulus scales ``s``
(``C_e -> s_e C_e``) and per-case load scales ``l``. Its VJP uses the
adjoint method: for a scalar objective ``L(u)``, solve ``K lambda = dL/du``
(``K`` is symmetric, so the forward solver and preconditioner are reused)
and

    dL/ds_e = -lambda_e^T (dK_e/ds_e) u_e = -V_e eps(lambda)_e . C_e eps(u)_e,
    dL/df   = lambda.

Importing this module enables 64-bit JAX (``jax_enable_x64``); FE solves to
tight tolerances need double precision.
"""
from __future__ import annotations

import contextlib
import hashlib
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import lax
from scipy.sparse.csgraph import connected_components

from craniofea.fem import kernels
from craniofea.mesh.primitives import face_normals_areas
from craniofea.model import LoadCase, Problem, Solution

jax.config.update("jax_enable_x64", True)

__all__ = [
    "JaxSolver",
    "DifferentiableSolver",
    "make_differentiable_solver",
    "assemble_load",
    "rigid_body_modes",
    "stiff_components",
    "DEFAULT_AMG_OPTIONS",
]

_GS = ("block_gauss_seidel", {"sweep": "symmetric", "iterations": 1})

#: PyAMG smoothed-aggregation settings for 3D elasticity (nodal 3x3 blocks).
DEFAULT_AMG_OPTIONS: dict = {
    "strength": ("symmetric", {"theta": 0.0}),
    "aggregate": "standard",
    "smooth": ("jacobi", {"omega": 4.0 / 3.0}),
    "presmoother": _GS,
    "postsmoother": _GS,
    "improve_candidates": None,
    "max_coarse": 600,
    "max_levels": 12,
    "keep": False,
}


def _single_threaded_blas():
    """Limit BLAS to one thread: sparse CG gains nothing from it and idle BLAS
    threads spinning between small dense calls can slow the solve by an order
    of magnitude on a loaded machine."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover
        return contextlib.nullcontext()
    return threadpool_limits(limits=1, user_api="blas")


# --------------------------------------------------------------------------- loads

def _scatter_rows(index: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Sum (K, 3) ``values`` into (n, 3) rows ``index``."""
    return np.stack([np.bincount(index, weights=values[:, k], minlength=n)
                     for k in range(3)], axis=1)


def assemble_load(problem: Problem, case: LoadCase, vol: np.ndarray) -> np.ndarray:
    """Consistent nodal load (N, 3) of one load case.

    ``vol`` are the element volumes (M,). Pressure acts along the inward
    normal and is lumped as ``p A / 3`` per face node, gravity as
    ``rho V / 4`` per element node (both exact for linear triangles /
    tetrahedra with constant data).
    """
    mesh = problem.mesh
    n = mesh.n_nodes
    f = np.zeros((n, 3))
    for set_name, p in case.pressures.items():
        faces = np.asarray(mesh.surface_sets[set_name], dtype=np.int64)
        normal, area = face_normals_areas(mesh.nodes, faces)
        p = np.broadcast_to(np.asarray(p, dtype=float), (len(faces),))
        fn = -(p * area / 3.0)[:, None] * normal
        f += _scatter_rows(faces.ravel(), np.repeat(fn, 3, axis=0), n)
    if case.gravity is not None:
        fe = (problem.material.mass_density * vol / 4.0)[:, None] * np.asarray(case.gravity)[None]
        f += _scatter_rows(mesh.tets.ravel(), np.repeat(fe, 4, axis=0), n)
    if case.nodal_forces is not None:
        f += np.asarray(case.nodal_forces, dtype=float)
    return f


# --------------------------------------------------------------------------- coarse spaces

def rigid_body_modes(nodes: np.ndarray) -> np.ndarray:
    """(3N, 6) rigid-body modes: three translations and three rotations.

    Rotations are taken about the centroid and scaled by the bounding-box
    size so all six columns have comparable magnitude.
    """
    x = nodes - nodes.mean(axis=0)
    x = x / max(float(np.ptp(nodes, axis=0).max()), 1e-300)
    n = len(nodes)
    R = np.zeros((n, 3, 6))
    R[:, 0, 0] = R[:, 1, 1] = R[:, 2, 2] = 1.0
    # columns 3..5: rotation about x, y, z  (omega x r)
    R[:, 1, 3], R[:, 2, 3] = -x[:, 2], x[:, 1]
    R[:, 0, 4], R[:, 2, 4] = x[:, 2], -x[:, 0]
    R[:, 0, 5], R[:, 1, 5] = -x[:, 1], x[:, 0]
    return R.reshape(3 * n, 6)


def stiff_components(tets: np.ndarray, C: np.ndarray, n_nodes: int, soft_ratio: float = 0.01,
                     min_nodes: int = 4) -> list[np.ndarray]:
    """Node sets of the connected stiff regions of a heterogeneous mesh.

    An element is *stiff* if ``tr C_e >= soft_ratio * q99(tr C)`` (99th
    percentile, robust to a few outliers). Stiff elements sharing a node are
    connected; nodes touching only soft elements belong to no component.
    Returns node-index arrays sorted by decreasing size, keeping components
    with at least ``min_nodes`` nodes.
    """
    k = np.trace(C, axis1=-2, axis2=-1)
    el = np.flatnonzero(k >= soft_ratio * np.percentile(k, 99))
    t = np.asarray(tets)[el]
    # a spanning set of edges (0-1, 0-2, 0-3) suffices for connectivity
    rows = np.repeat(t[:, 0], 3)
    cols = t[:, 1:].ravel()
    G = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes))
    _, lab = connected_components(G, directed=False)
    used = np.zeros(n_nodes, dtype=bool)
    used[t.ravel()] = True
    lab = np.where(used, lab, -1)
    order = np.argsort(lab, kind="stable")
    lab_sorted = lab[order]
    starts = np.flatnonzero(np.r_[True, lab_sorted[1:] != lab_sorted[:-1]])
    groups = np.split(order, starts[1:])
    comps = [g for g in groups if lab[g[0]] >= 0 and len(g) >= min_nodes]
    return sorted(comps, key=len, reverse=True)


def _deflation_space(nodes: np.ndarray, comps: list[np.ndarray], free: np.ndarray,
                     max_components: int) -> sp.csr_matrix | None:
    """Sparse (3N, 6k) rigid-body modes of each stiff component, zero on constrained dofs."""
    if len(comps) < 2:
        return None
    comps = comps[:max_components]
    rows, cols, vals = [], [], []
    for c, idx in enumerate(comps):
        R = rigid_body_modes(nodes[idx]).reshape(len(idx), 3, 6)
        r = (3 * idx[:, None, None] + np.arange(3)[None, :, None]) * np.ones((1, 1, 6), int)
        rows.append(r.ravel())
        cols.append(np.broadcast_to(6 * c + np.arange(6), R.shape).ravel())
        vals.append(R.ravel())
    n = 3 * len(nodes)
    Z = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, 6 * len(comps)))
    Z = sp.diags(free.ravel()) @ Z
    Z.eliminate_zeros()
    return Z.tocsr()


# --------------------------------------------------------------------------- jitted operators

@partial(jax.jit, static_argnames=("chunk",))
def _assemble_blocks(grads, C, w, inv, brow, bcol, diag_idx, cons, *, chunk):
    """Assemble BSR data (nnzb, 3, 3) and decouple the constrained dofs.

    Element blocks are scattered chunk by chunk (``inv`` maps element block
    ``(a, b)`` to its BSR slot). Constrained rows / columns are zeroed with
    their diagonal entry kept; empty diagonals (nodes in no element) get 1.
    """
    n_blocks = brow.shape[0]

    def body(i, data):
        sl = partial(lax.dynamic_slice_in_dim, start_index=i * chunk, slice_size=chunk)
        kb = kernels.stiffness_blocks(sl(grads), sl(C), sl(w))
        return data.at[sl(inv).reshape(-1)].add(kb.reshape(-1, 3, 3))

    data = lax.fori_loop(0, grads.shape[0] // chunk, body, jnp.zeros((n_blocks, 3, 3)))
    d = jnp.diagonal(data[diag_idx], axis1=1, axis2=2)              # (N, 3)
    empty = d == 0
    keep = 1.0 - jnp.maximum(cons, empty.astype(cons.dtype))
    data = data * keep[brow][:, :, None] * keep[bcol][:, None, :]
    dfix = jnp.where(empty, 1.0, d) * (1.0 - keep)
    return data.at[diag_idx].add(dfix[:, :, None] * jnp.eye(3))


@partial(jax.jit, static_argnames=("chunk",))
def _block_diagonal(grads, C, w, tets, cons, *, chunk):
    """Nodal 3x3 diagonal blocks of the decoupled operator, and the free mask."""
    n = cons.shape[0]

    def body(i, D):
        sl = partial(lax.dynamic_slice_in_dim, start_index=i * chunk, slice_size=chunk)
        B = kernels.b_blocks(sl(grads))
        CB = jnp.einsum("mkl,malj->makj", sl(C), B)
        kaa = jnp.einsum("maki,makj->maij", B, CB) * sl(w)[:, None, None, None]
        return D.at[sl(tets).reshape(-1)].add(kaa.reshape(-1, 3, 3))

    D = lax.fori_loop(0, grads.shape[0] // chunk, body, jnp.zeros((n, 3, 3)))
    d = jnp.diagonal(D, axis1=1, axis2=2)
    empty = d == 0
    keep = 1.0 - jnp.maximum(cons, empty.astype(cons.dtype))
    D = D * keep[:, :, None] * keep[:, None, :]
    D = D + (jnp.where(empty, 1.0, d) * (1.0 - keep))[:, :, None] * jnp.eye(3)
    return D, keep


def _inv3(D):
    """Batched inverse of (N, 3, 3) matrices by cofactors (no LU workspace)."""
    c0 = jnp.cross(D[:, 1], D[:, 2])
    c1 = jnp.cross(D[:, 2], D[:, 0])
    c2 = jnp.cross(D[:, 0], D[:, 1])
    det = jnp.einsum("ni,ni->n", D[:, 0], c0)
    return jnp.stack([c0, c1, c2], axis=-1) / det[:, None, None]


def _decoupled_matvec(grads, C, w, tets, keep, dcons, x):
    """``P K P x + (I - P) D x`` with ``P`` the free-dof projector."""
    return keep * kernels.apply_stiffness(grads, C, w, tets, keep * x) + dcons * x


def _jax_pcg(matvec: Callable, precond: Callable, b, x0, rtol: float, maxiter: int):
    """Preconditioned CG in ``lax.while_loop``; returns ``x, iterations, ||r|| / ||b||``."""
    bnorm = jnp.linalg.norm(b)
    tol = rtol * bnorm
    r0 = b - matvec(x0)
    z0 = precond(r0)

    def cond(st):
        _, r, _, _, _, k = st
        return (k < maxiter) & (jnp.linalg.norm(r) > tol)

    def body(st):
        x, r, z, p, rz, k = st
        Ap = matvec(p)
        alpha = rz / jnp.vdot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        z = precond(r)
        rz_new = jnp.vdot(r, z)
        p = z + (rz_new / rz) * p
        return x, r, z, p, rz_new, k + 1

    st = (x0, r0, z0, z0, jnp.vdot(r0, z0), jnp.asarray(0))
    x, r, *_, k = lax.while_loop(cond, body, st)
    return x, k, jnp.linalg.norm(r) / jnp.where(bnorm > 0, bnorm, 1.0)


@partial(jax.jit, static_argnames=("maxiter",))
def _jax_cg_solve(grads, C, w, tets, keep, dcons, Dinv, b, x0, rtol, *, maxiter):
    shape = b.shape
    mv = partial(_decoupled_matvec, grads, C, w, tets, keep, dcons)

    def matvec(x):
        return mv(x.reshape(shape)).reshape(-1)

    def precond(r):
        return jnp.einsum("nij,nj->ni", Dinv, r.reshape(shape)).reshape(-1)

    bf = b.reshape(-1)
    bnorm = jnp.linalg.norm(bf)

    def true_rel(x):
        return jnp.linalg.norm(bf - matvec(x)) / jnp.where(bnorm > 0, bnorm, 1.0)

    # restart from the current iterate while the true residual misses the tolerance
    def cond(st):
        x, k, rel, n = st
        return (true_rel(x) > rtol) & (k < maxiter) & (n < 4)

    def body(st):
        x, k, _, n = st
        x, dk, rel = _jax_pcg(matvec, precond, bf, x, rtol, maxiter - k)
        return x, k + dk, rel, n + 1

    x, k, rel, _ = lax.while_loop(cond, body, (x0.reshape(-1), jnp.asarray(0), jnp.asarray(1.0),
                                                jnp.asarray(0)))
    return x.reshape(shape), k, rel, true_rel(x)


_recover = jax.jit(kernels.recover)
_matvec = jax.jit(_decoupled_matvec)


@jax.jit
def _pair_energy(grads, C, vol, tets, lam, u):
    """``V_e eps(lam)_e . C_e eps(u)_e`` per element, i.e. ``lam_e^T K_e u_e``."""
    el = kernels.strain(grads, lam[tets])
    eu = kernels.strain(grads, u[tets])
    return vol * jnp.einsum("mi,mij,mj->m", el, C, eu)


# --------------------------------------------------------------------------- host-side CG / AMG

def _numpy_pcg(A, b: np.ndarray, M: Callable, x0: np.ndarray, rtol: float, maxiter: int,
               restarts: int = 3):
    """Preconditioned CG (Hestenes-Stiefel) on the host; returns ``x, iterations, rel. residual``.

    The recursively updated residual can drift from ``b - A x`` in finite
    precision; if the true residual misses the tolerance, CG is restarted
    from the current iterate (at most ``restarts`` times).
    """
    bnorm = float(np.linalg.norm(b))
    if bnorm == 0.0:
        return np.zeros_like(b), 0, 0.0
    x = x0.copy()
    k = 0
    for _ in range(restarts + 1):
        r = b - A @ x
        if np.linalg.norm(r) <= rtol * bnorm:
            break
        z = M(r)
        p = z.copy()
        rz = float(r @ z)
        while k < maxiter and np.linalg.norm(r) > rtol * bnorm:
            Ap = A @ p
            alpha = rz / float(p @ Ap)
            x += alpha * p
            r -= alpha * Ap
            z = M(r)
            rz_new = float(r @ z)
            p *= rz_new / rz
            p += z
            rz = rz_new
            k += 1
        if k >= maxiter:
            break
    return x, k, float(np.linalg.norm(b - A @ x)) / bnorm


class _AMGCG:
    """CG preconditioned by smoothed-aggregation AMG, optionally with A-DEF2 deflation.

    With deflation space ``Z``, ``E = Z^T A Z`` and ``Q = Z E^+ Z^T`` the
    preconditioner is ``P^T M^-1 + Q`` with ``P^T = I - Q A`` and the
    iteration starts from ``x0 = Q b`` (Tang et al. 2009, A-DEF2).
    :meth:`set_operator` swaps in a new matrix of the same pattern while
    keeping the AMG hierarchy (still a valid SPD preconditioner).
    """

    def __init__(self, A: sp.bsr_matrix, B: np.ndarray, options: dict,
                 Z: sp.csr_matrix | None = None):
        import pyamg

        t0 = time.perf_counter()
        with _single_threaded_blas():
            self.ml = pyamg.smoothed_aggregation_solver(A, B=B, symmetry="hermitian", **options)
            self.M = self.ml.aspreconditioner(cycle="V")
        self.Z = Z
        self.set_operator(A)
        self.setup_time = time.perf_counter() - t0
        self.levels = len(self.ml.levels)
        self.operator_complexity = float(self.ml.operator_complexity())
        self.n_deflation = 0 if Z is None else int(Z.shape[1])

    def set_operator(self, A) -> None:
        self.A = A
        if self.Z is None:
            return
        self.AZ = sp.csr_matrix(A @ self.Z)
        E = (self.Z.T @ self.AZ).toarray()
        w, V = np.linalg.eigh(0.5 * (E + E.T))
        ok = w > w.max() * 1e-12
        self.Einv = (V[:, ok] / w[ok]) @ V[:, ok].T

    def _Q(self, r: np.ndarray) -> np.ndarray:
        return self.Z @ (self.Einv @ (self.Z.T @ r))

    def _precond(self, r: np.ndarray) -> np.ndarray:
        z = self.M(r)
        if self.Z is None:
            return z
        return z - self.Z @ (self.Einv @ (self.AZ.T @ z)) + self._Q(r)

    def solve(self, b: np.ndarray, rtol: float, maxiter: int):
        with _single_threaded_blas():
            x0 = np.zeros_like(b) if self.Z is None else self._Q(b)
            return _numpy_pcg(self.A, b, self._precond, x0, rtol, maxiter)


# --------------------------------------------------------------------------- prepared system

@dataclass
class _Pattern:
    """BSR sparsity of a tet mesh (3x3 nodal blocks), shared by all materials."""

    indptr: np.ndarray
    indices: np.ndarray
    brow: np.ndarray
    inv: np.ndarray          # (M_pad, 16) element block (a, b) -> BSR slot
    diag_idx: np.ndarray     # (N,) slot of each diagonal block


def _block_pattern(tets: np.ndarray, n_nodes: int, m_pad: int) -> _Pattern:
    n = np.int64(n_nodes)
    t = tets.astype(np.int64)
    keys = (np.repeat(t, 4, axis=1) * n + np.tile(t, (1, 4))).ravel()
    diag_keys = np.arange(n, dtype=np.int64) * (n + 1)
    uniq, inv = np.unique(np.concatenate([keys, diag_keys]), return_inverse=True)
    inv = inv.ravel().astype(np.int32)
    brow = (uniq // n).astype(np.int32)
    indices = (uniq % n).astype(np.int32)
    indptr = np.searchsorted(brow, np.arange(n_nodes + 1)).astype(np.int32)
    inv_e = np.zeros((m_pad, 16), dtype=np.int32)
    inv_e[: len(tets)] = inv[: keys.size].reshape(-1, 16)
    return _Pattern(indptr, indices, brow, inv_e, inv[keys.size:])


class _System:
    """Element data of one problem, padded to whole chunks and placed on a device."""

    def __init__(self, problem: Problem, chunk_size: int, device, pattern_cache: dict | None):
        mesh = problem.mesh
        self.n_nodes, self.n_elems = mesh.n_nodes, mesh.n_elems
        M = self.n_elems
        self.chunk = int(min(chunk_size, max(M, 1)))
        self.m_pad = -(-M // self.chunk) * self.chunk
        pad = self.m_pad - M
        self.device = device

        tets = np.asarray(mesh.tets, dtype=np.int64)
        C = np.asarray(problem.material.C, dtype=float)
        if C.shape != (M, 6, 6):
            raise ValueError(f"Material.C must have shape ({M}, 6, 6), got {C.shape}")
        self.C_np = C
        put = partial(jax.device_put, device=device)
        x = put(np.asarray(mesh.nodes, dtype=float)[tets])
        grads, vol = jax.jit(kernels.tet4_geometry)(x)
        del x
        vol_np = np.asarray(vol)
        if np.any(vol_np <= 0):
            raise ValueError(f"{int(np.sum(vol_np <= 0))} tetrahedra have non-positive volume")
        self.vol = vol_np
        zpad = partial(np.pad, mode="constant")
        self.grads = put(jnp.pad(grads, ((0, pad), (0, 0), (0, 0))))
        self.vol_j = put(jnp.pad(vol, (0, pad)))
        self.C = put(zpad(C, ((0, pad), (0, 0), (0, 0))))
        self.tets_np = tets
        self.tets = put(zpad(tets.astype(np.int32), ((0, pad), (0, 0))))
        self.cons_np = problem.constrained_mask().astype(float)
        self.cons = put(self.cons_np)
        self.nodes = np.asarray(mesh.nodes, dtype=float)

        self._pattern_cache = pattern_cache
        self._pattern: _Pattern | None = None
        self._rbm: np.ndarray | None = None

    # -- assembled operator (AMG path)
    @property
    def pattern(self) -> _Pattern:
        if self._pattern is None:
            key = (self.n_nodes, self.m_pad,
                   hashlib.blake2b(np.ascontiguousarray(self.tets_np).tobytes(),
                                   digest_size=16).hexdigest())
            cache = self._pattern_cache if self._pattern_cache is not None else {}
            if key not in cache:
                cache.clear()
                cache[key] = _block_pattern(self.tets_np, self.n_nodes, self.m_pad)
            self._pattern = cache[key]
            p = self._pattern
            put = partial(jax.device_put, device=self.device)
            self._pat_dev = tuple(put(a) for a in (p.inv, p.brow, p.indices, p.diag_idx))
        return self._pattern

    def assemble(self, scale=None) -> jnp.ndarray:
        """Decoupled BSR block data (nnzb, 3, 3) for modulus scale ``scale`` (M_pad,)."""
        _ = self.pattern
        inv, brow, bcol, diag_idx = self._pat_dev
        w = self.vol_j if scale is None else self.vol_j * scale
        return _assemble_blocks(self.grads, self.C, w, inv, brow, bcol, diag_idx, self.cons,
                                chunk=self.chunk)

    def bsr(self, data) -> sp.bsr_matrix:
        p = self.pattern
        n = 3 * self.n_nodes
        return sp.bsr_matrix((np.asarray(data), p.indices, p.indptr), shape=(n, n))

    @property
    def rbm(self) -> np.ndarray:
        if self._rbm is None:
            self._rbm = rigid_body_modes(self.nodes)
        return self._rbm

    def deflation(self, soft_ratio: float, max_components: int) -> sp.csr_matrix | None:
        comps = stiff_components(self.tets_np, self.C_np, self.n_nodes, soft_ratio)
        return _deflation_space(self.nodes, comps, 1.0 - self.cons_np, max_components)

    def pad(self, scale) -> jnp.ndarray:
        scale = jnp.asarray(scale, dtype=jnp.float64)
        return jnp.pad(scale, (0, self.m_pad - self.n_elems))

    def recover(self, u: np.ndarray):
        eps, sig, sed = _recover(self.grads, self.C, self.tets, jnp.asarray(u))
        M = self.n_elems
        return np.asarray(eps)[:M], np.asarray(sig)[:M], np.asarray(sed)[:M]


# --------------------------------------------------------------------------- public solver

class JaxSolver:
    """Linear-elastic tet4 solver implementing the :class:`~craniofea.model.Solver` protocol.

    Parameters
    ----------
    method : ``"amg"`` (assembled BSR matrix, PyAMG smoothed-aggregation
        preconditioned CG with rigid-body deflation, CPU) or ``"jax-cg"``
        (matrix-free CG with nodal block-Jacobi preconditioning, runs on the
        default JAX device, i.e. the GPU when one is visible).
    rtol : relative residual tolerance ``||f - K u|| / ||f||``.
    maxiter : CG iteration cap (default 2000 for ``amg``, 50000 for ``jax-cg``).
    chunk_size : elements per assembly chunk (bounds peak memory).
    device : JAX device for the element kernels; default CPU for ``amg``
        and the default backend for ``jax-cg``.
    amg_options : overrides for :data:`DEFAULT_AMG_OPTIONS`.
    deflation : enable stiff-component rigid-body deflation (``amg`` only;
        inactive when the model has a single stiff component).
    soft_ratio : stiffness ratio below which an element counts as a
        compliant layer for the deflation space (see :func:`stiff_components`).
    max_components : largest number of stiff components used for deflation.

    ``solve`` returns one :class:`~craniofea.model.Solution` per load case
    with ``info`` holding ``iterations``, ``relative_residual``,
    ``converged`` and timings (``assembly_s``, ``setup_s``, ``solve_s``,
    ``time_s``). The sparsity pattern is cached between calls on the same
    mesh, so repeated solves with updated materials skip that step.
    """

    name = "jax"

    def __init__(self, method: str = "amg", rtol: float = 1e-10, maxiter: int | None = None,
                 chunk_size: int = 65536, device=None, amg_options: dict | None = None,
                 deflation: bool = True, soft_ratio: float = 0.01, max_components: int = 64):
        if method not in ("amg", "jax-cg"):
            raise ValueError(f"unknown method {method!r}; use 'amg' or 'jax-cg'")
        self.method = method
        self.rtol = float(rtol)
        self.maxiter = int(maxiter if maxiter is not None else (2000 if method == "amg" else 50000))
        self.chunk_size = int(chunk_size)
        if device is None:
            device = jax.devices("cpu")[0] if method == "amg" else jax.devices()[0]
        self.device = device
        self.amg_options = {**DEFAULT_AMG_OPTIONS, **(amg_options or {})}
        self.deflation = bool(deflation)
        self.soft_ratio = float(soft_ratio)
        self.max_components = int(max_components)
        self._pattern_cache: dict = {}

    def prepare(self, problem: Problem) -> _System:
        """Precompute element geometry and device arrays for ``problem``."""
        return _System(problem, self.chunk_size, self.device, self._pattern_cache)

    def build_amg(self, sys_: _System, A: sp.bsr_matrix) -> _AMGCG:
        Z = sys_.deflation(self.soft_ratio, self.max_components) if self.deflation else None
        return _AMGCG(A, sys_.rbm, self.amg_options, Z)

    def solve(self, problem: Problem, prescribed: np.ndarray | None = None) -> dict[str, Solution]:
        """Solve every load case of ``problem``.

        ``prescribed`` optionally gives (N, 3) displacement values for the
        constrained dofs (inhomogeneous Dirichlet data, e.g. patch tests);
        values on free dofs are ignored. By default constrained dofs are zero.
        """
        t0 = time.perf_counter()
        sys_ = self.prepare(problem)
        n = sys_.n_nodes
        F = [assemble_load(problem, c, sys_.vol) for c in problem.load_cases]
        free = 1.0 - sys_.cons_np
        uD = None if prescribed is None else np.asarray(prescribed, dtype=float) * sys_.cons_np
        if self.method == "amg":
            U, infos = self._solve_amg(sys_, F, free, uD)
        else:
            U, infos = self._solve_jax_cg(sys_, F, free, uD)
        out = {}
        for case, u, info in zip(problem.load_cases, U, infos, strict=True):
            eps, sig, sed = sys_.recover(u)
            info.update(solver=self.name, method=self.method, n_dofs=3 * n,
                        n_free_dofs=int(free.sum()), time_s=time.perf_counter() - t0)
            out[case.name] = Solution(u=u, strain=eps, stress=sig, sed=sed, info=info)
        return out

    def _unconstrained_K_times(self, sys_: _System, u: np.ndarray) -> np.ndarray:
        """``K u`` (N, 3) of the unconstrained operator (moves Dirichlet data to the rhs)."""
        ones = jnp.ones((sys_.n_nodes, 3))
        return np.asarray(_matvec(sys_.grads, sys_.C, sys_.vol_j, sys_.tets, ones,
                                  jnp.zeros_like(ones), jnp.asarray(u)))

    # -- assembled AMG-CG
    def _solve_amg(self, sys_: _System, F, free, uD):
        t0 = time.perf_counter()
        A = sys_.bsr(sys_.assemble())
        t_asm = time.perf_counter() - t0
        amg = self.build_amg(sys_, A)
        dcons = sys_.cons_np * A.diagonal().reshape(-1, 3)
        U, infos = [], []
        for f in F:
            ts = time.perf_counter()
            b = free * f
            if uD is not None:
                b = b - free * self._unconstrained_K_times(sys_, uD) + dcons * uD
            x, it, rel = amg.solve(b.ravel(), self.rtol, self.maxiter)
            u = free * x.reshape(-1, 3) + (0.0 if uD is None else uD)
            infos.append(self._info(it, rel, assembly_s=t_asm, setup_s=amg.setup_time,
                                    solve_s=time.perf_counter() - ts, amg_levels=amg.levels,
                                    operator_complexity=amg.operator_complexity,
                                    deflation_vectors=amg.n_deflation))
            U.append(u)
        return U, infos

    # -- matrix-free JAX CG
    def _solve_jax_cg(self, sys_: _System, F, free, uD):
        t0 = time.perf_counter()
        D, keep = _block_diagonal(sys_.grads, sys_.C, sys_.vol_j, sys_.tets, sys_.cons,
                                  chunk=sys_.chunk)
        dcons = jnp.diagonal(D, axis1=1, axis2=2) * (1.0 - keep)
        Dinv = _inv3(D)
        Dinv.block_until_ready()
        t_setup = time.perf_counter() - t0
        U, infos = [], []
        for f in F:
            ts = time.perf_counter()
            b = jnp.asarray(free * f)
            if uD is not None:
                b = b - keep * jnp.asarray(self._unconstrained_K_times(sys_, uD)) \
                    + dcons * jnp.asarray(uD)
            x, it, _, rel = _jax_cg_solve(sys_.grads, sys_.C, sys_.vol_j, sys_.tets, keep, dcons,
                                          Dinv, b, jnp.zeros_like(b), self.rtol,
                                          maxiter=self.maxiter)
            u = free * np.asarray(x) + (0.0 if uD is None else uD)
            infos.append(self._info(int(it), float(rel), setup_s=t_setup,
                                    solve_s=time.perf_counter() - ts, device=str(sys_.device)))
            U.append(u)
        return U, infos

    def _info(self, it: int, rel: float, **kw) -> dict:
        converged = bool(rel <= self.rtol * (1 + 1e-6)) or it == 0
        if not converged:
            warnings.warn(f"{self.method} CG did not converge: relative residual {rel:.2e} "
                          f"after {it} iterations", RuntimeWarning, stacklevel=3)
        return {"iterations": int(it), "relative_residual": float(rel),
                "converged": converged, **kw}


# --------------------------------------------------------------------------- differentiable map

class DifferentiableSolver:
    """Displacements and element fields as a differentiable JAX function.

    ``solver(modulus_scale, load_scale)`` returns a dict of JAX arrays

    ============  =====================  ======================================
    ``u``         (n_cases, N, 3)        nodal displacement
    ``strain``    (n_cases, M, 6)        Voigt, engineering shear
    ``stress``    (n_cases, M, 6)        ``s_e C_e strain``
    ``sed``       (n_cases, M)           ``0.5 stress . strain``
    ============  =====================  ======================================

    for the scaled problem ``C_e -> s_e C_e`` (``modulus_scale``, (M,)) and
    load cases multiplied by ``load_scale`` (n_cases,). The call is
    traceable (``jax.jit``, ``jax.grad``, ``jax.vjp``); reverse-mode
    gradients use one adjoint solve per load case,

        dL/ds_e = -sum_c V_e eps(lambda_c)_e . C_e eps(u_c)_e,   K(s) lambda_c = dL/du_c,

    and ``dL/dl_c = lambda_c . f_c``. Pressure magnitudes are load scales of
    a case that contains a single pressure set, so calibrating several
    pressures means splitting them into separate cases.

    With ``method="amg"`` the linear solves run on the host through
    ``jax.pure_callback`` (AMG-preconditioned CG). The hierarchy and
    deflation space are built at the first call and, with
    ``reuse_preconditioner=True``, reused for later scales; CG still
    converges to ``rtol`` on the current operator, only the iteration count
    changes. With ``method="jax-cg"`` the solves are matrix-free JAX CG and
    the whole map stays on device. Only first-order reverse mode is defined.

    ``history`` collects the iteration count and residual of every solve.
    Remaining keyword arguments are those of :class:`JaxSolver`.
    """

    def __init__(self, problem: Problem, method: str = "amg", rtol: float = 1e-10,
                 reuse_preconditioner: bool = True, **solver_kwargs):
        self._base = JaxSolver(method=method, rtol=rtol, **solver_kwargs)
        self.method, self.rtol, self.maxiter = method, self._base.rtol, self._base.maxiter
        self.reuse_preconditioner = reuse_preconditioner
        self.sys = self._base.prepare(problem)
        self.case_names: tuple[str, ...] = tuple(c.name for c in problem.load_cases)
        self.n_elems, self.n_nodes = self.sys.n_elems, self.sys.n_nodes
        free = 1.0 - self.sys.cons_np
        F0 = np.stack([free * assemble_load(problem, c, self.sys.vol)
                       for c in problem.load_cases])
        self.F0 = jax.device_put(F0, self.sys.device)
        self.free = jax.device_put(free, self.sys.device)
        self.history: list[dict] = []
        self._amg: _AMGCG | None = None
        self._solve = self._make_solve()

    # -- linear solves K(s) X = R for a stack of right-hand sides R (n, N, 3)
    def _linear_solve(self, s_pad, R):
        if self.method == "amg":
            data = self.sys.assemble(s_pad)
            return jax.pure_callback(self._host_solve, jax.ShapeDtypeStruct(R.shape, R.dtype),
                                     data, R, vmap_method="sequential")
        S = self.sys
        D, keep = _block_diagonal(S.grads, S.C, S.vol_j * s_pad, S.tets, S.cons,
                                  chunk=S.chunk)
        dcons = jnp.diagonal(D, axis1=1, axis2=2) * (1.0 - keep)
        Dinv = _inv3(D)

        def one(b):
            x, it, _, rel = _jax_cg_solve(S.grads, S.C, S.vol_j * s_pad, S.tets, keep, dcons,
                                          Dinv, b, jnp.zeros_like(b), self.rtol,
                                          maxiter=self.maxiter)
            jax.debug.callback(self._record, it, rel)
            return x

        return lax.map(one, R)

    def _record(self, it, rel) -> None:
        self.history.append({"iterations": int(it), "relative_residual": float(rel)})

    def _host_solve(self, data, R):
        A = self.sys.bsr(np.asarray(data))
        if self._amg is None or not self.reuse_preconditioner:
            self._amg = self._base.build_amg(self.sys, A)
        else:
            self._amg.set_operator(A)
        out = np.zeros(R.shape)
        for i, r in enumerate(np.asarray(R)):
            x, it, rel = self._amg.solve(r.ravel(), self.rtol, self.maxiter)
            out[i] = x.reshape(r.shape)
            self._record(it, rel)
        return out

    def _make_solve(self):
        S = self.sys
        pair = jax.vmap(lambda lam, u: _pair_energy(S.grads, S.C, S.vol_j, S.tets, lam, u))

        @jax.custom_vjp
        def solve(s_pad, F):
            return self._linear_solve(s_pad, F)

        def fwd(s_pad, F):
            U = self._linear_solve(s_pad, F)
            return U, (s_pad, U)

        def bwd(res, gU):
            s_pad, U = res
            Lam = self._linear_solve(s_pad, gU * self.free)
            return -pair(Lam, U).sum(axis=0), Lam

        solve.defvjp(fwd, bwd)
        return solve

    def __call__(self, modulus_scale, load_scale=None) -> dict[str, jnp.ndarray]:
        S = self.sys
        s = jnp.asarray(modulus_scale, dtype=jnp.float64)
        if s.shape != (self.n_elems,):
            raise ValueError(f"modulus_scale must have shape ({self.n_elems},), got {s.shape}")
        F = self.F0
        if load_scale is not None:
            F = jnp.asarray(load_scale, dtype=jnp.float64)[:, None, None] * F
        U = self._solve(S.pad(s), F)
        eps, sig, sed = jax.vmap(lambda u: kernels.recover(S.grads, S.C, S.tets, u))(U)
        M = self.n_elems
        return {"u": U, "strain": eps[:, :M], "stress": sig[:, :M] * s[None, :, None],
                "sed": sed[:, :M] * s[None]}

    def solution(self, out: dict, case: int | str = 0) -> Solution:
        """Package one case of a call result as a :class:`~craniofea.model.Solution`."""
        i = self.case_names.index(case) if isinstance(case, str) else int(case)
        return Solution(u=np.asarray(out["u"][i]), strain=np.asarray(out["strain"][i]),
                        stress=np.asarray(out["stress"][i]), sed=np.asarray(out["sed"][i]),
                        info={"solver": "jax", "method": self.method})


def make_differentiable_solver(problem: Problem, *, method: str = "amg", rtol: float = 1e-10,
                               **kwargs) -> DifferentiableSolver:
    """Build the differentiable map ``(modulus_scale, load_scale) -> {u, strain, stress, sed}``.

    Example
    -------
    >>> fem = make_differentiable_solver(problem)
    >>> def objective(s):
    ...     return fem(s)["sed"].sum()
    >>> g = jax.grad(objective)(jnp.ones(problem.mesh.n_elems))

    See :class:`DifferentiableSolver` for the adjoint formulation and
    keyword arguments.
    """
    return DifferentiableSolver(problem, method=method, rtol=rtol, **kwargs)
