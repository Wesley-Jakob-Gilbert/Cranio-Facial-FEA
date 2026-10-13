"""Verification problems with closed-form solutions, shared by tests and scripts.

Each builder returns a :class:`~craniofea.model.Problem` together with the
exact solution, so solver output can be compared against analysis:

* :func:`lame_cylinder` - thick-walled cylinder under internal pressure in
  plane strain (Lame 1852; Timoshenko & Goodier, *Theory of Elasticity*,
  sec. 28), meshed as a quarter annulus with symmetry conditions.
* :func:`cantilever` - end-loaded cantilever against Timoshenko beam theory
  with Cowper's (1966) shear coefficient for a rectangle.
* :func:`hanging_column` - prismatic column hanging under its own weight.
* :func:`jittered_box` - irregular tet mesh for patch tests.

:func:`l2_error` integrates errors with a conical-product Gauss rule that
is exact for polynomials of degree ``2 n - 3`` on each tetrahedron.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from craniofea.fem.constitutive import isotropic_C
from craniofea.mesh.primitives import (
    boundary_faces,
    box_mesh,
    face_normals_areas,
    orient_tets,
    tet_signed_volumes,
)
from craniofea.model import LoadCase, Material, Mesh, Problem


@dataclass
class VerificationCase:
    problem: Problem
    h: float                                              # characteristic element size
    u_exact: Callable[[np.ndarray], np.ndarray] | None    # (P, 3) points -> (P, 3)
    stress_exact: Callable[[np.ndarray], np.ndarray] | None  # (P, 3) -> (P, 6) Voigt
    meta: dict


# --------------------------------------------------------------------------- quadrature / errors

def tet_quadrature(n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric points (n^3, 4) and weights (n^3,) summing to 1 on a tetrahedron.

    Collapsed (Duffy) coordinates ``z = c, y = b (1 - c), x = a (1 - b)(1 - c)``
    with ``n``-point Gauss-Legendre rules in ``a, b, c``.
    """
    g, w = np.polynomial.legendre.leggauss(n)
    g, w = 0.5 * (g + 1.0), 0.5 * w
    A, B, Cc = np.meshgrid(g, g, g, indexing="ij")
    WA, WB, WC = np.meshgrid(w, w, w, indexing="ij")
    z = Cc
    y = B * (1 - Cc)
    x = A * (1 - B) * (1 - Cc)
    wt = WA * WB * WC * (1 - B) * (1 - Cc) ** 2 * 6.0
    lam = np.stack([1 - x - y - z, x, y, z], axis=-1).reshape(-1, 4)
    return lam, wt.ravel()


def l2_error(mesh: Mesh, u_h: np.ndarray, u_exact: Callable, n: int = 4) -> tuple[float, float]:
    """Absolute and relative L2 error of the piecewise-linear field ``u_h`` (N, 3)."""
    lam, wq = tet_quadrature(n)
    vol = tet_signed_volumes(mesh.nodes, mesh.tets)
    X = np.einsum("qa,mai->mqi", lam, mesh.nodes[mesh.tets])
    Uh = np.einsum("qa,mai->mqi", lam, u_h[mesh.tets])
    Ue = u_exact(X.reshape(-1, 3)).reshape(Uh.shape)
    w = vol[:, None] * wq[None]
    err = np.sqrt(np.sum(w * np.sum((Uh - Ue) ** 2, axis=-1)))
    ref = np.sqrt(np.sum(w * np.sum(Ue ** 2, axis=-1)))
    return float(err), float(err / ref)


def stress_l2_error(mesh: Mesh, sigma_h: np.ndarray, sigma_exact: Callable,
                    n: int = 4) -> tuple[float, float]:
    """Absolute and relative L2 error of element-constant stress (M, 6) (Frobenius norm)."""
    lam, wq = tet_quadrature(n)
    vol = tet_signed_volumes(mesh.nodes, mesh.tets)
    X = np.einsum("qa,mai->mqi", lam, mesh.nodes[mesh.tets])
    Se = sigma_exact(X.reshape(-1, 3)).reshape(X.shape[0], X.shape[1], 6)
    wv = np.array([1, 1, 1, 2, 2, 2], dtype=float)      # tensor (Frobenius) norm in Voigt form
    d = sigma_h[:, None, :] - Se
    w = vol[:, None] * wq[None]
    err = np.sqrt(np.sum(w * np.sum(wv * d ** 2, axis=-1)))
    ref = np.sqrt(np.sum(w * np.sum(wv * Se ** 2, axis=-1)))
    return float(err), float(err / ref)


def observed_rate(h: np.ndarray, e: np.ndarray) -> float:
    """Least-squares slope of ``log e`` against ``log h``."""
    return float(np.polyfit(np.log(h), np.log(e), 1)[0])


def traction_forces(mesh: Mesh, faces: np.ndarray, traction: np.ndarray) -> np.ndarray:
    """Consistent nodal forces (N, 3) of a uniform traction vector on boundary triangles."""
    _, area = face_normals_areas(mesh.nodes, faces)
    f = np.zeros((mesh.n_nodes, 3))
    for k in range(3):
        np.add.at(f, faces[:, k], (area / 3.0)[:, None] * np.asarray(traction)[None])
    return f


def _isotropic(mesh: Mesh, E: float, nu: float, rho: float = 1.0) -> Material:
    M = mesh.n_elems
    return Material(C=isotropic_C(np.full(M, E), nu), mass_density=np.full(M, rho),
                    E=np.full(M, E), nu=np.full(M, nu))


# --------------------------------------------------------------------------- Lame cylinder

def quarter_annulus_mesh(a: float, b: float, n_r: int, n_t: int, thickness: float) -> Mesh:
    """Quarter annulus ``a <= r <= b, 0 <= theta <= pi/2`` one element layer thick.

    A structured box in ``(r, theta, z)`` is mapped to Cartesian coordinates.
    Node sets: ``inner``, ``outer``, ``x0`` (theta = pi/2), ``y0`` (theta = 0);
    surface sets: ``inner``, ``outer``, ``all``.
    """
    box = box_mesh(1.0, 1.0, 1.0, n_r, n_t, 1)
    s, t, z = box.nodes.T
    r = a + (b - a) * s
    th = 0.5 * np.pi * t
    x, y = r * np.cos(th), r * np.sin(th)
    on_x0 = np.isclose(t, 1.0)
    x[on_x0] = 0.0
    nodes = np.column_stack([x, y, thickness * z])
    tets = orient_tets(nodes, box.tets)
    faces, _ = boundary_faces(nodes, tets)
    inner_n, outer_n = np.isclose(s, 0.0), np.isclose(s, 1.0)
    return Mesh(
        nodes=nodes, tets=tets,
        node_sets={"inner": np.flatnonzero(inner_n), "outer": np.flatnonzero(outer_n),
                   "x0": np.flatnonzero(on_x0), "y0": np.flatnonzero(np.isclose(t, 0.0))},
        surface_sets={"inner": faces[inner_n[faces].all(axis=1)],
                      "outer": faces[outer_n[faces].all(axis=1)], "all": faces},
        meta={"source": "quarter_annulus", "a": a, "b": b, "divisions": [n_r, n_t, 1]})


def lame_cylinder(level: int, a: float = 0.01, b: float = 0.02, p: float = 1e6,
                  E: float = 1e9, nu: float = 0.3) -> VerificationCase:
    """Thick-walled cylinder, internal pressure ``p``, plane strain.

    Exact solution with ``A = p a^2 / (b^2 - a^2)``:

        u_r = (1 + nu) A / E * ((1 - 2 nu) r + b^2 / r),
        sigma_rr = A (1 - b^2 / r^2),  sigma_tt = A (1 + b^2 / r^2),
        sigma_zz = 2 nu A.

    Symmetry: ``u_y = 0`` on ``theta = 0``, ``u_x = 0`` on ``theta = pi/2``;
    plane strain by ``u_z = 0`` everywhere. ``level`` k gives
    ``2k x 5k`` divisions in ``(r, theta)``.
    """
    n_r, n_t = 2 * level, 5 * level
    h = (b - a) / n_r
    mesh = quarter_annulus_mesh(a, b, n_r, n_t, thickness=h)
    fixed = np.zeros((mesh.n_nodes, 3), dtype=bool)
    fixed[:, 2] = True
    fixed[mesh.node_sets["y0"], 1] = True
    fixed[mesh.node_sets["x0"], 0] = True
    prob = Problem(mesh, _isotropic(mesh, E, nu), fixed_nodes=np.array([], dtype=np.int64),
                   fixed_dofs=fixed, load_cases=[LoadCase("pressure", pressures={"inner": p})])
    A = p * a**2 / (b**2 - a**2)

    def u_exact(X):
        r = np.hypot(X[:, 0], X[:, 1])
        ur = (1 + nu) * A / E * ((1 - 2 * nu) * r + b**2 / r)
        return np.column_stack([ur * X[:, 0] / r, ur * X[:, 1] / r, np.zeros(len(X))])

    def stress_exact(X):
        r2 = X[:, 0] ** 2 + X[:, 1] ** 2
        c2, s2, cs = X[:, 0] ** 2 / r2, X[:, 1] ** 2 / r2, X[:, 0] * X[:, 1] / r2
        srr, stt = A * (1 - b**2 / r2), A * (1 + b**2 / r2)
        z = np.zeros(len(X))
        return np.column_stack([srr * c2 + stt * s2, srr * s2 + stt * c2,
                                np.full(len(X), 2 * nu * A), z, z, (srr - stt) * cs])

    return VerificationCase(prob, h, u_exact, stress_exact,
                            {"a": a, "b": b, "p": p, "E": E, "nu": nu, "level": level})


# --------------------------------------------------------------------------- cantilever

def cantilever(level: int, L: float = 1.0, H: float = 0.1, E: float = 1e9, nu: float = 0.3,
               P: float = 100.0) -> VerificationCase:
    """Square-section cantilever ``L x H x H`` clamped at ``x = 0``, tip shear load ``P``.

    ``P`` is applied as a uniform traction ``-P / H^2`` (along ``-z``) on the
    free end. Timoshenko beam theory with Cowper's coefficient
    ``kappa = 10 (1 + nu) / (12 + 11 nu)`` gives the tip deflection

        delta = P L^3 / (3 E I) + P L / (kappa G A),   I = H^4 / 12,  A = H^2,

    reported in ``meta['delta_timoshenko']``. ``level`` k gives
    ``10k x k x k`` hexes (6 tets each).
    """
    nx, ny = 10 * level, level
    mesh = box_mesh(L, H, H, nx, ny, ny)
    f = traction_forces(mesh, mesh.surface_sets["xmax"], np.array([0.0, 0.0, -P / H**2]))
    prob = Problem(mesh, _isotropic(mesh, E, nu), fixed_nodes=mesh.node_sets["xmin"],
                   load_cases=[LoadCase("tip", nodal_forces=f)])
    G = E / (2 * (1 + nu))
    kappa = 10 * (1 + nu) / (12 + 11 * nu)
    I_, A_ = H**4 / 12.0, H**2
    bending, shear = P * L**3 / (3 * E * I_), P * L / (kappa * G * A_)
    return VerificationCase(prob, H / ny, None, None,
                            {"delta_timoshenko": bending + shear, "delta_bending": bending,
                             "delta_shear": shear, "tip_nodes": mesh.node_sets["xmax"],
                             "level": level})


# --------------------------------------------------------------------------- hanging column

def hanging_column(level: int, L: float = 1.0, W: float = 0.1, E: float = 1e7,
                   rho: float = 1000.0, g: float = 9.81) -> VerificationCase:
    """Column ``W x W x L`` hanging from its top face under gravity ``-g e_z``.

    With ``nu = 0`` the exact solution is uniaxial:
    ``sigma_zz = rho g z``, ``u_z = -rho g (L^2 - z^2) / (2 E)``,
    ``u_x = u_y = 0``. Supports: ``u_z = 0`` on the top face (roller) and
    symmetry rollers ``u_x = 0`` on ``x = 0`` and ``u_y = 0`` on ``y = 0``.
    ``level`` k gives ``k x k x 10k`` hexes.
    """
    nz, nx = 10 * level, level
    mesh = box_mesh(W, W, L, nx, nx, nz)
    fixed = np.zeros((mesh.n_nodes, 3), dtype=bool)
    fixed[mesh.node_sets["zmax"], 2] = True
    fixed[mesh.node_sets["xmin"], 0] = True
    fixed[mesh.node_sets["ymin"], 1] = True
    prob = Problem(mesh, _isotropic(mesh, E, 0.0, rho), fixed_nodes=np.array([], dtype=np.int64),
                   fixed_dofs=fixed,
                   load_cases=[LoadCase("gravity", gravity=np.array([0.0, 0.0, -g]))])

    def u_exact(X):
        out = np.zeros_like(X)
        out[:, 2] = -rho * g * (L**2 - X[:, 2] ** 2) / (2 * E)
        return out

    def stress_exact(X):
        out = np.zeros((len(X), 6))
        out[:, 2] = rho * g * X[:, 2]
        return out

    return VerificationCase(prob, L / nz, u_exact, stress_exact,
                            {"L": L, "E": E, "rho": rho, "g": g, "level": level})


# --------------------------------------------------------------------------- heterogeneous model

def random_rotations(n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 3, 3) uniformly distributed proper rotations (QR of Gaussian matrices)."""
    Q, R = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    Q = Q * np.sign(np.diagonal(R, axis1=1, axis2=2))[:, None, :]
    Q[np.linalg.det(Q) < 0, :, 0] *= -1
    return Q


def heterogeneous_box(divisions=(6, 4, 3), seed: int = 0) -> Problem:
    """Box with random orthotropic, randomly rotated elements and every load type.

    Elastic constants are drawn per element (E_i in 5-20 GPa, G_ij in
    2-6 GPa), material axes are random rotations, density varies. The
    ``xmin`` face is clamped and ``xmax`` is a roller in ``y``. Load cases:
    uniform plus per-face pressure, gravity, and random nodal forces.
    """
    from craniofea.fem.constitutive import orthotropic_C, rotate_C

    rng = np.random.default_rng(seed)
    m = box_mesh(0.03, 0.02, 0.015, *divisions)
    M = m.n_elems
    E = rng.uniform(5e9, 2e10, (3, M))
    G = rng.uniform(2e9, 6e9, (3, M))
    C_loc = orthotropic_C(*E, *G, 0.3, 0.25, 0.2)
    axes = random_rotations(M, rng)
    ortho = np.column_stack([*E, *G, np.full(M, 0.3), np.full(M, 0.25), np.full(M, 0.2)])
    mat = Material(C=rotate_C(C_loc, axes), mass_density=rng.uniform(1000, 2000, M),
                   ortho=ortho, axes=axes)
    roller = np.zeros((m.n_nodes, 3), dtype=bool)
    roller[m.node_sets["xmax"], 1] = True
    nf = np.zeros((m.n_nodes, 3))
    top = m.node_sets["zmax"]
    nf[top] = rng.normal(scale=50.0, size=(len(top), 3))
    cases = [
        LoadCase("pressure", pressures={"zmax": 1e6,
                                        "ymin": rng.uniform(0, 2e6, len(m.surface_sets["ymin"]))}),
        LoadCase("gravity", gravity=np.array([0.0, 0.0, -9.81e3])),
        LoadCase("nodal", nodal_forces=nf),
    ]
    return Problem(m, mat, fixed_nodes=m.node_sets["xmin"], fixed_dofs=roller, load_cases=cases)


def layered_box(divisions=(20, 14, 12), E_shell: float = 17e9, E_core: float = 0.5e9,
                E_suture: float = 50e6, nu: float = 0.3) -> Problem:
    """High-contrast box: stiff shell, soft core, compliant mid-plane slab.

    Mimics cortical / cancellous / suture contrast (340:10:1). The slab is
    two elements thick and cuts through shell and core at ``x = L / 2``; the
    ``xmin`` face is clamped and the top is loaded by 1 MPa.
    """
    dims = np.array([0.070, 0.050, 0.040])
    m = box_mesh(*dims, *divisions)
    xc = m.nodes[m.tets].mean(axis=1)
    h = dims[0] / divisions[0]
    depth = np.minimum(xc, dims - xc).min(axis=1)
    E = np.where(depth < 2 * h, E_shell, E_core)
    E = np.where(np.abs(xc[:, 0] - dims[0] / 2) < h, E_suture, E)
    mat = Material(C=isotropic_C(E, nu), mass_density=np.where(E == E_shell, 1900.0, 1100.0),
                   E=E, nu=np.full(m.n_elems, nu))
    cases = [LoadCase("top", pressures={"zmax": 1e6}, gravity=np.array([0.0, 0.0, -9.81])),
             LoadCase("side", pressures={"ymax": 5e5})]
    return Problem(m, mat, fixed_nodes=m.node_sets["xmin"], load_cases=cases)


# --------------------------------------------------------------------------- patch-test mesh

def jittered_box(n: int = 4, amplitude: float = 0.3, seed: int = 0) -> Mesh:
    """Unit-cube tet mesh with interior nodes randomly displaced by ``amplitude * h``.

    Boundary nodes stay on the faces, so face sets remain valid; every
    element keeps a positive volume.
    """
    m = box_mesh(1.0, 1.0, 1.0, n, n, n)
    rng = np.random.default_rng(seed)
    bnd = np.unique(m.surface_sets["all"])
    interior = np.setdiff1d(np.arange(m.n_nodes), bnd)
    m.nodes = m.nodes.copy()
    m.nodes[interior] += amplitude / n * rng.uniform(-1, 1, (len(interior), 3))
    if np.any(tet_signed_volumes(m.nodes, m.tets) <= 0):
        raise ValueError("jitter amplitude too large: inverted elements")
    m.node_sets["boundary"] = bnd
    m.node_sets["interior"] = interior
    return m
