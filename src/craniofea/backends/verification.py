"""Cross-verification problem shared by the backend tests and documentation.

:func:`benchmark_problem` builds a small box that exercises every part of the
data model a backend has to translate: a continuous isotropic modulus field, a
rotated orthotropic layer (all 21 stiffness constants non-zero), clamped and
roller (per-dof) supports, a per-face pressure field, gravity with a varying
mass density, and point forces. :func:`compare` reports max relative
differences between two solutions.
"""
from __future__ import annotations

import numpy as np

from craniofea.backends._common import orthotropic_C, rotate_stiffness
from craniofea.fem.constitutive import isotropic_C
from craniofea.mesh.primitives import box_mesh
from craniofea.model import LoadCase, Material, Problem, Solution

# [E1, E2, E3, G12, G13, G23, nu12, nu13, nu23] of the orthotropic layer, Pa.
ORTHO_LAYER = np.array([17e9, 11e9, 9e9, 4e9, 3.5e9, 3e9, 0.30, 0.25, 0.35])


def rotation_matrix(axis, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about `axis` by `angle` radians."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def benchmark_problem(load_scale: float = 1.0, divisions=(6, 4, 3)) -> Problem:
    """Heterogeneous 20 x 10 x 8 mm box with three load cases.

    - elements with centroid z < 4 mm: isotropic, E = 1..11 GPa varying along x,
      nu = 0.3; elements above: orthotropic ``ORTHO_LAYER`` with material axes
      rotated by 0.7 rad about (1, 2, 3);
    - mass density 1800..2000 kg/m^3 varying along y;
    - xmin clamped, ymin on rollers (u_y = 0);
    - load cases ``pressure`` (linearly varying per-face pressure on zmax plus a
      uniform one on xmax), ``gravity`` and ``nodal`` (three point forces).

    `load_scale` multiplies every load; the maximum strain is about
    ``1e-3 * load_scale``.
    """
    m = box_mesh(0.02, 0.01, 0.008, *divisions)
    M = m.n_elems
    cent = m.nodes[m.tets].mean(axis=1)
    C = isotropic_C(1e9 * (1 + 10 * cent[:, 0] / 0.02), 0.3)
    layer = cent[:, 2] > 0.004
    R = rotation_matrix([1, 2, 3], 0.7)
    C[layer] = rotate_stiffness(orthotropic_C(ORTHO_LAYER)[0], R.T)
    axes = np.broadcast_to(np.eye(3), (M, 3, 3)).copy()
    axes[layer] = R
    ortho = np.full((M, 9), np.nan)
    ortho[layer] = ORTHO_LAYER
    mat = Material(C=C, mass_density=1800 + 200 * cent[:, 1] / 0.01, ortho=ortho, axes=axes)

    fixed_dofs = np.zeros((m.n_nodes, 3), dtype=bool)
    fixed_dofs[m.node_sets["ymin"], 1] = True
    forces = np.zeros((m.n_nodes, 3))
    forces[m.node_sets["xmax"][:3]] = [1.0, -2.0, 0.5]
    nz = len(m.surface_sets["zmax"])
    s = float(load_scale)
    cases = [
        LoadCase("pressure", pressures={"zmax": s * 1e5 * (1 + np.arange(nz) / nz),
                                        "xmax": s * 2e4}),
        LoadCase("gravity", gravity=s * np.array([0.0, -9.81e3, -1.962e4])),
        LoadCase("nodal", nodal_forces=s * forces),
    ]
    return Problem(m, mat, fixed_nodes=m.node_sets["xmin"], fixed_dofs=fixed_dofs,
                   load_cases=cases)


def compare(ref: Solution, other: Solution) -> dict[str, float]:
    """Max relative differences ``max|a - b| / max|a|`` per field."""
    def rel(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))

    return {"u": rel(ref.u, other.u), "stress": rel(ref.stress, other.stress),
            "strain": rel(ref.strain, other.strain), "sed": rel(ref.sed, other.sed)}
