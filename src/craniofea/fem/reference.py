"""Small, dependency-light reference solver (NumPy + SciPy direct solve).

Deliberately simple and independent of the JAX implementation so it can act
as an oracle in tests. Intended for meshes up to ~1e5 dofs.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from craniofea.mesh.primitives import face_normals_areas
from craniofea.model import Problem, Solution


def tet4_gradients(nodes: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape-function gradients (M, 4, 3) and volumes (M,) of linear tets."""
    x = nodes[tets]
    X = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=1)  # rows
    vol = np.linalg.det(X) / 6.0
    G = np.transpose(np.linalg.inv(X), (0, 2, 1))     # row i = grad(lambda_{i+1})
    grads = np.concatenate([-G.sum(axis=1, keepdims=True), G], axis=1)
    return grads, vol


def tet4_B(grads: np.ndarray) -> np.ndarray:
    """Strain-displacement matrices (M, 6, 12), engineering shear."""
    M = grads.shape[0]
    B = np.zeros((M, 6, 12))
    for a in range(4):
        gx, gy, gz = grads[:, a, 0], grads[:, a, 1], grads[:, a, 2]
        c = 3 * a
        B[:, 0, c] = gx
        B[:, 1, c + 1] = gy
        B[:, 2, c + 2] = gz
        B[:, 3, c + 1], B[:, 3, c + 2] = gz, gy
        B[:, 4, c], B[:, 4, c + 2] = gz, gx
        B[:, 5, c], B[:, 5, c + 1] = gy, gx
    return B


def assemble_rhs(problem: Problem, case, vol: np.ndarray) -> np.ndarray:
    mesh = problem.mesh
    f = np.zeros((mesh.n_nodes, 3))
    for set_name, p in case.pressures.items():
        faces = mesh.surface_sets[set_name]
        n, a = face_normals_areas(mesh.nodes, faces)
        p = np.broadcast_to(np.asarray(p, dtype=float), (len(faces),))
        fn = -(p * a / 3.0)[:, None] * n
        for k in range(3):
            np.add.at(f, faces[:, k], fn)
    if case.gravity is not None:
        fe = (problem.material.mass_density * vol / 4.0)[:, None] * np.asarray(case.gravity)[None]
        for k in range(4):
            np.add.at(f, mesh.tets[:, k], fe)
    if case.nodal_forces is not None:
        f += case.nodal_forces
    return f.ravel()


class ReferenceSolver:
    name = "reference"

    def solve(self, problem: Problem) -> dict[str, Solution]:
        t0 = time.perf_counter()
        mesh = problem.mesh
        grads, vol = tet4_gradients(mesh.nodes, mesh.tets)
        B = tet4_B(grads)
        Ke = np.einsum("mki,mkl,mlj,m->mij", B, problem.material.C, B, vol)
        dofs = (3 * mesh.tets[:, :, None] + np.arange(3)).reshape(-1, 12)
        rows = np.repeat(dofs, 12, axis=1).ravel()
        cols = np.tile(dofs, (1, 12)).ravel()
        n = 3 * mesh.n_nodes
        K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
        free = ~problem.constrained_mask().ravel()
        Kff = K[free][:, free].tocsc()
        lu = spla.splu(Kff)
        out = {}
        for case in problem.load_cases:
            f = assemble_rhs(problem, case, vol)
            u = np.zeros(n)
            u[free] = lu.solve(f[free])
            ue = u[dofs]
            strain = np.einsum("mij,mj->mi", B, ue)
            stress = np.einsum("mij,mj->mi", problem.material.C, strain)
            sed = 0.5 * np.einsum("mi,mi->m", stress, strain)
            out[case.name] = Solution(u=u.reshape(-1, 3), strain=strain, stress=stress, sed=sed,
                                      info={"solver": self.name,
                                            "time_s": time.perf_counter() - t0})
        return out
