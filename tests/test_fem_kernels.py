"""JAX element kernels against the NumPy reference implementation."""
import jax.numpy as jnp
import numpy as np

from craniofea.fem import jax_solver as js
from craniofea.fem import kernels
from craniofea.fem.reference import tet4_B, tet4_gradients
from craniofea.fem.verification import heterogeneous_box, jittered_box


def test_geometry_and_element_stiffness_match_reference():
    prob = heterogeneous_box()
    m = prob.mesh
    g_ref, v_ref = tet4_gradients(m.nodes, m.tets)
    g, v = kernels.tet4_geometry(jnp.asarray(m.nodes[m.tets]))
    assert np.allclose(g, g_ref, rtol=0, atol=1e-12 * np.abs(g_ref).max())
    assert np.allclose(v, v_ref, rtol=1e-12, atol=0)
    B = tet4_B(g_ref)
    K_ref = np.einsum("mki,mkl,mlj,m->mij", B, prob.material.C, B, v_ref)
    K = np.asarray(kernels.element_stiffness(g, jnp.asarray(prob.material.C), v))
    assert np.allclose(K, K_ref, rtol=1e-12, atol=1e-12 * np.abs(K_ref).max())


def test_element_stiffness_annihilates_rigid_motions():
    m = jittered_box(2)
    prob = heterogeneous_box()
    g, v = kernels.tet4_geometry(jnp.asarray(m.nodes[m.tets]))
    C = jnp.asarray(prob.material.C[: m.n_elems])
    K = np.asarray(kernels.element_stiffness(g, C, v))
    for e in range(m.n_elems):
        R = js.rigid_body_modes(m.nodes[m.tets[e]])
        assert np.abs(K[e] @ R).max() < 1e-9 * np.abs(K[e]).max()


def test_matrix_free_operator_equals_assembled_matrix():
    prob = heterogeneous_box()
    solver = js.JaxSolver()
    S = solver.prepare(prob)
    A = S.bsr(S.assemble())
    x = np.random.default_rng(0).normal(size=(prob.mesh.n_nodes, 3))
    D, keep = js._block_diagonal(S.grads, S.C, S.vol_j, S.tets, S.cons)
    dcons = jnp.diagonal(D, axis1=1, axis2=2) * (1 - keep)
    y = np.asarray(js._matvec(S.grads, S.C, S.vol_j, S.tets, keep, dcons, jnp.asarray(x)))
    assert np.allclose(y.ravel(), A @ x.ravel(), rtol=1e-12, atol=1e-12 * np.abs(y).max())
    assert np.allclose(np.asarray(D).reshape(-1, 3, 3)[:, [0, 1, 2], [0, 1, 2]].ravel(),
                       A.diagonal(), rtol=1e-12)
    # symmetric operator
    assert abs(A - A.T).max() <= 1e-12 * abs(A).max()


def test_chunked_assembly_is_independent_of_chunk_size():
    prob = heterogeneous_box()
    A1 = js.JaxSolver(chunk_size=37).prepare(prob)
    A2 = js.JaxSolver(chunk_size=100000).prepare(prob)
    d1, d2 = A1.bsr(A1.assemble()), A2.bsr(A2.assemble())
    assert abs(d1 - d2).max() <= 1e-12 * abs(d1).max()
