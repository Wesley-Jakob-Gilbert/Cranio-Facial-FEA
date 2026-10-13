"""Vectorised JAX kernels for the linear tetrahedron (tet4 / C3D4).

For a tetrahedron with vertices ``x_0..x_3`` the barycentric shape functions
``N_a`` are linear, so their gradients ``g_a = grad N_a`` and the strain are
constant per element. With the edge matrix ``D`` whose rows are
``x_1 - x_0, x_2 - x_0, x_3 - x_0``,

    det D = 6 V,        [g_1 g_2 g_3] = D^-1,        g_0 = -(g_1 + g_2 + g_3).

The columns of ``D^-1`` are cofactor cross products, e.g.
``g_1 = (d_1 x d_2) / det D``, which avoids a batched LU factorisation.

Strain (engineering shear, Voigt ``[xx, yy, zz, yz, xz, xy]``) is
``eps = sum_a B_a u_a`` with the 6x3 nodal blocks

    B_a = [[gx, 0, 0], [0, gy, 0], [0, 0, gz], [0, gz, gy], [gz, 0, gx], [gy, gx, 0]],

and the element stiffness is ``K_e = V B^T C B`` (exact one-point
integration). The internal force of node ``a`` is ``f_a = V B_a^T sigma``,
i.e. ``f_a,i = V sigma_ij g_a,j``; the matrix-free operator uses this form.

Every function takes arrays with a leading element axis and contains no
Python loop over elements.
"""
from __future__ import annotations

import jax.numpy as jnp


def tet4_geometry(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Shape-function gradients (M, 4, 3) and signed volumes (M,) from (M, 4, 3) coordinates."""
    d = x[:, 1:] - x[:, :1]
    c = jnp.stack([jnp.cross(d[:, 1], d[:, 2]),
                   jnp.cross(d[:, 2], d[:, 0]),
                   jnp.cross(d[:, 0], d[:, 1])], axis=1)
    det = jnp.einsum("mi,mi->m", d[:, 0], c[:, 0])
    g = c / det[:, None, None]
    grads = jnp.concatenate([-g.sum(axis=1, keepdims=True), g], axis=1)
    return grads, det / 6.0


def b_blocks(grads: jnp.ndarray) -> jnp.ndarray:
    """Strain-displacement blocks ``B_a`` (M, 4, 6, 3), engineering shear."""
    gx, gy, gz = grads[..., 0], grads[..., 1], grads[..., 2]
    z = jnp.zeros_like(gx)
    rows = [
        [gx, z, z],
        [z, gy, z],
        [z, z, gz],
        [z, gz, gy],
        [gz, z, gx],
        [gy, gx, z],
    ]
    return jnp.stack([jnp.stack(r, axis=-1) for r in rows], axis=-2)


def stiffness_blocks(grads: jnp.ndarray, C: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Nodal 3x3 blocks ``K_ab = w B_a^T C B_b`` of the element stiffness, (M, 4, 4, 3, 3).

    ``w`` is the integration weight, normally the element volume (times an
    optional per-element modulus scale).
    """
    B = b_blocks(grads)
    CB = jnp.einsum("mkl,mblj->mbkj", C, B)
    return jnp.einsum("maki,mbkj->mabij", B, CB) * w[:, None, None, None, None]


def element_stiffness(grads: jnp.ndarray, C: jnp.ndarray, vol: jnp.ndarray) -> jnp.ndarray:
    """Element stiffness matrices (M, 12, 12) with dof order ``[u0x, u0y, u0z, u1x, ...]``."""
    kb = stiffness_blocks(grads, C, vol)
    return jnp.transpose(kb, (0, 1, 3, 2, 4)).reshape(-1, 12, 12)


def strain(grads: jnp.ndarray, ue: jnp.ndarray) -> jnp.ndarray:
    """Element strain (M, 6) from nodal displacements ``ue`` (M, 4, 3)."""
    H = jnp.einsum("mai,maj->mij", ue, grads)          # H_ij = du_i / dx_j
    return jnp.stack([H[:, 0, 0], H[:, 1, 1], H[:, 2, 2],
                      H[:, 1, 2] + H[:, 2, 1],
                      H[:, 0, 2] + H[:, 2, 0],
                      H[:, 0, 1] + H[:, 1, 0]], axis=-1)


def voigt_to_tensor(s: jnp.ndarray) -> jnp.ndarray:
    """(M, 6) stress-like Voigt vectors to symmetric (M, 3, 3) tensors."""
    return jnp.stack([jnp.stack([s[:, 0], s[:, 5], s[:, 4]], -1),
                      jnp.stack([s[:, 5], s[:, 1], s[:, 3]], -1),
                      jnp.stack([s[:, 4], s[:, 3], s[:, 2]], -1)], axis=-2)


def internal_forces(grads: jnp.ndarray, sigma_w: jnp.ndarray) -> jnp.ndarray:
    """Nodal forces ``f_a = B_a^T sigma_w`` (M, 4, 3) for volume-weighted stress (M, 6)."""
    return jnp.einsum("mij,maj->mai", voigt_to_tensor(sigma_w), grads)


def apply_stiffness(grads, C, w, tets, u):
    """Matrix-free ``K u`` (N, 3): gather, constitutive update, scatter-add."""
    eps = strain(grads, u[tets])
    sig = jnp.einsum("mij,mj->mi", C, eps) * w[:, None]
    fe = internal_forces(grads, sig)
    return jnp.zeros_like(u).at[tets.reshape(-1)].add(fe.reshape(-1, 3))


def recover(grads, C, tets, u):
    """Element strain, stress and strain-energy density for nodal displacements ``u`` (N, 3)."""
    eps = strain(grads, u[tets])
    sig = jnp.einsum("mij,mj->mi", C, eps)
    return eps, sig, 0.5 * jnp.einsum("mi,mi->m", sig, eps)
