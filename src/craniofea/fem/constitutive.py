"""Linear-elastic constitutive matrices in Voigt notation.

Conventions (see :mod:`craniofea.model`)
-----------------------------------------
Voigt order is ``[xx, yy, zz, yz, xz, xy]``. Stress holds tensor components,
strain holds *engineering* shear strains (``gamma_ij = 2 eps_ij``), so that

    sigma = C @ eps,        w = 0.5 * sigma . eps

with ``C`` the symmetric 6x6 stiffness. All functions are vectorised over
arbitrary leading dimensions ``(...)``, e.g. one matrix per element.

Material frames
---------------
Anisotropic constants are given in a local orthonormal frame
``(e1, e2, e3)``. Throughout the package that frame is stored as an
``axes`` array of shape ``(..., 3, 3)`` whose *rows* are ``e1, e2, e3``
expressed in global coordinates, so ``x_local = axes @ x_global``.
:func:`rotate_C` maps a local stiffness to the global frame with the Bond
transformation (Bond 1943; Ting, *Anisotropic Elasticity*, 1996, ch. 2).
"""
from __future__ import annotations

import numpy as np

# Voigt index -> tensor index pair, in the order [xx, yy, zz, yz, xz, xy].
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def isotropic_C(E, nu) -> np.ndarray:
    """(..., 6, 6) isotropic stiffness for engineering-shear Voigt strain.

    ``C = lambda 1 (x) 1 + 2 mu I`` with the Lame constants
    ``lambda = E nu / ((1 + nu)(1 - 2 nu))`` and ``mu = E / (2 (1 + nu))``;
    in engineering-shear Voigt form the shear diagonal is ``mu``.
    """
    E = np.asarray(E, dtype=float)
    nu = np.asarray(nu, dtype=float)
    E, nu = np.broadcast_arrays(E, nu)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.zeros(E.shape + (6, 6))
    C[..., :3, :3] = lam[..., None, None]
    for i in range(3):
        C[..., i, i] += 2 * mu
        C[..., 3 + i, 3 + i] = mu
    return C


def orthotropic_S(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23) -> np.ndarray:
    """(..., 6, 6) orthotropic compliance ``S = C^-1`` in the material frame.

    ``nu_ij`` is the Poisson ratio for loading along ``e_i`` and contraction
    along ``e_j`` (``nu_ij = -eps_j / eps_i`` under uniaxial ``sigma_i``), so
    symmetry of ``S`` gives ``nu_ji = nu_ij E_j / E_i``::

        S = [[ 1/E1,     -nu12/E1, -nu13/E1,  0,     0,     0    ],
             [-nu12/E1,   1/E2,    -nu23/E2,  0,     0,     0    ],
             [-nu13/E1,  -nu23/E2,  1/E3,     0,     0,     0    ],
             [ 0,         0,        0,        1/G23, 0,     0    ],
             [ 0,         0,        0,        0,     1/G13, 0    ],
             [ 0,         0,        0,        0,     0,     1/G12]]
    """
    args = np.broadcast_arrays(*(np.asarray(a, dtype=float)
                                 for a in (E1, E2, E3, G12, G13, G23, nu12, nu13, nu23)))
    E1, E2, E3, G12, G13, G23, nu12, nu13, nu23 = args
    S = np.zeros(E1.shape + (6, 6))
    S[..., 0, 0] = 1.0 / E1
    S[..., 1, 1] = 1.0 / E2
    S[..., 2, 2] = 1.0 / E3
    S[..., 0, 1] = S[..., 1, 0] = -nu12 / E1
    S[..., 0, 2] = S[..., 2, 0] = -nu13 / E1
    S[..., 1, 2] = S[..., 2, 1] = -nu23 / E2
    S[..., 3, 3] = 1.0 / G23
    S[..., 4, 4] = 1.0 / G13
    S[..., 5, 5] = 1.0 / G12
    return S


def orthotropic_C(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23) -> np.ndarray:
    """(..., 6, 6) orthotropic stiffness in the material frame.

    Obtained by inverting the compliance of :func:`orthotropic_S`. The
    constants must give a positive-definite compliance (thermodynamic
    stability, e.g. ``|nu_ij| < sqrt(E_i / E_j)`` and
    ``1 - nu12 nu21 - nu23 nu32 - nu13 nu31 - 2 nu21 nu32 nu13 > 0``);
    otherwise a ``ValueError`` is raised.
    """
    S = orthotropic_S(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23)
    # The normal block decouples from the diagonal shear block.
    if np.any(np.linalg.eigvalsh(S[..., :3, :3]) <= 0) or np.any(np.diagonal(S, 0, -2, -1) <= 0):
        raise ValueError("orthotropic constants do not define a positive-definite stiffness")
    C = np.zeros_like(S)
    C[..., :3, :3] = np.linalg.inv(S[..., :3, :3])
    for i in range(3, 6):
        C[..., i, i] = 1.0 / S[..., i, i]
    return C


def transversely_isotropic_C(E_p, E_t, nu_p, nu_pt, G_t) -> np.ndarray:
    """(..., 6, 6) transversely isotropic stiffness with symmetry axis ``e3``.

    Parameters
    ----------
    E_p, nu_p : in-plane (``e1``-``e2``) Young's modulus and Poisson ratio.
    E_t : modulus along the symmetry axis ``e3``.
    nu_pt : Poisson ratio for in-plane loading and contraction along ``e3``
        (``nu13 = nu23``); the reciprocal ratio is ``nu_tp = nu_pt E_t / E_p``.
    G_t : shear modulus in planes containing ``e3`` (``G13 = G23``).

    The in-plane shear modulus follows from isotropy in the plane,
    ``G12 = E_p / (2 (1 + nu_p))``. For bone the symmetry axis is usually
    the principal (osteonal / trabecular) direction.
    """
    E_p = np.asarray(E_p, dtype=float)
    nu_p = np.asarray(nu_p, dtype=float)
    G12 = E_p / (2.0 * (1.0 + nu_p))
    return orthotropic_C(E_p, E_p, E_t, G12, G_t, G_t, nu_p, nu_pt, nu_pt)


def bond_matrix(a) -> np.ndarray:
    """(..., 6, 6) Bond stress-transformation matrix of a rotation ``a``.

    For tensors related by ``sigma = a sigma' a^T`` the Voigt vectors obey
    ``sigma_v = M sigma'_v`` with

        M[I, J] = a[i,k] a[j,k]                      (J = (k, k))
        M[I, J] = a[i,k] a[j,l] + a[i,l] a[j,k]      (J = (k, l), k != l)

    where ``I = (i, j)``. Energy invariance then gives the engineering-strain
    transformation ``eps_v = M^-T eps'_v``. The map is a group homomorphism,
    ``M(a b) = M(a) M(b)``.
    """
    a = np.asarray(a, dtype=float)
    idx = np.array(VOIGT_PAIRS)
    row_i, row_j = idx[:, 0, None], idx[:, 1, None]   # I = (i, j) along rows
    col_k, col_l = idx[None, :, 0], idx[None, :, 1]   # J = (k, l) along columns
    aik = a[..., row_i, col_k]
    ajl = a[..., row_j, col_l]
    ail = a[..., row_i, col_l]
    ajk = a[..., row_j, col_k]
    return np.where(col_k != col_l, aik * ajl + ail * ajk, aik * ajl)


def rotate_C(C, axes) -> np.ndarray:
    """Rotate a local-frame stiffness into the global frame.

    Parameters
    ----------
    C : (..., 6, 6) stiffness expressed in the material frame.
    axes : (..., 3, 3) material axes as rows in global coordinates
        (``axes[..., i, :] = e_i``), proper orthogonal.

    Returns
    -------
    (..., 6, 6) global stiffness ``C_g = M C M^T`` with ``M`` the Bond
    matrix of ``a = axes^T`` (the local-to-global rotation, ``x = a x'``).
    ``rotate_C(rotate_C(C, R), swapaxes(R))`` recovers ``C``.
    """
    axes = np.asarray(axes, dtype=float)
    M = bond_matrix(np.swapaxes(axes, -1, -2))
    return M @ np.asarray(C, dtype=float) @ np.swapaxes(M, -1, -2)
