"""Linear-elastic constitutive matrices in Voigt notation [xx, yy, zz, yz, xz, xy]."""
from __future__ import annotations

import numpy as np


def isotropic_C(E, nu) -> np.ndarray:
    """(…, 6, 6) isotropic stiffness for engineering-shear Voigt strain."""
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
