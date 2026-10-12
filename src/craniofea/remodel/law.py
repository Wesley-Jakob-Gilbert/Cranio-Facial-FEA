"""Site-specific homeostatic remodeling law with a lazy zone.

Setpoint (attractor state, Beaupré, Orr & Carter 1990): each element's
homeostatic stimulus Psi*_e is its daily stimulus under the ``reference``
habit at the baseline density rho = 1. There is no global, hand-tuned
setpoint: the reference habit is in equilibrium by construction and every
other habit is evaluated relative to it.

Error and rate (trilinear "dead zone" law compared by Beaupré, Orr & Carter
1990, the "lazy zone" of Huiskes et al. 1987; written here for a relative
density instead of a surface apposition rate):

    S_e = Psi_e / Psi*_e - 1
    drho_e/dt = c_form (S_e - w)   if S_e >  w
              = 0                  if |S_e| <= w
              = c_res  (S_e + w)   if S_e < -w        [1/day]

Explicit Euler with step dt (days), re-solving the FE problem every step:

    rho^{k+1} = clip(rho^k + dt * drho/dt(rho^k), rho_min, rho_max)

Optional smoothing of S (default on): S is averaged to nodes (volume
weighted, bone elements only) and back to elements (mean of the 4 nodes).
This is a linear filter with a one-element-ring support that damps the
element-scale checkerboard pattern known to arise in element-based density
laws (see e.g. Garijo et al. 2014 on stability of remodeling schemes). It
leaves a spatially uniform S unchanged and maps S = 0 to 0.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from craniofea.model import Mesh


def stimulus_error(psi: np.ndarray, psi_star: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """S = Psi / Psi* - 1; elements with Psi* <= floor * max(Psi*) get S = 0.

    Unloaded elements carry no mechanical information relative to the
    reference habit and are held at their current density.
    """
    psi = np.asarray(psi, float)
    psi_star = np.asarray(psi_star, float)
    ok = psi_star > floor * psi_star.max()
    S = np.zeros_like(psi)
    S[ok] = psi[ok] / psi_star[ok] - 1.0
    return S


def nodal_smoother(mesh: Mesh, volume: np.ndarray, active: np.ndarray) -> sp.csr_matrix:
    """(M, M) sparse operator: element -> node (volume weighted over `active`) -> element."""
    M, N = mesh.n_elems, mesh.n_nodes
    rows = mesh.tets.ravel()
    cols = np.repeat(np.arange(M), 4)
    w = np.repeat(np.where(active, volume, 0.0), 4)
    A = sp.csr_matrix((w, (rows, cols)), shape=(N, M))           # node <- element
    wsum = np.asarray(A.sum(axis=1)).ravel()
    inv = np.where(wsum > 0, 1.0 / np.where(wsum > 0, wsum, 1.0), 0.0)
    A = sp.diags(inv) @ A
    B = sp.csr_matrix((np.full(4 * M, 0.25), (cols, rows)), shape=(M, N))  # element <- node
    return (B @ A).tocsr()


def remodel_rate(S: np.ndarray, lazy_zone: float, c_form: np.ndarray,
                 c_res: np.ndarray) -> np.ndarray:
    """Trilinear rate drho/dt (1/day) with dead zone half-width `lazy_zone`."""
    S = np.asarray(S, float)
    up = S > lazy_zone
    dn = S < -lazy_zone
    rate = np.zeros_like(S)
    rate[up] = (np.broadcast_to(c_form, S.shape)[up]) * (S[up] - lazy_zone)
    rate[dn] = (np.broadcast_to(c_res, S.shape)[dn]) * (S[dn] + lazy_zone)
    return rate


def update_density(rho: np.ndarray, rate: np.ndarray, dt: float,
                   rho_min: np.ndarray, rho_max: np.ndarray) -> np.ndarray:
    """Explicit Euler step with clamping to [rho_min, rho_max]."""
    return np.clip(np.asarray(rho, float) + dt * np.asarray(rate, float), rho_min, rho_max)
