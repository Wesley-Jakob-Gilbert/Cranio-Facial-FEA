"""Tissue layers and relative density -> per-element linear-elastic `Material`.

Model
-----
Every element belongs to exactly one layer (elem sets ``cortical``,
``cancellous``, ``suture_mid``). Bone layers carry a relative density
``rho`` (1.0 = baseline tissue at the start of the study) that scales the
stiffness with the power law of Carter & Hayes (1977):

    E(rho) = E_layer * rho ** n_layer        (cortical n ~ 3, cancellous n ~ 2)

The quadratic exponent for cancellous bone follows Rice, Cowin & Bowman
(1988); the cubic one for compact bone follows Carter & Hayes (1977). For an
orthotropic cortical layer all Young's and shear moduli are scaled by the same
factor ``rho ** n`` and the Poisson ratios are kept, so the anisotropy ratios
do not change with density (a simplifying assumption).

Cortical orthotropy: local axis 3 is the outward surface normal (elem field
``surface_normal``); axis 1 is the anatomical ``ap`` axis projected onto the
local tangent plane (fallback ``lr`` then ``si`` when the projection is
degenerate); axis 2 = axis 3 x axis 1. The local Voigt stiffness is rotated to
global coordinates with the engineering-strain Bond transformation.

Sutures: constant isotropic modulus, not remodeled. The legacy model's
heuristic softening/ossification rules are deliberately not carried over:
they had no calibrated rate constants, and sutural growth biology is outside
the scope of a strain-adaptive density model (see docs/METHODS.md).

Mass density: bone apparent mass density scales linearly with ``rho``
(rho is a relative apparent density); the suture value is constant. Mass only
enters through gravity, which is a minor load case.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from craniofea.fem.constitutive import isotropic_C
from craniofea.model import AnatomicalFrame, Material, Mesh

if TYPE_CHECKING:
    from craniofea.config import MaterialsConfig

try:  # provided by craniofea.fem.constitutive once the orthotropic kernels land
    from craniofea.fem.constitutive import orthotropic_C, rotate_C
except ImportError:  # pragma: no cover - exercised only before the merge
    orthotropic_C = None
    rotate_C = None

LAYERS = ("cortical", "cancellous", "suture_mid")
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


@dataclass(frozen=True)
class OrthoConstants:
    """Orthotropic engineering constants at rho = 1 (Pa); axis 3 = surface normal."""

    E1: float
    E2: float
    E3: float
    G12: float
    G13: float
    G23: float
    nu12: float
    nu13: float
    nu23: float

    def as_array(self) -> np.ndarray:
        return np.array([self.E1, self.E2, self.E3, self.G12, self.G13, self.G23,
                         self.nu12, self.nu13, self.nu23], dtype=float)


@dataclass(frozen=True)
class LayerProps:
    """Elastic and mass properties of one tissue layer at rho = 1."""

    E: float                     # isotropic Young's modulus, Pa
    nu: float                    # Poisson ratio
    n: float = 0.0               # density exponent (0 -> density independent)
    mass_density: float = 1000.0  # kg/m^3 at rho = 1
    ortho: OrthoConstants | None = None   # if set, the layer is orthotropic


# --------------------------------------------------------------------------- #
# Constitutive helpers (private fallbacks for the shared constitutive module)
# --------------------------------------------------------------------------- #

def _orthotropic_C(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23) -> np.ndarray:
    """(..., 6, 6) orthotropic stiffness in its material axes (engineering shear).

    Inverse of the compliance with S_ij = -nu_ij / E_i (i != j, normal block),
    S_ii = 1 / E_i, S_44 = 1/G23, S_55 = 1/G13, S_66 = 1/G12.
    """
    args = np.broadcast_arrays(*(np.asarray(a, dtype=float)
                                 for a in (E1, E2, E3, G12, G13, G23, nu12, nu13, nu23)))
    E1, E2, E3, G12, G13, G23, nu12, nu13, nu23 = args
    S = np.zeros(E1.shape + (6, 6))
    S[..., 0, 0], S[..., 1, 1], S[..., 2, 2] = 1 / E1, 1 / E2, 1 / E3
    S[..., 0, 1] = S[..., 1, 0] = -nu12 / E1
    S[..., 0, 2] = S[..., 2, 0] = -nu13 / E1
    S[..., 1, 2] = S[..., 2, 1] = -nu23 / E2
    S[..., 3, 3], S[..., 4, 4], S[..., 5, 5] = 1 / G23, 1 / G13, 1 / G12
    return np.linalg.inv(S)


def _strain_rotation(axes: np.ndarray) -> np.ndarray:
    """(..., 6, 6) T with eps_local = T @ eps_global (engineering-shear Voigt).

    `axes` (..., 3, 3) holds the local axes as rows in global coordinates, so a
    vector transforms as v_local = axes @ v_global.
    """
    R = np.asarray(axes, dtype=float)
    T = np.empty(R.shape[:-2] + (6, 6))
    for row, (a, b) in enumerate(_VOIGT_PAIRS):
        f = 1.0 if a == b else 2.0          # engineering shear factor of the local strain
        for col, (p, q) in enumerate(_VOIGT_PAIRS):
            if p == q:
                T[..., row, col] = f * R[..., a, p] * R[..., b, p]
            else:
                T[..., row, col] = 0.5 * f * (R[..., a, p] * R[..., b, q]
                                              + R[..., a, q] * R[..., b, p])
    return T


def _rotate_C(C: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Rotate a local Voigt stiffness to global axes: C_g = T^T C_l T."""
    T = _strain_rotation(axes)
    return np.einsum("...ki,...kl,...lj->...ij", T, C, T)


_ortho_C = orthotropic_C if orthotropic_C is not None else _orthotropic_C
_rot_C = rotate_C if rotate_C is not None else _rotate_C


def hill_modulus(C: np.ndarray) -> np.ndarray:
    """Isotropic-equivalent Young's modulus (Voigt-Reuss-Hill) of (..., 6, 6) C.

    K_V = [C11+C22+C33 + 2(C12+C13+C23)] / 9
    G_V = [C11+C22+C33 - (C12+C13+C23) + 3(C44+C55+C66)] / 15
    1/K_R = S11+S22+S33 + 2(S12+S13+S23)
    15/G_R = 4(S11+S22+S33) - 4(S12+S13+S23) + 3(S44+S55+S66)
    K = (K_V+K_R)/2, G = (G_V+G_R)/2, E = 9KG / (3K+G)   (Hill 1952)

    Rotation invariant and exact (E) for isotropic C, where Voigt = Reuss.
    Voigt shear entries use engineering shear, so S44 = 1/G23 etc.
    """
    C = np.asarray(C, dtype=float)
    S = np.linalg.inv(C)
    d = np.diagonal(C, axis1=-2, axis2=-1)
    ds = np.diagonal(S, axis1=-2, axis2=-1)
    off = C[..., 0, 1] + C[..., 0, 2] + C[..., 1, 2]
    offs = S[..., 0, 1] + S[..., 0, 2] + S[..., 1, 2]
    KV = (d[..., :3].sum(-1) + 2 * off) / 9
    GV = (d[..., :3].sum(-1) - off + 3 * d[..., 3:].sum(-1)) / 15
    KR = 1.0 / (ds[..., :3].sum(-1) + 2 * offs)
    GR = 15.0 / (4 * ds[..., :3].sum(-1) - 4 * offs + 3 * ds[..., 3:].sum(-1))
    K, G = 0.5 * (KV + KR), 0.5 * (GV + GR)
    return 9 * K * G / (3 * K + G)


# --------------------------------------------------------------------------- #
# Local material axes
# --------------------------------------------------------------------------- #

def cortical_axes(normals: np.ndarray, frame: AnatomicalFrame | None,
                  min_projection: float = 0.2) -> np.ndarray:
    """(M, 3, 3) right-handed local axes (rows): [tangent_ap, tangent_2, normal].

    axis 3 = unit normal; axis 1 = first of (ap, lr, si) whose tangential
    projection has norm >= `min_projection` (always exists: the three frame
    axes are orthonormal, so at least one has projection norm >= sqrt(2/3)).
    """
    n = np.asarray(normals, dtype=float)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    if frame is None:
        cands = np.eye(3)
    else:
        cands = np.stack([frame.ap, frame.lr, frame.si]).astype(float)
    e1 = np.zeros_like(n)
    done = np.zeros(len(n), dtype=bool)
    for c in cands:
        proj = c[None, :] - (n @ c)[:, None] * n
        norm = np.linalg.norm(proj, axis=1)
        take = ~done & (norm >= min_projection)
        e1[take] = proj[take] / norm[take, None]
        done |= take
    e2 = np.cross(n, e1)
    return np.stack([e1, e2, n], axis=1)


# --------------------------------------------------------------------------- #
# Material assembly
# --------------------------------------------------------------------------- #

def layer_index(mesh: Mesh) -> np.ndarray:
    """(M,) int layer id (index into LAYERS); raises if sets overlap or miss elements."""
    lid = np.full(mesh.n_elems, -1, dtype=np.int64)
    for k, name in enumerate(LAYERS):
        if name not in mesh.elem_sets:
            raise KeyError(f"mesh is missing elem set '{name}'")
        ids = np.asarray(mesh.elem_sets[name], dtype=np.int64)
        if np.any(lid[ids] >= 0):
            raise ValueError(f"elem set '{name}' overlaps another material layer")
        lid[ids] = k
    if np.any(lid < 0):
        raise ValueError(f"{int((lid < 0).sum())} elements belong to no material layer")
    return lid


def build_material(mesh: Mesh, density: np.ndarray | float,
                   cfg: MaterialsConfig | Mapping[str, LayerProps]) -> Material:
    """Assemble the per-element `Material` for a relative density field.

    Parameters
    ----------
    mesh : Mesh with elem sets ``cortical``, ``cancellous``, ``suture_mid`` and,
        if the cortical layer is orthotropic, elem field ``surface_normal``.
    density : (M,) relative density (or scalar); ignored on suture elements.
    cfg : `craniofea.config.MaterialsConfig`, or a mapping layer name ->
        LayerProps (see `layers_from_config`).

    Returns
    -------
    Material with global C (M, 6, 6), mass_density (M,), E/nu (NaN on
    orthotropic elements), and - when any layer is orthotropic - ortho (M, 9)
    and axes (M, 3, 3) for every element (isotropic elements get the
    isotropic constants and identity axes).
    """
    layers = cfg if isinstance(cfg, Mapping) else layers_from_config(cfg)
    missing = [k for k in LAYERS if k not in layers]
    if missing:
        raise KeyError(f"missing layer properties: {missing}")
    M = mesh.n_elems
    rho = np.broadcast_to(np.asarray(density, dtype=float), (M,)).copy()
    if np.any(~np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError("relative density must be finite and > 0")
    lid = layer_index(mesh)
    is_suture = lid == LAYERS.index("suture_mid")
    rho[is_suture] = 1.0

    props = [layers[k] for k in LAYERS]
    E0 = np.array([p.E for p in props])[lid]
    nu = np.array([p.nu for p in props])[lid]
    n = np.array([p.n for p in props])[lid]
    rhom = np.array([p.mass_density for p in props])[lid]
    scale = rho ** n

    E = E0 * scale
    C = isotropic_C(E, nu)
    mass = rhom * np.where(is_suture, 1.0, rho)

    any_ortho = any(p.ortho is not None for p in props)
    ortho_arr = axes_arr = None
    E_out, nu_out = E.copy(), nu.copy()
    if any_ortho:
        G = E / (2 * (1 + nu))
        ortho_arr = np.column_stack([E, E, E, G, G, G, nu, nu, nu])
        axes_arr = np.broadcast_to(np.eye(3), (M, 3, 3)).copy()
        for k, p in enumerate(props):
            if p.ortho is None:
                continue
            sel = np.flatnonzero(lid == k)
            if "surface_normal" not in mesh.elem_fields:
                raise KeyError("orthotropic layer requires elem field 'surface_normal'")
            consts = p.ortho.as_array()[None, :].repeat(len(sel), axis=0)
            consts[:, :6] *= scale[sel, None]
            ax = cortical_axes(mesh.elem_fields["surface_normal"][sel], mesh.frame)
            C_loc = _ortho_C(*consts.T)
            C[sel] = _rot_C(C_loc, ax)
            ortho_arr[sel] = consts
            axes_arr[sel] = ax
            E_out[sel] = np.nan
            nu_out[sel] = np.nan
    return Material(C=C, mass_density=mass, E=E_out, nu=nu_out, ortho=ortho_arr, axes=axes_arr)


def layers_from_config(mat_cfg: MaterialsConfig) -> dict[str, LayerProps]:
    """Translate a `craniofea.config.MaterialsConfig` into LayerProps."""
    c, s, u = mat_cfg.cortical, mat_cfg.cancellous, mat_cfg.suture
    ortho = None
    if c.anisotropy == "orthotropic":
        o = c.ortho
        ortho = OrthoConstants(o.E1, o.E2, o.E3, o.G12, o.G13, o.G23, o.nu12, o.nu13, o.nu23)
    return {
        "cortical": LayerProps(c.E, c.nu, c.n, c.mass_density, ortho),
        "cancellous": LayerProps(s.E, s.nu, s.n, s.mass_density),
        "suture_mid": LayerProps(u.E, u.nu, 0.0, u.mass_density),
    }
