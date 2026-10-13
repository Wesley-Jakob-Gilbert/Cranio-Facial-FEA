"""Constitutive matrices: symmetry, invertibility and rotation (Bond) consistency."""
import numpy as np
import pytest

from craniofea.fem.constitutive import (
    VOIGT_PAIRS,
    bond_matrix,
    isotropic_C,
    orthotropic_C,
    orthotropic_S,
    rotate_C,
    transversely_isotropic_C,
)
from craniofea.fem.verification import random_rotations

RNG = np.random.default_rng(42)
ORTHO = dict(E1=12e9, E2=8e9, E3=20e9, G12=3e9, G13=4.5e9, G23=5e9, nu12=0.35, nu13=0.2, nu23=0.25)


def _strain_tensor(v):
    """Engineering-shear Voigt strain -> symmetric tensor."""
    e = np.empty(v.shape[:-1] + (3, 3))
    for k, (i, j) in enumerate(VOIGT_PAIRS):
        e[..., i, j] = e[..., j, i] = v[..., k] if i == j else 0.5 * v[..., k]
    return e


def _stress_voigt(s):
    return np.stack([s[..., i, j] for i, j in VOIGT_PAIRS], axis=-1)


def test_isotropic_is_rotation_invariant():
    C = isotropic_C(17e9, 0.3)
    R = random_rotations(20, RNG)
    assert np.allclose(rotate_C(np.broadcast_to(C, (20, 6, 6)), R), C, rtol=0, atol=1e-6 * 17e9)


def test_orthotropic_uniaxial_compliance_recovers_constants():
    C = orthotropic_C(**ORTHO)
    S = np.linalg.inv(C)
    E = (ORTHO["E1"], ORTHO["E2"], ORTHO["E3"])
    nu = {(0, 1): ORTHO["nu12"], (0, 2): ORTHO["nu13"], (1, 2): ORTHO["nu23"]}
    for i in range(3):
        eps = S[:, i]                       # strain under unit uniaxial stress sigma_i
        assert np.isclose(eps[i], 1 / E[i], rtol=1e-12)
        for j in range(3):
            if (i, j) in nu:
                assert np.isclose(-eps[j] / eps[i], nu[(i, j)], rtol=1e-12)
            elif (j, i) in nu:              # reciprocity nu_ij / E_i = nu_ji / E_j
                assert np.isclose(-eps[j] / eps[i], nu[(j, i)] * E[i] / E[j], rtol=1e-12)
        assert np.allclose(eps[3:], 0)
    for k, G in zip((3, 4, 5), (ORTHO["G23"], ORTHO["G13"], ORTHO["G12"]), strict=True):
        assert np.isclose(S[k, k], 1 / G, rtol=1e-12)
    assert np.allclose(S, orthotropic_S(**ORTHO), rtol=1e-12, atol=0)


def test_orthotropic_is_symmetric_positive_definite_and_vectorised():
    n = 50
    args = {k: v * RNG.uniform(0.8, 1.2, n) for k, v in ORTHO.items()}
    C = orthotropic_C(**args)
    assert C.shape == (n, 6, 6)
    assert np.allclose(C, np.swapaxes(C, 1, 2), rtol=1e-14, atol=0)
    assert np.all(np.linalg.eigvalsh(C) > 0)


def test_orthotropic_rejects_unstable_constants():
    with pytest.raises(ValueError):
        orthotropic_C(**{**ORTHO, "nu12": 1.5})


def test_isotropic_limit_of_orthotropic():
    E, nu = 10e9, 0.3
    G = E / (2 * (1 + nu))
    assert np.allclose(orthotropic_C(E, E, E, G, G, G, nu, nu, nu), isotropic_C(E, nu),
                       rtol=1e-12, atol=1e-3)


def test_rotation_round_trip_and_homomorphism():
    C = orthotropic_C(**ORTHO)
    R = random_rotations(10, RNG)
    Cg = rotate_C(C, R)
    assert np.allclose(rotate_C(Cg, np.swapaxes(R, 1, 2)), C, rtol=0, atol=1e-6 * 20e9)
    a, b = random_rotations(2, RNG)
    assert np.allclose(bond_matrix(a @ b), bond_matrix(a) @ bond_matrix(b), atol=1e-12)


def test_rotated_stiffness_matches_tensor_transformation():
    """C_global eps = R^T (C_local (R eps R^T)) R for tensors in both frames."""
    C = orthotropic_C(**ORTHO)
    R = random_rotations(8, RNG)                  # rows: material axes in global coordinates
    Cg = rotate_C(C, R)
    eps_g = RNG.normal(size=(8, 6)) * 1e-3
    e_loc = R @ _strain_tensor(eps_g) @ np.swapaxes(R, 1, 2)
    eps_loc = np.stack([e_loc[:, i, j] * (1 if i == j else 2) for i, j in VOIGT_PAIRS], -1)
    sig_loc = np.einsum("ij,mj->mi", C, eps_loc)
    s_loc = np.zeros((8, 3, 3))
    for k, (i, j) in enumerate(VOIGT_PAIRS):
        s_loc[:, i, j] = s_loc[:, j, i] = sig_loc[:, k]
    sig_g = _stress_voigt(np.swapaxes(R, 1, 2) @ s_loc @ R)
    assert np.allclose(np.einsum("mij,mj->mi", Cg, eps_g), sig_g, rtol=1e-12, atol=1e-3)


def test_transverse_isotropy_about_e3():
    C = transversely_isotropic_C(E_p=10e9, E_t=20e9, nu_p=0.3, nu_pt=0.25, G_t=5e9)
    th = 0.7
    Rz = np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1]])
    assert np.allclose(rotate_C(C, Rz), C, rtol=0, atol=1e-6 * 20e9)
    S = np.linalg.inv(C)
    assert np.isclose(1 / S[2, 2], 20e9) and np.isclose(1 / S[0, 0], 10e9)
