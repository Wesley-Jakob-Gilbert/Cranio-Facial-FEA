import itertools

import numpy as np
import pytest

from craniofea.config import MaterialsConfig, from_dict
from craniofea.fem.constitutive import isotropic_C
from craniofea.materials import (
    LayerProps,
    OrthoConstants,
    _orthotropic_C,
    _rotate_C,
    _strain_rotation,
    build_material,
    cortical_axes,
    hill_modulus,
)
from craniofea.remodel.demo import labeled_box_mesh

ORTHO = OrthoConstants(20e9, 13e9, 11e9, 5e9, 4e9, 3.5e9, 0.3, 0.3, 0.3)


def _random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _layers(ortho=None):
    return {"cortical": LayerProps(15e9, 0.3, 3.0, 1850.0, ortho),
            "cancellous": LayerProps(1e9, 0.3, 2.0, 1000.0),
            "suture_mid": LayerProps(1e7, 0.45, 0.0, 1100.0)}


@pytest.fixture(scope="module")
def mesh():
    return labeled_box_mesh(nx=6, ny=4, nz=3)


def test_density_modulus_power_law(mesh):
    rng = np.random.default_rng(1)
    rho = rng.uniform(0.5, 1.5, mesh.n_elems)
    mat = build_material(mesh, rho, _layers())
    cort, canc, sut = (mesh.elem_sets[k] for k in ("cortical", "cancellous", "suture_mid"))
    assert np.allclose(mat.E[cort], 15e9 * rho[cort] ** 3)
    assert np.allclose(mat.E[canc], 1e9 * rho[canc] ** 2)
    assert np.allclose(mat.E[sut], 1e7)                       # sutures are not remodeled
    assert np.allclose(mat.C, isotropic_C(mat.E, mat.nu))
    assert np.allclose(mat.mass_density[canc], 1000.0 * rho[canc])
    assert np.allclose(mat.mass_density[sut], 1100.0)
    assert mat.ortho is None and mat.axes is None


def test_build_material_accepts_config(mesh):
    cfg = from_dict(MaterialsConfig, {"cancellous": {"E": "2.0e9"}})
    mat = build_material(mesh, 1.0, cfg)
    assert np.allclose(mat.E[mesh.elem_sets["cancellous"]], 2e9)


def test_layer_sets_must_partition(mesh):
    bad = labeled_box_mesh(nx=6, ny=4, nz=3)
    bad.elem_sets["cancellous"] = np.append(bad.elem_sets["cancellous"],
                                            bad.elem_sets["cortical"][:1])
    with pytest.raises(ValueError, match="overlaps"):
        build_material(bad, 1.0, _layers())
    bad.elem_sets["cancellous"] = bad.elem_sets["cancellous"][:-2]
    with pytest.raises(ValueError, match="no material layer"):
        build_material(bad, 1.0, _layers())
    with pytest.raises(ValueError):
        build_material(mesh, -1.0, _layers())


def test_orthotropic_reduces_to_isotropic():
    E, nu = 7e9, 0.25
    G = E / (2 * (1 + nu))
    assert np.allclose(_orthotropic_C(E, E, E, G, G, G, nu, nu, nu), isotropic_C(E, nu))


def test_strain_rotation_matches_tensor_rotation():
    rng = np.random.default_rng(0)
    R = _random_rotation(rng)
    e = rng.normal(size=(3, 3))
    e = e + e.T
    def voigt(t):
        return np.array([t[0, 0], t[1, 1], t[2, 2], 2 * t[1, 2], 2 * t[0, 2], 2 * t[0, 1]])
    assert np.allclose(_strain_rotation(R) @ voigt(e), voigt(R @ e @ R.T))


def test_rotate_C_invariants():
    rng = np.random.default_rng(2)
    R = _random_rotation(rng)
    Ci = isotropic_C(10e9, 0.3)
    assert np.allclose(_rotate_C(Ci, R), Ci, rtol=1e-12, atol=1e-3)
    Cl = _orthotropic_C(*ORTHO.as_array())
    Cg = _rotate_C(Cl, R)
    eg = rng.normal(size=6) * 1e-3
    el = _strain_rotation(R) @ eg
    assert np.isclose(eg @ Cg @ eg, el @ Cl @ el, rtol=1e-12)     # energy invariance
    assert np.allclose(Cg, Cg.T, rtol=1e-12, atol=1e-3)
    assert np.isclose(hill_modulus(Cg), hill_modulus(Cl), rtol=1e-12)


def test_hill_modulus_isotropic_exact():
    assert np.allclose(hill_modulus(isotropic_C(np.array([1e9, 2e10]), 0.3)), [1e9, 2e10])


def test_cortical_axes_frames(mesh):
    normals = np.array([[0, 0, 1.0], [0, 1.0, 0], [1, 1, 0] / np.sqrt(2), [0, 0.999, 0.0447]])
    ax = cortical_axes(normals, mesh.frame)
    for a, n in zip(ax, normals, strict=True):
        assert np.allclose(a @ a.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(a), 1.0)
        assert np.allclose(a[2], n / np.linalg.norm(n))
    assert np.allclose(ax[0, 0], mesh.frame.ap)          # ap lies in the plate
    assert np.allclose(ax[1, 0], mesh.frame.lr)          # ap is the normal: fall back to lr


def test_orthotropic_cortex(mesh):
    rho = np.full(mesh.n_elems, 0.8)
    m1 = build_material(mesh, 1.0, _layers(ORTHO))
    m2 = build_material(mesh, rho, _layers(ORTHO))
    cort = mesh.elem_sets["cortical"]
    canc = mesh.elem_sets["cancellous"]
    assert np.all(np.isnan(m1.E[cort])) and np.all(np.isfinite(m1.E[canc]))
    assert m1.ortho.shape == (mesh.n_elems, 9) and m1.axes.shape == (mesh.n_elems, 3, 3)
    assert np.allclose(m2.C[cort], 0.8 ** 3 * m1.C[cort])     # uniform scaling of all moduli
    assert np.allclose(m2.ortho[cort, 6:], 0.3)               # Poisson ratios unchanged
    # stiffness along the outward normal equals E3-direction stiffness of the local C
    Cl = _orthotropic_C(*ORTHO.as_array())
    for e in cort[:20]:
        n = mesh.elem_fields["surface_normal"][e]
        i = int(np.argmax(np.abs(n)))
        assert np.isclose(m1.C[e, i, i], Cl[2, 2], rtol=1e-10)
    assert np.all(np.linalg.eigvalsh(m1.C) > 0)


@pytest.mark.parametrize("perm", list(itertools.permutations(range(3)))[:3])
def test_hill_modulus_rotation_invariant(perm):
    P = np.eye(3)[list(perm)]
    if np.linalg.det(P) < 0:
        P[0] *= -1
    Cl = _orthotropic_C(*ORTHO.as_array())
    assert np.isclose(hill_modulus(_rotate_C(Cl, P)), hill_modulus(Cl))
