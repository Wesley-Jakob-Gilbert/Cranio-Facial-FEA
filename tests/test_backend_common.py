"""Solver-independent helpers of the external backends (no executables needed)."""
import io

import numpy as np
import pytest

from craniofea.backends._common import (
    TET_FACE_NODES,
    TET_FACE_OPPOSITE,
    collect_face_pressures,
    equilibrium_residual,
    face_owners,
    isotropic_parameters,
    orthotropic_C,
    orthotropic_parameters,
    rotate_stiffness,
    voigt_stress_rotation,
    voigt_to_tensor,
    write_id_list,
    write_rows,
)
from craniofea.backends.verification import ORTHO_LAYER, benchmark_problem, rotation_matrix
from craniofea.fem.constitutive import isotropic_C
from craniofea.fem.reference import ReferenceSolver
from craniofea.mesh.primitives import box_mesh


def test_c3d4_face_numbering_matches_calculix_definition():
    # CalculiX C3D4: face 1 = 1-2-3, face 2 = 1-4-2, face 3 = 2-4-3, face 4 = 3-4-1.
    expected = [(1, 2, 3), (1, 4, 2), (2, 4, 3), (3, 4, 1)]
    assert [tuple(f + 1) for f in TET_FACE_NODES] == expected


def test_face_owners_recovers_boundary_faces_with_outward_sign():
    m = box_mesh(1.0, 0.5, 0.4, 3, 2, 2)
    faces = m.surface_sets["all"]
    elem, lface, sign = face_owners(m.nodes, m.tets, faces)
    owned = m.tets[elem[:, None], TET_FACE_NODES[lface]]
    assert np.array_equal(np.sort(owned, axis=1), np.sort(faces, axis=1))
    assert np.all(sign == 1)                        # surface sets are outward
    _, _, sign_flipped = face_owners(m.nodes, m.tets, faces[:, [0, 2, 1]])
    assert np.all(sign_flipped == -1)


def test_face_owners_internal_face_picks_outward_element():
    m = box_mesh(1.0, 1.0, 1.0, 2, 1, 1)
    all_faces = m.tets[:, TET_FACE_NODES].reshape(-1, 3)
    key = np.sort(all_faces, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    internal = all_faces[cnt[inv.ravel()] == 2][:4]
    for tri in (internal, internal[:, [0, 2, 1]]):
        e, lf, s = face_owners(m.nodes, m.tets, tri)
        assert np.all(s == 1)
        x = m.nodes[tri]
        n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
        opp = m.nodes[m.tets[e, TET_FACE_OPPOSITE[lf]]]
        assert np.all(np.einsum("ij,ij->i", n, opp - x[:, 0]) < 0)


def test_face_owners_rejects_unknown_triangle():
    m = box_mesh(1.0, 1.0, 1.0, 1, 1, 1)
    with pytest.raises(ValueError, match="not faces"):
        face_owners(m.nodes, m.tets, np.array([[1, 2, 4]]))     # corner-cut, not a Kuhn face


def test_collect_face_pressures_sums_overlapping_sets():
    m = box_mesh(1.0, 1.0, 1.0, 2, 2, 2)
    m.surface_sets["zmax_again"] = m.surface_sets["zmax"]
    nf = len(m.surface_sets["zmax"])
    per_face = np.arange(nf, dtype=float)
    e, lf, p = collect_face_pressures(m, {"zmax": per_face, "zmax_again": 2.0, "xmin": 5.0})
    assert len(e) == nf + len(m.surface_sets["xmin"])
    e_z, lf_z, _ = face_owners(m.nodes, m.tets, m.surface_sets["zmax"])
    lookup = dict(zip(4 * e + lf, p, strict=True))
    assert np.allclose([lookup[k] for k in 4 * e_z + lf_z], per_face + 2.0)


def test_voigt_rotation_preserves_isotropy_and_energy():
    R = rotation_matrix([0.3, -1.0, 2.0], 1.1)
    C = isotropic_C(7e9, 0.27)
    assert np.allclose(rotate_stiffness(C, R), C, rtol=0, atol=1e-6 * C.max())
    Cl = orthotropic_C(ORTHO_LAYER)[0]
    Cg = rotate_stiffness(Cl, R.T)
    assert np.allclose(rotate_stiffness(Cg, R), Cl, atol=1e-6 * Cl.max())
    # Stress transforms as a tensor.
    s = np.array([1.0, -2.0, 0.5, 0.3, -0.7, 1.2])
    T = voigt_stress_rotation(R)
    assert np.allclose(voigt_to_tensor(T @ s), R @ voigt_to_tensor(s) @ R.T)


def test_isotropic_and_orthotropic_parameter_recovery():
    E = np.array([1e9, 3.5e9, 17e9])
    nu = np.array([0.2, 0.3, 0.45])
    iso, E2, nu2 = isotropic_parameters(isotropic_C(E, nu))
    assert iso.all() and np.allclose(E2, E, rtol=1e-12) and np.allclose(nu2, nu, rtol=1e-12)
    Cl = orthotropic_C(ORTHO_LAYER)
    iso, _, _ = isotropic_parameters(Cl)
    assert not iso.any()
    ok, eng = orthotropic_parameters(Cl)
    assert ok.all() and np.allclose(eng[0], ORTHO_LAYER, rtol=1e-12)
    R = rotation_matrix([1, 1, 0], 0.4)
    ok, _ = orthotropic_parameters(rotate_stiffness(Cl, R.T))
    assert not ok.any()                              # rotated: not orthotropic in global axes


def test_write_rows_and_id_list_format():
    buf = io.StringIO()
    write_rows(buf, "%d, %.3g\n", np.array([[1, 0.5], [2, 1.25]]), chunk=1)
    assert buf.getvalue() == "1, 0.5\n2, 1.25\n"
    buf = io.StringIO()
    write_id_list(buf, np.arange(1, 24), per_line=10)
    lines = buf.getvalue().splitlines()
    assert lines[0].endswith(",") and lines[1].endswith(",") and not lines[2].endswith(",")
    assert [int(t) for t in buf.getvalue().replace("\n", "").split(",")] == list(range(1, 24))
    buf = io.StringIO()
    write_id_list(buf, np.arange(1, 11), per_line=10)
    assert buf.getvalue() == ",".join(map(str, range(1, 11))) + "\n"


def test_equilibrium_residual_is_tiny_for_exact_solution_and_large_otherwise():
    prob = benchmark_problem()
    ref = ReferenceSolver().solve(prob)
    for case in prob.load_cases:
        assert equilibrium_residual(prob, case, ref[case.name].stress) < 1e-10
    case = prob.load_cases[0]
    bad = ref[case.name].stress.copy()
    bad[7] *= 1.01
    assert equilibrium_residual(prob, case, bad) > 1e-4
