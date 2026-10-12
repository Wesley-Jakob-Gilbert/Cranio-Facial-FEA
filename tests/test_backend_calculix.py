"""CalculiX backend: deck generation, .dat parsing and cross-verification."""
import re

import numpy as np
import pytest

from craniofea.backends._common import (
    BackendError,
    isotropic_parameters,
    orthotropic_C,
    rotate_stiffness,
)
from craniofea.backends.calculix import (
    ANISO_INDEX,
    CalculixSolver,
    aniso_constants,
    aniso_to_voigt,
    dat_to_solution,
    group_materials,
    parse_dat,
    unique_rows,
    write_inp,
)
from craniofea.backends.verification import (
    ORTHO_LAYER,
    benchmark_problem,
    compare,
    rotation_matrix,
)
from craniofea.fem.constitutive import isotropic_C
from craniofea.fem.reference import ReferenceSolver
from craniofea.mesh.primitives import box_mesh
from craniofea.model import LoadCase, Material, Problem

HAVE_CCX = CalculixSolver().executable() is not None

# Excerpt of a ccx 2.17 .dat file (2 nodes, 2 elements); one value uses the
# Fortran form of a three-digit exponent ('E' dropped).
DAT_FIXTURE = """
 displacements (vx,vy,vz) for set NALL and time  0.1000000E+01

         1  0.000000E+00  0.000000E+00  0.000000E+00
         2 -8.818169E-06 -5.509547E-07 -1.264377E-04

 stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EALL and time  0.1000000E+01

         1   1 -1.672881E+04 -8.484893E+03 -1.788391E+04 -1.094196E+04 -2.525353E+04 -1.189045E+04
         2   1 -2.946819E+04 -2.021530E+04 -4.861811E+04 -9.404552E+02 -3.525504E+04  1.000000-120

 strains (elem, integ.pnt.,exx,eyy,ezz,exy,exz,eyz) for set EALL and time  0.1000000E+01

         1   1 -8.818169E-06  1.898922E-06 -1.031980E-05 -1.422455E-05 -3.282959E-05 -1.545758E-05
         2   1 -8.818169E-06  3.210595E-06 -3.371307E-05 -1.222592E-06 -4.583155E-05 -4.416785E-06

 internal energy density (elem, integ.pnt.,energy) for set EALL and time  0.1000000E+01

         1   1  1.326487E+00
         2   1  2.548958E+00
"""


# --------------------------------------------------------------------------- #
# Pure-Python tests
# --------------------------------------------------------------------------- #
def test_aniso_order_matches_calculix_manual():
    # D1111 D1122 D2222 D1133 D2233 D3333 D1112 D2212 D3312 D1212 D1113 D2213
    # D3313 D1213 D1313 D1123 D2223 D3323 D1223 D1323 D2323
    names = ["1111", "1122", "2222", "1133", "2233", "3333", "1112", "2212", "3312", "1212",
             "1113", "2213", "3313", "1213", "1313", "1123", "2223", "3323", "1223", "1323",
             "2323"]
    voigt = {"11": 0, "22": 1, "33": 2, "23": 3, "13": 4, "12": 5}
    expected = [(voigt[n[:2]], voigt[n[2:]]) for n in names]
    assert [tuple(p) for p in ANISO_INDEX] == expected


def test_aniso_round_trip_and_isotropic_shear_constant():
    C = rotate_stiffness(orthotropic_C(ORTHO_LAYER)[0], rotation_matrix([1, 2, 3], 0.7).T)
    D = aniso_constants(C)
    assert D.shape == (21,) and np.all(D != 0)
    assert np.allclose(aniso_to_voigt(D), C, rtol=0, atol=1e-12 * np.abs(C).max())
    # Tensor constant D1212 equals the engineering-shear modulus G (no factor 2).
    Ciso = isotropic_C(10e9, 0.25)
    assert aniso_constants(Ciso)[9] == pytest.approx(10e9 / (2 * 1.25))


def test_unique_rows_groups_identical_rows_in_first_occurrence_order():
    key = np.array([[2.0, 1.0], [1.0, 0.0], [2.0, 1.0], [0.0, 3.0], [1.0, 0.0]])
    first, inv = unique_rows(key)
    assert first.tolist() == [0, 1, 3]
    assert inv.tolist() == [0, 1, 0, 2, 1]


def test_group_materials_exact_and_binned():
    E = np.repeat([1e9, 2e9, 3e9], 4)
    C = isotropic_C(E, 0.3)
    t = group_materials(C, np.ones(12), max_materials=10)
    assert t.exact and t.n_materials == 3
    assert np.array_equal(t.C[t.elem_material], C)
    # Density matters only when requested (gravity present).
    rho = np.arange(12.0)
    assert group_materials(C, rho, use_density=True).n_materials == 12
    assert group_materials(C, rho, use_density=False).n_materials == 3
    # Continuous field: binned, approximation reported.
    Ec = np.geomspace(1e9, 1e10, 500)
    tb = group_materials(isotropic_C(Ec, 0.3), np.ones(500), max_materials=50)
    assert not tb.exact and tb.n_materials <= 50
    assert 0 < tb.stats["max_rel_C_error"] < 0.03
    # Different orientations are never merged into one bin.
    R = rotation_matrix([0, 0, 1], 0.5)
    Cl = orthotropic_C(ORTHO_LAYER)[0]
    Cmix = np.stack([Cl, Cl, Cl, rotate_stiffness(Cl, R.T), rotate_stiffness(Cl, R.T),
                     rotate_stiffness(Cl, R.T)]) * np.array([1, 2, 3, 1, 2, 3.0])[:, None, None]
    tm = group_materials(Cmix, np.ones(6), max_materials=2)
    assert tm.n_materials == 2 and tm.stats["n_shapes"] == 2
    assert np.array_equal(tm.elem_material[:3], [tm.elem_material[0]] * 3)
    assert tm.elem_material[0] != tm.elem_material[3]
    with pytest.raises(ValueError, match="orientations"):
        group_materials(Cmix, np.ones(6), max_materials=1)


def test_deck_contents(tmp_path):
    prob = benchmark_problem()
    m = prob.mesh
    decks, table = write_inp(prob, tmp_path)
    model = (tmp_path / "model.inp").read_text()
    assert len(decks) == 3 and table.exact
    assert all(len(line) <= 132 for line in model.splitlines())
    node_block = model.split("*NODE, NSET=NALL\n")[1].split("*")[0].strip().splitlines()
    assert len(node_block) == m.n_nodes
    n_elem_lines = sum(len(b.split("*")[0].strip().splitlines())
                       for b in model.split("*ELEMENT, TYPE=C3D4")[1:])
    assert n_elem_lines == m.n_elems + table.n_materials       # + one header line each
    # The rotated orthotropic layer is written as ANISO; with gravity present the
    # material key includes the (varying) mass density.
    layer = ~isotropic_parameters(prob.material.C)[0]
    n_layer_mats = len(np.unique(prob.material.mass_density[layer]))
    assert model.count("*ELASTIC, TYPE=ANISO") == n_layer_mats
    assert model.count("*ELASTIC\n") == table.n_materials - n_layer_mats
    assert model.count("*MATERIAL") == table.n_materials
    bnd = model.split("*BOUNDARY\n")[1].strip().splitlines()
    assert len(bnd) == prob.constrained_mask().sum()

    press = decks[0].read_text()
    dl = re.findall(r"^(\d+), P([1-4]), (\S+)$", press, re.MULTILINE)
    assert len(dl) == len(m.surface_sets["zmax"]) + len(m.surface_sets["xmax"])
    grav = decks[1].read_text()
    assert re.search(r"^EALL, GRAV, ", grav, re.MULTILINE)
    nodal = decks[2].read_text()
    assert len(re.findall(r"^\d+, [123], ", nodal.split("*CLOAD\n")[1], re.MULTILINE)) == 9
    for d in decks:
        assert "*INCLUDE, INPUT=model.inp" in d.read_text()


def test_parse_dat_fixture_and_convert_to_contract():
    blocks = parse_dat(DAT_FIXTURE)
    assert blocks["displacements"].shape == (2, 4)
    assert blocks["stresses"].shape == (2, 8)
    assert blocks["stresses"][1, 7] == pytest.approx(1e-120)
    sol = dat_to_solution(DAT_FIXTURE, n_nodes=2, n_elems=2)
    assert sol.u[1, 2] == pytest.approx(-1.264377e-4)
    # ccx (xx, yy, zz, xy, xz, yz) -> contract (xx, yy, zz, yz, xz, xy)
    assert sol.stress[0].tolist() == pytest.approx(
        [-1.672881e4, -8.484893e3, -1.788391e4, -1.189045e4, -2.525353e4, -1.094196e4])
    # tensor shear strain -> engineering shear strain
    assert sol.strain[0, 3] == pytest.approx(2 * -1.545758e-5)
    assert sol.strain[0, 5] == pytest.approx(2 * -1.422455e-5)
    assert sol.strain[0, 0] == pytest.approx(-8.818169e-6)
    assert sol.sed.tolist() == pytest.approx([1.326487, 2.548958])


def test_dat_missing_blocks_raise():
    with pytest.raises(BackendError, match="lacks blocks"):
        dat_to_solution(DAT_FIXTURE.split(" strains")[0], 2, 2)


def test_missing_executable_raises():
    m = box_mesh(1, 1, 1, 1, 1, 1)
    prob = Problem(m, Material(isotropic_C(np.full(6, 1e9), 0.3), np.ones(6)),
                   fixed_nodes=m.node_sets["xmin"], load_cases=[LoadCase("p")])
    with pytest.raises(BackendError, match="not found"):
        CalculixSolver(ccx="/nonexistent/ccx").solve(prob)


# --------------------------------------------------------------------------- #
# Cross-verification against the reference solver (requires ccx)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def benchmark_results():
    prob = benchmark_problem()
    return prob, ReferenceSolver().solve(prob), CalculixSolver(n_threads=2).solve(prob)


@pytest.mark.parametrize("case", ["pressure", "gravity", "nodal"])
@pytest.mark.calculix
@pytest.mark.skipif(not HAVE_CCX, reason="CalculiX (ccx) not installed")
def test_calculix_matches_reference(benchmark_results, case):
    """Heterogeneous isotropic + rotated orthotropic box, per-face pressure, gravity,
    point loads, clamped + roller supports. Agreement is limited by the 7 significant
    digits of the .dat output."""
    _, ref, ccx = benchmark_results
    err = compare(ref[case], ccx[case])
    assert max(err.values()) < 2e-6, err
    assert ccx[case].info["materials_exact"]
    assert ccx[case].info["equilibrium_residual"] < 1e-5


@pytest.mark.calculix
@pytest.mark.skipif(not HAVE_CCX, reason="CalculiX (ccx) not installed")
def test_calculix_binned_materials_match_reference_with_binned_stiffness():
    m = box_mesh(0.02, 0.01, 0.01, 8, 4, 4)
    cent = m.nodes[m.tets].mean(axis=1)
    E = 1e9 * np.exp(2.3 * cent[:, 0] / 0.02 + 0.2 * np.sin(900 * cent[:, 1]))
    mat = Material(isotropic_C(E, 0.3), np.full(m.n_elems, 1800.0))
    prob = Problem(m, mat, fixed_nodes=m.node_sets["xmin"],
                   load_cases=[LoadCase("p", pressures={"xmax": 1e5})])
    sol = CalculixSolver(max_materials=40, n_threads=2).solve(prob)["p"]
    info = sol.info
    assert not info["materials_exact"] and info["n_materials"] <= 40
    stats = info["material_binning"]
    assert 0 < stats["max_rel_C_error"] < 0.05
    # The deck is an exact model of the binned stiffness ...
    t = group_materials(mat.C, mat.mass_density, use_density=False, max_materials=40)
    binned = Problem(m, Material(t.C[t.elem_material], mat.mass_density),
                     fixed_nodes=prob.fixed_nodes, load_cases=prob.load_cases)
    assert max(compare(ReferenceSolver().solve(binned)["p"], sol).values()) < 2e-6
    # ... and close to the continuous field.
    assert compare(ReferenceSolver().solve(prob)["p"], sol)["u"] < 1e-2


@pytest.mark.calculix
@pytest.mark.skipif(not HAVE_CCX, reason="CalculiX (ccx) not installed")
def test_calculix_error_message_shows_ccx_output(tmp_path):
    m = box_mesh(1, 1, 1, 1, 1, 1)
    # No supports: singular stiffness matrix.
    prob = Problem(m, Material(isotropic_C(np.full(6, 1e9), 0.3), np.ones(6)),
                   fixed_nodes=np.array([], dtype=int),
                   load_cases=[LoadCase("p", pressures={"zmax": 1.0})])
    with pytest.raises(BackendError) as exc:
        CalculixSolver(work_dir=tmp_path).solve(prob)
    assert "load case 'p'" in str(exc.value)
