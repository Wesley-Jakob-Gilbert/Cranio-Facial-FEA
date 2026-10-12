"""FEBio backend: .feb (spec 4.0) structure, logfile parsing and cross-verification."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from craniofea.backends._common import BackendError
from craniofea.backends.febio import (
    ORTHO_PARAMS,
    FebioSolver,
    read_logfile_data,
    split_materials,
    write_feb,
)
from craniofea.backends.verification import ORTHO_LAYER, benchmark_problem, compare
from craniofea.fem.constitutive import isotropic_C
from craniofea.fem.reference import ReferenceSolver
from craniofea.mesh.primitives import box_mesh
from craniofea.model import LoadCase, Material, Problem

HAVE_FEBIO = FebioSolver().executable() is not None

SECTIONS = ["Module", "Control", "Material", "Mesh", "MeshDomains", "MeshData", "Boundary",
            "Loads", "LoadData", "Output"]


def _ids(text: str) -> list[int]:
    return [int(t) for t in text.replace("\n", "").split(",") if t.strip()]


@pytest.fixture(scope="module")
def problem():
    return benchmark_problem()


@pytest.fixture(scope="module")
def feb_trees(problem, tmp_path_factory):
    d = tmp_path_factory.mktemp("feb")
    return [ET.parse(write_feb(problem, d / f"c{i}.feb", i)).getroot()
            for i in range(len(problem.load_cases))]


def test_feb_top_level_structure(feb_trees):
    for root in feb_trees:
        assert root.tag == "febio_spec" and root.get("version") == "4.0"
        assert [c.tag for c in root] == SECTIONS
        assert root.find("Module").get("type") == "solid"
        ctrl = root.find("Control")
        assert ctrl.findtext("analysis") == "STATIC"
        assert ctrl.find("solver").get("type") == "solid"
        assert ctrl.find("solver/qn_method/max_ups").text == "0"      # full Newton


def test_feb_mesh_and_materials(problem, feb_trees):
    root = feb_trees[0]
    m, mat = problem.mesh, problem.material
    nodes = root.findall("Mesh/Nodes/node")
    assert len(nodes) == m.n_nodes
    assert np.allclose([float(v) for v in nodes[5].text.split(",")], m.nodes[5], rtol=1e-15)
    parts = {e.get("name"): e for e in root.findall("Mesh/Elements")}
    assert set(parts) == {"iso", "ortho"}
    assert all(p.get("type") == "tet4" for p in parts.values())
    ids = sorted(int(e.get("id")) for p in parts.values() for e in p)
    assert ids == list(range(1, m.n_elems + 1))
    first = parts["ortho"][0]
    eid = int(first.get("id")) - 1
    assert _ids(first.text) == (m.tets[eid] + 1).tolist()

    mats = {x.get("name"): x for x in root.findall("Material/material")}
    assert mats["iso"].get("type") == "isotropic elastic"
    assert mats["ortho"].get("type") == "orthotropic elastic"
    assert mats["iso"].find("E").get("type") == "map"            # continuous E field
    assert float(mats["iso"].findtext("v")) == pytest.approx(0.3, rel=1e-14)   # constant
    domains = {d.get("name"): d.get("mat") for d in root.findall("MeshDomains/SolidDomain")}
    assert domains == {"iso": "iso", "ortho": "ortho"}

    # ElementData values follow the element order of the part.
    data = {d.get("name"): d for d in root.findall("MeshData/ElementData") if d.get("name")}
    iso_ids = np.array([int(e.get("id")) - 1 for e in parts["iso"]])
    E_map = np.array([float(e.text) for e in data["iso_E"]])
    assert np.allclose(E_map, mat.C[iso_ids, 3, 3] * 2 * 1.3, rtol=1e-12)
    rho_map = np.array([float(e.text) for e in data["iso_density"]])
    assert np.allclose(rho_map, mat.mass_density[iso_ids], rtol=1e-15)
    axes = root.find("MeshData/ElementData[@type='mat_axis']")
    assert axes.get("elem_set") == "ortho" and len(axes) == len(parts["ortho"])
    a = [float(v) for v in axes[0].findtext("a").split(",")]
    assert np.allclose(a, mat.axes[eid, 0])


def test_feb_boundary_conditions_cover_constrained_dofs(problem, feb_trees):
    root = feb_trees[0]
    sets = {s.get("name"): _ids(s.text) for s in root.findall("Mesh/NodeSet")}
    mask = np.zeros((problem.mesh.n_nodes, 3), dtype=bool)
    for bc in root.findall("Boundary/bc"):
        assert bc.get("type") == "zero displacement"
        dofs = [bc.findtext(f"{a}_dof") == "1" for a in "xyz"]
        mask[np.array(sets[bc.get("node_set")]) - 1] |= np.array(dofs)
    assert np.array_equal(mask, problem.constrained_mask())


def test_feb_loads(problem, feb_trees):
    m = problem.mesh
    press, grav, nodal = feb_trees
    loads = {s.get("surface"): s for s in press.findall("Loads/surface_load")}
    assert set(loads) == {"p_zmax", "p_xmax"}
    assert all(s.get("type") == "pressure" and s.findtext("linear") == "1"
               for s in loads.values())
    tris = press.findall("Mesh/Surface[@name='p_zmax']/tri3")
    assert len(tris) == len(m.surface_sets["zmax"])
    # Node order is kept, so FEBio's facet normal is the outward contract normal.
    assert _ids(tris[0].text) == (m.surface_sets["zmax"][0] + 1).tolist()
    assert loads["p_zmax"].find("pressure").get("type") == "map"
    vals = [float(f.text) for f in press.find("MeshData/SurfaceData[@surface='p_zmax']")]
    assert np.allclose(vals, problem.load_cases[0].pressures["zmax"])
    assert float(loads["p_xmax"].findtext("pressure")) == pytest.approx(2e4)

    force = [float(v) for v in grav.find("Loads/body_load/force").text.split(",")]
    assert np.allclose(force, -problem.load_cases[1].gravity)       # FEBio sign convention
    assert grav.find("Loads/body_load").get("type") == "body force"

    nl = nodal.find("Loads/nodal_load")
    assert nl.get("type") == "nodal_force" and nl.find("value").get("type") == "map"
    rows = nodal.findall("MeshData/NodeData/node")
    loaded = _ids(nodal.find("Mesh/NodeSet[@name='loaded_nodes']").text)
    F = problem.load_cases[2].nodal_forces
    assert len(rows) == len(loaded) == 3
    assert np.allclose([[float(v) for v in r.text.split(",")] for r in rows],
                       F[np.array(loaded) - 1])


def test_split_materials_recovers_febio_orthotropic_constants(problem):
    split = split_materials(problem.material)
    E1, E2, E3, G12, G13, G23, n12, n13, n23 = ORTHO_LAYER
    expected = dict(zip(ORTHO_PARAMS, [E1, E2, E3, G12, G23, G13, n12, n23, n13 * E3 / E1],
                        strict=True))
    got = dict(zip(ORTHO_PARAMS, split.ortho[0], strict=True))
    assert got == pytest.approx(expected, rel=1e-10)
    assert len(split.iso_elems) + len(split.ortho_elems) == problem.mesh.n_elems


def test_split_materials_rejects_general_anisotropy():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    C = (A @ A.T + 6 * np.eye(6))[None] * 1e9
    with pytest.raises(ValueError, match="anisotropic"):
        split_materials(Material(C=C, mass_density=np.ones(1)))


def test_read_logfile_data_takes_last_record(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("*Step  = 0\n*Time  = 0\n*Data  = ux;uy;uz\n1,0,0,0\n2,0,0,0\n"
                 "*Step  = 1\n*Time  = 1\n*Data  = ux;uy;uz\n2,4,5,6\n1,1.5e-07,-2,3\n")
    assert read_logfile_data(p, 2, 3).tolist() == [[1.5e-7, -2, 3], [4, 5, 6]]
    with pytest.raises(BackendError):
        read_logfile_data(p, 3, 3)


def test_missing_executable_raises():
    m = box_mesh(1, 1, 1, 1, 1, 1)
    prob = Problem(m, Material(isotropic_C(np.full(6, 1e9), 0.3), np.ones(6)),
                   fixed_nodes=m.node_sets["xmin"], load_cases=[LoadCase("p")])
    with pytest.raises(BackendError, match="not found"):
        FebioSolver(febio="/nonexistent/febio4").solve(prob)


# --------------------------------------------------------------------------- #
# Cross-verification against the reference solver (requires FEBio)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_strain_results():
    prob = benchmark_problem(load_scale=1e-3)          # max strain ~1e-6
    return ReferenceSolver().solve(prob), FebioSolver(n_threads=2).solve(prob)


@pytest.mark.parametrize("case", ["pressure", "gravity", "nodal"])
@pytest.mark.febio
@pytest.mark.skipif(not HAVE_FEBIO, reason="FEBio (febio4) not installed")
def test_febio_matches_reference_at_small_strain(small_strain_results, case):
    """FEBio's St. Venant-Kirchhoff materials equal linear elasticity to O(strain)."""
    ref, fe = small_strain_results
    err = compare(ref[case], fe[case])
    assert max(err.values()) < 2e-5, err


@pytest.mark.febio
@pytest.mark.skipif(not HAVE_FEBIO, reason="FEBio (febio4) not installed")
def test_febio_deviation_is_first_order_in_load():
    """The remaining difference is geometric/material nonlinearity: it scales with the
    load. A mapping error (wrong sign, axis, unit or face) would not."""
    errs = []
    for s in (1e-2, 1e-1):
        prob = benchmark_problem(load_scale=s)
        prob.load_cases = prob.load_cases[:1]
        ref = ReferenceSolver().solve(prob)["pressure"]
        errs.append(compare(ref, FebioSolver(n_threads=2).solve(prob)["pressure"])["u"])
    assert 7 < errs[1] / errs[0] < 13, errs
