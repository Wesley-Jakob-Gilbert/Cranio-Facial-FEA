import numpy as np
import pytest

from craniofea.config import StudyConfig
from craniofea.fem.reference import ReferenceSolver
from craniofea.mesh.primitives import tet_signed_volumes
from craniofea.remodel.demo import labeled_box_mesh
from craniofea.remodel.law import nodal_smoother, remodel_rate, stimulus_error, update_density
from craniofea.remodel.simulate import run_study

FAST = {"remodeling.n_steps": 3, "remodeling.days": 90.0}


@pytest.fixture(scope="module")
def mesh():
    return labeled_box_mesh(nx=6, ny=4, nz=3)


def test_lazy_zone_and_linear_branches():
    S = np.array([-0.5, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3])
    r = remodel_rate(S, 0.1, c_form=2.0, c_res=3.0)
    assert np.allclose(r, [3 * -0.4, 0, 0, 0, 0, 0, 2 * 0.2])
    assert np.all(remodel_rate(S, 0.0, 1.0, 1.0) == S)


def test_update_density_clamps():
    rho = update_density(np.array([1.0, 1.0, 1.0]), np.array([1.0, -1.0, 0.001]), 10.0,
                         np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5]))
    assert np.allclose(rho, [1.5, 0.5, 1.01])


def test_stimulus_error_floor():
    S = stimulus_error(np.array([2.0, 1.0, 5.0]), np.array([1.0, 1e-12, 2.0]), floor=1e-6)
    assert np.allclose(S, [1.0, 0.0, 1.5])


def test_smoother_preserves_constants_and_zero(mesh):
    vol = tet_signed_volumes(mesh.nodes, mesh.tets)
    active = np.ones(mesh.n_elems, bool)
    A = nodal_smoother(mesh, vol, active)
    assert np.allclose(A @ np.full(mesh.n_elems, 0.3), 0.3)
    assert np.all(A @ np.zeros(mesh.n_elems) == 0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=mesh.n_elems)
    assert np.std(A @ x) < np.std(x)                     # damps element-scale noise


def test_homeostasis_reference_is_exactly_stationary(mesh):
    res = run_study(mesh, StudyConfig().with_overrides(FAST), ReferenceSolver(),
                    scenarios=["reference"])
    assert np.all(res.error["reference"] == 0.0)
    assert np.all(res.rho["reference"] == 1.0)


def test_more_sustained_load_more_formation(mesh):
    cfg = StudyConfig().with_overrides({
        **FAST, "remodeling.lazy_zone": 0.0, "stimulus.static_equivalent_cycles": 1e6,
        "loading.scenarios": {
            "reference": {"resting_pressure": 200.0},
            "low": {"resting_pressure": 1000.0},
            "high": {"resting_pressure": 4000.0},
            "none": {"resting_pressure": 0.0}},
        "sensitivity.contrast": ["high", "none"]})
    res = run_study(mesh, cfg, ReferenceSolver())
    v = {s: res.final_mean_drho(s, "region_palatal_vault") for s in ("none", "low", "high")}
    assert v["high"] > v["low"] > 0 > v["none"]
    # pointwise only for the first update: later, local stiffening redistributes
    # strain (stress shielding), so element-wise ordering need not persist
    assert np.all(res.rho["high"][1] >= res.rho["low"][1])
    assert np.all(res.rho["low"][1] >= res.rho["none"][1])


def test_lazy_zone_suppresses_small_contrast(mesh):
    """Default Carter weighting (1 cycle/day for sustained loads): mewing stays in the lazy zone."""
    res = run_study(mesh, StudyConfig().with_overrides(FAST), ReferenceSolver())
    for s in ("mewing", "mouth_breathing"):
        assert np.abs(res.error[s]).max() < 0.1
        assert np.all(res.rho[s] == 1.0)


def test_no_static_weight_makes_postures_identical(mesh):
    cfg = StudyConfig().with_overrides({**FAST, "stimulus.static_equivalent_cycles": 0.0,
                                        "remodeling.lazy_zone": 0.0})
    res = run_study(mesh, cfg, ReferenceSolver(), scenarios=["mewing", "mouth_breathing"])
    assert np.array_equal(res.rho["mewing"], res.rho["mouth_breathing"])


def test_bounds_respected_and_suture_fixed(mesh):
    cfg = StudyConfig().with_overrides({
        **FAST, "remodeling.lazy_zone": 0.0, "stimulus.static_equivalent_cycles": 1e9,
        "remodeling.formation_rate": {"cortical": 1.0, "cancellous": 1.0},
        "remodeling.resorption_rate": {"cortical": 1.0, "cancellous": 1.0}})
    res = run_study(mesh, cfg, ReferenceSolver(), scenarios=["mewing", "mouth_breathing"])
    canc, cort, sut = (mesh.elem_sets[k] for k in ("cancellous", "cortical", "suture_mid"))
    for r in res.rho.values():
        assert r[:, canc].max() <= 2.0 and r[:, canc].min() >= 0.25
        assert r[:, cort].max() <= 1.15 and r[:, cort].min() >= 0.5
        assert np.all(r[:, sut] == 1.0)
    assert np.isclose(res.rho["mewing"][-1, canc].max(), 2.0)      # clamp reached


def test_determinism_and_summary(mesh, tmp_path):
    cfg = StudyConfig().with_overrides({**FAST, "stimulus.static_equivalent_cycles": 1e6})
    a = run_study(mesh, cfg, ReferenceSolver(), scenarios=["mewing"])
    b = run_study(mesh, cfg, ReferenceSolver(), scenarios=["mewing"])
    assert np.array_equal(a.rho["mewing"], b.rho["mewing"])
    assert a.manifest["config_hash"] == b.manifest["config_hash"]
    rows = [r for r in a.summary if r["step"] == 3]
    assert {r["region"] for r in rows} == {"region_palatal_vault", "region_alveolar_anterior",
                                           "region_alveolar_posterior", "region_basal"}
    for r in rows:
        assert 0 <= r["frac_forming"] + r["frac_resorbing"] <= 1
    out = a.save(tmp_path / "res")
    for f in ("setpoint.npz", "density_mewing.npz", "summary.csv", "manifest.json",
              "stimulus_shares.json"):
        assert (out / f).is_file()
    d = np.load(out / "density_mewing.npz")
    assert d["rho"].shape == (4, mesh.n_elems) and np.allclose(d["times"], [0, 30, 60, 90])
