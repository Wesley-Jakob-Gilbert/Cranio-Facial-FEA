import numpy as np

from craniofea.config import ParamRange, StudyConfig
from craniofea.fem.reference import ReferenceSolver
from craniofea.remodel.demo import labeled_box_mesh
from craniofea.remodel.sensitivity import (
    critical_static_cycles,
    morris,
    morris_design,
    scale_unit,
    sweep_static_cycles,
)
from craniofea.remodel.simulate import run_study


def test_morris_design_is_one_at_a_time():
    rng = np.random.default_rng(3)
    X, order, sign, delta = morris_design(5, 8, 4, rng)
    assert np.isclose(delta, 4 / 6)
    assert X.min() >= 0 and X.max() <= 1
    steps = np.diff(X, axis=1)                         # (r, k, k)
    assert np.all((np.abs(steps) > 1e-12).sum(axis=2) == 1)
    assert np.allclose(np.abs(steps).sum(axis=2), delta)
    for t in range(8):
        assert sorted(order[t]) == list(range(5))


def test_morris_on_analytic_function():
    params = {"a": ParamRange(0.0, 1.0), "b": ParamRange(0.0, 1.0), "c": ParamRange(0.0, 2.0)}

    def f(x):
        return 3.0 * x["a"] + x["b"] ** 2 + 0.0 * x["c"]

    res = morris(f, params, trajectories=20, levels=4, seed=1)
    i = {n: k for k, n in enumerate(res.names)}
    assert np.isclose(res.mu_star[i["a"]], 3.0) and np.isclose(res.sigma[i["a"]], 0.0)
    assert np.isclose(res.mu_star[i["c"]], 0.0)
    assert 0 < res.mu_star[i["b"]] < 3.0 and res.sigma[i["b"]] > 0
    # EE of b^2 is 2 x + delta: its mean over the symmetric grid design is 1 (= hi^2 - lo^2)
    assert np.isclose(res.mu[i["b"]], 1.0, atol=0.25)
    assert res.table()[0]["parameter"] == "a"
    again = morris(f, params, trajectories=20, levels=4, seed=1)
    assert np.array_equal(res.effects, again.effects)


def test_log_scaling():
    x = np.array([[0.0, 0.5, 1.0]]).T
    v = scale_unit(x, [ParamRange(1.0, 1e4, log=True)])
    assert np.allclose(v.ravel(), [1.0, 100.0, 1e4])


def test_critical_static_cycles_matches_lazy_zone_edge():
    mesh = labeled_box_mesh(nx=6, ny=4, nz=3)
    cfg = StudyConfig().with_overrides({"remodeling.smoothing": False})
    crit = critical_static_cycles(mesh, cfg, ReferenceSolver())
    n = crit["n_crit"]
    e = int(np.argmin(n))
    assert np.isfinite(n[e]) and n[e] > 1.0
    for factor, inside in ((0.99, True), (1.01, False)):
        c = cfg.with_overrides({"stimulus.static_equivalent_cycles": factor * n[e],
                                "remodeling.n_steps": 1, "remodeling.days": 1.0})
        S = run_study(mesh, c, ReferenceSolver(), scenarios=["mewing"]).error["mewing"][0]
        assert (S[e] <= cfg.remodeling.lazy_zone) == inside
    assert crit["region_palatal_vault"]["min"] == n[mesh.elem_sets["region_palatal_vault"]].min()


def test_static_cycles_sweep_monotone_contrast():
    mesh = labeled_box_mesh(nx=6, ny=4, nz=3)
    cfg = StudyConfig().with_overrides({"remodeling.n_steps": 2, "remodeling.days": 60.0})
    rows = sweep_static_cycles(mesh, cfg, ReferenceSolver(), values=[1.0, 1e6, 1e8])
    r = [row["response"] for row in rows]
    assert r[0] == 0.0                      # lazy zone at the Carter default
    assert 0.0 <= r[1] < r[2]
    assert set(rows[0]["by_region"]) >= {"region_palatal_vault", "region_basal"}
