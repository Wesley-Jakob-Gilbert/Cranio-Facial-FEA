"""Failure modes must be loud: a broken solve must never look like 'no density change'."""
from pathlib import Path

import numpy as np
import pytest

from craniofea.config import ConfigError, load_config
from craniofea.remodel.demo import labeled_box_mesh
from craniofea.remodel.law import stimulus_error
from craniofea.remodel.simulate import run_study

ROOT = Path(__file__).resolve().parents[1]


def test_stimulus_error_rejects_non_finite():
    with pytest.raises(FloatingPointError):
        stimulus_error(np.array([5.0, 5.0, 5.0]), np.array([1.0, np.nan, 2.0]))
    with pytest.raises(FloatingPointError):
        stimulus_error(np.array([np.inf, 1.0]), np.array([1.0, 1.0]))


def test_unconverged_solve_aborts_study():
    from craniofea.fem.jax_solver import JaxSolver

    cfg = load_config(ROOT / "configs" / "study_template.yaml").with_overrides(
        {"remodeling.n_steps": 1})
    mesh = labeled_box_mesh(nx=6, ny=5, nz=3)
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError, match="did not converge"):
        run_study(mesh, cfg, JaxSolver(method="jax-cg", maxiter=2), scenarios=["mewing"])


def test_unknown_region_rejected_at_load_time():
    cfg = load_config(ROOT / "configs" / "study_template.yaml")
    with pytest.raises(ConfigError):
        cfg.with_overrides({"sensitivity.region": "region_palate"})
