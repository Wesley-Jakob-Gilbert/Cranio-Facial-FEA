from pathlib import Path

import pytest

from craniofea.config import ConfigError, StudyConfig, from_dict, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["study.yaml", "study_template.yaml"])
def test_shipped_configs_load(name):
    cfg = load_config(ROOT / "configs" / name)
    assert "reference" in cfg.loading.scenarios
    assert cfg.loading.scenarios["mewing"].resting_pressure == 2000.0
    assert cfg.stimulus.static_equivalent_cycles == 1.0
    assert isinstance(cfg.materials.cortical.E, float)


def test_main_study_settings():
    cfg = load_config(ROOT / "configs" / "study.yaml")
    assert cfg.mesh.path == "data/derived/patient3_maxilla.npz"
    assert cfg.mesh.fallback == ["template"]
    assert cfg.materials.cortical.anisotropy == "orthotropic"
    assert 1.0 in cfg.sensitivity.static_cycles and max(cfg.sensitivity.static_cycles) >= 1e4


def test_defaults_valid_and_numeric_strings():
    cfg = from_dict(StudyConfig, {"materials": {"cortical": {"E": "1.5e10"}}})
    assert cfg.materials.cortical.E == 1.5e10


@pytest.mark.parametrize("data, match", [
    ({"materials": {"cortical": {"Young": 1.0}}}, "unknown key"),
    ({"materials": {"cortical": {"E": -1.0}}}, "E must be > 0"),
    ({"materials": {"cancellous": {"nu": 0.5}}}, "nu must lie"),
    ({"materials": {"cortical": {"anisotropy": "transverse"}}}, "anisotropy"),
    ({"materials": {"cortical": {"ortho": {"nu12": 0.9, "nu13": 0.9, "nu23": 0.9}}}},
     "positive definite"),
    ({"loading": {"scenarios": {"mewing": {"resting_pressure": 2000.0}}}}, "reference"),
    ({"loading": {"scenarios": {"reference": {}}}}, "resting_pressure"),
    ({"remodeling": {"lazy_zone": 1.5}}, "lazy_zone"),
    ({"remodeling": {"rho_min": {"cortical": 1.2, "cancellous": 0.2}}}, "rho bounds"),
    ({"remodeling": {"n_steps": 2.5}}, "integer"),
    ({"remodeling": {"smoothing": "yes"}}, "true/false"),
    ({"stimulus": {"m": 0.5}}, "m must be"),
    ({"mesh": {"source": "file"}}, "mesh.path"),
    ({"mesh": {"source": "ct"}}, "mesh source"),
    ({"sensitivity": {"contrast": ["mewing", "snoring"]}}, "snoring"),
    ({"sensitivity": {"morris": {"levels": 3}}}, "even"),
    ({"sensitivity": {"morris": {"parameters": {"x": {"lo": 2.0, "hi": 1.0}}}}}, "hi > lo"),
])
def test_validation_errors(data, match):
    with pytest.raises(ConfigError, match=match):
        from_dict(StudyConfig, data)


def test_error_message_has_path():
    with pytest.raises(ConfigError, match=r"materials\.suture"):
        from_dict(StudyConfig, {"materials": {"suture": {"E": "soft"}}})


def test_overrides_and_hash():
    cfg = StudyConfig()
    c2 = cfg.with_overrides({"stimulus.m": 3.0, "loading.scenarios.mewing.resting_pressure": 1.0})
    assert c2.stimulus.m == 3.0 and c2.loading.scenarios["mewing"].resting_pressure == 1.0
    assert cfg.stimulus.m == 4.0                         # original untouched
    assert cfg.hash() == StudyConfig().hash() != c2.hash()
    with pytest.raises(ConfigError, match="unknown config path"):
        cfg.with_overrides({"stimulus.n": 1.0})
    with pytest.raises(ConfigError):
        cfg.with_overrides({"stimulus.m": -1.0})


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml")
