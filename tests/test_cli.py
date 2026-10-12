import json

import pytest
import yaml

from craniofea.cli import main
from craniofea.fem.reference import ReferenceSolver
from craniofea.remodel.demo import labeled_box_mesh
from craniofea.solvers import available_solvers, get_solver

BOX = {"nx": 6, "ny": 4, "nz": 3}


def _write_cfg(tmp_path, **extra):
    cfg = {"name": "tiny", "output_dir": str(tmp_path / "out"),
           "mesh": {"source": "box", "box": BOX},
           "remodeling": {"n_steps": 2, "days": 60.0},
           "sensitivity": {"static_cycles": [1.0, 1.0e+6],
                           "morris": {"trajectories": 2, "parameters": {
                               "stimulus.m": {"lo": 3.0, "hi": 5.0},
                               "remodeling.lazy_zone": {"lo": 0.0, "hi": 0.2}}}}}
    cfg.update(extra)
    p = tmp_path / "tiny.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_solver_registry():
    assert isinstance(get_solver("reference"), ReferenceSolver)
    assert {"reference", "jax", "calculix", "febio"} <= set(available_solvers())
    with pytest.raises(ValueError, match="unknown solver"):
        get_solver("ansys")


def test_info(tmp_path, capsys):
    path = tmp_path / "box.npz"
    labeled_box_mesh(**BOX).save(path)
    assert main(["info", str(path)]) == 0
    out = capsys.readouterr().out
    assert "tets" in out and "palate" in out and "region_palatal_vault" in out and "frame" in out


def test_study_command(tmp_path, capsys):
    cfg = _write_cfg(tmp_path)
    out = tmp_path / "res"
    assert main(["study", str(cfg), "--solver", "reference", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "contrast mewing - mouth_breathing" in text
    man = json.loads((out / "manifest.json").read_text())
    assert man["solver"] == "reference" and len(man["config_hash"]) == 64
    assert {"density_reference.npz", "density_mewing.npz", "density_mouth_breathing.npz",
            "summary.csv"} <= {p.name for p in out.iterdir()}


def test_sensitivity_command(tmp_path, capsys):
    cfg = _write_cfg(tmp_path)
    out = tmp_path / "sens"
    assert main(["sensitivity", str(cfg), "--out", str(out), "--what", "all"]) == 0
    sweep = json.loads((out / "static_cycles_sweep.json").read_text())
    assert [r["static_equivalent_cycles"] for r in sweep] == [1.0, 1e6]
    crit = json.loads((out / "critical_static_cycles.json").read_text())
    assert crit["region_palatal_vault"]["min"] > 1.0
    morris = json.loads((out / "morris.json").read_text())
    assert {r["parameter"] for r in morris["table"]} == {"stimulus.m", "remodeling.lazy_zone"}
    assert morris["meta"]["evaluations"] == 2 * 3


def test_bad_config_exit_code(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text("stimulus: {m: -2}\n")
    assert main(["study", str(p)]) == 2
    assert "m must be" in capsys.readouterr().err


def test_mesh_fallback_is_recorded(tmp_path):
    cfg = _write_cfg(tmp_path, mesh={"source": "file", "path": str(tmp_path / "missing.npz"),
                                     "fallback": ["box"], "box": BOX})
    out = tmp_path / "fb"
    assert main(["study", str(cfg), "--out", str(out)]) == 0
    man = json.loads((out / "manifest.json").read_text())
    assert "mesh_fallback_from" in man["mesh"]["meta"]
