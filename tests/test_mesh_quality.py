import numpy as np
import pytest

from craniofea.mesh.primitives import box_mesh
from craniofea.mesh.quality import format_report, quality_report, tet_quality
from craniofea.mesh.template import build_template_mesh

REGULAR = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)


def test_regular_tet_metrics():
    q = tet_quality(REGULAR, np.array([[0, 1, 2, 3]]))
    assert np.isclose(q["radius_ratio"][0], 1.0)
    assert np.isclose(q["aspect_ratio"][0], 1.0)
    assert np.isclose(q["min_dihedral"][0], np.degrees(np.arccos(1 / 3)))
    assert np.isclose(q["max_dihedral"][0], np.degrees(np.arccos(1 / 3)))


def test_sliver_detected():
    # Four nearly coplanar points spread over a square: a classic sliver.
    sliver = np.array([[0, 0, 0], [1, 0, 0.01], [1, 1, 0], [0, 1, 0.01]], float)
    q = tet_quality(sliver, np.array([[0, 1, 2, 3]]))
    assert q["radius_ratio"][0] < 0.05
    assert q["min_dihedral"][0] < 3.0
    assert q["max_dihedral"][0] > 170.0


def test_orientation_does_not_change_shape_metrics():
    t = np.array([[0, 1, 2, 3], [0, 2, 1, 3]])
    q = tet_quality(REGULAR, t)
    assert q["volume"][0] == pytest.approx(-q["volume"][1])
    assert np.allclose(q["radius_ratio"], 1.0)


def test_report_box_and_template():
    rows = {"box": quality_report(box_mesh(1, 1, 1, 3, 3, 3)),
            "template": quality_report(build_template_mesh())}
    assert rows["box"]["negative_volumes"] == 0
    assert np.isclose(rows["box"]["volume_mm3"], 1e9)
    # Kuhn tets of a cube: radius ratio 0.717, min dihedral 45 deg.
    assert np.isclose(rows["box"]["radius_ratio_min"], 0.7174, atol=1e-3)
    assert np.isclose(rows["box"]["min_dihedral_min_deg"], 45.0)
    t = rows["template"]
    assert t["negative_volumes"] == 0
    assert t["radius_ratio_min"] > 0.05 and t["min_dihedral_min_deg"] > 3.0
    table = format_report(rows)
    assert table.splitlines()[0].startswith("| metric | box | template |")
