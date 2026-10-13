import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from craniofea.mesh.frame import detect_frame, frame_from_override, parse_axis
from craniofea.mesh.template import build_template_mesh
from craniofea.model import Mesh


@pytest.fixture(scope="module")
def tpl() -> Mesh:
    return build_template_mesh({"plan_size": 3.0, "layers": 4})


def _angle(u, v) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def _transformed(m: Mesh, M: np.ndarray, t: np.ndarray) -> Mesh:
    nodes = m.nodes @ M.T + t
    tets = m.tets.copy()
    if np.linalg.det(M) < 0:            # keep tets positively oriented after a reflection
        tets[:, [1, 2]] = tets[:, [2, 1]]
    return Mesh(nodes=nodes, tets=tets)


def test_template_frame_recovered(tpl):
    f = detect_frame(tpl)
    for k, e in (("lr", [1, 0, 0]), ("ap", [0, 1, 0]), ("si", [0, 0, 1])):
        assert _angle(getattr(f, k), e) < 3.0, k
    assert tpl.meta["frame"]["method"] == "auto"
    assert np.isclose(np.linalg.det(np.stack([f.lr, f.ap, f.si])), 1.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_frame_rotation_equivariant(tpl, seed):
    R = Rotation.random(random_state=seed).as_matrix()
    m = _transformed(tpl, R, np.array([0.05, -0.02, 0.3]))
    f = detect_frame(m)
    for k, e in (("lr", [1, 0, 0]), ("ap", [0, 1, 0]), ("si", [0, 0, 1])):
        assert _angle(getattr(f, k), R @ np.array(e, float)) < 3.0, (seed, k)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_frame_under_reflection(tpl, axis):
    M = np.eye(3)
    M[axis, axis] = -1.0
    R = Rotation.from_euler("xyz", [20, -35, 50], degrees=True).as_matrix()
    M = R @ M
    f = detect_frame(_transformed(tpl, M, np.zeros(3)))
    # ap and si follow the anatomy; lr = ap x si keeps the frame right-handed,
    # i.e. a mirrored scan is read as the mirror-image patient.
    assert _angle(f.ap, M @ [0, 1, 0]) < 3.0
    assert _angle(f.si, M @ [0, 0, 1]) < 3.0
    assert _angle(f.lr, -(M @ np.array([1.0, 0, 0]))) < 3.0


def test_parse_axis():
    assert np.array_equal(parse_axis("-y"), [0, -1, 0])
    assert np.array_equal(parse_axis("z"), [0, 0, 1])
    assert np.allclose(parse_axis([2, 0, 0]), [1, 0, 0])


def test_override_completes_right_handed_frame():
    # Open-Full-Jaw files are DICOM LPS: +y posterior, +z superior -> lr = -x.
    f = frame_from_override({"ap": "-y", "si": "+z"}, origin=np.zeros(3))
    assert np.allclose(f.lr, [-1, 0, 0])
    f = frame_from_override({"lr": "+x", "si": "+z"}, origin=np.zeros(3))
    assert np.allclose(f.ap, [0, 1, 0])
    with pytest.raises(ValueError):
        frame_from_override({"lr": "+x", "ap": "+y", "si": "-z"}, origin=np.zeros(3))
    with pytest.raises(ValueError):
        frame_from_override({"si": "+z"}, origin=np.zeros(3))


def test_detect_frame_with_override_records_agreement(tpl):
    m = Mesh(nodes=tpl.nodes, tets=tpl.tets)
    f = detect_frame(m, {"ap": "+y", "si": "+z"})
    assert np.allclose(f.lr, [1, 0, 0])
    assert m.meta["frame"]["method"] == "override"
    assert max(m.meta["frame"]["auto_vs_override_deg"].values()) < 3.0
