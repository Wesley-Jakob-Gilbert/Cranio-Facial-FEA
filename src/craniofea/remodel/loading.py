"""Daily loading history of the maxilla as explicit FE load cases.

A habit (scenario) is represented by the set of distinct mechanical events of
a typical day, each an FE load case with a daily cycle count:

1. ``rest_posture`` (sustained): tongue resting pressure on the ``palate``
   surface, linear anterior-posterior gradient along ``frame.ap``.
2. ``swallow`` (cyclic): peak tongue-palate pressure during swallowing.
3. ``chew`` (cyclic): bite reaction on the ``occlusal`` surface, a traction
   along +si (optionally tilted toward +ap) distributed along the arch with a
   posterior (molar) bias.
4. ``gravity`` (sustained): self weight along -si (minor).

Sustained events are flagged ``static``; their daily cycle weight is chosen
by the stimulus model (`static_equivalent_cycles`). Cases with zero load are
omitted (they contribute nothing to the stimulus).

Pressure gradient: with t in [0, 1] the normalised AP coordinate of a face
centroid over the loaded set (0 = most posterior), the face pressure is

    p_f = p_mean * g(t_f) / <g>_A,    g(t) = 1 + (ratio - 1) t

where <g>_A is the area-weighted mean of g, so that the resultant normal
force equals p_mean times the loaded area for any ratio.

Bite traction: face weights w_f = A_f * h(t_f), h(t) = molar_weight + (1 -
molar_weight) t, normalised to sum 1; each face carries F w_f d (d the unit
force direction), lumped equally on its three nodes. The resultant is
exactly F d.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from craniofea.config import LoadingConfig
from craniofea.mesh.primitives import face_normals_areas
from craniofea.model import LoadCase, Mesh


@dataclass
class DailyLoad:
    """One event of the daily loading history."""

    case: LoadCase
    static: bool            # sustained load: cycle weight set by the stimulus model

    @property
    def name(self) -> str:
        return self.case.name


def _require(mesh: Mesh, surfaces: tuple[str, ...]) -> None:
    if mesh.frame is None:
        raise ValueError("loading requires mesh.frame (anatomical axes)")
    for s in surfaces:
        if s not in mesh.surface_sets or len(mesh.surface_sets[s]) == 0:
            raise KeyError(f"mesh is missing (or has empty) surface set '{s}'")


def ap_coordinate(mesh: Mesh, faces: np.ndarray) -> np.ndarray:
    """Normalised AP position t in [0, 1] of face centroids (0 = posterior)."""
    c = mesh.nodes[faces].mean(axis=1)
    s = (c - mesh.frame.origin) @ np.asarray(mesh.frame.ap, dtype=float)
    span = s.max() - s.min()
    return np.zeros_like(s) if span <= 0 else (s - s.min()) / span


def graded_pressure(mesh: Mesh, set_name: str, mean_pressure: float,
                    ap_ratio: float) -> np.ndarray:
    """(F,) per-face pressure with anterior/posterior ratio and preserved resultant."""
    faces = mesh.surface_sets[set_name]
    _, area = face_normals_areas(mesh.nodes, faces)
    g = 1.0 + (ap_ratio - 1.0) * ap_coordinate(mesh, faces)
    return mean_pressure * g * area.sum() / (g * area).sum()


def bite_nodal_forces(mesh: Mesh, total_force: float, molar_weight: float,
                      inclination_deg: float = 0.0, set_name: str = "occlusal") -> np.ndarray:
    """(N, 3) nodal forces of the bite reaction; resultant = total_force * d."""
    faces = mesh.surface_sets[set_name]
    _, area = face_normals_areas(mesh.nodes, faces)
    t = ap_coordinate(mesh, faces)
    w = area * (molar_weight + (1.0 - molar_weight) * t)
    w = w / w.sum()
    th = np.deg2rad(inclination_deg)
    si, ap = np.asarray(mesh.frame.si, float), np.asarray(mesh.frame.ap, float)
    d = np.cos(th) * si + np.sin(th) * ap
    f = np.zeros((mesh.n_nodes, 3))
    fe = (total_force * w / 3.0)[:, None] * d[None, :]
    for k in range(3):
        np.add.at(f, faces[:, k], fe)
    return f


def build_daily_loading(mesh: Mesh, cfg: LoadingConfig, scenario: str) -> list[DailyLoad]:
    """Load history of one scenario (see module docstring)."""
    if scenario not in cfg.scenarios:
        raise KeyError(f"unknown scenario {scenario!r}; defined: {sorted(cfg.scenarios)}")
    _require(mesh, ("palate", "occlusal"))
    sc = cfg.scenarios[scenario]
    out: list[DailyLoad] = []

    if sc.resting_pressure > 0:
        p = graded_pressure(mesh, "palate", sc.resting_pressure, cfg.resting.ap_ratio)
        out.append(DailyLoad(LoadCase("rest_posture", pressures={"palate": p},
                                      cycles_per_day=1.0), static=True))

    sw = cfg.swallowing
    p_sw = sw.pressure * sc.swallowing_scale
    if p_sw > 0 and sw.cycles_per_day > 0:
        p = graded_pressure(mesh, "palate", p_sw, sw.ap_ratio)
        out.append(DailyLoad(LoadCase("swallow", pressures={"palate": p},
                                      cycles_per_day=sw.cycles_per_day), static=False))

    ch = cfg.mastication
    F = ch.force * sc.mastication_scale
    if F > 0 and ch.cycles_per_day > 0:
        f = bite_nodal_forces(mesh, F, ch.molar_weight, ch.inclination_deg)
        out.append(DailyLoad(LoadCase("chew", nodal_forces=f, cycles_per_day=ch.cycles_per_day),
                             static=False))

    gr = cfg.gravity
    if gr.enabled and gr.g > 0:
        gvec = -gr.g * np.asarray(mesh.frame.si, dtype=float)
        out.append(DailyLoad(LoadCase("gravity", gravity=gvec, cycles_per_day=1.0), static=True))

    if not out:
        raise ValueError(f"scenario {scenario!r} has no non-zero load case")
    return out
