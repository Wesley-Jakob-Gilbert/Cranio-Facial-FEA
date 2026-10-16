"""Typed study configuration (dataclasses) loaded from YAML, with validation.

Every physical constant of the material, loading, stimulus and remodeling
models lives here so that each one can be varied in sensitivity studies. SI
units throughout (Pa, N, m, kg/m^3), time in days. Default values and their
sources are tabulated in docs/METHODS.md.

Unknown keys are rejected (typos must not silently fall back to defaults).
Numeric strings such as ``1.0e9`` (which YAML 1.1 parses as text) are
accepted for float fields.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np


class ConfigError(ValueError):
    """Invalid study configuration."""


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _positive(name: str, v: float) -> None:
    _check(np.isfinite(v) and v > 0, f"{name} must be > 0 (got {v})")


def _nonneg(name: str, v: float) -> None:
    _check(np.isfinite(v) and v >= 0, f"{name} must be >= 0 (got {v})")


def _poisson(name: str, v: float) -> None:
    _check(-1.0 < v < 0.5, f"{name} must lie in (-1, 0.5) (got {v})")


# --------------------------------------------------------------------------- #
# Materials
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OrthoConfig:
    """Cortical orthotropic constants at rho = 1. Axis 1: anatomical `ap`
    projected on the cortical plate; axis 2: in-plane transverse; axis 3:
    outward surface normal (plate thickness direction)."""

    E1: float = 20.0e9
    E2: float = 13.0e9
    E3: float = 11.0e9
    G12: float = 5.0e9
    G13: float = 4.0e9
    G23: float = 3.5e9
    nu12: float = 0.30
    nu13: float = 0.30
    nu23: float = 0.30

    def __post_init__(self):
        for k in ("E1", "E2", "E3", "G12", "G13", "G23"):
            _positive(k, getattr(self, k))
        E1, E2, E3 = self.E1, self.E2, self.E3
        n12, n13, n23 = self.nu12, self.nu13, self.nu23
        S = np.array([[1 / E1, -n12 / E1, -n13 / E1],
                      [-n12 / E1, 1 / E2, -n23 / E2],
                      [-n13 / E1, -n23 / E2, 1 / E3]])
        _check(bool(np.all(np.linalg.eigvalsh(S) > 0)),
               "orthotropic constants are not positive definite")


@dataclass(frozen=True)
class CorticalConfig:
    E: float = 15.0e9
    nu: float = 0.30
    n: float = 3.0
    mass_density: float = 1850.0
    anisotropy: str = "isotropic"          # "isotropic" | "orthotropic"
    ortho: OrthoConfig = field(default_factory=OrthoConfig)

    def __post_init__(self):
        _positive("E", self.E)
        _poisson("nu", self.nu)
        _nonneg("n", self.n)
        _positive("mass_density", self.mass_density)
        _check(self.anisotropy in ("isotropic", "orthotropic"),
               f"anisotropy must be 'isotropic' or 'orthotropic' (got {self.anisotropy!r})")


@dataclass(frozen=True)
class CancellousConfig:
    E: float = 1.0e9
    nu: float = 0.30
    n: float = 2.0
    mass_density: float = 1000.0

    def __post_init__(self):
        _positive("E", self.E)
        _poisson("nu", self.nu)
        _nonneg("n", self.n)
        _positive("mass_density", self.mass_density)


@dataclass(frozen=True)
class SutureConfig:
    E: float = 1.0e7
    nu: float = 0.45
    mass_density: float = 1100.0

    def __post_init__(self):
        _positive("E", self.E)
        _poisson("nu", self.nu)
        _positive("mass_density", self.mass_density)


@dataclass(frozen=True)
class MaterialsConfig:
    cortical: CorticalConfig = field(default_factory=CorticalConfig)
    cancellous: CancellousConfig = field(default_factory=CancellousConfig)
    suture: SutureConfig = field(default_factory=SutureConfig)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RestingPostureConfig:
    """Sustained tongue-palate contact; magnitude is set per scenario.
    ap_ratio = anterior / posterior pressure (linear along frame.ap)."""

    ap_ratio: float = 2.0

    def __post_init__(self):
        _positive("ap_ratio", self.ap_ratio)


@dataclass(frozen=True)
class SwallowingConfig:
    pressure: float = 15.0e3          # mean peak palatal pressure, Pa
    cycles_per_day: float = 600.0
    ap_ratio: float = 1.0

    def __post_init__(self):
        _nonneg("pressure", self.pressure)
        _nonneg("cycles_per_day", self.cycles_per_day)
        _positive("ap_ratio", self.ap_ratio)


@dataclass(frozen=True)
class MasticationConfig:
    force: float = 100.0              # total bite reaction on the arch, N
    cycles_per_day: float = 1500.0
    molar_weight: float = 3.0         # posterior / anterior traction ratio along the arch
    inclination_deg: float = 0.0      # tilt of the force from +si toward +ap

    def __post_init__(self):
        _nonneg("force", self.force)
        _nonneg("cycles_per_day", self.cycles_per_day)
        _positive("molar_weight", self.molar_weight)
        _check(-90.0 < self.inclination_deg < 90.0, "inclination_deg must lie in (-90, 90)")


@dataclass(frozen=True)
class GravityConfig:
    enabled: bool = True
    g: float = 9.81

    def __post_init__(self):
        _nonneg("g", self.g)


@dataclass(frozen=True)
class ScenarioConfig:
    resting_pressure: float           # mean sustained palatal pressure, Pa
    swallowing_scale: float = 1.0     # multiplies swallowing pressure
    mastication_scale: float = 1.0    # multiplies bite force
    description: str = ""

    def __post_init__(self):
        _nonneg("resting_pressure", self.resting_pressure)
        _nonneg("swallowing_scale", self.swallowing_scale)
        _nonneg("mastication_scale", self.mastication_scale)


def _default_scenarios() -> dict[str, ScenarioConfig]:
    return {
        "reference": ScenarioConfig(200.0, description="habitual resting tongue posture"),
        "mewing": ScenarioConfig(2000.0, description="sustained tongue-on-palate posture"),
        "mouth_breathing": ScenarioConfig(0.0, description="tongue low, no palatal contact"),
    }


@dataclass(frozen=True)
class LoadingConfig:
    resting: RestingPostureConfig = field(default_factory=RestingPostureConfig)
    swallowing: SwallowingConfig = field(default_factory=SwallowingConfig)
    mastication: MasticationConfig = field(default_factory=MasticationConfig)
    gravity: GravityConfig = field(default_factory=GravityConfig)
    scenarios: dict[str, ScenarioConfig] = field(default_factory=_default_scenarios)

    def __post_init__(self):
        _check("reference" in self.scenarios,
               "loading.scenarios must contain 'reference' (it defines the setpoint)")


# --------------------------------------------------------------------------- #
# Stimulus and remodeling
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StimulusConfig:
    m: float = 4.0                         # cycle-number exponent
    static_equivalent_cycles: float = 1.0  # daily cycle weight of a sustained load
    setpoint_floor: float = 1e-6           # Psi* below floor * max(Psi*) -> no signal

    def __post_init__(self):
        _check(np.isfinite(self.m) and self.m >= 1.0, f"m must be >= 1 (got {self.m})")
        _nonneg("static_equivalent_cycles", self.static_equivalent_cycles)
        _check(0.0 <= self.setpoint_floor < 1.0, "setpoint_floor must lie in [0, 1)")


@dataclass(frozen=True)
class LayerValues:
    cortical: float
    cancellous: float


@dataclass(frozen=True)
class RemodelingConfig:
    lazy_zone: float = 0.10
    # relative density per day per unit stimulus error beyond the lazy zone
    formation_rate: LayerValues = field(default_factory=lambda: LayerValues(1.1e-4, 7.7e-4))
    resorption_rate: LayerValues = field(default_factory=lambda: LayerValues(1.1e-4, 7.7e-4))
    rho_min: LayerValues = field(default_factory=lambda: LayerValues(0.5, 0.25))
    rho_max: LayerValues = field(default_factory=lambda: LayerValues(1.15, 2.0))
    days: float = 365.0
    n_steps: int = 12
    smoothing: bool = True

    def __post_init__(self):
        _check(0.0 <= self.lazy_zone < 1.0, f"lazy_zone must lie in [0, 1) (got {self.lazy_zone})")
        for nm in ("formation_rate", "resorption_rate"):
            v = getattr(self, nm)
            _nonneg(f"{nm}.cortical", v.cortical)
            _nonneg(f"{nm}.cancellous", v.cancellous)
        for layer in ("cortical", "cancellous"):
            lo, hi = getattr(self.rho_min, layer), getattr(self.rho_max, layer)
            _check(0.0 < lo <= 1.0 <= hi,
                   f"rho bounds for {layer} must satisfy 0 < rho_min <= 1 <= rho_max "
                   f"(got {lo}, {hi})")
        _positive("days", self.days)
        _check(self.n_steps >= 1, "n_steps must be >= 1")

    @property
    def dt(self) -> float:
        return self.days / self.n_steps


# --------------------------------------------------------------------------- #
# Mesh, solver, sensitivity, study
# --------------------------------------------------------------------------- #

_MESH_SOURCES = ("file", "template", "box")


@dataclass(frozen=True)
class MeshConfig:
    source: str = "file"               # "file" | "template" | "box"
    path: str | None = None            # for source "file"
    fallback: list[str] = field(default_factory=list)   # tried in order if source fails
    template: dict = field(default_factory=dict)        # kwargs of build_template_mesh
    box: dict = field(default_factory=dict)             # kwargs of labeled_box_mesh

    def __post_init__(self):
        for s in (self.source, *self.fallback):
            _check(s in _MESH_SOURCES, f"mesh source must be one of {_MESH_SOURCES} (got {s!r})")
        _check(self.source != "file" or bool(self.path), "mesh.path is required for source 'file'")


@dataclass(frozen=True)
class SolverConfig:
    name: str = "reference"
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ParamRange:
    lo: float
    hi: float
    log: bool = False

    def __post_init__(self):
        _check(self.hi > self.lo, f"parameter range needs hi > lo (got {self.lo}, {self.hi})")
        _check(not self.log or self.lo > 0, "log-scaled parameter range needs lo > 0")


def _default_morris_params() -> dict[str, ParamRange]:
    return {
        "loading.scenarios.mewing.resting_pressure": ParamRange(1.0e3, 4.0e3),
        "loading.scenarios.reference.resting_pressure": ParamRange(50.0, 500.0),
        "loading.swallowing.pressure": ParamRange(5.0e3, 30.0e3),
        "loading.swallowing.cycles_per_day": ParamRange(230.0, 1000.0),
        "loading.mastication.force": ParamRange(50.0, 250.0),
        "loading.mastication.cycles_per_day": ParamRange(200.0, 3300.0),
        "stimulus.m": ParamRange(2.0, 8.0),
        "stimulus.static_equivalent_cycles": ParamRange(1.0, 1.0e6, log=True),
        "remodeling.lazy_zone": ParamRange(0.0, 0.3),
        "remodeling.formation_rate.cancellous": ParamRange(2.0e-4, 2.0e-3, log=True),
        "materials.cortical.E": ParamRange(10.0e9, 20.0e9),
        "materials.cancellous.E": ParamRange(0.3e9, 2.0e9),
        "materials.suture.E": ParamRange(1.0e6, 1.0e8, log=True),
    }


@dataclass(frozen=True)
class MorrisConfig:
    trajectories: int = 10
    levels: int = 4
    seed: int = 0
    parameters: dict[str, ParamRange] = field(default_factory=_default_morris_params)

    def __post_init__(self):
        _check(self.trajectories >= 2, "morris.trajectories must be >= 2")
        _check(self.levels >= 2 and self.levels % 2 == 0, "morris.levels must be even and >= 2")
        _check(len(self.parameters) >= 1, "morris.parameters must not be empty")


_REGIONS = ("region_palatal_vault", "region_alveolar_anterior", "region_alveolar_posterior",
            "region_basal")


@dataclass(frozen=True)
class SensitivityConfig:
    static_cycles: list[float] = field(
        default_factory=lambda: [1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    region: str = "region_palatal_vault"
    contrast: list[str] = field(default_factory=lambda: ["mewing", "mouth_breathing"])
    morris: MorrisConfig = field(default_factory=MorrisConfig)

    def __post_init__(self):
        _check(len(self.static_cycles) >= 1, "sensitivity.static_cycles must not be empty")
        for v in self.static_cycles:
            _nonneg("static_cycles entry", v)
        _check(len(self.contrast) == 2, "sensitivity.contrast must name two scenarios")
        _check(self.region in _REGIONS,
               f"sensitivity.region must be one of {_REGIONS} (got {self.region!r})")


@dataclass(frozen=True)
class StudyConfig:
    name: str = "study"
    mesh: MeshConfig = field(default_factory=lambda: MeshConfig(source="box"))
    solver: SolverConfig = field(default_factory=SolverConfig)
    materials: MaterialsConfig = field(default_factory=MaterialsConfig)
    loading: LoadingConfig = field(default_factory=LoadingConfig)
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    remodeling: RemodelingConfig = field(default_factory=RemodelingConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    output_dir: str = "results/study"

    def __post_init__(self):
        for s in self.sensitivity.contrast:
            _check(s in self.loading.scenarios,
                   f"sensitivity.contrast scenario {s!r} is not defined in loading.scenarios")

    # -- conversion helpers ------------------------------------------------- #
    def to_dict(self) -> dict:
        return to_dict(self)

    def hash(self) -> str:
        """SHA-256 of the canonical JSON form (key-sorted) of the config."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=float).encode()
        return hashlib.sha256(blob).hexdigest()

    def with_overrides(self, overrides: dict[str, Any]) -> StudyConfig:
        """New config with dotted-path overrides, e.g. {'stimulus.m': 3.0}."""
        d = copy.deepcopy(self.to_dict())
        for path, value in overrides.items():
            keys = path.split(".")
            node = d
            for k in keys[:-1]:
                if not isinstance(node, dict) or k not in node:
                    raise ConfigError(f"unknown config path {path!r}")
                node = node[k]
            if not isinstance(node, dict) or keys[-1] not in node:
                raise ConfigError(f"unknown config path {path!r}")
            node[keys[-1]] = value
        return from_dict(StudyConfig, d)


# --------------------------------------------------------------------------- #
# Generic (de)serialisation
# --------------------------------------------------------------------------- #

def to_dict(obj) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


def _convert(tp, value, path: str):
    origin = typing.get_origin(tp)
    if origin in (Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _convert(inner[0], value, path)
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value, path)
    if origin is dict or tp is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        args = typing.get_args(tp)
        if not args:
            return dict(value)
        return {str(k): _convert(args[1], v, f"{path}.{k}") for k, v in value.items()}
    if origin is list or tp is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        (arg,) = typing.get_args(tp) or (Any,)
        return [_convert(arg, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{path}: expected a number, got {value!r}") from None
    if tp is int:
        if isinstance(value, bool) or not float(value).is_integer():
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return int(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    return value


def from_dict(cls, data: dict | None, path: str = ""):
    """Build dataclass `cls` from a (YAML) mapping, recursively and with validation."""
    data = {} if data is None else data
    if dataclasses.is_dataclass(data):
        return data
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"{path or cls.__name__}: unknown key(s) {unknown}; "
                          f"allowed: {sorted(names)}")
    kwargs = {k: _convert(hints[k], v, f"{path}.{k}" if path else k) for k, v in data.items()}
    try:
        return cls(**kwargs)
    except ConfigError as e:
        raise ConfigError(f"{path or cls.__name__}: {e}") from None
    except TypeError as e:
        raise ConfigError(f"{path or cls.__name__}: {e}") from None


def load_config(path: str | Path) -> StudyConfig:
    """Read and validate a study YAML file."""
    import yaml

    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return from_dict(StudyConfig, data)
