"""P000 dimensions are SI internally; optimization coordinates are in [0, 1]."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

WIDTH_BOUNDS_M = (0.0005, 0.002)
HEIGHT_BOUNDS_M = (0.0005, 0.002)
MESHES = {"coarse": (50, 8, 8), "medium": (100, 16, 16), "fine": (200, 32, 32)}


@dataclass(frozen=True)
class ChannelConfig:
    width_m: float = 0.001
    height_m: float = 0.001
    length_m: float = 0.05
    flow_rate_m3_s: float = 1e-8
    density_kg_m3: float = 998.2
    kinematic_viscosity_m2_s: float = 1e-6
    cells: tuple[int, int, int] = MESHES["coarse"]
    max_iterations: int = 3000
    residual_tolerance: float = 1e-7
    mass_balance_tolerance: float = 1e-5
    reference_tolerance: float = 0.08

    def __post_init__(self):
        for name in ("width_m", "height_m", "length_m", "flow_rate_m3_s",
                     "density_kg_m3", "kinematic_viscosity_m2_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name, bounds in (("width_m", WIDTH_BOUNDS_M), ("height_m", HEIGHT_BOUNDS_M)):
            if not bounds[0] <= getattr(self, name) <= bounds[1]:
                raise ValueError(f"{name} must lie in {bounds} m for P000")
        cells = tuple(self.cells)
        if len(cells) != 3 or any(type(n) is not int or n < 2 for n in cells):
            raise ValueError("cells must contain three integers >= 2")
        if math.prod(cells) > 2_000_000:
            raise ValueError("P000 mesh limit is 2,000,000 cells")
        object.__setattr__(self, "cells", cells)
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        for name in ("residual_tolerance", "mass_balance_tolerance", "reference_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
        if self.reynolds_number > 100:
            raise ValueError("P000 verification is limited to Reynolds number <= 100")
        if self.length_m / self.hydraulic_diameter_m < 20:
            raise ValueError("P000 requires L/Dh >= 20 to limit entrance effects")

    @property
    def mean_velocity_m_s(self):
        return self.flow_rate_m3_s / (self.width_m * self.height_m)

    @property
    def hydraulic_diameter_m(self):
        return 2 * self.width_m * self.height_m / (self.width_m + self.height_m)

    @property
    def reynolds_number(self):
        return self.mean_velocity_m_s * self.hydraulic_diameter_m / self.kinematic_viscosity_m2_s

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as stream:
            return cls(**json.load(stream))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def decode_design(x, base=None):
    """Map [width, height] in [0,1]^2 linearly to [0.5,2] mm, without clipping."""
    values = tuple(x)
    if len(values) != 2:
        raise ValueError("P000 requires two coordinates: [width, height]")
    if any(isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Design coordinates must be finite and within [0, 1]")
    w = WIDTH_BOUNDS_M[0] + float(values[0]) * (WIDTH_BOUNDS_M[1] - WIDTH_BOUNDS_M[0])
    h = HEIGHT_BOUNDS_M[0] + float(values[1]) * (HEIGHT_BOUNDS_M[1] - HEIGHT_BOUNDS_M[0])
    return replace(base or ChannelConfig(), width_m=w, height_m=h)


def reference_pressure_drop(config):
    """Fully developed laminar rectangular-duct pressure drop, in Pa.

    Q = b*a^3*DeltaP/(12*mu*L) *
        [1 - 192*a/(pi^5*b) * sum_odd(tanh(n*pi*b/(2*a))/n^5)].
    a <= b; 100 odd terms make truncation negligible relative to mesh error.
    The CFD uses a uniform inlet, so its total drop also includes entrance effects.
    """
    a, b = sorted((config.width_m, config.height_m))
    series = math.fsum(math.tanh(n * math.pi * b / (2 * a)) / n**5 for n in range(1, 200, 2))
    correction = 1 - 192 * a * series / (math.pi**5 * b)
    mu = config.density_kg_m3 * config.kinematic_viscosity_m2_s
    return 12 * mu * config.length_m * config.flow_rate_m3_s / (b * a**3 * correction)
