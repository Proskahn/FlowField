"""Jin et al., Fuel 404 (2026) 136144: dimensionless 2D reconstruction."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

DOI = '10.1016/j.fuel.2025.136144'
MODEL_REVISION = 'jin-2d-flow-heat-v1'
WEIGHTS = ((.25, .75), (.4, .6), (.6, .4), (.8, .2), (1., 0.))


@dataclass(frozen=True)
class TopologyConfig:
    mesh: str = 'coarse'
    design_nx: int = 16
    design_ny: int = 16
    mirror_y: bool = True
    scheme: int = 4
    reynolds: float = 130.
    prandtl: float = 6.78
    volume_limit: float = .635
    ramp_q: float = .01
    projection_beta: float = 8.
    projection_threshold: float = .5
    pumping_power: float = 1.
    # Not supplied by the paper. Explicit reconstruction choices, not fitted data.
    darcy: float = 1e-4
    heat_coefficient: float = 1.
    filter_radius: float = .15
    iterations: int = 4000
    write_interval: int = 200
    diagnostic_interval: int = 10
    outer_tolerance: float = 1e-7
    balance_tolerance: float = 1e-5
    velocity_relaxation: float = .6
    pressure_relaxation: float = .3
    power_relaxation: float = .2
    volume_penalty: float = 100.

    def __post_init__(self):
        for key in ('design_nx', 'design_ny', 'scheme', 'iterations', 'write_interval', 'diagnostic_interval'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.mesh not in ('coarse', 'medium', 'fine') or self.scheme not in range(1, 6):
            raise ValueError('Unknown mesh or paper weighting scheme (1..5)')
        if type(self.mirror_y) is not bool or self.design_ny % 2 or min(self.design_nx,self.design_ny) < 2:
            raise ValueError('Use an even design_ny >= 2, design_nx >= 2 and boolean mirror_y')
        for key, value in asdict(self).items():
            if key in ('mesh','mirror_y','design_nx','design_ny','scheme','iterations','write_interval','diagnostic_interval'):
                continue
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if not 0 < self.volume_limit < 1 or not 0 < self.projection_threshold < 1:
            raise ValueError('Fractions must be in (0,1)')
        if any(not 0 < v <= 1 for v in (self.velocity_relaxation,self.pressure_relaxation,self.power_relaxation)):
            raise ValueError('Relaxation factors must be in (0,1]')
        if self.outer_tolerance > 1e-7 or self.balance_tolerance > 1e-5:
            raise ValueError('Qualification tolerances can be tightened, not relaxed')

    @property
    def dimension(self):
        return self.design_nx * self.design_ny // (2 if self.mirror_y else 1)

    @property
    def weights(self): return WEIGHTS[self.scheme-1]
    @property
    def alpha_s(self): return (1+1/self.reynolds)/self.darcy
    def to_dict(self): return asdict(self)
    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(dict(config=self.to_dict(),revision=MODEL_REVISION),sort_keys=True,allow_nan=False).encode()).hexdigest()

