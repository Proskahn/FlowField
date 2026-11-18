"""Chen et al. (2025): reported inputs and explicit reconstruction choices."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

DOI = '10.1002/ese3.70173'
MODEL_REVISION = 'chen-brinkman-disk-v1'
BOUNDS = ((70., 115.), (.3, 1.2), (1.6, 2.24))
PARAMETERS = ('angle_deg', 'flow_lpm', 'thickness_mm')


@dataclass(frozen=True)
class ChenConfig:
    mesh: str = 'coarse'
    diameter_mm: float = 160.  # Section 3.8; the parameter study uses 100 mm.
    inlet_width_mm: float = 4.
    inlet_height_mm: float = 1.5
    manifold_width_mm: float = 1.  # Not dimensioned; declared reconstruction.
    port_length_mm: float = 4.    # Not dimensioned; declared reconstruction.
    measurement_height_mm: float = 1.
    porosity: float = .8
    permeability_m2: float = 2.5e-9
    inertial_coefficient_m_inv: float = 0.  # C2 is not reported; Darcy limit.
    density_kg_m3: float = 1050.
    dynamic_viscosity_pa_s: float = .0012
    max_iterations: int = 3000
    residual_tolerance: float = 1e-6
    balance_tolerance: float = 1e-4
    steady_tolerance: float = 1e-3
    stability_window: int = 20
    write_interval: int = 100

    def __post_init__(self):
        if self.mesh not in ('coarse', 'medium', 'fine'):
            raise ValueError('mesh must be coarse, medium or fine')
        for key, value in asdict(self).items():
            if key == 'mesh':
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f'{key} must be finite numeric')
            if value < 0 or (value == 0 and key != 'inertial_coefficient_m_inv'):
                raise ValueError(f'{key} must be positive (C2 may be zero)')
        for key in ('max_iterations', 'stability_window', 'write_interval'):
            if type(getattr(self, key)) is not int:
                raise ValueError(f'{key} must be an integer')
        if not .2 <= self.porosity <= .9:
            raise ValueError('Paper porosity range is 0.2 to 0.9')
        if not 2 <= self.inlet_width_mm <= 6 or not 1 <= self.inlet_height_mm <= 2:
            raise ValueError('Paper inlet dimensions are 2..6 mm wide and 1..2 mm high')
        if self.diameter_mm not in (100., 160.):
            raise ValueError('Use the reported 100 or 160 mm diameter')
        if self.manifold_width_mm >= self.diameter_mm/10 or self.port_length_mm <= self.manifold_width_mm:
            raise ValueError('Require a narrow manifold and a longer inlet/outlet stem')
        if self.measurement_height_mm >= 1.6:
            raise ValueError('Measurement plane must be inside every optimization design')
        if self.residual_tolerance > 1e-5 or self.balance_tolerance > 1e-3 or self.steady_tolerance > 1e-3:
            raise ValueError('Numerical qualification tolerances may only be tightened')
        if self.stability_window < 5:
            raise ValueError('At least five stationary samples are required')

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(dict(config=asdict(self), revision=MODEL_REVISION),
            sort_keys=True, allow_nan=False).encode()).hexdigest()


def physical_geometry(angle, flow, thickness, config):
    values = (angle, flow, thickness)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('Design values must be finite numbers')
    if not 70 <= angle <= 115 or not .1 <= flow <= 1.2 or not 1 <= thickness <= 2.44:
        raise ValueError('Physical study range: angle 70..115 deg, flow 0.1..1.2 LPM, thickness 1..2.44 mm')
    if thickness <= config.measurement_height_mm:
        raise ValueError('The 1 mm measurement plane must lie inside the diffusion layer, not on its wall')
    if thickness < config.inlet_height_mm:
        raise ValueError('This reconstruction requires inlet height <= diffusion layer thickness')
    x = [(v-lo)/(hi-lo) for v, (lo, hi) in zip(values, BOUNDS)]
    return dict(zip(PARAMETERS, values), normalized_design=x,
        within_optimization_bounds=all(-1e-12 <= v <= 1+1e-12 for v in x),
        diameter_mm=config.diameter_mm, inlet_width_mm=config.inlet_width_mm,
        inlet_height_mm=config.inlet_height_mm, manifold_width_mm=config.manifold_width_mm,
        port_length_mm=config.port_length_mm, flow_m3_s=flow*1e-3/60,
        inlet_speed_m_s=flow*1e-3/60/(config.inlet_width_mm*config.inlet_height_mm*1e-6),
        constraints={})


def geometry(x, config):
    return physical_geometry(*(lo+(hi-lo)*v for v, (lo, hi) in zip(x, BOUNDS)), config)
