"""Source geometry and explicitly auxiliary hydraulic conditions."""
from dataclasses import dataclass,asdict
import hashlib,json,math

HYDRAULIC_REVISION='channel-only-laminar-v1'

@dataclass(frozen=True)
class HydraulicConfig:
    mesh: str='coarse'
    flow_rate_m3_s: float=1e-6
    density_kg_m3: float=1.
    viscosity_pa_s: float=2.07e-5
    max_iterations: int=3000
    residual_tolerance: float=1e-8
    balance_tolerance: float=1e-6
    steady_tolerance: float=1e-5

    def __post_init__(self):
        if self.mesh not in ('coarse','medium','fine'):raise ValueError('Unknown mesh')
        if type(self.max_iterations) is not int or self.max_iterations<50:raise ValueError('Need at least 50 iterations')
        for k,v in asdict(self).items():
            if k in ('mesh','max_iterations'):continue
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:raise ValueError(f'Invalid {k}')
        if self.residual_tolerance>1e-8 or self.balance_tolerance>1e-6 or self.steady_tolerance>1e-5:
            raise ValueError('Tolerances may be tightened, not relaxed')
    @property
    def cells_per_width(self):return {'coarse':4,'medium':8,'fine':12}[self.mesh]
    @property
    def mean_velocity_m_s(self):return self.flow_rate_m3_s/1e-6
    @property
    def reynolds_number(self):return self.density_kg_m3*self.mean_velocity_m_s*.001/self.viscosity_pa_s
    def to_dict(self):return asdict(self)
    @property
    def fingerprint(self):return hashlib.sha256(json.dumps(dict(config=asdict(self),revision=HYDRAULIC_REVISION),sort_keys=True).encode()).hexdigest()
