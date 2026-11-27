"""Behrou 2019 reference inputs; numerical and IOH choices are versioned."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math

DOI = '10.1016/j.ijheatmasstransfer.2019.01.050'
MODEL_REVISION = 'depth-averaged-cathode-v1'
CONTINUATION = ((1.,10000.),(2.,1000.),(3.,100.),(5.,10.))
SCALABLE_MAPPING = 'bilinear_cell_filter_v1'


@dataclass(frozen=True)
class TopologyConfig:
    # Geometry
    length_m: float = .02236
    inlet_width_m: float = .0012
    outlet_width_m: float = .0012
    channel_depth_m: float = .001
    gdl_depth_m: float = .00038

    # Physical parameters
    open_circuit_voltage_v: float = 1.22
    ohmic_resistance_ohm_m2: float = 7e-5
    temperature_k: float = 343.15
    oxygen_reference_mol_m3: float = 30.
    exchange_current_a_m2: float = .17
    cathodic_alpha: float = .5
    gas_density_kg_m3: float = 1.
    viscosity_pa_s: float = 2.07e-5
    gdl_porosity: float = .8
    gdl_permeability_m2: float = 1.38e-11

    # Initial conditions
    # case.py starts velocity/pressure at zero and species at their inlet fractions.

    # Boundary conditions
    # The outlet pressure, wall and symmetry conditions are fixed in case.py.
    inlet_pressure_pa: float = 5.
    cell_voltage_v: float = .4
    oxygen_inlet: float = .228
    water_inlet: float = .023

    # Mesh and discretization
    mesh: str = 'coarse'
    design_nx: int = 32
    design_ny: int = 64

    # Topology parameterization
    design_mapping: str = 'nodal_filter'
    filter_radius_m: float = .00035
    projection_beta: float = 50.
    stage: int = 1  # At 5 Pa, the paper skips stage 0.

    # Solver controls
    iterations: int = 3000
    outer_tolerance: float = 1e-7

    # Objectives and constraints
    volume_limit: float = .5
    power_weight: float = 1.
    uniformity_weight: float = 0.
    # The weight is weird. However, the original paper also use this weight.???
    power_scale_w: float = 1.
    current_std_scale_a_m2: float = 1.
    volume_penalty: float = 10.

    # Diagnostics and output
    write_interval: int = 200
    balance_tolerance: float = 1e-5

    def __post_init__(self):
        for name in ('design_nx','design_ny','iterations','write_interval','stage'):
            if type(getattr(self,name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if not (self.design_nx>=2 and self.design_ny>=2 and self.iterations>=1 and self.write_interval>=1):
            raise ValueError('Grid sizes and iteration counts must be positive')
        if self.stage not in range(4) or self.mesh not in ('coarse','medium','fine'):
            raise ValueError('Unknown continuation stage or mesh')
        if self.design_mapping not in ('nodal_filter', SCALABLE_MAPPING):
            raise ValueError('Unknown P003 design_mapping')
        for key,v in asdict(self).items():
            if key in ('mesh','design_mapping','stage','design_nx','design_ny','iterations','write_interval'):
                continue
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
                raise ValueError(f'{key} must be finite and nonnegative')
            if v == 0 and key not in ('projection_beta','uniformity_weight','power_weight','volume_penalty','water_inlet'):
                raise ValueError(f'{key} must be positive')
        if not (0<self.cathodic_alpha<1 and 0<self.gdl_porosity<1 and 0<self.volume_limit<=1):
            raise ValueError('Invalid fraction')
        if self.oxygen_inlet+self.water_inlet>=1 or self.cell_voltage_v>=self.open_circuit_voltage_v:
            raise ValueError('Need positive nitrogen and positive fuel-cell overpotential')
        if min(self.inlet_width_m,self.outlet_width_m)<=0 or max(self.inlet_width_m,self.outlet_width_m)>=self.width_m/2:
            raise ValueError('Ports must fit the half-domain')
        if self.power_weight+self.uniformity_weight<=0:
            raise ValueError('At least one objective weight must be positive')
        if self.outer_tolerance>1e-7 or self.balance_tolerance>1e-5:
            raise ValueError('Convergence tolerances may be tightened, not relaxed')
        # Every CFD centroid must have at least one nodal filter neighbor.
        if (self.design_mapping == 'nodal_filter'
                and self.filter_radius_m <= .5*math.hypot(self.width_m/self.design_nx,self.length_m/self.design_ny)):
            raise ValueError('Design grid is too coarse for the filter radius')
        if self.design_mapping == SCALABLE_MAPPING:
            from .design import grid
            x,y=grid(self)
            if self.design_nx+1 > len(x)-1 or self.design_ny+1 > len(y)-1:
                raise ValueError('Scalable design points must not exceed CFD cells along either axis; '
                                 'choose fewer design points or a finer CFD mesh')

    @property
    def width_m(self): return self.length_m/2
    @property
    def depth_m(self): return self.channel_depth_m+self.gdl_depth_m
    @property
    def dimension(self): return (self.design_nx+1)*(self.design_ny+1)
    @property
    def area_m2(self): return self.width_m*self.length_m
    @property
    def interpolation(self):
        p,qa=CONTINUATION[self.stage]
        h=self.depth_m;g=self.gdl_depth_m;c=self.channel_depth_m
        return dict(simp_p=p,ramp_q=qa,eps_rib=self.gdl_porosity*g/h,
                    eps_channel=(self.gdl_porosity*g+c)/h,
                    diffusion_rib=self.gdl_porosity**1.5*g/h,
                    diffusion_channel=(self.gdl_porosity**1.5*g+c)/h,
                    alpha_rib=self.viscosity_pa_s/(self.gdl_permeability_m2*g/h),alpha_channel=0.)
    def to_dict(self):
        values=asdict(self)
        # Preserve archived configurations, fingerprints and numerical results
        # for the original paper mapping. The new mapping versions itself.
        if self.design_mapping == 'nodal_filter':values.pop('design_mapping')
        return values
    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(dict(config=self.to_dict(),revision=MODEL_REVISION),sort_keys=True,allow_nan=False).encode()).hexdigest()


def current_density(oxygen_mass_fraction, config):
    """Unique nonnegative solution of Eqs. 14–15, with C= rho*w/M_O2."""
    if not math.isfinite(oxygen_mass_fraction) or not 0<=oxygen_mass_fraction<=1:
        raise ValueError('Invalid oxygen mass fraction')
    pref=config.exchange_current_a_m2*config.gas_density_kg_m3*oxygen_mass_fraction/.032/config.oxygen_reference_mol_m3
    a=config.cathodic_alpha;v=config.open_circuit_voltage_v-config.cell_voltage_v
    f=96485.3/(8.3144*config.temperature_k);lo=0.;hi=v/config.ohmic_resistance_ohm_m2
    for _ in range(75):
        mid=(lo+hi)/2;eta=v-config.ohmic_resistance_ohm_m2*mid
        rhs=pref*(math.exp(a*f*eta)-math.exp(-(1-a)*f*eta))
        if mid>rhs:hi=mid
        else:lo=mid
    return (lo+hi)/2
