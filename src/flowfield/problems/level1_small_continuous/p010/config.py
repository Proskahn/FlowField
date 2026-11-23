"""Xie et al. ESCFF PEMFC: paper inputs and explicit closure estimates (SI)."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

DOI='10.1016/j.ijheatmasstransfer.2025.127792'
MODEL_REVISION='escff-pemfc-coupled-v1'
PARAMETERS=('temperature_K','humidity_percent','cathode_stoichiometry','gdl_porosity')
BOUNDS=((333.15,353.15),(80.,100.),(1.,4.),(.2,.8))
BASELINE=(333.15,90.,2.2,.6)
FLOW_FIELDS=('ESCFF-1','ESCFF-2','ESCFF-3','ESCFF-4','serpentine','CDLIS')

@dataclass(frozen=True)
class PEMFCConfig:
    # Geometry
    # Fixed dimensions are in decode() and geometry.py; operating design bounds
    # are declared in PARAMETERS and BOUNDS above.
    flow_field: str='ESCFF-3'

    # Physical parameters
    open_circuit_voltage: float=.972
    cl_porosity: float=.4
    ionomer_fraction: float=.3
    permeability_gdl: float=1e-12
    permeability_cl: float=1e-13
    electronic_conductivity_gdl: float=1000.
    electronic_conductivity_cl: float=1000.
    electronic_conductivity_bp: float=20000.
    exchange_current_anode: float=1e9  # Volumetric reference, A/m3; unreported.
    exchange_current_cathode: float=1e4
    kinetic_reference_temperature: float=333.15
    activation_energy_anode: float=20000.
    activation_energy_cathode: float=66000.
    transfer_coefficient: float=.5
    concentration_exponent: float=3.  # Printed Eqs. 9/10; Table 3 instead says 1.
    oxygen_diffusivity: float=2.5e-5
    hydrogen_diffusivity: float=8e-5
    vapor_diffusivity_air: float=2.8e-5
    air_viscosity: float=2e-5
    hydrogen_viscosity: float=9.6e-6
    liquid_density: float=983.2
    liquid_viscosity: float=.000466
    liquid_conductivity: float=.65
    surface_tension: float=.0662
    contact_angle_deg: float=110.
    channel_liquid_dispersion: float=1e-5
    phase_change_rate: float=100.
    sorption_rate: float=10.
    membrane_dry_density: float=2000.
    membrane_equivalent_weight: float=1.1  # Table 1100 interpreted as g/mol -> kg/mol.
    membrane_diffusivity: float=2e-10
    latent_heat: float=2.36e6

    # Initial conditions
    # case.py initializes temperature/humidity from the design, charge from
    # open_circuit_voltage, and fixed velocity, pressure and liquid saturation.

    # Boundary conditions
    # Temperature, humidity and cathode stoichiometry are design variables.
    # reference_current_density sets the stoichiometric inlet-flow reference.
    cell_voltage: float=.4
    outlet_pressure: float=101325.
    anode_stoichiometry: float=1.1
    reference_current_density: float=15000.  # Unreported Iref, A/m2. Never fitted to Pnet.

    # Mesh and discretization
    mesh: str='coarse'

    # Solver controls
    linear_solver: str='iterative'
    iterations: int=3000
    minimum_iterations: int=60
    processes: int=1
    liquid_pseudo_time: float=5.  # Numerical false-transient storage; absent at convergence.
    velocity_relaxation: float=.5
    scalar_relaxation: float=.3
    field_tolerance: float=1e-5
    voltage_ramp_iterations: int=40

    # Diagnostics and output
    write_interval: int=100
    balance_tolerance: float=1e-3

    def __post_init__(self):
        if self.mesh not in ('coarse','medium','fine'):raise ValueError('Unknown mesh')
        if self.flow_field not in FLOW_FIELDS:raise ValueError('Unknown flow field')
        if self.linear_solver not in ('direct','iterative'):raise ValueError('Unknown linear solver')
        for k,v in asdict(self).items():
            if k in ('mesh','flow_field','linear_solver'):continue
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
                raise ValueError(f'{k} must be positive and finite')
        for k in ('iterations','minimum_iterations','write_interval','processes','voltage_ramp_iterations'):
            if type(getattr(self,k)) is not int:raise ValueError(f'{k} must be an integer')
        if self.processes!=1:raise ValueError('P010 supports one CPU process per case')
        if not .3<=self.cell_voltage<self.open_circuit_voltage<=1.1:raise ValueError('Invalid fuel-cell voltage')
        for k in ('cl_porosity','ionomer_fraction','velocity_relaxation','scalar_relaxation'):
            if not 0<getattr(self,k)<1:raise ValueError(f'{k} must be in (0,1)')
        if not 90<self.contact_angle_deg<180:raise ValueError('This closure requires hydrophobic porous media')
        if self.field_tolerance>1e-5 or self.balance_tolerance>1e-3:raise ValueError('Acceptance limits may only be tightened')
    def to_dict(self):return asdict(self)
    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(dict(revision=MODEL_REVISION,config=self.to_dict()),sort_keys=True,allow_nan=False).encode()).hexdigest()

def normalized(values):return [(v-lo)/(hi-lo) for v,(lo,hi) in zip(values,BOUNDS)]

def decode(x,config):
    return dict(normalized_design=list(x),**{k:lo+(hi-lo)*v for k,v,(lo,hi) in zip(PARAMETERS,x,BOUNDS)},
        flow_field=config.flow_field,active_area_m2=.023**2,footprint_mm=[23.,23.],channel_depth_mm=1.,constraints={})
