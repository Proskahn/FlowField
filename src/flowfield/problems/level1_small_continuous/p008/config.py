"""Li et al., JECE 2026: reported parameters and declared reconstruction inputs."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

DOI='10.1016/j.jece.2026.122524'
MODEL_REVISION='aem-mixture-darcy-thermal-v2'

@dataclass(frozen=True)
class AEMConfig:
    mesh: str='coarse'
    linear_solver: str='iterative'
    energy_form: str='lte_phase_advection'  # See model notes on printed Eq45.
    iterations: int=2500
    minimum_iterations: int=30
    write_interval: int=100
    processes: int=1
    cell_voltage: float=2.2
    inlet_flow_ml_min: float=50.
    inlet_temperature: float=333.15
    ambient_temperature: float=293.15
    external_heat_transfer: float=50.
    outlet_pressure: float=101325.
    koh_molarity: float=1.
    koh_conductivity_scale: float=1.  # Literal Eq32 in SI; 100 tests S/cm interpretation.
    membrane_water_content: float=18.
    oxygen_channel_dispersion: float=2e-5  # Unreported effective phase dispersion, m2/s.
    hydrogen_channel_dispersion: float=6e-5
    liquid_density: float=1030.  # Estimated 1 M KOH at 60 C; not tabulated.
    liquid_viscosity: float=.00055
    liquid_cp: float=4000.
    liquid_conductivity: float=.65
    oxygen_viscosity: float=2.2e-5
    hydrogen_viscosity: float=9.7e-6
    oxygen_cp: float=930.
    hydrogen_cp: float=14300.
    oxygen_conductivity: float=.028
    hydrogen_conductivity: float=.20
    porosity_ptl: float=.6
    porosity_cl: float=.5
    permeability_ptl: float=1e-12
    permeability_cl: float=1e-13
    surface_tension: float=.06624
    contact_angle_deg: float=60.
    maximum_packing: float=.99
    exchange_current_anode: float=21.9
    exchange_current_cathode: float=13200.
    active_area_anode: float=3e6
    active_area_cathode: float=3e6
    activation_energy_anode: float=76000.
    activation_energy_cathode: float=18000.
    kinetic_reference_temperature: float=298.15  # Unreported reference temperature.
    velocity_relaxation: float=.5
    thermal_relaxation: float=.3
    field_tolerance: float=1e-5
    balance_tolerance: float=1e-3
    ewm_calibration: dict | None=None

    def __post_init__(self):
        if self.linear_solver not in ('direct','iterative'):raise ValueError('Unknown linear solver')
        if self.energy_form not in ('lte_phase_advection','paper_printed'):raise ValueError('Unknown energy formulation')
        if self.mesh not in ('coarse','medium','fine'):raise ValueError('Unknown mesh')
        for k,v in asdict(self).items():
            if k in ('mesh','energy_form','linear_solver','ewm_calibration'):continue
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0:raise ValueError(f'{k} must be positive and finite')
        for k in ('iterations','minimum_iterations','write_interval','processes'):
            if type(getattr(self,k)) is not int:raise ValueError(f'{k} must be an integer')
        if self.processes!=1:raise ValueError('P008 currently supports serial coupled solves only')
        for k in ('porosity_ptl','porosity_cl','maximum_packing','velocity_relaxation','thermal_relaxation'):
            if not 0<getattr(self,k)<1:raise ValueError(f'{k} must be in (0,1)')
        if not 1.4<=self.cell_voltage<=2.2:raise ValueError('Voltage range is 1.4--2.2 V')
        if not 0<self.contact_angle_deg<90:raise ValueError('This revision requires a wetting porous layer (0 < theta < 90)')
        if self.field_tolerance>1e-5 or self.balance_tolerance>1e-3:raise ValueError('Acceptance tolerances may only be tightened')
        if not 273.15<self.inlet_temperature<=333.15:raise ValueError('Paper neglects evaporation up to 60 C')
        if self.ewm_calibration is not None:
            from .ewm import INDICATORS,score
            if not isinstance(self.ewm_calibration,dict):raise ValueError('Embed the EWM calibration object in the configuration')
            if list(zip(self.ewm_calibration.get('names',[]),self.ewm_calibration.get('directions',[])))!=list(INDICATORS):raise ValueError('P008 requires all nine paper indicators')
            score(self.ewm_calibration['minimum'],self.ewm_calibration)
    def to_dict(self):return asdict(self)
    @property
    def fingerprint(self):
        data=dict(config=self.to_dict(),revision=MODEL_REVISION)
        return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()

def geometry(x,config):
    # Three paper widths; depth is continuous. Half ribs close the 20 mm plate.
    index=min(2,int(3*x[0]));n=(8,10,12)[index];w=12/n;r=8/n;d=.4+.6*x[1]
    return dict(normalized_design=list(x),channels=n,width_mm=w,rib_mm=r,depth_mm=d,
        depth_width_ratio=d/w,active_area_m2=400e-6,footprint_mm=[20.,20.],
        centers_mm=[(i+.5)*20/n for i in range(n)],constraints={})
