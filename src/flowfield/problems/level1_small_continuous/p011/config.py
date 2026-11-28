"""Li et al. (2026): paper constants and explicit reconstruction closures, SI.

Four endpoint widths span the feasible manifold of the eight printed sizes.
Heights are dependent quantities, never independently optimized or rounded.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

DOI = '10.1016/j.renene.2026.125531'
MODEL_REVISION = 'tapered-pemfc-two-phase-v1'
PARAMETERS = ('cathode_inlet_width_mm', 'cathode_outlet_width_mm',
              'anode_inlet_width_mm', 'anode_outlet_width_mm')
BOUNDS = ((1/1.8, 1.8),) * 4
BASELINE = (1., 1., 1., 1.)
REFERENCE_WIDTHS = {'CFC': BASELINE, 'GA': (1.395, 1.533, .750, .798),
                    'IGA': (1.329, 1.612, .779, 1.013)}


@dataclass(frozen=True)
class TaperedConfig:
    # Table 2 values. Unreported material/closure values are estimates below.
    temperature_K: float = 343.14
    humidity_percent: float = 100.
    cathode_stoichiometry: float = 2.
    anode_stoichiometry: float = 2.
    gdl_porosity: float = .4
    permeability_gdl: float = 1.76e-11
    electronic_conductivity_gdl: float = 1250.
    exchange_current_anode: float = 1.05e10
    exchange_current_cathode: float = 100.
    reference_concentration: float = 40.88
    hydrogen_diffusivity: float = 1.111e-4
    nitrogen_water_diffusivity: float = 3.107e-5
    oxygen_water_diffusivity: float = 3.405e-5
    oxygen_nitrogen_diffusivity: float = 2.897e-5
    outlet_pressure: float = 101325.
    cell_voltage: float = .4
    objective_scale: float = 1e5

    # Explicit estimates; these are not calibrated to published performance.
    reference_current_density: float = 10000.
    open_circuit_voltage: float = 1.18
    plate_thickness_mm: float = .5
    cl_porosity: float = .4
    ionomer_fraction: float = .3
    permeability_cl: float = 1e-13
    electronic_conductivity_cl: float = 1250.
    electronic_conductivity_bp: float = 20000.
    kinetic_reference_temperature: float = 343.14
    diffusion_reference_temperature: float = 343.14
    activation_energy_anode: float = 20000.
    activation_energy_cathode: float = 66000.
    transfer_coefficient: float = .5
    concentration_exponent: float = 1.
    air_viscosity: float = 2e-5
    hydrogen_viscosity: float = 9.6e-6
    liquid_density: float = 978.
    liquid_viscosity: float = .000404
    liquid_conductivity: float = .65
    surface_tension: float = .0644
    contact_angle_deg: float = 110.
    channel_liquid_dispersion: float = 1e-5
    phase_change_rate: float = 100.
    sorption_rate: float = 10.
    membrane_dry_density: float = 2000.
    membrane_equivalent_weight: float = 1.1
    membrane_diffusivity: float = 2e-10
    latent_heat: float = 2.33e6
    # Liquid-product HHV, W/A (=V). Subtract vaporization before adding
    # explicit phase-change heat; split reversible heat equally between CLs.
    thermoneutral_voltage: float = 1.48

    mesh: str = 'coarse'
    linear_solver: str = 'iterative'
    iterations: int = 3000
    minimum_iterations: int = 60
    processes: int = 1
    liquid_pseudo_time: float = 5.
    velocity_relaxation: float = .5
    scalar_relaxation: float = .3
    field_tolerance: float = 1e-5
    voltage_ramp_iterations: int = 80
    write_interval: int = 100
    balance_tolerance: float = 1e-3

    def __post_init__(self):
        if self.mesh not in ('coarse', 'medium', 'fine'):
            raise ValueError('Unknown mesh')
        if self.linear_solver not in ('direct', 'iterative'):
            raise ValueError('Unknown linear solver')
        for key, value in asdict(self).items():
            if key in ('mesh', 'linear_solver'):
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        for key in ('iterations', 'minimum_iterations', 'processes', 'voltage_ramp_iterations', 'write_interval'):
            if type(getattr(self, key)) is not int:
                raise ValueError(f'{key} must be an integer')
        if self.processes != 1:
            raise ValueError('P011 supports one CPU process per case')
        if self.minimum_iterations > self.iterations or self.voltage_ramp_iterations > self.iterations:
            raise ValueError('Iteration budget must cover the ramp and minimum iterations')
        if not .2 <= self.cell_voltage < self.open_circuit_voltage < self.thermoneutral_voltage:
            raise ValueError('Invalid fuel-cell voltages')
        if not 273.15 < self.temperature_K < 373.15 or not 0 < self.humidity_percent <= 100:
            raise ValueError('Temperature or humidity is outside supported bounds')
        for key in ('gdl_porosity', 'cl_porosity', 'ionomer_fraction', 'velocity_relaxation', 'scalar_relaxation', 'transfer_coefficient'):
            if not 0 < getattr(self, key) < 1:
                raise ValueError(f'{key} must be in (0, 1)')
        if not 90 < self.contact_angle_deg < 180:
            raise ValueError('Capillary closure requires hydrophobic porous media')
        if self.field_tolerance > 1e-5 or self.balance_tolerance > 1e-3:
            raise ValueError('Acceptance limits may only be tightened')
        c = self.temperature_K - 273.15
        psat = 101325*10**(-2.1794+.02953*c-9.1837e-5*c*c+1.4454e-7*c**3)
        if psat*self.humidity_percent*.01 >= self.outlet_pressure:
            raise ValueError('Humidified feed must retain positive reactant partial pressure')

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(dict(revision=MODEL_REVISION, config=self.to_dict()),
                                         sort_keys=True, allow_nan=False).encode()).hexdigest()


def normalized(widths):
    values = list(widths)
    if len(values) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                              or not math.isfinite(v) or not lo <= v <= hi
                              for v, (lo, hi) in zip(values, BOUNDS)):
        raise ValueError('Provide four endpoint widths in [1/1.8, 1.8] mm')
    return [(v-lo)/(hi-lo) for v, (lo, hi) in zip(values, BOUNDS)]


def decode(x, config):
    x = list(x)
    if len(x) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                          or not math.isfinite(v) or not 0 <= v <= 1 for v in x):
        raise ValueError('Provide four normalized variables in [0, 1]')
    g = {key: lo+(hi-lo)*v for key, v, (lo, hi) in zip(PARAMETERS, x, BOUNDS)}
    g.update({key.replace('width', 'height'): 1/value for key, value in tuple(g.items())})
    g.update(normalized_design=x, length_mm=20., pitch_mm=2., active_area_m2=40e-6,
             temperature_K=config.temperature_K, humidity_percent=config.humidity_percent,
             cathode_stoichiometry=config.cathode_stoichiometry, gdl_porosity=config.gdl_porosity,
             endpoint_area_mm2=1., constraints={})
    for side in ('cathode', 'anode'):
        g[side+'_taper_angle_deg'] = math.degrees(math.atan(
            (g[side+'_inlet_height_mm']-g[side+'_outlet_height_mm'])/g['length_mm']))
    return g
