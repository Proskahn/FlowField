"""Zhou et al. (2026): reported inputs and explicit reconstruction choices."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math

DOI = '10.1016/j.ijhydene.2026.157728'
MODEL_REVISION = 'manifold-two-fluid-thermal-v2'
# Wall-clock allowance per command, separate from the nonlinear iteration budget.
DEFAULT_TIMEOUT_S = 14400
PARAMETERS = ('win', 'wmid', 'wout')
BASELINE_X = (0.375,)*3
# Ordered from least to most resolved. Search meshes retain the coarse x/z grid.
SCREENING_SPACING_MM = {'screening-coarse': 2.0, 'screening': 1.5}
MESH_LEVELS = (*SCREENING_SPACING_MM, 'coarse', 'medium', 'fine')
REFERENCE_FIELD_TOLERANCE = 1e-5
SEARCH_FIELD_TOLERANCE = 1e-4


@dataclass(frozen=True)
class ManifoldConfig:
    # Geometry
    # Channel widths are decoded from the design in geometry() below.
    manifold_width_mm: float = 1.0

    # Physical parameters
    liquid_density: float = 970.
    oxygen_density: float = 1.42
    hydrogen_density: float = .09
    liquid_viscosity: float = .000355
    oxygen_viscosity: float = 2.3e-5
    hydrogen_viscosity: float = 1.0e-5
    liquid_cp: float = 4195.
    oxygen_cp: float = 930.
    hydrogen_cp: float = 14400.
    liquid_conductivity: float = .67
    oxygen_conductivity: float = .030
    hydrogen_conductivity: float = .21
    solid_conductivity: float = 15.2
    catalyst_conductivity: float = 2.
    membrane_conductivity: float = .21
    electron_conductivity: float = 1000.
    porosity_ptl: float = .4
    porosity_cl: float = .3
    permeability_ptl: float = 1e-12
    permeability_cl: float = 1e-13
    permeability_channel: float = 1e-6  # Unreported equivalent-channel Brinkman permeability.
    surface_tension: float = .0627
    contact_angle_deg: float = 70.
    exchange_current_anode: float = .1
    exchange_current_cathode: float = 1e4
    active_area_anode: float = 1e5
    active_area_cathode: float = 1e5
    activation_energy_anode: float = 54000.
    activation_energy_cathode: float = 24360.
    kinetic_reference_temperature: float = 353.
    ionomer_fraction_cl: float = .3
    ionomer_density: float = 2000.
    ionomer_equivalent_weight: float = 1.1
    water_diffusivity_factor: float = 1.

    # Initial conditions
    # Field initialization is in geometry.py:case_files; warm starts are separate.
    initial_voltage: float = 2.2  # Numerical initialization, capped at the requested voltage.

    # Boundary conditions
    cell_voltage: float = 2.2
    inlet_flow_ml_min: float = 15.
    inlet_temperature: float = 353.
    inlet_liquid_saturation: float = .9
    ambient_temperature: float = 298.15
    external_heat_transfer: float = 20.
    membrane_water_boundary: float = 22.

    # Mesh and discretization
    mesh: str = 'coarse'

    # Solver controls
    iterations: int = 3500
    minimum_iterations: int = 60
    processes: int = 2  # Verified fresh-start coarse baseline; serial can be selected explicitly.
    pseudo_dt: float = 1e-6  # Numerical phase storage; not a physical time step.
    phase_numerical_courant: float = .001
    field_tolerance: float = 1e-5

    # Objectives and constraints
    contact_tolerance: float = .05  # Paper Eq. 29 does not give delta.

    # Diagnostics and output
    write_interval: int = 250
    balance_tolerance: float = 1e-3

    def __post_init__(self):
        if self.mesh not in MESH_LEVELS:
            raise ValueError('mesh must be one of: '+', '.join(MESH_LEVELS))
        for key, value in asdict(self).items():
            if key == 'mesh':
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        for key in ('iterations', 'write_interval', 'processes', 'minimum_iterations'):
            if type(getattr(self, key)) is not int:
                raise ValueError(f'{key} must be an integer')
        if not 1 <= self.manifold_width_mm <= 3:
            raise ValueError('Paper manifold range is 1 to 3 mm')
        if not 1.4 <= self.cell_voltage <= 2.2:
            raise ValueError('Paper voltage range is 1.4 to 2.2 V')
        if not 1.4 <= self.initial_voltage <= 2.2:
            raise ValueError('Initial continuation voltage must be between 1.4 and 2.2 V')
        for key in ('contact_tolerance', 'inlet_liquid_saturation', 'porosity_ptl', 'porosity_cl', 'ionomer_fraction_cl'):
            if not 0 < getattr(self, key) < 1:
                raise ValueError(f'{key} must be between zero and one')
        if not 0 < self.contact_angle_deg < 180 or self.contact_angle_deg == 90:
            raise ValueError('Use a nonzero capillary slope: contact angle in (0,180), excluding 90')
        if not 273.15 < self.inlet_temperature < 373.15:
            raise ValueError('Liquid-water inlet must lie between freezing and boiling')
        if self.field_tolerance > SEARCH_FIELD_TOLERANCE:
            raise ValueError('field_tolerance must be <= 1e-4; values above 1e-5 are search settings')
        if self.balance_tolerance > 1e-3:
            raise ValueError('Conservation tolerance must be <= 1e-3, including search runs')

    @property
    def fidelity(self):
        """Describe numerical settings, not a convergence certificate.

        Derived metadata deliberately stays out of to_dict/fingerprint, keeping
        existing case and optimizer identities unchanged. New meshes/tolerances
        are already distinguished by those existing configuration fields.
        """
        search = self.mesh in SCREENING_SPACING_MM or self.field_tolerance > REFERENCE_FIELD_TOLERANCE
        return dict(purpose='search' if search else 'reference', mesh=self.mesh,
                    field_tolerance=self.field_tolerance, balance_tolerance=self.balance_tolerance,
                    reference_recheck_required=search)

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(dict(config=self.to_dict(), revision=MODEL_REVISION),
            sort_keys=True, allow_nan=False).encode()).hexdigest()


def spline_widths(controls):
    """Natural cubic spline at nine fixed centers (end second derivatives zero).

    Three equally spaced knots at channels 1, 5 and 9. The end condition is
    unreported in the paper; it is part of this benchmark's versioned definition.
    """
    a, b, c = controls
    second = 1.5*(a-2*b+c)
    values = []
    for i in range(9):
        t = i/4
        if t <= 1:
            values.append(a*(1-t)+b*t+second*(t**3-t)/6)
        else:
            t -= 1
            values.append(b*(1-t)+c*t+second*((1-t)**3-(1-t))/6)
    return values


def geometry(design, config):
    controls = [.7+.8*v for v in design]
    widths = spline_widths(controls)
    centers = [1.+2*i for i in range(9)]
    ribs = [centers[i+1]-centers[i]-(widths[i]+widths[i+1])/2 for i in range(8)]
    ribs += [centers[0]-widths[0]/2, 18-centers[-1]-widths[-1]/2]
    length = 25-2*config.manifold_width_mm
    # The contact comparison uses the uniform baseline at THIS manifold width.
    contact = (18-sum(widths))*length
    reference = 9*length
    return dict(normalized_design=list(design), controls_mm=dict(zip(PARAMETERS, controls)),
        widths_mm=widths, centers_mm=centers, ribs_mm=ribs,
        manifold_width_mm=config.manifold_width_mm, clear_branch_length_mm=length,
        contact_area_mm2=contact, reference_contact_area_mm2=reference,
        constraints=dict(contact_area=abs(contact/reference-1)-config.contact_tolerance,
                         minimum_width=.7-min(widths), maximum_width=max(widths)-1.5,
                         positive_rib=1e-6-min(ribs)),
        footprint_mm=[18.,25.], active_area_m2=450e-6)
