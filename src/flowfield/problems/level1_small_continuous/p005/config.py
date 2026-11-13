"""Wu et al. 2024 physical-only VOF reconstruction; SI units throughout."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math

DOI = "10.1016/j.egyai.2024.100411"
MODEL_REVISION = "dual-layer-vof-v1"
BASELINE_X = (1/3, 5/7)  # HB=1 mm, HD=3 mm

@dataclass(frozen=True)
class DualLayerConfig:
    mesh: str = "coarse"
    oxygen_velocity_m_s: float = 5.0
    water_velocity_m_s: float = .00174
    length_m: float = .020
    base_width_m: float = .001
    degassing_width_m: float = .0016
    gas_inlet_side_m: float = .0001  # Unreported: explicit reconstruction assumption
    gas_inlet_z_m: float = .001    # Unreported: explicit reconstruction assumption
    temperature_k: float = 273.15
    reference_pressure_pa: float = 101325.
    water_density_kg_m3: float = 999.84
    oxygen_density_kg_m3: float = 1.429
    water_viscosity_pa_s: float = .0017918
    oxygen_viscosity_pa_s: float = .00001919
    surface_tension_n_m: float = .0756
    gravity_m_s2: float = 9.8
    wall_water_contact_angle_deg: float = 70.
    electrode_water_contact_angle_deg: float = 30.
    end_time_s: float = .2
    max_delta_t_s: float = .00001
    max_courant: float = .25
    write_interval_s: float = .01
    monitor_interval_s: float = .0001
    gas_threshold: float = .5
    min_bubble_volume_m3: float = 1e-11
    detached_fraction: float = .5
    persistence_s: float = .0002
    gas_balance_tolerance: float = .02
    continuity_tolerance: float = .001
    alpha_tolerance: float = 1e-6

    def __post_init__(self):
        if self.mesh not in ("coarse","medium","fine"):
            raise ValueError("mesh must be coarse, medium or fine")
        for key,v in asdict(self).items():
            if key == "mesh": continue
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
                raise ValueError(f"{key} must be positive and finite")
        if self.temperature_k != 273.15:
            raise ValueError("v1 is isothermal at Table 2's 273.15 K; change properties in a new model revision")
        if self.degassing_width_m <= self.base_width_m or self.gas_inlet_side_m >= self.base_width_m:
            raise ValueError("Need gas inlet narrower than base, and base narrower than degassing layer")
        if not self.gas_inlet_side_m/2 < self.gas_inlet_z_m < self.length_m-self.gas_inlet_side_m/2:
            raise ValueError("Gas inlet must lie inside the bottom electrode")
        if self.water_density_kg_m3 <= self.oxygen_density_kg_m3:
            raise ValueError("Need water density greater than oxygen density")
        if not 0 < self.gas_threshold < 1 or not 0 < self.detached_fraction <= 1:
            raise ValueError("Invalid gas detection fractions")
        if max(self.wall_water_contact_angle_deg,self.electrode_water_contact_angle_deg)>=180:
            raise ValueError("Contact angles must lie strictly between 0 and 180 degrees")
        if self.max_courant>.5 or self.gas_balance_tolerance>.02 or self.continuity_tolerance>.001 or self.alpha_tolerance>1e-6:
            raise ValueError("Verification tolerances and Courant limit may be tightened, not relaxed")

    @property
    def refinement(self): return {"coarse":1,"medium":2,"fine":4}[self.mesh]
    def to_dict(self): return asdict(self)
    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(dict(config=self.to_dict(),revision=MODEL_REVISION),sort_keys=True,allow_nan=False).encode()).hexdigest()
