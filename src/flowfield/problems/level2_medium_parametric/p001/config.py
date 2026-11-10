"""Zhao et al. (2025) data and explicit, provisional reconstruction settings."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

DOI = "10.1016/j.ijhydene.2025.03.237"
PARAMETERS = ("x1_1", "x1_2", "x2", "x3", "h1_1", "h1_2")
BOUNDS_MM = ((31., 37.), (28., 33.), (27., 32.), (28., 31.), (13., 17.), (13., 17.))
# Figure 7(b)/8(b) example, NOT the unreported optimal design.
BASELINE_MM = (35., 32., 30., 30., 15., 15.)
BASELINE_X = tuple((v-lo)/(hi-lo) for v, (lo, hi) in zip(BASELINE_MM, BOUNDS_MM))
MESHES = {"coarse": (0.003, 0.006), "medium": (0.0015, 0.003), "fine": (0.00075, 0.0015)}
MODEL_REVISION = "ppfc-reconstruction-v1"


@dataclass(frozen=True)
class PPFCConfig:
    """SI units. Unreported geometry is an assumption, never paper ground truth.

    Mesh levels are local engineering choices, not the paper's Mesh-1..5.
    This configuration is deliberately separate from P000 ChannelConfig.
    """
    mesh: str = "coarse"
    inlet_velocity_m_s: float = 0.5
    density_kg_m3: float = 1290.4769
    dynamic_viscosity_pa_s: float = 0.0025491
    turbulence_intensity: float = 0.05
    port_width_m: float = 0.010
    port_length_m: float = 0.015
    sphere_cap_height_m: float = 0.003
    turbulence_length_m: float = 0.001
    max_iterations: int = 2000
    residual_tolerance: float = 1e-6
    mass_balance_tolerance: float = 1e-4
    stability_tolerance: float = 1e-4
    stability_window: int = 20

    def __post_init__(self):
        if self.mesh not in MESHES:
            raise ValueError(f"mesh must be one of {tuple(MESHES)}")
        for key, value in asdict(self).items():
            if key == "mesh":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not 0.1 <= self.inlet_velocity_m_s <= 0.5:
            raise ValueError("Paper inlet velocity range is 0.1 to 0.5 m/s")
        if not 0.001 <= self.port_width_m <= 0.025 or not 0.001 <= self.port_length_m <= 0.05:
            raise ValueError("Reconstruction ports require width 1..25 mm and length 1..50 mm")
        if not 0.0005 <= self.sphere_cap_height_m <= 0.005:
            raise ValueError("Reconstruction cap height must be 0.5..5 mm")
        if type(self.max_iterations) is not int or type(self.stability_window) is not int:
            raise ValueError("Iteration counts must be integers")
        if self.max_iterations <= self.stability_window or self.stability_window < 2:
            raise ValueError("Need at least two stability samples and more solver iterations")
        if self.turbulence_intensity >= 1 or any(v >= 1 for v in
                (self.residual_tolerance, self.mass_balance_tolerance, self.stability_tolerance)):
            raise ValueError("Intensity and numerical tolerances must be below one")

    @property
    def kinematic_viscosity_m2_s(self):
        return self.dynamic_viscosity_pa_s / self.density_kg_m3

    @property
    def flow_rate_m3_s(self):
        return self.inlet_velocity_m_s * self.port_width_m * 0.0075

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps({"revision": MODEL_REVISION, **asdict(self)},
            sort_keys=True, allow_nan=False).encode()).hexdigest()


def uniformity(speeds, areas):
    """Equations 5/6, for nonnegative scalar speed and positive area weights.

    Zero flow has undefined uniformity and must not receive a perfect score.
    """
    speeds, areas = list(speeds), list(areas)
    if not speeds or len(speeds) != len(areas):
        raise ValueError("Need equally sized, nonempty speed and area arrays")
    if any(not math.isfinite(v) or v < 0 for v in speeds) or any(not math.isfinite(a) or a <= 0 for a in areas):
        raise ValueError("Speeds must be nonnegative and areas positive, all finite")
    area = math.fsum(areas)
    mean = math.fsum(v*a for v, a in zip(speeds, areas)) / area
    if mean <= 0:
        raise ValueError("Uniformity is undefined for zero mean speed")
    return 1 - math.fsum(abs(v-mean)*a for v, a in zip(speeds, areas))/(2*mean*area)
