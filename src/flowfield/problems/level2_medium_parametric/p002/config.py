"""Paper inputs and explicitly identified reconstruction choices for P002."""

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from types import MappingProxyType

DOI = '10.1016/j.ijhydene.2023.08.026'
MODEL_REVISION = 'pem-reconstruction-v2-graded-mesh'
PARAMETERS = ('W1', 'W2', 'W3', 'W4', 'W5', 'Ha', 'Hc')
BOUNDS_MM = ((.7, 1.9),)*5 + ((.5, 1.4),)*2
BASELINE_MM = (1.,)*7
PAPER_OPTIMUM_MM = (.8465, .7388, .8904, .8315, .7938, 1.2477, 1.3979)


def normalize(mm):
    values = tuple(mm)
    if len(values) != 7:
        raise ValueError('Expected five widths and two heights')
    for v, (lo, hi) in zip(values, BOUNDS_MM):
        if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError('Physical dimensions are outside the paper bounds')
    return tuple((v-lo)/(hi-lo) for v, (lo, hi) in zip(values, BOUNDS_MM))


BASELINE_X = normalize(BASELINE_MM)
PAPER_OPTIMUM_X = normalize(PAPER_OPTIMUM_MM)


def reference():
    return json.loads((Path(__file__).parent/'data/reference.json').read_text())


@dataclass(frozen=True)
class PEMConfig:
    mesh: str = 'coarse'
    scalarization: str = 'total_power_w'
    iterations: int = 1000
    write_interval: int = 100
    model_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mesh not in ('coarse', 'medium', 'fine'):
            raise ValueError('Unknown P002 mesh level')
        if self.scalarization not in ('total_power_w', 'voltage_v', 'pressure_sum_pa'):
            raise ValueError('Unknown P002 scalarization')
        for name in ('iterations', 'write_interval'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        overrides = dict(self.model_overrides)
        defaults = reference()['model']
        if set(overrides)-set(defaults):
            raise ValueError(f'Unknown model keys: {sorted(set(overrides)-set(defaults))}')
        model = defaults | overrides
        for name, v in model.items():
            if name in ('coveragePressureMode', 'includeNernstConcentration', 'reversibleHeatSign'):
                continue
            if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('minimumIterations', 'maxChargeNewton'):
            if type(model[name]) is not int:
                raise ValueError(f'{name} must be an integer')
        for name in ('porosityPTL','porosityCL','alphaAnodic','alphaCathodic','concentrationFloorRatio'):
            if not 0 < model[name] < 1:
                raise ValueError(f'{name} must lie strictly between 0 and 1')
        if model['coveragePressureMode'] not in ('dimensionless','paperSI'):
            raise ValueError('Unknown coverage pressure closure')
        if type(model['includeNernstConcentration']) is not bool:
            raise ValueError('includeNernstConcentration must be boolean')
        if isinstance(model['reversibleHeatSign'], bool) or model['reversibleHeatSign'] not in (-1,1):
            raise ValueError('reversibleHeatSign must be -1 or 1')
        t = model['inletTemperature']
        if not 273.15 < t < 373.15:
            raise ValueError('Liquid property model requires 273.15 < T < 373.15 K')
        psat = 133.322368*10**(8.07131-1730.63/(233.426+t-273.15))
        if model['operatingPressure'] <= psat:
            raise ValueError('Operating pressure must exceed inlet saturation pressure')
        if model['outerTolerance'] > 1e-5 or model['balanceTolerance'] > 1e-3:
            raise ValueError('P002 convergence tolerances may be tightened, not relaxed')
        object.__setattr__(self, 'model_overrides', MappingProxyType(overrides))

    @property
    def model(self):
        return reference()['model'] | dict(self.model_overrides)

    def to_dict(self):
        return dict(mesh=self.mesh, scalarization=self.scalarization, iterations=self.iterations,
                    write_interval=self.write_interval, model_overrides=dict(self.model_overrides))

    @property
    def fingerprint(self):
        data = dict(config=self.to_dict(), reference=reference(), revision=MODEL_REVISION)
        return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @property
    def objective_unit(self):
        return {'total_power_w':'W', 'voltage_v':'V', 'pressure_sum_pa':'Pa'}[self.scalarization]


def parameters_for(geometry, config):
    g = reference()['geometry']
    for name in ('channel_width_mm','channel_height_mm','single_length_mm','single_pitch_mm'):
        g.pop(name)
    mm = geometry['dimensions_mm']
    g.update(channel_widths_mm=[mm[k] for k in PARAMETERS[:5]],
             anode_height_mm=mm['Ha'], cathode_height_mm=mm['Hc'])
    return dict(geometry=g, model=config.model, iterations=config.iterations,
                write_interval=config.write_interval)


def scalar_objective(voltage, pressure_sum, config):
    if not all(math.isfinite(v) and v > 0 for v in (voltage, pressure_sum)):
        raise ValueError('Expected positive finite voltage and pressure drop')
    if config.scalarization == 'voltage_v':
        return voltage
    if config.scalarization == 'pressure_sum_pa':
        return pressure_sum
    current = config.model['currentDensity']*121e-6
    flow = config.model['inletFlow_mL_min']*1e-6/60
    return current*voltage + flow*pressure_sum
