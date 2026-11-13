"""P001: six-variable alkaline PPFC, provisional OpenFOAM reconstruction."""

import math
from pathlib import Path

from ....problem import CFDProblem
from ....runtime import Runtime, SimulationError
from .config import BASELINE_X, BOUNDS_MM, DOI, MODEL_REVISION, PARAMETERS, PPFCConfig


class P001Problem(CFDProblem):
    config_type = PPFCConfig

    @property
    def case_definition(self):
        from .evaluation import CASE
        return CASE

    def initial_design(self):
        return list(BASELINE_X)

    def __init__(self, output_root='runs', *, config=None, runtime=None, failure_penalty=None):
        super().__init__('P001', 6, [0.]*6, [1.]*6)
        self.config = config if config is not None else PPFCConfig()
        if not isinstance(self.config, PPFCConfig):
            raise TypeError('P001 requires PPFCConfig')
        self.runtime = runtime or Runtime(timeout_s=1800)
        self.output_root = Path(output_root)
        if failure_penalty is not None and (not math.isfinite(failure_penalty) or failure_penalty <= 0):
            raise ValueError('failure_penalty must be positive and finite')
        self.failure_penalty = failure_penalty

    @property
    def ioh_name(self):
        policy = 'raise' if self.failure_penalty is None else format(self.failure_penalty, '.12g')
        return f'P001_{MODEL_REVISION}_{self.config.fingerprint[:12]}_failure_{policy}'

    def decode(self, x):
        design = self.validate_design(x)
        mm = {name:lo+v*(hi-lo) for name,v,(lo,hi) in zip(PARAMETERS, design, BOUNDS_MM)}
        return {'normalized_design':design, 'spacings_mm':mm,
                'spacings_m':{k:v*.001 for k,v in mm.items()},
                'diameter_m':.4, 'height_m':.0075, 'sphere_diameter_m':.015,
                'prism_side_m':.013, 'region_boundaries_y_m':[-.135,-.045,.045,.135]}

    def evaluate_raw(self, x):
        from .evaluation import evaluate
        from ....evaluation import write_json
        self.last_result = None
        geometry = self.decode(x)
        metadata = {'problem_id':self.name, 'problem_name':self.ioh_name,
            'dimension':6, 'design':geometry['normalized_design'], 'geometry':geometry,
            'config':self.config.to_dict(), 'config_sha256':self.config.fingerprint,
            'paper_doi':DOI, 'replication_status':'provisional_geometry_reconstruction',
            'paper_validated':False, 'objective_direction':'minimize', 'objective_unit':'dimensionless'}
        try:
            result = evaluate(geometry, self.config, self.output_root, self.runtime)
        except SimulationError as exc:
            if self.failure_penalty is None:
                raise
            metadata.update(valid_cfd=False, case_directory=exc.case_directory, error=str(exc))
            result = {'status':'penalized', 'objective':float(self.failure_penalty),
                      'objectives':{}, 'constraints':{}, 'error':str(exc)}
        else:
            metadata.update(valid_cfd=True, case_directory=result['case_directory'],
                            runtime=result['runtime'], metrics=result['metrics'])
        result['metadata'] = metadata
        if metadata.get('case_directory'):
            write_json(Path(metadata['case_directory'])/'evaluation.json', result)
        self.last_result = result
        return result


__all__ = ['P001Problem', 'PPFCConfig', 'BASELINE_X']
