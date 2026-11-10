"""P002: seven-variable PEM electrolyzer, coupled reconstruction of Zhuang 2024."""

import math
from pathlib import Path

from ....base import FlowFieldProblem
from ....runtime import Runtime, SimulationError
from .config import (BASELINE_X, PAPER_OPTIMUM_X, BOUNDS_MM, DOI, MODEL_REVISION,
                     PARAMETERS, PEMConfig)


class P002Problem(FlowFieldProblem):
    def __init__(self, output_root='runs', *, config=None, runtime=None, failure_penalty=None):
        super().__init__('P002', 7, [0.]*7, [1.]*7)
        self.config = config if config is not None else PEMConfig()
        if not isinstance(self.config, PEMConfig):
            raise TypeError('P002 requires PEMConfig')
        self.runtime = runtime or Runtime(timeout_s=1800)
        self.output_root = Path(output_root)
        if failure_penalty is not None and (isinstance(failure_penalty, bool)
                or not math.isfinite(failure_penalty) or failure_penalty <= 0):
            raise ValueError('failure_penalty must be positive and finite')
        self.failure_penalty = failure_penalty

    @property
    def ioh_name(self):
        policy = 'raise' if self.failure_penalty is None else format(self.failure_penalty, '.12g')
        return f'P002_{MODEL_REVISION}_{self.config.scalarization}_{self.config.fingerprint[:12]}_failure_{policy}'

    def decode(self, x):
        design = self.validate_design(x)
        mm = {k:lo+v*(hi-lo) for k,v,(lo,hi) in zip(PARAMETERS,design,BOUNDS_MM)}
        return dict(normalized_design=design, dimensions_mm=mm,
                    dimensions_m={k:v*.001 for k,v in mm.items()},
                    active_area_m2=121e-6, plate_size_mm=11., branch_length_mm=8.,
                    branch_centers_x_mm=[1.5,3.5,5.5,7.5,9.5], manifold_width_mm=1.)

    def evaluate_raw(self, x):
        from .evaluation import evaluate
        from ....evaluation import write_json
        self.last_result = None
        geometry = self.decode(x)
        metadata = dict(problem_id=self.name, problem_name=self.ioh_name, dimension=7,
            design=geometry['normalized_design'], geometry=geometry,
            config=self.config.to_dict(), config_sha256=self.config.fingerprint,
            paper_doi=DOI, replication_status='provisional_coupled_reconstruction',
            paper_validated=False, objective_direction='minimize',
            objective_unit=self.config.objective_unit, scalarization=self.config.scalarization)
        try:
            result = evaluate(geometry, self.config, self.output_root, self.runtime)
        except SimulationError as exc:
            if self.failure_penalty is None:
                raise
            metadata.update(valid_cfd=False, case_directory=exc.case_directory, error=str(exc))
            result = dict(status='penalized', objective=float(self.failure_penalty),
                          objectives={}, constraints={}, error=str(exc))
        else:
            metadata.update(valid_cfd=True, case_directory=result['case_directory'],
                            runtime=result['runtime'], metrics=result['metrics'])
        result['metadata'] = metadata
        if metadata.get('case_directory'):
            write_json(Path(metadata['case_directory'])/'evaluation.json', result)
        self.last_result = result
        return result


__all__ = ['P002Problem', 'PEMConfig', 'BASELINE_X', 'PAPER_OPTIMUM_X']
