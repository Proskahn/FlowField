"""P006: three-control, fixed-manifold two-fluid thermal optimization."""
import math
from pathlib import Path
from ....problem import CFDProblem
from ....runtime import Runtime, SimulationError
from .config import ManifoldConfig, BASELINE_X, DOI, MODEL_REVISION, DEFAULT_TIMEOUT_S, geometry as decode_geometry


class P006Problem(CFDProblem):
    config_type=ManifoldConfig
    requires_qualified_baseline=True

    def __init__(self,output_root='runs',*,config=None,runtime=None,failure_penalty=None):
        super().__init__('P006',3,[0.]*3,[1.]*3)
        self.config=config if config is not None else ManifoldConfig()
        if not isinstance(self.config,ManifoldConfig):raise TypeError('P006 requires ManifoldConfig')
        self.runtime=runtime or Runtime(timeout_s=DEFAULT_TIMEOUT_S)
        self.output_root=Path(output_root)
        if failure_penalty is not None and (isinstance(failure_penalty,bool) or not math.isfinite(failure_penalty) or failure_penalty<=0):
            raise ValueError('failure_penalty must be positive and finite')
        self.failure_penalty=failure_penalty

    @property
    def case_definition(self):
        from .evaluation import CASE
        return CASE

    @property
    def ioh_name(self):
        policy='raise' if self.failure_penalty is None else format(self.failure_penalty,'.12g')
        return f'P006_{MODEL_REVISION}_{self.config.fingerprint[:12]}_failure_{policy}'

    def initial_design(self):return list(BASELINE_X)

    def preflight(self):
        from .runtime import eigen_headers
        eigen_headers()

    def decode(self,x):return decode_geometry(self.validate_design(x),self.config)

    def geometric_constraints(self,x):return self.decode(x)['constraints']

    def evaluate_raw(self,x):
        from ....evaluation import CaseEvaluator
        self.last_result=None
        g=self.decode(x)
        try:
            result=CaseEvaluator(self.case_definition,self.runtime).evaluate(g,self.config,output_root=self.output_root)
        except SimulationError as exc:
            if self.failure_penalty is None:raise
            result=dict(status='penalized',objective=self.failure_penalty,objectives={},constraints={},
                        error=str(exc),case_directory=exc.case_directory)
        result['metadata']=dict(problem_id=self.name,problem_name=self.ioh_name,dimension=3,
            design=g['normalized_design'],geometry=g,config=self.config.to_dict(),
            config_sha256=self.config.fingerprint,fidelity=self.config.fidelity,paper_doi=DOI,paper_validated=False,
            replication_status='provisional_two_fluid_thermal_reconstruction',
            objective_unit='K',valid_cfd=result['status']=='ok',case_directory=result['case_directory'])
        self.last_result=result
        return result

__all__=['P006Problem','ManifoldConfig']
