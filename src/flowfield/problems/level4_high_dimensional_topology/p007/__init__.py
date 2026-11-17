"""P007: Jin et al. 2D flow/heat density topology. No full-cell electrochemistry."""
from pathlib import Path
import math
from ....problem import CFDProblem
from ....runtime import Runtime,SimulationError
from .config import TopologyConfig,MODEL_REVISION,DOI
from .design import decode,volume_gradient,repair_candidates


class P007Problem(CFDProblem):
    config_type=TopologyConfig
    requires_qualified_baseline=True

    def __init__(self,output_root='runs',*,config=None,runtime=None,failure_penalty=None):
        self.config=config if config is not None else TopologyConfig()
        if not isinstance(self.config,TopologyConfig):raise TypeError('P007 requires TopologyConfig')
        n=self.config.dimension;super().__init__('P007',n,[0.]*n,[1.]*n)
        self.output_root=Path(output_root);self.runtime=runtime or Runtime(timeout_s=1800)
        if failure_penalty is not None and (isinstance(failure_penalty,bool) or not math.isfinite(failure_penalty) or failure_penalty<=0):
            raise ValueError('failure_penalty must be positive and finite')
        self.failure_penalty=failure_penalty

    @property
    def case_definition(self):
        from .evaluation import CASE
        return CASE
    @property
    def ioh_name(self):return f'P007_{MODEL_REVISION}_{self.config.fingerprint[:12]}_failure_{self.failure_penalty}'
    def initial_design(self):return [.5]*self.dimension
    def decode(self,x):return decode(self.validate_design(x),self.config)
    def volume_gradient(self,x):return volume_gradient(self.validate_design(x),self.config)
    @property
    def candidate_repair(self):
        return dict(method='helmholtz_projected_volume_radial',revision='1',constraint='fluid_volume',limit=self.config.volume_limit)
    def repair_candidates(self,candidates,*,anchor=None):return repair_candidates(candidates,self.config,anchor=anchor)
    def preflight(self):
        try:import scipy.sparse.linalg
        except ImportError as exc:raise SimulationError('P007 requires scipy: python -m pip install -e ".[p007]"') from exc
    def evaluate_raw(self,x):
        from ....evaluation import CaseEvaluator,write_json
        self.last_result=None;g=self.decode(x)
        try:result=CaseEvaluator(self.case_definition,self.runtime).evaluate(g,self.config,output_root=self.output_root)
        except SimulationError as exc:
            # A failed baseline must never initialize an optimizer on a penalty.
            if self.failure_penalty is None or g['normalized_design']==self.initial_design():raise
            result=dict(status='penalized',objective=float(self.failure_penalty),objectives={},
                constraints={'fluid_volume':g['volume_fraction']-self.config.volume_limit},
                case_directory=exc.case_directory,error=str(exc))
        result['metadata']=dict(problem_id=self.name,problem_name=self.ioh_name,dimension=self.dimension,
            design=g['normalized_design'],config=self.config.to_dict(),config_sha256=self.config.fingerprint,
            paper_doi=DOI,paper_validated=False,replication_status='2d_equation_reconstruction_with_declared_estimates',
            objective_direction='minimize',objective_unit='dimensionless',cfd_adjoint_available=False,
            valid_cfd=result['status']=='ok',case_directory=result.get('case_directory'))
        if result.get('case_directory'):write_json(Path(result['case_directory'])/'evaluation.json',result)
        self.last_result=result;return result


__all__=['P007Problem','TopologyConfig']
