"""P008: Li et al. 3D AEM channel depth/width and entropy-weighted performance."""
from pathlib import Path
import math
from ....problem import CFDProblem
from ....runtime import Runtime,SimulationError
from .config import AEMConfig,DOI,MODEL_REVISION,geometry as decode_geometry
class P008Problem(CFDProblem):
    config_type=AEMConfig
    requires_qualified_baseline=True
    def __init__(self,output_root='runs',*,config=None,runtime=None,failure_penalty=None):
        super().__init__('P008',2,[0.,0.],[1.,1.])
        self.config=config if config is not None else AEMConfig()
        if not isinstance(self.config,AEMConfig):raise TypeError('P008 requires AEMConfig')
        self.output_root=Path(output_root);self.runtime=runtime or Runtime(timeout_s=14400)
        if failure_penalty is not None and (isinstance(failure_penalty,bool) or not math.isfinite(failure_penalty) or failure_penalty<=0):raise ValueError('Invalid failure penalty')
        self.failure_penalty=failure_penalty
    @property
    def case_definition(self):
        from .evaluation import CASE
        return CASE
    @property
    def ioh_name(self):return f'P008_{MODEL_REVISION}_{self.config.fingerprint[:12]}_failure_{self.failure_penalty}'
    def initial_design(self):return [1/6,2/3]
    def decode(self,x):return decode_geometry(self.validate_design(x),self.config)
    def geometric_constraints(self,x):return self.decode(x)['constraints']
    def preflight(self):
        from .runtime import eigen_headers
        eigen_headers()
    def validate_optimization(self):
        if not self.config.ewm_calibration:raise ValueError('P008 optimization requires a frozen EWM calibration from qualified CFD cases; run flowfield-p008 paper-study or calibrate first')
        from dataclasses import replace
        cal=self.config.ewm_calibration
        if cal.get('qualified_cfd_panel') is not True or cal.get('model_revision')!=MODEL_REVISION or cal.get('config_sha256')!=replace(self.config,ewm_calibration=None).fingerprint:
            raise ValueError('EWM calibration must come from qualified P008 cases with this model and configuration')
    def evaluate_raw(self,x):
        from ....evaluation import CaseEvaluator,write_json
        self.last_result=None;g=self.decode(x)
        try:result=CaseEvaluator(self.case_definition,self.runtime).evaluate(g,self.config,output_root=self.output_root)
        except SimulationError as exc:
            if self.failure_penalty is None or g['normalized_design']==self.initial_design():raise
            result=dict(status='penalized',objective=self.failure_penalty,objectives={},constraints={},error=str(exc),case_directory=exc.case_directory)
        result['metadata']=dict(problem_id=self.name,problem_name=self.ioh_name,dimension=2,design=g['normalized_design'],
            geometry=g,config=self.config.to_dict(),config_sha256=self.config.fingerprint,paper_doi=DOI,
            paper_validated=False,valid_cfd=result['status']=='ok',replication_status='coupled_3d_equation_reconstruction',
            objective_unit='negative_entropy_score' if self.config.ewm_calibration else 'negative_A_per_cm2',case_directory=result.get('case_directory'))
        if result.get('case_directory'):write_json(Path(result['case_directory'])/'evaluation.json',result)
        self.last_result=result;return result
__all__=['P008Problem','AEMConfig']
