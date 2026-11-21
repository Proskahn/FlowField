"""P010: coupled ESCFF PEM fuel-cell CFD and four-variable net-power optimization."""
from pathlib import Path
from ....problem import CFDProblem
from ....runtime import Runtime
from .config import PEMFCConfig,BASELINE,normalized,decode,MODEL_REVISION

class P010Problem(CFDProblem):
    config_type=PEMFCConfig
    requires_qualified_baseline=True
    def __init__(self,output_root='runs',*,config=None,runtime=None):
        super().__init__('P010',4,[0.]*4,[1.]*4)
        self.config=config if config is not None else PEMFCConfig()
        if not isinstance(self.config,PEMFCConfig):raise TypeError('P010 requires PEMFCConfig')
        self.output_root=Path(output_root);self.runtime=runtime or Runtime(timeout_s=14400)
    @property
    def case_definition(self):
        from .evaluation import CASE
        return CASE
    @property
    def ioh_name(self):return f'P010_{MODEL_REVISION}_{self.config.fingerprint[:12]}'
    def initial_design(self):return normalized(BASELINE)
    def decode(self,x):return decode(self.validate_design(x),self.config)
    def case_arguments(self,x):return self.decode(x),self.config
    def preflight(self):
        from .runtime import eigen_headers
        eigen_headers()
    def evaluate_raw(self,x):
        from ....evaluation import CaseEvaluator
        self.last_result=None;g=self.decode(x)
        result=CaseEvaluator(self.case_definition,self.runtime).evaluate(g,self.config,output_root=self.output_root)
        result['metadata']=dict(problem_id='P010',problem_name=self.ioh_name,design=g['normalized_design'],
            dimension=4,geometry=g,config=self.config.to_dict(),model_revision=MODEL_REVISION,
            valid_cfd=result['status']=='ok',paper_validated=False,objective_unit='negative_W',case_directory=result['case_directory'])
        self.last_result=result;return result

__all__=['P010Problem','PEMFCConfig']
