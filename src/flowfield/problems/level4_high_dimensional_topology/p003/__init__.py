"""P003: density-field topology of a depth-averaged PEMFC cathode."""
from pathlib import Path
import math
from ....base import FlowFieldProblem
from ....runtime import Runtime,SimulationError
from .config import TopologyConfig,DOI,MODEL_REVISION
from .design import decode as decode_field,preset,volume_gradient


class P003Problem(FlowFieldProblem):
    def __init__(self,output_root='runs',*,config=None,runtime=None,failure_penalty=None):
        self.config=config if config is not None else TopologyConfig()
        if not isinstance(self.config,TopologyConfig):raise TypeError('P003 requires TopologyConfig')
        n=self.config.dimension
        super().__init__('P003',n,[0.]*n,[1.]*n)
        self.output_root=Path(output_root);self.runtime=runtime or Runtime(timeout_s=1800)
        if failure_penalty is not None and (isinstance(failure_penalty,bool) or not math.isfinite(failure_penalty) or failure_penalty<=0):
            raise ValueError('failure_penalty must be positive and finite')
        self.failure_penalty=failure_penalty
    @property
    def ioh_name(self):
        policy='raise' if self.failure_penalty is None else format(self.failure_penalty,'.12g')
        return f'P003_{MODEL_REVISION}_{self.config.fingerprint[:12]}_failure_{policy}'
    def decode(self,x):return decode_field(self.validate_design(x),self.config)
    def initial_design(self):return preset('uniform',self.config)
    def preset(self,name):return preset(name,self.config)
    def volume_gradient(self,x):return volume_gradient(self.validate_design(x),self.config)
    def evaluate_raw(self,x):
        from .evaluation import evaluate
        from ....evaluation import write_json
        self.last_result=None;geometry=self.decode(x)
        metadata=dict(problem_id=self.name,problem_name=self.ioh_name,dimension=self.dimension,
            design=geometry['normalized_design'],config=self.config.to_dict(),config_sha256=self.config.fingerprint,
            paper_doi=DOI,paper_validated=False,replication_status='provisional_depth_averaged_forward_model',
            objective_direction='minimize',objective_unit='dimensionless',
            volume_fraction=geometry['volume_fraction'],cfd_adjoint_available=False)
        try:result=evaluate(geometry,self.config,self.output_root,self.runtime)
        except SimulationError as exc:
            if self.failure_penalty is None:raise
            metadata.update(valid_cfd=False,case_directory=exc.case_directory,error=str(exc))
            result=dict(status='penalized',objective=float(self.failure_penalty),objectives={},
                        constraints={'channel_volume':geometry['volume_fraction']-self.config.volume_limit},error=str(exc))
        else:metadata.update(valid_cfd=True,case_directory=result['case_directory'],runtime=result['runtime'])
        result['metadata']=metadata
        if metadata.get('case_directory'):write_json(Path(metadata['case_directory'])/'evaluation.json',result)
        self.last_result=result
        return result


__all__=['P003Problem','TopologyConfig']
