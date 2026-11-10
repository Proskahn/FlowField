"""P004 discrete paths; current data/surrogate and auxiliary hydraulics stay distinct."""
from pathlib import Path
from ....base import FlowFieldProblem
from ....runtime import Runtime
from .library import catalog,path_from_id,path_id,validate_path,serpentine,DOI,LIBRARY_REVISION
from .geometry import decode as decode_geometry
from .config import HydraulicConfig,HYDRAULIC_REVISION
from .dataset import CurrentDataset,MissingPaperDataError


class P004Problem(FlowFieldProblem):
    variable_type='integer'
    def __init__(self,output_root='runs',*,dataset=None,model=None):
        super().__init__('P004',49,[0]*49,[48]*49)
        if dataset is not None and model is not None:raise ValueError('Choose a dataset or a trained surrogate explicitly')
        self.output_root=Path(output_root)
        self.dataset=CurrentDataset(dataset) if dataset is not None else None
        if model is not None:
            from .surrogate import Predictor
            self.model=Predictor(model)
        else:self.model=None
    @property
    def ioh_name(self):
        backend='missing_current_data'
        if self.dataset:backend='tabulated_'+self.dataset.sha256[:16]
        if self.model:backend='surrogate_'+self.model.sha256[:16]
        return f'P004_{LIBRARY_REVISION}_{backend}'
    def validate_design(self,x):return validate_path(super().validate_design(x))
    def decode(self,x):return decode_geometry(self.validate_design(x))
    def initial_design(self):return serpentine()
    def design_from_id(self,identifier):return path_from_id(identifier)
    def sample_design(self,rng):
        ids=sorted(self.dataset.labels) if self.dataset else range(len(catalog()))
        return path_from_id(ids[rng.randrange(len(ids))])
    def evaluate_raw(self,x):
        self.last_result=None;geometry=self.decode(x);identifier=geometry['library_id']
        if self.dataset:
            current=self.dataset.value(identifier);kind='tabulated_external_current_density'
            source=dict(dataset_sha256=self.dataset.sha256,declared_source=self.dataset.source)
        elif self.model:
            current=float(self.model.predict([geometry['path']])[0]);kind='trained_surrogate_prediction'
            source=dict(model_sha256=self.model.sha256,training_dataset_sha256=self.model.metadata['dataset_sha256'])
        else:raise MissingPaperDataError('P004 current-density evaluation needs an explicitly supplied dataset or trained model. Supplementary Table S1 and author labels are unavailable. Use P004-hydraulic only for the separate channel-flow verification.')
        result=dict(status='ok',objective=-current,objectives={'negative_mean_current_density_A_m2':-current},constraints={},
            metrics={'mean_current_density_A_m2':current,'power_density_W_m2':.475*current},
            metadata=dict(problem_id=self.name,problem_name=self.ioh_name,dimension=49,variable_type='integer',
                library_id=identifier,path=geometry['path'],evaluation_kind=kind,voltage_v=.475,paper_doi=DOI,
                paper_validated=False,executed_cfd=False,objective_unit='A/m²',**source))
        self.last_result=result;return result


class P004HydraulicProblem(P004Problem):
    """Auxiliary pressure-drop benchmark; never substitutes for fuel-cell current."""
    def __init__(self,output_root='runs',*,config=None,runtime=None):
        super().__init__(output_root);self.name='P004-hydraulic'
        self.config=config if config is not None else HydraulicConfig()
        if not isinstance(self.config,HydraulicConfig):raise TypeError('Expected HydraulicConfig')
        self.runtime=runtime or Runtime(timeout_s=1800)
    @property
    def ioh_name(self):return f'P004_hydraulic_{HYDRAULIC_REVISION}_{self.config.fingerprint[:16]}'
    def evaluate_raw(self,x):
        from .hydraulics import evaluate
        from ....evaluation import write_json
        self.last_result=None;geometry=self.decode(x)
        result=evaluate(geometry['path'],self.output_root,self.config,self.runtime)
        result['metadata']=dict(problem_id=self.name,problem_name=self.ioh_name,dimension=49,variable_type='integer',
            library_id=geometry['library_id'],path=geometry['path'],evaluation_kind='auxiliary_channel_only_CFD',
            config_sha256=self.config.fingerprint,paper_doi=DOI,paper_validated=False,
            executed_cfd=True,electrochemistry_solved=False,objective_unit='Pa')
        write_json(Path(result['case_directory'])/'evaluation.json',result);self.last_result=result;return result


__all__=['P004Problem','P004HydraulicProblem','HydraulicConfig','MissingPaperDataError']
