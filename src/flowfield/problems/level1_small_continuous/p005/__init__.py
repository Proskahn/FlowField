"""P005: physical-only dual-layer oxygen-bubble detachment benchmark (Level I)."""
import math
from pathlib import Path
from ....problem import CFDProblem
from ....runtime import Runtime,SimulationError
from .config import DualLayerConfig,BASELINE_X,DOI,MODEL_REVISION
from .geometry import decode as decode_geometry


class P005Problem(CFDProblem):
    config_type = DualLayerConfig

    @property
    def case_definition(self):
        from .evaluation import CASE
        return CASE

    def __init__(self,output_root='runs',*,config=None,runtime=None,failure_penalty=None):
        self.config=config or DualLayerConfig()
        if not isinstance(self.config,DualLayerConfig):raise TypeError('P005 requires DualLayerConfig')
        super().__init__('P005',2,[0.,0.],[1.,1.])
        self.output_root=Path(output_root);self.runtime=runtime or Runtime(timeout_s=1800)
        if failure_penalty is not None and (isinstance(failure_penalty,bool) or not math.isfinite(failure_penalty) or failure_penalty<=2*self.config.end_time_s):
            raise ValueError('Failure penalty must be finite and greater than the non-detachment score')
        self.failure_penalty=failure_penalty
    @property
    def ioh_name(self):
        policy='raise' if self.failure_penalty is None else format(self.failure_penalty,'.12g')
        return f'P005_{MODEL_REVISION}_{self.config.fingerprint[:12]}_failure_{policy}'
    def initial_design(self):return list(BASELINE_X)
    def decode(self,x):return decode_geometry(self.validate_design(x),self.config)
    def evaluate_raw(self,x):
        from .evaluation import evaluate
        from ....evaluation import write_json
        self.last_result=None;g=self.decode(x)
        metadata=dict(problem_id=self.name,problem_name=self.ioh_name,dimension=self.dimension,
            design=g['normalized_design'],config=self.config.to_dict(),config_sha256=self.config.fingerprint,
            paper_doi=DOI,paper_validated=False,replication_status='provisional_physical_VOF_reconstruction',
            objective_direction='minimize',objective_unit='s',data_driven_model=False,
            detachment_criterion='persistent_resolved_component_disconnected_from_electrode_and_gas_inlet',
            scalarization='detachment_time_if_observed_else_2_times_observation_horizon',
            unreported_inputs=['gas_inlet_side_m','gas_inlet_z_m','fluid_properties','initial_flow',
                'contact_angle_phase_convention','event_thresholds','outlet_hydrostatic_convention'])
        try:result=evaluate(g,self.config,self.output_root,self.runtime)
        except SimulationError as exc:
            if self.failure_penalty is None:raise
            result=dict(status='penalized',objective=float(self.failure_penalty),objectives={},constraints={},error=str(exc))
            metadata.update(valid_cfd=False,case_directory=exc.case_directory,error=str(exc))
        else:metadata.update(valid_cfd=True,case_directory=result['case_directory'],runtime=result['runtime'])
        result['metadata']=metadata
        if metadata.get('case_directory'):write_json(Path(metadata['case_directory'])/'evaluation.json',result)
        self.last_result=result;return result


__all__=['P005Problem','DualLayerConfig','BASELINE_X']
