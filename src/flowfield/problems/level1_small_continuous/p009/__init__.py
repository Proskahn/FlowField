"""P009: Chen's single-phase porous oxygen-side flow and V/delta-p objective."""
import math
from pathlib import Path
from ....problem import CFDProblem
from ....runtime import Runtime, SimulationError
from .config import ChenConfig, DOI, MODEL_REVISION, geometry, physical_geometry


class P009Problem(CFDProblem):
    config_type = ChenConfig
    requires_qualified_baseline = True

    def __init__(self, output_root='runs', *, config=None, runtime=None, failure_penalty=None):
        super().__init__('P009', 3, [0.]*3, [1.]*3)
        self.config = config if config is not None else ChenConfig()
        if not isinstance(self.config, ChenConfig):
            raise TypeError('P009 requires ChenConfig')
        self.output_root = Path(output_root)
        self.runtime = runtime or Runtime(timeout_s=14400)
        if failure_penalty is not None and (isinstance(failure_penalty, bool) or not math.isfinite(failure_penalty) or failure_penalty <= 0):
            raise ValueError('Failure penalty must be positive and finite')
        self.failure_penalty = failure_penalty

    @property
    def case_definition(self):
        from .evaluation import CASE
        return CASE

    @property
    def ioh_name(self):
        return f'P009_{MODEL_REVISION}_{self.config.fingerprint[:12]}_failure_{self.failure_penalty}'

    def decode(self, x):
        return geometry(self.validate_design(x), self.config)

    def preflight(self):
        import importlib.metadata
        try:
            version = importlib.metadata.version('gmsh')
        except importlib.metadata.PackageNotFoundError as exc:
            raise SimulationError('P009 needs python -m pip install -e ".[p009]"') from exc
        if version != '4.13.1':
            raise SimulationError('P009 mesh reproducibility requires gmsh==4.13.1')

    def evaluate_physical(self, angle, flow, thickness):
        return self._evaluate(physical_geometry(angle, flow, thickness, self.config), penalize=False)

    def evaluate_raw(self, x):
        g = self.decode(x)
        return self._evaluate(g, penalize=self.validate_design(x) != self.initial_design())

    def _evaluate(self, g, *, penalize):
        from ....evaluation import CaseEvaluator, write_json
        self.last_result = None
        try:
            result = CaseEvaluator(self.case_definition, self.runtime).evaluate(g, self.config, output_root=self.output_root)
        except SimulationError as exc:
            if not penalize or self.failure_penalty is None:
                raise
            result = dict(status='penalized', objective=self.failure_penalty, objectives={}, constraints={},
                error=str(exc), case_directory=exc.case_directory)
        result['metadata'] = dict(problem_id=self.name, problem_name=self.ioh_name, dimension=3,
            design=g['normalized_design'], geometry=g, config=self.config.to_dict(),
            config_sha256=self.config.fingerprint, paper_doi=DOI, paper_validated=False,
            valid_cfd=result['status']=='ok', objective_unit='negative_m_per_s_per_bar',
            replication_status='provisional_hydrodynamic_reconstruction')
        if result.get('case_directory'):
            write_json(Path(result['case_directory'])/'evaluation.json', result)
        self.last_result = result
        return result


__all__ = ['P009Problem', 'ChenConfig']
