"""Shared execution lifecycle with source pinning and strict result extraction."""
import hashlib
import json
from pathlib import Path
from ....evaluation import CaseDefinition, CaseEvaluator
from ....runtime import SimulationError
from ....mesh import PolyMesh
from .case import INPUTS, SOLVER_INPUTS, generate_case
from .config import MODEL_REVISION, TopologyConfig
from .metrics import extract_metrics
from .runtime import run_phase


def _validate(case,manifest,config):
    from . import P007Problem
    geometry=json.loads((case/'design.json').read_text())
    if geometry!=P007Problem(config=config).decode(geometry['normalized_design']):
        raise ValueError('P007 design differs from its declared coordinates/configuration')
    for name in SOLVER_INPUTS:
        if (case/'solver'/name).read_bytes()!=(Path(__file__).parent/'solver'/name).read_bytes():
            raise ValueError('P007 solver source revision changed; generate a fresh case')


def _execute(case,config,manifest,runtime,info,result):
    PolyMesh.from_case(case)
    log=runtime.run(case,'checkMesh',('-allGeometry','-allTopology'),info=info).read_text()
    if 'Mesh OK.' not in log or 'Failed ' in log:raise SimulationError('P007 mesh check failed')
    result['mesh_validation']={'openfoam_all_checks_passed':True}
    run_phase(case,'build',runtime,info)
    result['solver_binary_sha256']=hashlib.sha256((case/'bin/flowfieldP007Foam').read_bytes()).hexdigest()
    run_phase(case,'solve',runtime,info)


CASE=CaseDefinition(name='P007',config_type=TopologyConfig,inputs=frozenset(INPUTS),commands=('checkMesh',),
    validate=_validate,execute=_execute,generate=generate_case,extract=extract_metrics,
    revision=MODEL_REVISION,execution_outputs=('bin','log.checkMesh'))


def run_case(case,runtime=None):return CaseEvaluator(CASE,runtime).run_case(case)

