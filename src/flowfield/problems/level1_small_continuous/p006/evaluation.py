"""Shared execution lifecycle with model-specific validation."""
import hashlib
from pathlib import Path
from ....evaluation import CaseDefinition
from .case import generate_case,INPUTS,SOLVER_INPUTS
from .config import ManifoldConfig,MODEL_REVISION,DEFAULT_TIMEOUT_S
from .runtime import run_phase,eigen_headers


def validate(case,manifest,config):
    from . import P006Problem
    g=manifest['design_geometry']
    if g!=P006Problem(config=config).decode(g['normalized_design']):raise ValueError('Geometry was changed')
    # Older manifests have no descriptive fidelity metadata; their immutable
    # configuration still supplies the exact numerical settings.
    if 'fidelity' in manifest and manifest['fidelity']!=config.fidelity:
        raise ValueError('Fidelity metadata differs from the recorded configuration')
    for name in SOLVER_INPUTS:
        if (case/'solver'/name).read_bytes()!=(Path(__file__).parent/'solver'/name).read_bytes():
            raise ValueError('Solver sources differ from this model revision')


def execute(case,config,manifest,runtime,info,result):
    _,result['eigen']=eigen_headers()
    log=runtime.run(case,'checkMesh',('-allTopology','-meshQuality'),info=info).read_text()
    if 'Mesh OK.' not in log or 'Failed' in log:raise ValueError('checkMesh did not pass')
    result['mesh_validation']=dict(passed=True)
    run_phase(case,'build',runtime,info,config.processes)
    result['solver_binary_sha256']=hashlib.sha256((case/'bin/flowfieldP006Foam').read_bytes()).hexdigest()
    run_phase(case,'solve',runtime,info,config.processes)


def extract(case,config):
    from .metrics import extract_metrics
    return extract_metrics(case,config)


CASE=CaseDefinition(name='P006',config_type=ManifoldConfig,inputs=frozenset(INPUTS),
    commands=('checkMesh',),validate=validate,execute=execute,generate=generate_case,
    extract=extract,revision=MODEL_REVISION,execution_outputs=('bin','log.checkMesh'),timeout_s=DEFAULT_TIMEOUT_S)
