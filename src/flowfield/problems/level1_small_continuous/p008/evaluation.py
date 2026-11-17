"""Shared execution lifecycle with model-specific validation."""
import hashlib
from pathlib import Path
from ....evaluation import CaseDefinition
from .case import generate_case,INPUTS,SOLVER_INPUTS
from .config import AEMConfig,MODEL_REVISION
from .runtime import run_phase,eigen_headers


def validate(case,manifest,config):
    from . import P008Problem
    import json
    if json.loads((case/'calibration.json').read_text())!=config.ewm_calibration:raise ValueError('Calibration differs from the immutable configuration')
    g=manifest['design_geometry']
    if g!=P008Problem(config=config).decode(g['normalized_design']):raise ValueError('Geometry was changed')
    for name in SOLVER_INPUTS:
        if (case/'solver'/name).read_bytes()!=(Path(__file__).parent/'solver'/name).read_bytes():
            raise ValueError('Solver sources differ from this model revision')


def execute(case,config,manifest,runtime,info,result):
    _,result['eigen']=eigen_headers()
    log=runtime.run(case,'checkMesh',('-allTopology','-meshQuality'),info=info).read_text()
    from .mesh_checks import assess_mesh
    result['mesh_validation']=assess_mesh(case,log,manifest)
    run_phase(case,'build',runtime,info,config.processes)
    result['solver_binary_sha256']=hashlib.sha256((case/'bin/flowfieldP008Foam').read_bytes()).hexdigest()
    run_phase(case,'solve',runtime,info,config.processes)


def extract(case,config):
    from .metrics import extract_metrics
    return extract_metrics(case,config)


CASE=CaseDefinition(name='P008',config_type=AEMConfig,inputs=frozenset(INPUTS),
    commands=('checkMesh',),validate=validate,execute=execute,generate=generate_case,
    extract=extract,revision=MODEL_REVISION,execution_outputs=('bin','log.checkMesh'),timeout_s=14400)
