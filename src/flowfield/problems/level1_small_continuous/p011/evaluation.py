"""Build/execute the coupled model and independently qualify saved fields."""
import hashlib
from pathlib import Path
from ....evaluation import CaseDefinition
from .case import generate_case,INPUTS,SOLVER_INPUTS
from .config import TaperedConfig,MODEL_REVISION,decode
from .runtime import eigen_headers,run_phase

def validate(case,manifest,config):
    g=manifest['design_geometry']
    if g!=decode(g['normalized_design'],config):raise ValueError('Case design was changed')
    root=Path(__file__).parent
    if manifest.get('geometry_source_sha256')!=hashlib.sha256((root/'geometry.py').read_bytes()).hexdigest():raise ValueError('Mesher revision changed; generate a fresh case')
    for name in SOLVER_INPUTS:
        if (case/'solver'/name).read_bytes()!=(root/'solver'/name).read_bytes():raise ValueError('Solver sources changed; generate a fresh case')

def execute(case,config,manifest,runtime,info,result):
    _,result['eigen']=eigen_headers()
    log=runtime.run(case,'checkMesh',('-allTopology','-meshQuality'),info=info).read_text()
    if 'Mesh OK.' not in log or 'Failed' in log:raise ValueError('checkMesh did not pass')
    result['mesh_validation']=dict(passed=True)
    run_phase(case,'build',runtime,info)
    result['solver_binary_sha256']=hashlib.sha256((case/'bin/flowfieldP011Foam').read_bytes()).hexdigest()
    run_phase(case,'solve',runtime,info)

def extract(case,config):
    from .metrics import extract_metrics
    return extract_metrics(case,config)

CASE=CaseDefinition(name='P011',config_type=TaperedConfig,inputs=frozenset(INPUTS),commands=('checkMesh',),
 validate=validate,execute=execute,generate=generate_case,extract=extract,revision=MODEL_REVISION,
 execution_outputs=('bin','log.checkMesh'),timeout_s=14400)
