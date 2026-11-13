"""Hash-checked fresh-case VOF execution with preserved failed-run evidence."""
import hashlib
import json
from pathlib import Path
from ....evaluation import CaseDefinition, CaseEvaluator
from ....runtime import SimulationError
from .config import DualLayerConfig,MODEL_REVISION
from .case import generate_case,INPUTS,MONITOR_INPUTS
from .runtime import build_monitor
from .metrics import extract_metrics,verify_mesh_log
from .geometry import block_mesh

from ....mesh import PolyMesh


def _validate(case, m, c):
    from . import P005Problem
    g=json.loads((case/'geometry.json').read_text())
    if g!=m['geometry'] or g!=P005Problem(config=c).decode(g['normalized_design']):
        raise ValueError('P005 geometry does not match declared design')
    if m.get('mesh')!=block_mesh(g,c)[1]:
        raise ValueError('Mesh manifest differs from the physical configuration')
    for name in MONITOR_INPUTS:
        if (case/'monitor'/name).read_bytes()!=(Path(__file__).parent/'monitor'/name).read_bytes():
            raise ValueError('Monitor source revision differs; generate a fresh case')


def _execute(case, c, m, runtime, info, result):
    runtime.run(case,'blockMesh',('-noFunctionObjects',),info=info)
    PolyMesh.from_case(case)
    log=runtime.run(case,'checkMesh',('-allGeometry','-allTopology','-noFunctionObjects'),info=info).read_text()
    result['mesh_validation']=verify_mesh_log(log,m['mesh'])
    build_monitor(case,runtime,info)
    result['monitor_library_sha256']=hashlib.sha256((case/'lib/libflowfieldP005.so').read_bytes()).hexdigest()
    runtime.run(case,'interFoam',info=info)


extract_results = extract_metrics


CASE = CaseDefinition(
    name='P005', config_type=DualLayerConfig, inputs=frozenset(INPUTS),
    commands=('blockMesh', 'checkMesh', 'interFoam'), validate=_validate, execute=_execute,
    generate=generate_case, extract=extract_results, revision=MODEL_REVISION, execution_outputs=('lib', 'constant/polyMesh', 'log.blockMesh'))


def run_case(case, runtime=None):
    return CaseEvaluator(CASE, runtime).run_case(case)


def evaluate(geometry, config, output_root, runtime):
    return CaseEvaluator(CASE, runtime).evaluate(geometry, config, output_root=output_root)
