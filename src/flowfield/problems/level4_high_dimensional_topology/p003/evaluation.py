"""Fresh-case execution with input hashes, per-case compilation and strict verification."""

import hashlib
import json
from pathlib import Path

from ....evaluation import CaseDefinition, CaseEvaluator
from ....runtime import SimulationError
from .case import INPUTS, SOLVER_INPUTS, generate_case
from .config import MODEL_REVISION, TopologyConfig
from .metrics import extract_metrics
from .runtime import run_phase

from ....mesh import PolyMesh


def _validate(case, manifest, config):
    from . import P003Problem
    geometry = json.loads((case/'design.json').read_text())
    if geometry != P003Problem(config=config).decode(geometry['normalized_design']):
        raise ValueError('P003 design differs from its declared coordinates and configuration')
    for name in SOLVER_INPUTS:
        if (case/'solver'/name).read_bytes() != (Path(__file__).parent/'solver'/name).read_bytes():
            raise ValueError('Case uses a different solver source revision')


def _execute(case, config, manifest, runtime, info, result):
    result.update(objective_unit='dimensionless', scalarization='weighted_criteria_plus_explicit_volume_penalty')
    PolyMesh.from_case(case)
    mesh_log = runtime.run(case, 'checkMesh', ('-allGeometry','-allTopology'), info=info).read_text()
    if 'Mesh OK.' not in mesh_log or 'Failed ' in mesh_log:
        raise SimulationError('Mesh check failed')
    result['mesh_validation'] = {'openfoam_all_checks_passed':True}
    run_phase(case, 'build', runtime, info)
    result['solver_binary_sha256'] = hashlib.sha256((case/'bin/flowfieldP003Foam').read_bytes()).hexdigest()
    run_phase(case, 'solve', runtime, info)


extract_results = extract_metrics


CASE = CaseDefinition(
    name='P003', config_type=TopologyConfig, inputs=frozenset(INPUTS),
    commands=('checkMesh',), validate=_validate, execute=_execute,
    generate=generate_case, extract=extract_results, revision=MODEL_REVISION, execution_outputs=('bin', 'log.checkMesh'))


def run_case(case, runtime=None):
    return CaseEvaluator(CASE, runtime).run_case(case)


def evaluate(geometry, config, output_root, runtime):
    return CaseEvaluator(CASE, runtime).evaluate(geometry, config, output_root=output_root)
