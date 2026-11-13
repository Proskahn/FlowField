"""Fresh-case execution with input hashes, per-case compilation and strict verification."""

import hashlib
import json
from pathlib import Path

from ....evaluation import CaseDefinition, CaseEvaluator
from ....runtime import SimulationError
from .case import INPUTS, SOLVER_INPUTS, generate_case
from .config import MODEL_REVISION, PEMConfig, parameters_for
from .metrics import extract_metrics
from .mesh_checks import assess_mesh
from .runtime import run_phase

from ....mesh import PolyMesh


def _validate(case, manifest, config):
    from . import P002Problem
    geometry = manifest['design_geometry']
    if (geometry != P002Problem(config=config).decode(geometry['normalized_design'])
            or manifest['parameters'] != parameters_for(geometry,config)):
        raise ValueError('P002 parameters differ from the declared design and configuration')
    for name in SOLVER_INPUTS:
        if (case/'solver'/name).read_bytes() != (Path(__file__).parent/'solver'/name).read_bytes():
            raise ValueError('Case uses a different solver source revision')


def _execute(case, config, manifest, runtime, info, result):
    result.update(objective_unit=config.objective_unit, scalarization=config.scalarization)
    PolyMesh.from_case(case)
    mesh_log = runtime.run(case, 'checkMesh', ('-allGeometry','-allTopology'), info=info).read_text()
    result['mesh_validation'] = assess_mesh(case, mesh_log, manifest)
    run_phase(case, 'build', runtime, info)
    result['solver_binary_sha256'] = hashlib.sha256((case/'bin/flowfieldP002Foam').read_bytes()).hexdigest()
    run_phase(case, 'solve', runtime, info)


extract_results = extract_metrics


CASE = CaseDefinition(
    name='P002', config_type=PEMConfig, inputs=frozenset(INPUTS),
    commands=('checkMesh',), validate=_validate, execute=_execute,
    generate=generate_case, extract=extract_results, revision=MODEL_REVISION, execution_outputs=('bin', 'log.checkMesh'))


def run_case(case, runtime=None):
    return CaseEvaluator(CASE, runtime).run_case(case)


def evaluate(geometry, config, output_root, runtime):
    return CaseEvaluator(CASE, runtime).evaluate(geometry, config, output_root=output_root)
