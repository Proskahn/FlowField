"""Build and solve a fresh P001 reconstruction, preserving all failure records."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

from ....evaluation import CaseDefinition, CaseEvaluator
from ....runtime import SimulationError
from .case import generate_case
from .config import MODEL_REVISION, PPFCConfig
from .metrics import extract_metrics

INPUTS = {'0/U', '0/p', '0/k', '0/epsilon', '0/nut', 'constant/transportProperties',
          'constant/turbulenceProperties', 'system/controlDict', 'system/fvSchemes',
          'system/fvSolution', 'layout.svg', 'units.json'}

from ....mesh import PolyMesh


def _validate(case, manifest, config):
    # Regenerate design-dependent inputs in memory and validate the exposed layout.
    from .geometry import unit_layout, layout_svg
    geometry = manifest['geometry']
    from . import P001Problem
    if geometry != P001Problem(config=config).decode(geometry['normalized_design']):
        raise ValueError('P001 geometry manifest differs from its declared design')
    if json.loads((case/'units.json').read_text()) != unit_layout(geometry) or (case/'layout.svg').read_text() != layout_svg(geometry):
        raise ValueError('P001 geometry inputs were modified')


def _execute(case, config, manifest, runtime, info, result):
    result.update(objective_direction='minimize', objective_unit='dimensionless', paper_validated=False)
    # Explicit source path prevents an unrelated editable install being used
    # in the mesher subprocess. Running it separately bounds CAD/mesh time.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    source_root = str(Path(__file__).resolve().parents[4])
    env['PYTHONPATH'] = source_root + os.pathsep + env.get('PYTHONPATH', '')
    with (case/'log.gmsh').open('x') as log:
        subprocess.run([sys.executable, '-m', __package__+'.geometry', str(case)],
            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=runtime.timeout_s, check=True)
    runtime.run(case, 'gmshToFoam', ('mesh.msh',), info=info)
    boundary = case/'constant/polyMesh/boundary'
    text = boundary.read_text()
    text, count = re.subn(r'(\bwalls\s*\{\s*type\s+)patch;', r'\1wall;', text)
    if count != 1 and not re.search(r'\bwalls\s*\{\s*type\s+wall;', text):
        raise SimulationError('Cannot identify the imported wall patch')
    boundary.write_text(text)
    result['mesh_sha256'] = PolyMesh.from_case(case).checksums()
    mesh_log = runtime.run(case, 'checkMesh', ('-allGeometry', '-allTopology'), info=info).read_text()
    if 'Mesh OK.' not in mesh_log or 'Failed ' in mesh_log:
        raise SimulationError('Mesh validation failed; simpleFoam was not started')
    runtime.run(case, 'simpleFoam', info=info)


def extract_results(case, config):
    metrics = extract_metrics(case, config)
    return dict(objective=-metrics['uniformity'], metrics=metrics,
                objectives={'negative_uniformity': -metrics['uniformity']}, constraints={})


CASE = CaseDefinition(
    name='P001', config_type=PPFCConfig, inputs=frozenset(INPUTS),
    commands=('gmshToFoam', 'checkMesh', 'simpleFoam'), validate=_validate, execute=_execute,
    generate=generate_case, extract=extract_results, revision=MODEL_REVISION, execution_outputs=('constant/polyMesh', 'mesh.msh', 'log.gmsh'))


def run_case(case, runtime=None):
    return CaseEvaluator(CASE, runtime).run_case(case)


def evaluate(geometry, config, output_root, runtime):
    return CaseEvaluator(CASE, runtime).evaluate(geometry, config, output_root=output_root)
