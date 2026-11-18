"""Shared immutable case lifecycle; actual OpenFOAM CFD only."""
import os
from pathlib import Path
import re
import subprocess
import sys
from ....evaluation import CaseDefinition
from ....mesh import PolyMesh
from ....runtime import SimulationError
from .case import INPUTS, generate_case
from .config import ChenConfig, MODEL_REVISION, physical_geometry
from .metrics import extract_metrics


def validate(case, manifest, config):
    g = manifest['geometry']
    expected = physical_geometry(g['angle_deg'], g['flow_lpm'], g['thickness_mm'], config)
    if g != expected:
        raise ValueError('P009 geometry differs from its declared physical design')
    if (case/'mesher.py').read_text() != Path(__file__).with_name('geometry.py').read_text():
        raise ValueError('P009 mesher source revision differs; preserve this case and generate a fresh one')


def execute(case, config, manifest, runtime, info, result):
    result.update(objective_direction='minimize', objective_unit='negative_m_per_s_per_bar', paper_validated=False)
    print(f'[{case.name}] Meshing P009...', flush=True)
    with (case/'log.gmsh').open('x') as stream:
        subprocess.run([sys.executable, str(case/'mesher.py'), str(case)],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), cwd=case, stdout=stream,
            stderr=subprocess.STDOUT, check=True, timeout=runtime.timeout_s)
    runtime.run(case, 'gmshToFoam', ('mesh.msh',), info=info)
    boundary = case/'constant/polyMesh/boundary'
    text = boundary.read_text()
    text, count = re.subn(r'(\bwalls\s*\{\s*type\s+)patch;', r'\1wall;', text)
    if count != 1 and not re.search(r'\bwalls\s*\{\s*type\s+wall;', text):
        raise SimulationError('Cannot identify P009 wall patch')
    boundary.write_text(text)
    result['mesh_sha256'] = PolyMesh.from_case(case).checksums()
    mesh_log = runtime.run(case, 'checkMesh', ('-allTopology', '-meshQuality'), info=info).read_text()
    if 'Mesh OK.' not in mesh_log or re.search(r'Failed\s+\d+\s+mesh checks', mesh_log):
        raise SimulationError('P009 mesh failed; inspect log.checkMesh')
    print(f'[{case.name}] Solving P009; residuals in log.simpleFoam', flush=True)
    runtime.run(case, 'simpleFoam', info=info)


def extract(case, config):
    metrics = extract_metrics(case, config)
    return dict(objective=-metrics['performance_m_s_per_bar'], metrics=metrics,
        objectives=dict(negative_performance=-metrics['performance_m_s_per_bar'],
            pressure_drop_pa=metrics['pressure_drop_pa'], flow_cv=metrics['flow_cv']), constraints={})


CASE = CaseDefinition(name='P009', config_type=ChenConfig, inputs=INPUTS,
    commands=('gmshToFoam', 'checkMesh', 'simpleFoam'), validate=validate, execute=execute,
    generate=generate_case, extract=extract, revision=MODEL_REVISION,
    execution_outputs=('constant/polyMesh', 'mesh.msh', 'log.gmsh'), timeout_s=14400)
