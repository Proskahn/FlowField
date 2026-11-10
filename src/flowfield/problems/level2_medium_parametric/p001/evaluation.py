"""Build and solve a fresh P001 reconstruction, preserving all failure records."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from ....evaluation import write_json
from ....runtime import Runtime, SimulationError
from .case import generate_case
from .config import MODEL_REVISION, PPFCConfig
from .metrics import extract_metrics

INPUTS = {'0/U', '0/p', '0/k', '0/epsilon', '0/nut', 'constant/transportProperties',
          'constant/turbulenceProperties', 'system/controlDict', 'system/fvSchemes',
          'system/fvSolution', 'layout.svg', 'units.json'}


def run_case(case, runtime=None):
    case = Path(case).resolve()
    runtime = runtime or Runtime(timeout_s=1800)
    manifest = json.loads((case/'case.json').read_text())
    config = PPFCConfig(**manifest['config'])
    if (manifest.get('problem') != 'P001' or manifest.get('model_revision') != MODEL_REVISION
            or manifest.get('config_sha256') != config.fingerprint or set(manifest.get('input_sha256', {})) != INPUTS):
        raise ValueError('Invalid P001 reconstruction manifest')
    # Regenerate design-dependent inputs in memory and validate the exposed layout.
    from .geometry import unit_layout, layout_svg
    geometry = manifest['geometry']
    from . import P001Problem
    if geometry != P001Problem(config=config).decode(geometry['normalized_design']):
        raise ValueError('P001 geometry manifest differs from its declared design')
    if json.loads((case/'units.json').read_text()) != unit_layout(geometry) or (case/'layout.svg').read_text() != layout_svg(geometry):
        raise ValueError('P001 geometry inputs were modified')
    for name, digest in manifest['input_sha256'].items():
        if hashlib.sha256((case/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Case input changed: {name}; generate a fresh case')
    if any((case/name).exists() for name in ('postProcessing', 'constant/polyMesh', 'result.json', 'mesh.msh', 'log.gmsh')):
        raise FileExistsError('P001 case already contains execution data; generate a fresh case')
    (case/'.execution-started').open('x').close()
    result = {**manifest, 'status':'failed', 'case_directory':str(case),
        'created_utc':datetime.now(timezone.utc).isoformat(), 'objective_direction':'minimize',
        'objective_unit':'dimensionless', 'paper_validated':False}
    started = time.monotonic()
    try:
        info = runtime.resolve(commands=('gmshToFoam', 'checkMesh', 'simpleFoam'))
        result['runtime'] = info
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
        result['mesh_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((case/'constant/polyMesh').iterdir()) if p.is_file()}
        mesh_log = runtime.run(case, 'checkMesh', ('-allGeometry', '-allTopology'), info=info).read_text()
        if 'Mesh OK.' not in mesh_log or 'Failed ' in mesh_log:
            raise SimulationError('Mesh validation failed; simpleFoam was not started')
        runtime.run(case, 'simpleFoam', info=info)
        metrics = extract_metrics(case, config)
        result.update(status='ok', objective=-metrics['uniformity'], metrics=metrics,
            objectives={'negative_uniformity':-metrics['uniformity']}, constraints={})
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        result['error'] = str(exc)
        raise SimulationError(f'P001 failed in {case}: {exc}; see log.gmsh and log.*', case_directory=case) from exc
    except KeyboardInterrupt:
        result['error'] = 'Interrupted by user'
        raise
    finally:
        result['wall_time_s'] = time.monotonic()-started
        write_json(case/'result.json', result)
    return result


def evaluate(geometry, config, output_root, runtime):
    case = Path(output_root).resolve()/('P001-'+uuid.uuid4().hex)
    generate_case(case, geometry, config)
    return run_case(case, runtime)
