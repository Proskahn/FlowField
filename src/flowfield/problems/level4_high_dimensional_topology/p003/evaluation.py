"""Fresh-case execution with input hashes, per-case compilation and strict verification."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

from ....evaluation import write_json
from ....runtime import Runtime, SimulationError
from .case import INPUTS, SOLVER_INPUTS, generate_case
from .config import MODEL_REVISION, TopologyConfig
from .metrics import extract_metrics
from .runtime import run_phase


def run_case(case, runtime=None):
    from . import P003Problem
    case = Path(case).resolve()
    runtime = runtime or Runtime(timeout_s=1800)
    manifest = json.loads((case/'case.json').read_text())
    config = TopologyConfig(**manifest['config'])
    if (manifest.get('problem') != 'P003' or manifest.get('model_revision') != MODEL_REVISION
            or manifest.get('config_sha256') != config.fingerprint
            or set(manifest.get('input_sha256', {})) != INPUTS):
        raise ValueError('Invalid P003 manifest')
    geometry = json.loads((case/'design.json').read_text())
    if geometry != P003Problem(config=config).decode(geometry['normalized_design']):
        raise ValueError('P003 design differs from its declared coordinates and configuration')
    for name,digest in manifest['input_sha256'].items():
        if hashlib.sha256((case/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Case input changed: {name}; generate a fresh case')
    for name in SOLVER_INPUTS:
        if (case/'solver'/name).read_bytes() != (Path(__file__).parent/'solver'/name).read_bytes():
            raise ValueError('Case uses a different solver source revision')
    if any((case/name).exists() for name in ('postProcessing','result.json','bin','log.checkMesh')):
        raise FileExistsError('Case already contains execution data; generate a fresh case')
    if any(p.is_dir() and p.name.replace('.','',1).isdigit() and float(p.name)>0 for p in case.iterdir()):
        raise FileExistsError('Case contains old fields; generate a fresh case')
    (case/'.execution-started').open('x').close()
    result = dict(manifest, status='failed', case_directory=str(case),
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  objective_unit='dimensionless', scalarization='weighted_criteria_plus_explicit_volume_penalty')
    started = time.monotonic()
    try:
        info = runtime.resolve(commands=('checkMesh',))
        result['runtime'] = info
        mesh_log = runtime.run(case, 'checkMesh', ('-allGeometry','-allTopology'), info=info).read_text()
        if 'Mesh OK.' not in mesh_log or 'Failed ' in mesh_log:
            raise SimulationError('Mesh check failed')
        result['mesh_validation'] = {'openfoam_all_checks_passed':True}
        run_phase(case, 'build', runtime, info)
        result['solver_binary_sha256'] = hashlib.sha256((case/'bin/flowfieldP003Foam').read_bytes()).hexdigest()
        run_phase(case, 'solve', runtime, info)
        result.update(extract_metrics(case,config), status='ok')
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        result['error'] = str(exc)
        raise SimulationError(f'P003 failed in {case}: {exc}', case_directory=case) from exc
    except KeyboardInterrupt:
        result['error'] = 'Interrupted by user'
        raise
    finally:
        result['wall_time_s'] = time.monotonic()-started
        write_json(case/'result.json',result)
    return result


def evaluate(geometry, config, output_root, runtime):
    case = Path(output_root).resolve()/('P003-'+uuid.uuid4().hex)
    generate_case(case,geometry,config)
    return run_case(case,runtime)
