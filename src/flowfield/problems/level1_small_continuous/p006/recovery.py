"""Recover a completed native MPI state without modifying its original case."""
import json
import os
from pathlib import Path
import re
import shutil

from ....evaluation import write_json
from ....runtime import SimulationError
from ....stability.report import atomic
from .config import ManifoldConfig
from .case import generate_case
from .runtime import _run_command


def recover_completed(source, output, *, timeout=600):
    from . import P006Problem
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(f'Recovery output must be new: {output}')
    if source in output.parents:raise ValueError('Recovery output cannot be inside the source case')
    manifest=json.loads((source/'case.json').read_text())
    status=json.loads((source/'postProcessing/p006/status.json').read_text())
    result=json.loads((source/'result.json').read_text())
    log=(source/'log.flowfieldP006Foam').read_text()
    if manifest.get('problem')!='P006':raise ValueError('Recovery requires a P006 case')
    config=ManifoldConfig(**manifest['config'])
    if config.fingerprint!=manifest['config_sha256']:raise ValueError('Source configuration fingerprint differs')
    endings=list(re.finditer(r'^End\s*$',log,re.M))
    starts=list(re.finditer(r'^(?:Iteration = |Create time)',log,re.M))
    if (not endings or (starts and starts[-1].start()>endings[-1].start()) or 'FOAM FATAL' in log
            or result.get('status') not in ('ok','failed')):
        raise ValueError('Source must have finished; running, interrupted or fatal cases cannot be recovered')
    step=status.get('iteration')
    if (type(step) is not int or step<1 or status.get('cells')!=manifest['cells']
            or status.get('processes')!=config.processes):
        raise ValueError('Source completion metadata is inconsistent')
    problem=P006Problem(config=config)
    geometry=problem.decode(manifest['design_geometry']['normalized_design'])
    reconstruct=None
    if config.processes>1:
        if os.environ.get('WM_PROJECT_VERSION') not in ('2312','v2312'):
            raise SimulationError('Load native OpenCFD OpenFOAM v2312 before recovering MPI fields')
        reconstruct=shutil.which('reconstructPar')
        if not reconstruct:raise SimulationError('Missing native OpenFOAM command: reconstructPar')
    output.mkdir(parents=True,exist_ok=False)
    copied=output/'recovered';continued=output/'continuation'
    record=dict(status='preparing',source_case=str(source),source_iteration=step,
                recovered_case=str(copied),continuation_case=str(continued),
                original_execution_status=result['status'],qualification_inherited=False)
    write_json(output/'recovery.json',record)
    try:
        print(f'Copying completed P006 case to {copied}',flush=True)
        shutil.copytree(source,copied)
        if reconstruct:
            reconstruction_log=copied/'log.reconstructPar.recovery'
            record['reconstruction_log']=str(reconstruction_log)
            print(f'Reconstructing saved iteration {step}; log: {reconstruction_log}',flush=True)
            with reconstruction_log.open('x') as stream:
                _run_command([reconstruct,'-case',str(copied),'-time',str(step)],
                             cwd=copied,env=os.environ.copy(),stdout=stream,timeout=timeout)
        # The existing loader verifies input hashes, complete field addressing,
        # physical bounds and coupling evidence before creating a destination.
        generate_case(continued,geometry,config,initial_state=copied)
        record['status']='prepared'
        print(f'Prepared continuation: {continued}. It must solve and qualify independently.',flush=True)
    except BaseException as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        atomic(output/'recovery.json',json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record
