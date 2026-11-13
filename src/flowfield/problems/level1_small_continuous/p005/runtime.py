"""Compile the diagnostic library into the individual case; never modify OpenFOAM."""
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from ....runtime import SimulationError


def build_monitor(case,runtime,info):
    case=Path(case).resolve();(case/'lib').mkdir(exist_ok=False)
    env=dict(os.environ,FOAM_USER_LIBBIN=str(case/'lib'))
    name='flowfield-'+uuid.uuid4().hex
    if info['backend']=='native':
        wmake=shutil.which('wmake')
        if not wmake:raise SimulationError('P005 requires v2312 wmake for its diagnostic library')
        argv=[wmake,'libso',str(case/'monitor')]
    else:
        if ',' in str(case):raise SimulationError('Docker bind paths cannot contain commas')
        argv=['docker','run','--rm','--pull','never','--init','--name',name,
            '--user',f'{os.getuid()}:{os.getgid()}','--env','HOME=/tmp','--env','USER=foam',
            '--workdir','/case','--mount',f'type=bind,source={case},target=/case',
            '--entrypoint','/bin/bash',info['image_id'],'-c',
            'source /usr/lib/openfoam/openfoam2312/etc/bashrc || exit $?; '
            '[[ "$WM_PROJECT_VERSION" == v2312 || "$WM_PROJECT_VERSION" == 2312 ]] || exit 90; '
            'export FOAM_USER_LIBBIN=/case/lib; exec wmake libso /case/monitor']
    log=case/'log.wmake'
    try:
        with log.open('x') as stream:
            subprocess.run(argv,cwd=case,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=runtime.timeout_s)
    except (OSError,subprocess.SubprocessError,KeyboardInterrupt) as exc:
        if info['backend']=='docker':
            try:subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=30)
            except (OSError,subprocess.SubprocessError):pass
        if isinstance(exc,KeyboardInterrupt):raise
        raise SimulationError(f'P005 monitor compilation failed; inspect {log}',case_directory=case) from exc
