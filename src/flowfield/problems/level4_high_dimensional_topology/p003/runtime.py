"""Bounded build and execution of the bundled P003 solver, in a case-only mount."""

import os
from pathlib import Path
import shutil
import subprocess
import uuid

from ....runtime import SimulationError


def run_phase(case, phase, runtime, info):
    if phase not in ('build', 'solve'):
        raise ValueError('Unsupported P003 phase')
    case = Path(case).resolve()
    name = 'flowfield-' + uuid.uuid4().hex
    log = case/('log.wmake' if phase == 'build' else 'log.flowfieldP003Foam')
    env = dict(os.environ, FOAM_USER_APPBIN=str(case/'bin'))
    (case/'bin').mkdir(exist_ok=True)
    if info['backend'] == 'native':
        if phase == 'build':
            wmake = shutil.which('wmake')
            if not wmake:
                raise SimulationError('Native v2312 wmake is required to build P003')
            argv = [wmake, str(case/'solver')]
        else:
            argv = [str(case/'bin/flowfieldP003Foam'), '-case', str(case)]
    else:
        if ',' in str(case):
            raise SimulationError('Docker bind paths cannot contain commas')
        operation = ('export FOAM_USER_APPBIN=/case/bin; exec wmake /case/solver'
                     if phase == 'build' else 'exec /case/bin/flowfieldP003Foam -case /case')
        argv = ['docker','run','--rm','--pull','never','--init','--name',name,
                '--user',f'{os.getuid()}:{os.getgid()}','--env','HOME=/tmp','--env','USER=foam',
                '--workdir','/case','--mount',f'type=bind,source={case},target=/case',
                '--entrypoint','/bin/bash',info['image_id'],'-c',
                'source /usr/lib/openfoam/openfoam2312/etc/bashrc || exit $?; '
                '[[ "$WM_PROJECT_VERSION" == v2312 || "$WM_PROJECT_VERSION" == 2312 ]] || exit 90; '+operation]
    try:
        with log.open('x') as stream:
            subprocess.run(argv, cwd=case, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=runtime.timeout_s)
    except (OSError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        if info['backend'] == 'docker':
            try:
                subprocess.run(['docker','rm','-f',name], capture_output=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise SimulationError(f'P003 {phase} failed; inspect {log}', case_directory=case) from exc
    return log
