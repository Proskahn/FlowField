"""Bounded build and execution of the bundled P007 solver, in a case-only mount."""

import os
from pathlib import Path
import shutil
import subprocess
import uuid
import threading


def _progress(log, stop, label):
    with log.open() as stream:
        while not stop.is_set():
            line=stream.readline()
            if not line:
                stop.wait(.2);continue
            if line.startswith(('Iteration =','P007:','CONVERGED:','ITERATION LIMIT:')):
                print(f'[{label}] {line.rstrip()}',flush=True)

from ....runtime import SimulationError


def run_phase(case, phase, runtime, info):
    if phase not in ('build', 'solve'):
        raise ValueError('Unsupported P007 phase')
    case = Path(case).resolve()
    name = 'flowfield-' + uuid.uuid4().hex
    log = case/('log.wmake' if phase == 'build' else 'log.flowfieldP007Foam')
    env = dict(os.environ, FOAM_USER_APPBIN=str(case/'bin'))
    (case/'bin').mkdir(exist_ok=True)
    if info['backend'] == 'native':
        if phase == 'build':
            wmake = shutil.which('wmake')
            if not wmake:
                raise SimulationError('Native v2312 wmake is required to build P007')
            argv = [wmake, str(case/'solver')]
        else:
            argv = [str(case/'bin/flowfieldP007Foam'), '-case', str(case)]
    else:
        if ',' in str(case):
            raise SimulationError('Docker bind paths cannot contain commas')
        operation = ('export FOAM_USER_APPBIN=/case/bin; exec wmake /case/solver'
                     if phase == 'build' else 'exec /case/bin/flowfieldP007Foam -case /case')
        argv = ['docker','run','--rm','--pull','never','--init','--name',name,
                '--user',f'{os.getuid()}:{os.getgid()}','--env','HOME=/tmp','--env','USER=foam',
                '--workdir','/case','--mount',f'type=bind,source={case},target=/case',
                '--entrypoint','/bin/bash',info['image_id'],'-c',
                'source /usr/lib/openfoam/openfoam2312/etc/bashrc || exit $?; '
                '[[ "$WM_PROJECT_VERSION" == v2312 || "$WM_PROJECT_VERSION" == 2312 ]] || exit 90; '+operation]
    try:
        with log.open('x') as stream:
            stop=threading.Event()
            watcher=threading.Thread(target=_progress,args=(log,stop,case.name),daemon=True) if phase=='solve' else None
            if watcher:watcher.start()
            try:
                subprocess.run(argv, cwd=case, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=runtime.timeout_s)
            finally:
                stop.set()
                if watcher:watcher.join(timeout=2)
    except (OSError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        if info['backend'] == 'docker':
            try:
                subprocess.run(['docker','rm','-f',name], capture_output=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise SimulationError(f'P007 {phase} failed; inspect {log}', case_directory=case) from exc
    return log
