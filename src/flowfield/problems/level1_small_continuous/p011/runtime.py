"""Per-case build and serial solve on native v2312 or the installed image."""
import os
import hashlib
import json
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path
import shutil
import re
import subprocess
import threading
import time
from ..p006.runtime import _run_command
import uuid
from ....runtime import SimulationError


def iteration_limit_reason(case):
    """Distinguish a normal unqualified budget exit from an MPI/build crash."""
    try:
        config=json.loads((case/'case.json').read_text())['config']
        status=json.loads((case/'postProcessing/p011/status.json').read_text())
        log=(case/'log.flowfieldP011Foam').read_text()
        if (status.get('converged') is False and status.get('iteration')==config['iterations']
                and re.search(r'^End\s*$',log,re.M) and 'FOAM FATAL' not in log):
            return (f'P011 reached its {config["iterations"]}-iteration limit without satisfying all convergence checks. '
                    'The saved state is unqualified; inspect the histories before generating another case. ')
    except (OSError,ValueError,TypeError,KeyError):pass
    return None


def eigen_headers():
    try:
        package=distribution('cmeel-eigen')
    except PackageNotFoundError as exc:
        raise SimulationError('P011 requires Eigen headers in the active Python environment: '
                              'python -m pip install cmeel-eigen==3.4.1') from exc
    path=Path(package.locate_file('cmeel.prefix/include/eigen3')).resolve()
    if not (path/'Eigen/SparseLU').is_file():raise SimulationError('Installed Eigen headers are incomplete')
    digest=hashlib.sha256()
    for file in sorted((path/'Eigen').rglob('*')):
        if file.is_file():
            digest.update(file.relative_to(path).as_posix().encode()+b'\0'+file.read_bytes())
    return path,dict(version=package.version,headers_sha256=digest.hexdigest())


def _show_progress(log,stop):
    last=time.monotonic();fatal=False
    with log.open() as source:
        while True:
            line=source.readline()
            if not line:
                if stop.is_set():break
                if time.monotonic()-last>=30:
                    print(f'[{log.parent.name}] Still running; progress log: {log.name}',flush=True)
                    last=time.monotonic()
                stop.wait(.2);continue
            if '--> FOAM FATAL' in line or 'error:' in line:fatal=True
            if fatal or line.startswith(('P011:', 'CONVERGED:')):
                print(f'[{log.parent.name}] {line.rstrip()}',flush=True);last=time.monotonic()


def run_phase(case,phase,runtime,info,processes=1):
    if phase not in ('build','solve'):raise ValueError('Unknown phase')
    case=Path(case).resolve();name='flowfield-'+uuid.uuid4().hex
    (case/'bin').mkdir(exist_ok=True)
    log=case/('log.wmake' if phase=='build' else 'log.flowfieldP011Foam')
    env=dict(os.environ,FOAM_USER_APPBIN=str(case/'bin'))
    eigen=None
    if phase=='build':
        eigen,_=eigen_headers()
        env['FLOWFIELD_EIGEN_INCLUDE']=str(eigen)
    if phase=='build':
        script='export FOAM_USER_APPBIN="$1/bin"; exec wmake "$1/solver"'
    elif processes==1:
        script='exec "$1/bin/flowfieldP011Foam" -case "$1"'
    else:raise ValueError('P011 currently supports one process per case')
    if info['backend']=='native':
        for tool in (('wmake',) if phase=='build' else ('mpirun','decomposePar','reconstructPar') if processes>1 else ()):
            if not shutil.which(tool):raise SimulationError(f'P011 needs {tool} from the loaded v2312 environment')
        argv=['bash','-c',script,'bash',str(case),str(processes)]
    else:
        if ',' in str(case):raise SimulationError('Docker paths cannot contain commas')
        extra=[]
        if eigen is not None:
            if ',' in str(eigen):raise SimulationError('Docker header paths cannot contain commas')
            extra=['--env','FLOWFIELD_EIGEN_INCLUDE=/opt/flowfield-eigen','--mount',
                   f'type=bind,source={eigen},target=/opt/flowfield-eigen,readonly']
        argv=['docker','run','--rm','--pull','never','--init','--name',name,
            '--user',f'{os.getuid()}:{os.getgid()}','--env','HOME=/tmp','--env','USER=foam',
            '--workdir','/case','--mount',f'type=bind,source={case},target=/case',*extra,
            '--entrypoint','/bin/bash',info['image_id'],'-c',
            'p011_args=("$@"); set --; source /usr/lib/openfoam/openfoam2312/etc/bashrc || exit $?; '
            '[[ "$WM_PROJECT_VERSION" == v2312 || "$WM_PROJECT_VERSION" == 2312 ]] || exit 90; '
            'set -- "${p011_args[@]}"; '+script,'bash','/case',str(processes)]
    try:
        with log.open('x') as stream:
            stop=threading.Event();watcher=None
            if phase in ('build','solve'):
                watcher=threading.Thread(target=_show_progress,args=(log,stop),daemon=True);watcher.start()
            try:
                print(f'[{case.name}] P011 {phase} started; timeout={runtime.timeout_s:g} s; log={log.name}',flush=True)
                _run_command(argv,cwd=case,env=env,stdout=stream,timeout=runtime.timeout_s)
            finally:
                stop.set()
                if watcher is not None:watcher.join(timeout=2)
    except (OSError,subprocess.SubprocessError,KeyboardInterrupt) as exc:
        if info['backend']=='docker':
            try:subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=30)
            except (OSError,subprocess.SubprocessError):pass
        if isinstance(exc,KeyboardInterrupt):raise
        reason=iteration_limit_reason(case) if phase=='solve' and isinstance(exc,subprocess.CalledProcessError) and exc.returncode==2 else None
        raise SimulationError(reason or f'P011 {phase} failed; inspect {log}',case_directory=case) from exc
    return log
