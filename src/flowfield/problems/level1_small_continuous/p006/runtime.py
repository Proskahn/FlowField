"""Per-case build and serial/MPI solve on native v2312 or the installed image."""
import os
import hashlib
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path
import shutil
import subprocess
import threading
import uuid
from ....runtime import SimulationError


def eigen_headers():
    try:
        package=distribution('cmeel-eigen')
    except PackageNotFoundError as exc:
        raise SimulationError('P006 requires Eigen headers in the active Python environment: '
                              'python -m pip install cmeel-eigen==3.4.1') from exc
    path=Path(package.locate_file('cmeel.prefix/include/eigen3')).resolve()
    if not (path/'Eigen/SparseLU').is_file():raise SimulationError('Installed Eigen headers are incomplete')
    digest=hashlib.sha256()
    for file in sorted((path/'Eigen').rglob('*')):
        if file.is_file():
            digest.update(file.relative_to(path).as_posix().encode()+b'\0'+file.read_bytes())
    return path,dict(version=package.version,headers_sha256=digest.hexdigest())


def _show_progress(log,stop):
    """Show every tenth iteration and the full fatal-error diagnostic block."""
    show=False;fatal=False
    with log.open() as source:
        while True:
            line=source.readline()
            if not line:
                if stop.is_set():break
                stop.wait(.2);continue
            # OpenFOAM puts the reason and source location on the lines after
            # this marker. MPI may also prefix it with a rank such as "[0]".
            if '--> FOAM FATAL' in line:fatal=True
            if fatal:
                print(f'[{log.parent.name}] {line.rstrip()}',flush=True)
            elif line.startswith('Iteration = '):
                try:iteration=int(line.split('=',1)[1]);show=iteration==1 or iteration%10==0
                except ValueError:show=False
                if show:print(f'[{log.parent.name}] {line.strip()}',flush=True)
            elif (show and line.startswith('P006:')) or line.startswith(
                    ('CONVERGED:', 'Momentum continuation:', 'Voltage continuation:',
                     'Enabling coupled energy solve')):
                print(f'[{log.parent.name}] {line.strip()}',flush=True)


def run_phase(case,phase,runtime,info,processes=1):
    if phase not in ('build','solve'):raise ValueError('Unknown phase')
    case=Path(case).resolve();name='flowfield-'+uuid.uuid4().hex
    (case/'bin').mkdir(exist_ok=True)
    log=case/('log.wmake' if phase=='build' else 'log.flowfieldP006Foam')
    env=dict(os.environ,FOAM_USER_APPBIN=str(case/'bin'))
    eigen=None
    if phase=='build':
        eigen,_=eigen_headers()
        env['FLOWFIELD_EIGEN_INCLUDE']=str(eigen)
    if phase=='build':
        script='export FOAM_USER_APPBIN="$1/bin"; exec wmake "$1/solver"'
    elif processes==1:
        script='exec "$1/bin/flowfieldP006Foam" -case "$1"'
    else:
        script=('decomposePar -case "$1" > "$1/log.decomposePar" 2>&1 || exit $?; '
            'mpirun -np "$2" "$1/bin/flowfieldP006Foam" -case "$1" -parallel; '
            'p006_solve_status=$?; '
            'if [[ "$p006_solve_status" == 0 || "$p006_solve_status" == 2 ]] '
            '&& [[ -f "$1/postProcessing/p006/status.json" ]]; then '
            'reconstructPar -case "$1" -latestTime > "$1/log.reconstructPar" 2>&1 || exit $?; fi; '
            'exit "$p006_solve_status"')
    if info['backend']=='native':
        for tool in (('wmake',) if phase=='build' else ('mpirun','decomposePar','reconstructPar') if processes>1 else ()):
            if not shutil.which(tool):raise SimulationError(f'P006 needs {tool} from the loaded v2312 environment')
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
            'p006_args=("$@"); set --; source /usr/lib/openfoam/openfoam2312/etc/bashrc || exit $?; '
            '[[ "$WM_PROJECT_VERSION" == v2312 || "$WM_PROJECT_VERSION" == 2312 ]] || exit 90; '
            'set -- "${p006_args[@]}"; '+script,'bash','/case',str(processes)]
    try:
        with log.open('x') as stream:
            stop=threading.Event();watcher=None
            if phase=='solve':
                watcher=threading.Thread(target=_show_progress,args=(log,stop),daemon=True);watcher.start()
            try:
                subprocess.run(argv,cwd=case,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=runtime.timeout_s)
            finally:
                stop.set()
                if watcher is not None:watcher.join(timeout=2)
    except (OSError,subprocess.SubprocessError,KeyboardInterrupt) as exc:
        if info['backend']=='docker':
            try:subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=30)
            except (OSError,subprocess.SubprocessError):pass
        if isinstance(exc,KeyboardInterrupt):raise
        raise SimulationError(f'P006 {phase} failed; inspect {log}',case_directory=case) from exc
    return log
