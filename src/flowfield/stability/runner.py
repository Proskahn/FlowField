"""Sequential baseline/study execution with immutable cases and resumable reports."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import threading

from ..problem import CFDProblem
from ..registry import create_problem,list_problems
from ..runtime import Runtime
from .audit import inspect_case
from .compare import compare
from .diagnostics import scan_log,history
from .profiles import PROFILES,experiments
from .report import atomic,overall,write_reports


def source_digest():
    root=Path(__file__).resolve().parents[1];h=hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file() and (p.suffix in ('.py','.C','.H','.json') or p.name in ('files','options')):
            h.update(p.relative_to(root).as_posix().encode()+b'\0'+p.read_bytes())
    return h.hexdigest()


def make_plan(names,configs=None,studies=(),designs=None,processes=None):
    configs=configs or {};designs=designs or {};entries=[]
    if len(names)!=len(set(names)):raise ValueError('Duplicate problem selection')
    unknown=(set(configs)|set(designs))-set(names)
    if unknown:raise ValueError('Configuration/design supplied for unselected problems: '+', '.join(sorted(unknown)))
    if any(s not in ('repeat','mesh','numerics','parallel') for s in studies):raise ValueError('Unknown study')
    if processes is not None and ('P006' not in names or type(processes) is not int or processes<1):
        raise ValueError('--processes requires P006 and a positive integer; other implementations are serial.')
    for name in names:
        if name not in list_problems():raise ValueError(f'Unknown problem: {name}')
        if name=='P004':
            if name in configs or name in designs:raise ValueError('P004 is a data/surrogate problem, not an executable CFD stability target.')
            entries.append(dict(problem=name,not_applicable='P004 uses external data or a surrogate. P004-hydraulic is the separate CFD problem.'));continue
        problem=create_problem(name,config=configs.get(name))
        if not isinstance(problem,CFDProblem):raise ValueError(f'{name} is not a CFD problem')
        if name not in PROFILES:raise ValueError(f'No stability profile registered for {name}')
        if processes is not None and name=='P006':
            from dataclasses import replace
            problem=create_problem(name,config=replace(problem.config,processes=processes))
        design=problem.validate_design(designs.get(name,problem.initial_design()))
        jobs,unavailable=experiments(name,problem.config,studies)
        entries.append(dict(problem=name,design=design,unsupported_studies=unavailable,
            jobs=[dict(experiment=label,study=study,config=cfg.to_dict(),config_sha256=cfg.fingerprint) for label,study,cfg in jobs]))
    return dict(schema_version=1,source_sha256=source_digest(),requested_studies=list(studies),problems=entries,
                case_count=sum(len(e.get('jobs',[])) for e in entries),execution='Sequential cases; P006 uses its configured MPI ranks. No optimizer.')


def preflight(problem):
    problem.preflight()
    info=problem.runtime.resolve(problem.case_definition.commands)
    if problem.name=='P001' and importlib.util.find_spec('gmsh') is None:
        raise RuntimeError('P001 requires gmsh: python -m pip install gmsh==4.13.1')
    if info['backend']=='native':
        required=[]
        if problem.name in ('P002','P003','P005','P006','P007','P008','P010'):required.append('wmake')
        if getattr(problem.config,'processes',1)>1:required.extend(('mpirun','decomposePar','reconstructPar'))
        missing=[n for n in required if not shutil.which(n)]
        if missing:raise RuntimeError('Missing native runtime tools: '+', '.join(missing))
    return info


def progress(case,name,stop,interval):
    previous=None
    while not stop.wait(interval):
        try:
            log,csv_path=PROFILES[name];r=scan_log(case/log,tail_only=True)
            h=history(case/csv_path,unfinished=True) if csv_path else {};last=h.get('last',{})
            step=last.get('iteration',last.get('time_s',r['last_step']))
            value=(step,last.get('momentum_fraction'),last.get('thermal_active'))
            if step is not None and value!=previous:
                extra=f'; momentum={value[1]}, heat={value[2]}' if name=='P006' else ''
                print(f'[check {name}/{case.name}] step={step:g}{extra}',flush=True);previous=value
        except (OSError,ValueError,KeyError):pass  # Writer may be between file creation and first complete record.


def run_suite(plan,output,*,resume=False,**kwargs):
    """Lock only this suite; a second runner must never share its reports/cases."""
    import fcntl  # OpenFOAM execution targets here are Linux and macOS.
    output=Path(output).resolve()
    if not resume:output.mkdir(parents=True,exist_ok=False)
    elif not output.is_dir():raise ValueError('Resume directory does not exist')
    with (output/'.checker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('Another checker is already running this suite') from exc
        try:return _run_suite(plan,output,resume=resume,**kwargs)
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def _run_suite(plan,output,*,runtime=None,resume=False,progress_interval=30,rtol=.02,atol=1e-10,max_field_bytes=128*1024*1024):
    runtime=runtime or Runtime(timeout_s=14400);output=Path(output).resolve()
    request=dict(plan=plan,backend=runtime.backend,image=runtime.image,timeout_s=runtime.timeout_s,
                 comparison_rtol=rtol,comparison_atol=atol,max_field_bytes=max_field_bytes)
    # Configurations/designs may contain tuples; compare their persisted JSON
    # representation so an identical resume is not rejected as tuple != list.
    request=json.loads(json.dumps(request,allow_nan=False))
    if resume:
        if not (output/'plan.json').is_file() or json.loads((output/'plan.json').read_text())!=request:
            raise ValueError('Resume requires the identical plan, source, configurations and runtime settings.')
    else:
        atomic(output/'plan.json',json.dumps(request,indent=2)+'\n')
    report=dict(schema_version=1,status='running',created_utc=datetime.now(timezone.utc).isoformat(),cases=[],comparisons=[],
        coverage=dict(requested_studies=plan['requested_studies'],mesh_independence_established=False,
            scope='Only selected baseline/design configurations and requested perturbations are checked.',optimization_run=False))
    write_reports(output,report)
    record=None;case=None
    try:
        for entry in plan['problems']:
            name=entry['problem']
            if 'not_applicable' in entry:
                report['cases'].append(dict(problem=name,experiment='baseline',audit=dict(status='not_applicable',qualified=False,reason=entry['not_applicable'])))
                write_reports(output,report);continue
            done={}
            for job in entry['jobs']:
                label=job['experiment'];case=output/name/label
                record=dict(problem=name,experiment=label,study=job['study'],audit=dict(status='running',qualified=False,case_directory=str(case)))
                report['cases'].append(record);write_reports(output,report)
                print(f'[check {name}/{label}] starting',flush=True)
                problem=create_problem(name,config=job['config'],runtime=runtime);error=None
                try:
                    if not (case/'case.json').is_file():
                        preflight(problem);problem.write_case(entry['design'],case)
                    # Never rerun a finished/started case, including a failed one.
                    if not (case/'.execution-started').exists() and not (case/'result.json').exists():
                        preflight(problem)
                        stop=threading.Event();watcher=threading.Thread(target=progress,args=(case,name,stop,progress_interval),daemon=True);watcher.start()
                        try:problem.run_case(case)
                        finally:stop.set();watcher.join(timeout=5)
                    elif not resume:raise ValueError('Unexpected execution data in a new suite case')
                except (OSError,ValueError,RuntimeError,TypeError,ImportError,subprocess.SubprocessError) as exc:error=f'{type(exc).__name__}: {exc}'
                audit=inspect_case(case,max_field_bytes=max_field_bytes) if (case/'case.json').is_file() else dict(status='blocked',qualified=False,case_directory=str(case))
                if error:
                    audit['execution_error']=error
                    if not (case/'.execution-started').exists():audit.update(status='blocked',error=error)
                record['audit']=audit;done[label]=audit
                if job['study']!='baseline':
                    reference=done['baseline']
                    if job['study']=='mesh':
                        earlier=[j['experiment'] for j in entry['jobs'][:entry['jobs'].index(job)] if j['study']=='mesh']
                        if earlier:reference=done[earlier[-1]]
                    report['comparisons'].append(compare(name,job['study'],reference,audit,rtol,atol))
                print(f'[check {name}/{label}] {audit["status"]}',flush=True)
                write_reports(output,report)
            for item in entry['unsupported_studies']:
                report['comparisons'].append(dict(problem=name,status='not_applicable',**item))
    except KeyboardInterrupt:
        if record is not None:
            record['audit']=inspect_case(case,max_field_bytes=max_field_bytes) if (case/'case.json').is_file() else dict(status='incomplete',qualified=False,case_directory=str(case))
        report.update(status='interrupted');write_reports(output,report);raise
    report['status']=overall(report['cases'],report['comparisons']);write_reports(output,report)
    return report
