"""Read-only inspection: no solver launch and no modification of case files."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from ..evaluation import CaseEvaluator
from ..registry import create_problem
from .diagnostics import scan_log,history,trends
from .fields import final_fields
from .profiles import PROFILES,limits


def _json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def mesh_check(case,name,manifest):
    p=case/'log.checkMesh'
    if not p.is_file():return dict(status='pending',message='Mesh check has not been recorded.')
    text=p.read_text(errors='replace')
    if 'Mesh OK.' not in text and not re.search(r'Failed\s+\d+\s+mesh checks|FOAM FATAL|^End\s*$',text,re.M):
        return dict(status='pending',message='Mesh check has not completed.')
    try:
        if name=='P002':
            from ..problems.level2_medium_parametric.p002.mesh_checks import assess_mesh
            data=assess_mesh(case,text,manifest)
        elif name=='P005':
            from ..problems.level1_small_continuous.p005.metrics import verify_mesh_log
            data=verify_mesh_log(text,manifest['mesh'])
        else:
            if 'Mesh OK.' not in text or re.search(r'Failed\s+\d+\s+mesh checks',text):raise ValueError('Mesh has not passed its declared criteria.')
            data={}
        return dict(status='passed',criteria='Problem-declared mesh criteria; not a claim of mesh independence.',details=data)
    except (ValueError,RuntimeError,OSError,KeyError) as exc:return dict(status='failed',message=str(exc))


def inspect_case(case, *, max_field_bytes=128*1024*1024):
    """Isolate malformed/unreadable case evidence from the remaining suite."""
    try:return _inspect_case(case,max_field_bytes=max_field_bytes)
    except (OSError,ValueError,TypeError,KeyError,AttributeError,ArithmeticError,EOFError) as exc:
        return dict(case_directory=str(Path(case).resolve()),status='inconclusive',qualified=False,
                    error=f'Cannot finish inspection: {type(exc).__name__}: {exc}',
                    checks=[dict(name='inspection_integrity',status='inconclusive',message=str(exc))],warnings=[])


def _inspect_case(case, *, max_field_bytes=128*1024*1024):
    case=Path(case).resolve()
    out=dict(case_directory=str(case),status='incomplete',qualified=False,
        inspected_utc=datetime.now(timezone.utc).isoformat(),checks=[],warnings=[],
        scope='Numerical evidence for this case only; no proof of mathematical stability or paper/mesh validation.')
    checks=out['checks']
    def check(key,status,message,**kw):checks.append(dict(name=key,status=status,message=message,**kw))
    try:
        manifest=_json(case/'case.json')
        if not isinstance(manifest,dict):raise ValueError('Missing case.json; select a generated benchmark case or its parent directory.')
        name=manifest['problem'];out['problem']=name
        if name not in PROFILES:
            out.update(status='not_applicable',reason='No registered CFD stability profile for this case.');return out
        problem=create_problem(name,config=manifest['config']);config=problem.config
        out['config']=config.to_dict();out['model_revision']=manifest.get('model_revision')
    except (OSError,ValueError,TypeError,KeyError) as exc:
        out.update(status='incompatible',error=str(exc));return out
    try:
        CaseEvaluator(problem.case_definition)._load_case(case)
        check('inputs_and_sources','passed','Input hashes, configuration, geometry and current solver revision match.')
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        check('inputs_and_sources','failed',str(exc));out['status']='incompatible'
    result=None
    try:
        result=_json(case/'result.json')
        if result is not None and not isinstance(result,dict):
            result=None;raise ValueError('Expected a JSON object')
    except (OSError,ValueError) as exc:check('execution_record','failed',f'Malformed result record: {exc}')
    log_name,history_name=PROFILES[name];log=scan_log(case/log_name);out['log']=log
    terminal=bool(result is not None or log['ended'] or log['fatal'])
    interrupted=bool(result and result.get('status')=='failed' and result.get('error')=='Interrupted by user')
    final_assessment=terminal and not interrupted
    out['execution_state']='finished_record' if result is not None else 'ended_log' if log['ended'] else 'fatal_log' if log['fatal'] else 'unfinished_or_running'
    if log['exists']:
        age=max(0,time.time()-(case/log_name).stat().st_mtime);out['log_age_seconds']=age
        if not terminal and age>300:out['warnings'].append('No log update for over five minutes; process liveness cannot be determined from files.')
    check('solver_log','failed' if log['fatal'] or log['errors'] else 'passed' if log['ended'] else 'pending',
          '; '.join(log['fatal']+log['errors']) or ('Normal End marker.' if log['ended'] else 'No normal completion yet.'))
    mesh=mesh_check(case,name,manifest);out['mesh']=mesh
    check('mesh',mesh['status'],mesh.get('message','Declared mesh-quality checks passed.'))
    thresholds=limits(name,config);out['limits']=thresholds
    if history_name:
        h=history(case/history_name,unfinished=not final_assessment);out['history']=h
        if h['errors']:check('history_integrity','failed','; '.join(h['errors']))
        else:check('history_integrity','passed' if h['samples'] else 'pending',f'{h["samples"]} finite ordered diagnostic samples.')
        rows=h['recent'];last=h['last'];out['last_step']=last.get('iteration',last.get('time_s',log['last_step']))
        active_thresholds=dict(thresholds)
        if name=='P006' and last:
            completion=_json(case/'postProcessing/p006/status.json')
            if (log['ended'] and isinstance(completion,dict) and completion.get('converged') is False
                    and completion.get('iteration')==config.iterations and last.get('iteration')==config.iterations):
                out['stop_reason']='iteration_limit'
                check('iteration_budget','failed',
                      'Iteration budget exhausted before full acceptance; a completed saved state may initialize a new continuation case.',
                      value=completion['iteration'],limit=config.iterations)
            beta=last.get('momentum_fraction');thermal=last.get('thermal_active')
            stage='momentum_initialization' if beta is not None and beta<1 else 'flow_initialization' if thermal==0 else 'coupled' if beta==1 and thermal==1 else 'unknown'
            out['stage']=stage
            active=beta==1 and thermal==1 and abs(last.get('applied_voltage_V',0)-config.cell_voltage)<1e-12
            check('full_physical_model','passed' if active else 'failed' if final_assessment else 'pending',
                f'Momentum fraction={beta}; heat active={thermal}; voltage={last.get("applied_voltage_V")}.',
                required=dict(momentum_fraction=1,thermal_active=1,voltage=config.cell_voltage))
            if not active:
                out['warnings'].append('Initialization temperatures/objectives are not accepted results; tiny linear residuals do not prove full-model convergence.')
                active_thresholds.update(field_change=.001)
                if beta==1:active_thresholds.update(local_liquid_balance=.01,local_gas_balance=.01)
        for key,limit in thresholds.items():
            values=[r[key] for r in rows[-5:] if key in r]
            if name=='P005':
                bounds=h['ranges'].get(key,{})
                ok=bool(bounds) and 0<=bounds['min']<=bounds['max']<=limit
            else:ok=len(values)==5 and all(0<=v<limit for v in values)
            check(key,'passed' if ok else 'failed' if final_assessment else 'pending',
                  'All transient samples against model limit.' if name=='P005' else 'Last five available samples against final model limit.',value=values[-1] if values else None,
                  recent_max=max(values) if values else None,limit=limit,consecutive_passes=sum(0<=v<limit for v in values))
        if name=='P005' and h['samples']:
            ranges=h['ranges'];lo=ranges.get('min_alpha_water',{}).get('min');hi=ranges.get('max_alpha_water',{}).get('max')
            ok=lo is not None and hi is not None and lo>=-config.alpha_tolerance and hi<=1+config.alpha_tolerance
            check('transient_phase_bounds','passed' if ok else 'failed','All diagnostic samples must keep phase fraction within its declared bounds.',minimum=lo,maximum=hi)
            for key,limit in thresholds.items():
                if key in ranges and (ranges[key]['min']<0 or ranges[key]['max']>limit):
                    check('transient_'+key,'failed','Conservation limit exceeded during the transient.',value=ranges[key]['max'],limit=limit)
        if name in ('P002','P006') and last and 'bounded_cells' in last:
            check('recent_bounded_cells','passed' if last['bounded_cells']==0 else 'failed' if final_assessment else 'pending',
                  'Final state must not depend on clipping fields.',value=last['bounded_cells'],limit=0)
        out['trends']=trends(rows,active_thresholds)
        for key,v in out['trends'].items():
            if v['indicator'] in ('stalled','growing'):out['warnings'].append(f'{key}: possible {v["indicator"]} behavior in the current stage (trend heuristic).')
    else:out['last_step']=log['last_step'];out['trends']={}
    if name=='P005':
        for key,entry in log['courant'].items():
            if entry['maximum']>config.max_courant*(1+1e-6):
                out['warnings'].append(f'{key} reached {entry["maximum"]:g}, above configured target {config.max_courant:g}; assess with the halved-step study. This alone is not proof of instability.')
        if not log['courant']:out['warnings'].append('No Courant-number evidence found in solver log.')
    if result is not None:
        check('execution_record','passed' if result.get('status')=='ok' else 'pending' if interrupted else 'failed',
              f'Execution status: {result.get("status")}. '+str(result.get('error','')))
        if result.get('config_sha256')!=manifest.get('config_sha256') or result.get('input_sha256')!=manifest.get('input_sha256'):
            check('result_provenance','failed','Result record belongs to different inputs.')
    if terminal and out['status']!='incompatible' and log['ended']:
        try:
            # These two legacy extractors normally write an audit JSON. Disable
            # that side effect for read-only inspection of existing/live cases.
            if name=='P002':
                from ..problems.level2_medium_parametric.p002.metrics import extract_metrics
                extracted=extract_metrics(case,config,write_report=False)
            elif name=='P003':
                from ..problems.level4_high_dimensional_topology.p003.metrics import extract_metrics
                extracted=extract_metrics(case,config,write_report=False)
            else:extracted=problem.case_definition.extract(case,config)
            out['extracted']=extracted
            check('model_acceptance','passed','Problem-owned convergence, conservation and output consistency checks passed; objective recomputed.')
            metrics=extracted['metrics']
            steady_limits={}
            if name in ('P000','P001'):
                steady_limits=dict.fromkeys(('mass_balance_relative_error','flow_rate_relative_error'),config.mass_balance_tolerance)
                if name=='P001':steady_limits['uniformity_final_window_range']=config.stability_tolerance
            elif name=='P004-hydraulic':
                steady_limits=dict.fromkeys(('mass_balance_relative_error','imposed_flow_relative_error','sealed_wall_relative_flux'),config.balance_tolerance)
                steady_limits['pressure_last20_relative_change']=config.steady_tolerance
            for key,limit in steady_limits.items():
                check(key,'passed' if 0<=metrics[key]<=limit else 'failed','Model-owned final conservation or stationarity measure.',value=metrics[key],limit=limit)
            if name=='P005' and extracted['metrics']['right_censored']:
                out['warnings'].append('VOF observation horizon completed without detachment. This is a censored event, not a measured detachment time or a steady solution.')
            if name=='P000' and not extracted['metrics']['reference_check_passed']:
                out['warnings'].append('Pressure differs from the analytical reference beyond its configured tolerance; numerical convergence is distinct from accuracy.')
        except (OSError,ValueError,RuntimeError,KeyError,TypeError,IndexError,AttributeError) as exc:
            check('model_acceptance','failed',str(exc))
        mesh_count=re.search(r'^\s*cells:\s*(\d+)',(case/'log.checkMesh').read_text(errors='replace'),re.M) if (case/'log.checkMesh').is_file() else None
        f=final_fields(case,name,config,log['last_step'],int(mesh_count[1]) if mesh_count else None,max_field_bytes)
        out['saved_fields']=f;check('saved_fields',f['status'],f.get('reason','; '.join(f.get('errors',[])+f.get('unchecked',[])) or 'Saved field values and physical bounds checked.'))
    else:check('model_acceptance','pending','Full model extraction requires a completed, compatible run.')
    failed=any(c['status']=='failed' for c in checks)
    if out['status']=='incompatible':pass
    elif failed:out['status']='failed'
    elif interrupted:out['status']='interrupted'
    elif terminal and any(c['status'] in ('pending','inconclusive','incomplete') for c in checks):out['status']='inconclusive'
    elif terminal:
        out['status']='passed';out['qualified']=True
    else:out['status']='initializing' if out.get('stage') in ('momentum_initialization','flow_initialization') else 'incomplete'
    return out


def inspect_paths(paths, **kwargs):
    cases=set()
    for value in paths:
        p=Path(value).resolve()
        if (p/'case.json').is_file():cases.add(p)
        elif p.is_dir():
            found={f.parent for f in p.rglob('case.json')}
            if not found:raise ValueError(f'No case.json found under {p}')
            cases.update(found)
        else:raise ValueError(f'Case path is not a directory: {p}')
    if not cases:raise ValueError('No case.json found in the supplied case directories or roots.')
    return [inspect_case(p,**kwargs) for p in sorted(cases)]
