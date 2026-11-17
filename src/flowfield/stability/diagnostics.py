"""Streaming log/history readers. Trends are indicators, never convergence proof."""
from collections import deque
import csv
import math
from pathlib import Path
import re
import statistics

NUMBER=r'[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?'
TIME=re.compile(r'^\s*(?:\[\d+\]\s*)?(?:Time|Iteration) = (\S+)\s*$')
RESIDUAL=re.compile(r'Solving for ([\w.]+), Initial residual = ([^,]+), Final residual = ([^,]+), No Iterations (\d+)')
FATAL=re.compile(r'FOAM FATAL|Floating point exception|signal SIGFPE|Segmentation fault',re.I)


def scan_log(path, tail_only=False):
    out=dict(exists=path.is_file(),last_step=None,ended=False,fatal=[],errors=[],residuals={},
             courant={},delta_t={},time_records=0,scope='recent_tail' if tail_only else 'complete_log')
    if not out['exists']: return out
    previous=None;last_end=False
    with path.open('rb') as stream:
        if tail_only:
            stream.seek(max(0,path.stat().st_size-131072))
            if stream.tell():stream.readline()
        for raw in stream:
            line=raw.decode('utf-8',errors='replace').strip()
            if FATAL.search(line) and 'trapping enabled' not in line.lower() and len(out['fatal'])<8:out['fatal'].append(line[:500])
            match=TIME.match(line)
            if match:
                try:step=float(match[1])
                except ValueError:step=float('nan')
                if not math.isfinite(step) or step<0:
                    if len(out['errors'])<20:out['errors'].append('Invalid solver time/iteration')
                elif previous is not None and step<=previous:
                    if 'Repeated/decreasing solver time: possible mixed or restarted log' not in out['errors']:
                        out['errors'].append('Repeated/decreasing solver time: possible mixed or restarted log')
                previous=step;out['last_step']=step if math.isfinite(step) else None
                out['time_records']+=1;last_end=False
            if re.fullmatch(r'End',line):last_end=True
            match=RESIDUAL.search(line)
            if match:
                field=match[1]
                try:a,b=float(match[2]),float(match[3])
                except ValueError:a=b=float('nan')
                if not all(math.isfinite(v) and v>=0 for v in (a,b)):
                    if len(out['errors'])<20:out['errors'].append(f'Nonfinite/negative linear residual: {field}')
                    continue
                item=out['residuals'].setdefault(field,dict(first_initial=a,last_initial=a,
                    last_final=b,maximum_initial=a,solves=0,recent=[]))
                item.update(last_initial=a,last_final=b,maximum_initial=max(item['maximum_initial'],a),solves=item['solves']+1)
                item['recent'].append([out['last_step'],a]);item['recent']=item['recent'][-100:]
            match=re.search(r'((?:Interface )?Courant Number) mean:\s*(\S+)\s+max:\s*(\S+)',line,re.I)
            if match:
                try:value=float(match[3])
                except ValueError:value=float('nan')
                if not math.isfinite(value) or value<0:
                    if len(out['errors'])<20:out['errors'].append('Nonfinite/negative Courant number')
                else:
                    key=match[1].lower();entry=out['courant'].setdefault(key,dict(maximum=value,last=value,samples=0))
                    entry.update(maximum=max(entry['maximum'],value),last=value,samples=entry['samples']+1)
            match=re.fullmatch(r'deltaT = (\S+)',line)
            if match:
                try:value=float(match[1])
                except ValueError:value=float('nan')
                if not math.isfinite(value) or value<=0:
                    if len(out['errors'])<20:out['errors'].append('Nonpositive/nonfinite time step')
                else:
                    entry=out['delta_t']
                    entry.update(last=value,minimum=min(entry.get('minimum',value),value),maximum=max(entry.get('maximum',value),value))
    out['ended']=last_end
    return out


def history(path, unfinished=False, keep=250):
    result=dict(path=str(path),samples=0,last={},recent=[],errors=[],ranges={},partial_last_row_ignored=False)
    if not path.is_file():return result
    rows=deque(maxlen=keep);previous=None
    # A writer can be halfway through its last CSV line. Only this final
    # unterminated line is ignored, and only for an unfinished run.
    with path.open(newline='',errors='replace') as stream:
        header=stream.readline();names=next(csv.reader([header]),[])
        if not names or len(names)!=len(set(names)):
            result['errors'].append('Missing/duplicate history columns');return result
        key='time_s' if 'time_s' in names else 'iteration'
        for line in stream:
            if not line.strip():continue
            if unfinished and not line.endswith('\n'):
                result['partial_last_row_ignored']=True;break
            try:
                cells=next(csv.reader([line]))
                if len(cells)!=len(names):raise ValueError('Wrong number of CSV columns')
                row=dict(zip(names,map(float,cells)))
                if key not in row or not all(math.isfinite(v) for v in row.values()):raise ValueError('Missing/nonfinite history value')
                step=row[key]
                if step<0 or (previous is not None and step<=previous):raise ValueError('Repeated/decreasing history steps')
                if key=='iteration' and step!=result['samples']+1:raise ValueError('Nonconsecutive iteration history')
                previous=step;result['samples']+=1;rows.append(row)
                for name,value in row.items():
                    bounds=result['ranges'].setdefault(name,dict(min=value,max=value))
                    bounds.update(min=min(bounds['min'],value),max=max(bounds['max'],value))
            except (ValueError,csv.Error) as exc:
                result['errors'].append(f'History row {result["samples"]+1}: {exc}');break
    result['recent']=list(rows)
    if rows:result['last']=rows[-1]
    return result


def trends(rows, thresholds, window=20):
    if not rows:return {}
    # Never compare residuals across different P006 continuation models.
    last=rows[-1];stage_keys=[k for k in ('momentum_fraction','thermal_active','applied_voltage_V') if k in last]
    start=len(rows)-1
    while start>0 and all(rows[start-1].get(k)==last[k] for k in stage_keys):start-=1
    rows=rows[start:];result={}
    for key,limit in thresholds.items():
        if key=='energy_balance' and last.get('thermal_active')==0:continue
        samples=[r[key] for r in rows if key in r]
        if len(samples)<2*window:continue
        a,b=samples[-2*window:-window],samples[-window:]
        ma,mb=statistics.median(a),statistics.median(b)
        ratio=mb/max(ma,limit*1e-6)
        state='below_limit' if max(b)<limit else 'growing' if ratio>2 else 'stalled' if ratio>=.95 else 'decreasing'
        result[key]=dict(indicator=state,previous_median=ma,recent_median=mb,median_ratio=ratio,
            recent_min=min(b),recent_max=max(b),limit=limit,window=window,
            note='Heuristic over consecutive samples in the same model stage; not a stability proof.')
    return result
