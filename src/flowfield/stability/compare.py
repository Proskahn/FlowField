"""Numerical sensitivity of accepted raw physical outputs, not optimizer penalties."""
import math
from .profiles import observables


def compare(name,study,reference,repeat,rtol=.02,atol=1e-10):
    out=dict(problem=name,study=study,reference=reference.get('case_directory'),repeat=repeat.get('case_directory'),
             status='inconclusive',relative_tolerance=rtol,absolute_tolerance=atol)
    if not reference.get('qualified') or not repeat.get('qualified'):
        out['reason']='Both simulations must independently qualify; failed/penalized/incomplete cases cannot be compared.';return out
    a,b=reference['extracted'],repeat['extracted']
    if name=='P005':
        ma,mb=a['metrics'],b['metrics']
        if ma['right_censored'] or mb['right_censored']:
            out['reason']='At least one detachment time is right-censored. Equal penalty scores or horizons do not establish event-time convergence.';return out
        out['event_brackets']=[ma['detachment_bracket_s'],mb['detachment_bracket_s']]
    try:x,y=observables(name,a),observables(name,b)
    except (KeyError,TypeError):
        out['reason']='Missing comparable physical observables.';return out
    if not x or x.keys()!=y.keys():out['reason']='Missing comparable physical observables.';return out
    checks={}
    for key,value in x.items():
        other=y[key]
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (value,other)):
            out['reason']='Nonfinite comparison observable.';return out
        tolerance=atol+rtol*max(abs(value),abs(other))
        checks[key]=dict(reference=value,repeat=other,absolute_difference=abs(other-value),
            relative_difference=abs(other-value)/max(abs(value),abs(other),1e-30),allowed_difference=tolerance,
            passed=abs(other-value)<=tolerance)
    out.update(status='passed' if all(v['passed'] for v in checks.values()) else 'failed',observables=checks,
               interpretation='Sensitivity screen at the declared tolerance, not a mesh-independence or stability proof.')
    if name=='P005' and out['status']=='failed':
        (al,ah),(bl,bh)=out['event_brackets']
        if max(al,bl)<=min(ah,bh):
            out.update(status='inconclusive',reason='Event brackets overlap: monitor resolution cannot resolve the requested difference.')
    return out
