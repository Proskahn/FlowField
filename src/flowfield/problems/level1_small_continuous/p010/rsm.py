"""Quadratic Box-Behnken response surfaces, with auditable uncertainty limits."""
import itertools
import math
from .config import PARAMETERS,BOUNDS,normalized
from .reference import ROWS,REPORTED_OPTIMUM,printed_polynomial

def features(x):
    import numpy as np
    a=np.atleast_2d(np.asarray(x,dtype=float))
    if a.shape[1]!=4 or not np.isfinite(a).all():raise ValueError('Expected finite four-factor designs')
    z=2*a-1
    return np.column_stack([np.ones(len(z)),*z.T,*(z*z).T,*[z[:,i]*z[:,j] for i in range(4) for j in range(i+1,4)]])

def fit(x,y,*,source):
    import numpy as np
    X=features(x);y=np.asarray(y,dtype=float)
    if y.shape!=(len(X),) or not np.isfinite(y).all():raise ValueError('Invalid response values')
    coef,_,rank,_=np.linalg.lstsq(X,y,rcond=None)
    if rank!=15 or len(X)<=rank:raise ValueError('A full-rank, overdetermined four-factor quadratic is required')
    residual=y-X@coef;sse=float(residual@residual);sst=float(np.sum((y-y.mean())**2));df=int(len(X)-rank)
    cov=np.linalg.inv(X.T@X)*sse/df;hat=np.einsum('ij,ji->i',X,np.linalg.solve(X.T@X,X.T))
    press=float(np.sum((residual/(1-hat))**2)) if np.max(hat)<1-1e-10 else None
    groups={tuple(v) for v in np.asarray(x).tolist()};group_errors=[];unavailable=[]
    for group in sorted(groups):
        held=np.all(np.asarray(x)==np.array(group),axis=1);train=X[~held]
        beta,_,r,_=np.linalg.lstsq(train,y[~held],rcond=None)
        if r<15:unavailable.append(list(group))
        else:group_errors.extend((y[held]-X[held]@beta).tolist())
    return dict(schema_version=1,source=source,coefficient_order=['1',*PARAMETERS,
        *[v+'^2' for v in PARAMETERS],*[PARAMETERS[i]+'*'+PARAMETERS[j] for i in range(4) for j in range(i+1,4)]],
        coding='z=2*normalized_design-1; physical bounds are Table 5',coefficients=coef.tolist(),
        covariance=cov.tolist(),observations=len(y),distinct_designs=len(groups),rank=int(rank),residual_dof=df,
        residual_variance=sse/df,sse=sse,r_squared=1-sse/sst if sst else None,
        adjusted_r_squared=1-sse/df/(sst/(len(y)-1)) if sst else None,
        press=press,predicted_r_squared=1-press/sst if press is not None and sst else None,
        group_validation=dict(status='incomplete' if unavailable else 'complete',
            rank_deficient_held_out_designs=unavailable,partial_sse=float(sum(v*v for v in group_errors))),
        uncertainty_scope='OLS model-based intervals; not CFD mesh uncertainty, physical validation, or independent replicate evidence')

def predict(model,x):
    import numpy as np
    return float(features(x)[0]@np.asarray(model['coefficients']))

def maximize(model):
    """Enumerate stationary points on all 81 faces of the four-dimensional box."""
    import numpy as np
    c=np.array(model['coefficients']);H=np.diag(2*c[5:9]);index=9
    for i in range(4):
        for j in range(i+1,4):H[i,j]=H[j,i]=c[index];index+=1
    best=None;checked=0
    for active in itertools.product((-1,0,1),repeat=4):
        z=np.array(active,dtype=float);free=[i for i,v in enumerate(active) if v==0];fixed=[i for i,v in enumerate(active) if v!=0]
        if free:
            a=H[np.ix_(free,free)];b=-c[1:5][free]-H[np.ix_(free,fixed)]@z[fixed]
            if np.linalg.matrix_rank(a)<len(free):continue
            z[free]=np.linalg.solve(a,b)
        if np.any(abs(z)>1+1e-10):continue
        x=np.clip((z+1)/2,0,1);value=predict(model,x);checked+=1
        if best is None or value>best['predicted_net_power_W']:
            best=dict(normalized_design=x.tolist(),physical_design={k:lo+(hi-lo)*v for k,v,(lo,hi) in zip(PARAMETERS,x,BOUNDS)},predicted_net_power_W=value)
    if best is None:raise ValueError('No feasible stationary/corner candidate')
    best.update(method='Exact quadratic box face enumeration',candidates_checked=checked)
    return best

def interval(model,x,confidence=.95):
    import numpy as np
    from scipy.stats import t
    if not 0<confidence<1:raise ValueError('Confidence must lie in (0,1)')
    row=features(x)[0];mean=predict(model,x);variance=max(float(row@np.asarray(model['covariance'])@row),0.)
    q=float(t.ppf((1+confidence)/2,model['residual_dof']))
    return dict(confidence=confidence,predicted_net_power_W=mean,
        mean_response_interval_W=[mean-q*math.sqrt(variance),mean+q*math.sqrt(variance)],
        individual_response_interval_W=[mean-q*math.sqrt(variance+model['residual_variance']),mean+q*math.sqrt(variance+model['residual_variance'])],
        scope=model['uncertainty_scope'])

def reference_audit():
    x=[normalized(row[:4]) for row in ROWS];model=fit(x,[row[4] for row in ROWS],source='Published Table 6; no CFD executed')
    optimum=maximize(model);reported=normalized(REPORTED_OPTIMUM)
    return dict(status='reference_data_audit_only',paper_validated=False,cfd_executed=False,
        model=model,optimum=optimum,optimum_intervals=interval(model,optimum['normalized_design']),
        reported_optimum=dict(physical_design=list(REPORTED_OPTIMUM),reported_prediction_W=3.11097,
            rounded_equation_prediction_W=printed_polynomial(REPORTED_OPTIMUM),refitted_prediction_W=predict(model,reported),
            reported_validation_mean_W=3.09460),
        discrepancies=['Table 3 concentration exponent is 1, printed Eqs. 9/10 use 3.',
            'Reference current, diffusivities, permeabilities, phase-change/sorption parameters and kinetic prefactors are not supplied.',
            'Printed Darcy/source signs require a consistent signed-current convention.',
            'Repeated center responses differ; deterministic CFD reuse cannot manufacture that experimental variance.',
            'Rounded physical-unit polynomial coefficients lose precision; use the centered Table 6 fit for this audit.'])
