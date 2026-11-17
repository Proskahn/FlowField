"""Li et al. 2026 entropy weighting, Eqs. 54--60, with fixed-panel calibration.

The paper's score is a property of a comparison panel, not an intrinsic f(x).
Fit once on a declared panel before scoring an optimization trajectory.
"""
import hashlib
import json
import math

INDICATORS = (
    ('current_density_A_cm2', 'benefit'),
    ('membrane_temperature_uniformity', 'cost'),
    ('temperature_gradient_K_m', 'cost'),
    ('cathode_mean_velocity_m_s', 'benefit'),
    ('anode_mean_velocity_m_s', 'benefit'),
    ('cathode_cl_gas_fraction', 'cost'),
    ('anode_cl_gas_fraction', 'cost'),
    ('cathode_pressure_drop_Pa', 'cost'),
    ('anode_pressure_drop_Pa', 'cost'),
)


def _rows(records, names):
    rows=[]
    for record in records:
        if isinstance(record,dict):
            row=[record[k] for k in names]
        else:row=list(record)
        if len(row)!=len(names) or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in row):
            raise ValueError('Entropy weighting requires complete finite numeric rows')
        rows.append([float(x) for x in row])
    if not rows:raise ValueError('No EWM observations')
    return rows


def fit(records, indicators=INDICATORS, *, epsilon=1e-4):
    """Fit the paper's min-max normalization and entropy weights to one panel.

    Constant columns receive zero information/weight. If every column is
    constant, no ranking is defined; reject instead of inventing equal weights.
    """
    if not indicators or len({k for k,_ in indicators})!=len(indicators):raise ValueError('Unique indicators required')
    if any(d not in ('benefit','cost') for _,d in indicators):raise ValueError('Unknown indicator direction')
    if isinstance(epsilon,bool) or not isinstance(epsilon,(int,float)) or not math.isfinite(epsilon) or epsilon<=0:raise ValueError('epsilon must be positive')
    names=[k for k,_ in indicators];rows=_rows(records,names);m=len(rows)
    if m<2:raise ValueError('Entropy weights require at least two observations')
    lo=[min(row[j] for row in rows) for j in range(len(names))]
    hi=[max(row[j] for row in rows) for j in range(len(names))]
    z=[]
    for row in rows:
        z.append([((value-lo[j]) if indicators[j][1]=='benefit' else (hi[j]-value))/(hi[j]-lo[j])+epsilon
                  if hi[j]>lo[j] else epsilon for j,value in enumerate(row)])
    entropy=[];information=[]
    for j in range(len(names)):
        total=math.fsum(row[j] for row in z);p=[row[j]/total for row in z]
        e=-math.fsum(v*math.log(v) for v in p)/math.log(m)
        entropy.append(e);information.append(max(0.,1-e) if hi[j]>lo[j] else 0.)
    total=math.fsum(information)
    if total<=1e-14:raise ValueError('EWM panel contains no discriminating information')
    weights=[v/total for v in information]
    panel_digest=hashlib.sha256(json.dumps(rows,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return dict(schema_version=1,method='paper_minmax_entropy',names=names,
        directions=[d for _,d in indicators],minimum=lo,maximum=hi,epsilon=epsilon,
        entropy=entropy,weights=weights,panel_count=m,panel_sha256=panel_digest,
        scores=[math.fsum(w*v for w,v in zip(weights,row)) for row in z])


def score(record, calibration):
    """Score with frozen reference bounds/weights; never refit or silently clip.

    Extrapolation beyond the calibration range is reported. Its score may lie
    outside [epsilon,1+epsilon]; that is preferable to changing past scores.
    """
    names=calibration['names']
    if not names or len(set(names))!=len(names):raise ValueError('Unique nonempty calibration indicators required')
    row=_rows([record],names)[0]
    n=len(names)
    if any(len(calibration[k])!=n for k in ('minimum','maximum','weights','directions')):
        raise ValueError('Malformed EWM calibration')
    weights=calibration['weights'];epsilon=calibration['epsilon']
    if isinstance(epsilon,bool) or not isinstance(epsilon,(int,float)) or not math.isfinite(epsilon) or epsilon<=0:raise ValueError('Invalid calibration epsilon')
    if any(isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w<0 for w in weights) or not math.isclose(sum(weights),1,abs_tol=1e-10):
        raise ValueError('Invalid entropy weights')
    normalized=[];outside=[]
    for i,value in enumerate(row):
        lo,hi=calibration['minimum'][i],calibration['maximum'][i]
        if not math.isfinite(lo+hi) or hi<lo:raise ValueError('Invalid calibration bounds')
        direction=calibration['directions'][i]
        if direction not in ('benefit','cost'):raise ValueError('Invalid calibration direction')
        if hi==lo:
            if weights[i]!=0:raise ValueError('Constant indicator must have zero weight')
            z=epsilon
        else:z=((value-lo) if direction=='benefit' else (hi-value))/(hi-lo)+epsilon
        normalized.append(z)
        if value<lo or value>hi:outside.append(names[i])
    return dict(score=math.fsum(w*z for w,z in zip(weights,normalized)),normalized=normalized,
        outside_calibration=outside,panel_sha256=calibration['panel_sha256'])
