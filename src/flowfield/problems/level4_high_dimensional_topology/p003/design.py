"""Nodal design -> linear radial filter -> Eq. 40 projection -> porous properties."""
from functools import lru_cache
import math


def project(value,beta):
    return -math.expm1(-beta*value)+value*math.exp(-beta)


def inverse_project(value,beta):
    if not 0<=value<=1:raise ValueError('Projection target must lie in [0,1]')
    if value in (0,1):return float(value)
    lo,hi=0.,1.
    for _ in range(70):
        mid=(lo+hi)/2
        if project(mid,beta)<value:lo=mid
        else:hi=mid
    return (lo+hi)/2


def grid(config):
    factor={'coarse':1,'medium':2,'fine':3}[config.mesh]
    # CFD resolution is independent of nodal design resolution.
    spacing=config.width_m/(32*factor)
    knots=sorted({0.,config.inlet_width_m,config.width_m-config.outlet_width_m,config.width_m})
    x=[0.]
    for a,b in zip(knots,knots[1:]):
        n=math.ceil((b-a)/spacing-1e-12)
        x.extend(a+(b-a)*i/n for i in range(1,n+1))
    y=[config.length_m*j/(64*factor) for j in range(64*factor+1)]
    return x,y


@lru_cache(maxsize=12)
def stencil(config):
    x,y=grid(config);dx=config.width_m/config.design_nx;dy=config.length_m/config.design_ny
    r=config.filter_radius_m;rows=[];areas=[]
    for j in range(len(y)-1):
        cy=(y[j]+y[j+1])/2
        for i in range(len(x)-1):
            cx=(x[i]+x[i+1])/2;entries=[]
            for jj in range(max(0,math.ceil((cy-r)/dy)),min(config.design_ny,math.floor((cy+r)/dy))+1):
                for ii in range(max(0,math.ceil((cx-r)/dx)),min(config.design_nx,math.floor((cx+r)/dx))+1):
                    w=max(0.,1-math.hypot(ii*dx-cx,jj*dy-cy)/r)
                    if w>0:entries.append((jj*(config.design_nx+1)+ii,w))
            total=sum(v for _,v in entries)
            if total<=0:raise ValueError('Filter contains an empty neighborhood')
            rows.append(tuple((n,w/total) for n,w in entries));areas.append((x[i+1]-x[i])*(y[j+1]-y[j]))
    return tuple(rows),tuple(areas)


def decode(values,config):
    rows,areas=stencil(config)
    filtered=[sum(values[n]*w for n,w in row) for row in rows]
    indicator=[project(v,config.projection_beta) for v in filtered]
    a=config.interpolation;p=a['simp_p'];q=a['ramp_q']
    return dict(normalized_design=list(values),node_shape=[config.design_ny+1,config.design_nx+1],
        flattening='C order: node iy*(nx+1)+ix; x varies fastest',
        filtered=filtered,indicator=indicator,cell_areas_m2=list(areas),
        porosity=[a['eps_rib']+(a['eps_channel']-a['eps_rib'])*s for s in indicator],
        diffusivity_factor=[a['diffusion_rib']+(a['diffusion_channel']-a['diffusion_rib'])*s**p for s in indicator],
        brinkman_pa_s_m2=[a['alpha_rib']*(1-s)/(1+q*s) for s in indicator],
        volume_fraction=sum(v*a for v,a in zip(indicator,areas))/sum(areas),
        area_m2=config.area_m2,depth_m=config.depth_m)


def volume_gradient(values,config):
    """Exact filter/projection chain rule for the volume constraint; no CFD adjoint."""
    rows,areas=stencil(config);gradient=[0.]*config.dimension;total=sum(areas);b=config.projection_beta
    for row,area in zip(rows,areas):
        f=sum(values[n]*w for n,w in row);deriv=b*math.exp(-b*f)+math.exp(-b)
        for n,w in row:gradient[n]+=area/total*deriv*w
    return gradient


@lru_cache(maxsize=4)
def _repair_filter(config):
    # Optional optimizer dependencies; importing/running a standalone CFD case
    # continues to work without NumPy or SciPy.
    import numpy as np
    from scipy.sparse import csr_matrix
    rows, areas = stencil(config)
    indices, weights, offsets = [], [], [0]
    for row in rows:
        indices.extend(n for n, _ in row)
        weights.extend(w for _, w in row)
        offsets.append(len(indices))
    matrix = csr_matrix((weights, indices, offsets), shape=(len(rows), config.dimension))
    areas = np.asarray(areas)
    return matrix, areas/areas.sum()


def repair_candidates(candidates, config, *, anchor=None):
    """Enforce projected, area-weighted channel volume without running CFD.

    Infeasible rows move along anchor + s*(row-anchor), taking the largest
    feasible s in [0,1]. Nonnegative filter weights and a monotone projection
    make volume monotone on that segment. A trust-region lower corner is a
    valid anchor whenever its center is feasible, and keeps repair in the box.
    Feasible rows are unchanged. Work in chunks to bound temporary memory.
    """
    import numpy as np
    values = np.asarray(candidates)
    if (values.ndim != 2 or values.shape[1] != config.dimension or values.dtype.kind not in 'fiu'
            or not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1)):
        raise ValueError('P003 repair requires finite design rows within [0, 1]')
    origin = np.zeros(config.dimension) if anchor is None else np.asarray(anchor)
    if (origin.shape != (config.dimension,) or origin.dtype.kind not in 'fiu'
            or not np.all(np.isfinite(origin)) or np.any(origin < 0) or np.any(origin > 1)
            or np.any(values < origin)):
        raise ValueError('P003 repair anchor must be a componentwise lower bound on the candidates')
    matrix, areas = _repair_filter(config)
    beta, limit = config.projection_beta, config.volume_limit
    def volume(filtered):
        return np.sum((-np.expm1(-beta*filtered) + filtered*math.exp(-beta))*areas, axis=-1)
    base = matrix @ origin
    base_volume = float(volume(base))
    if base_volume > limit + 1e-12:
        raise ValueError('P003 repair anchor violates the channel-volume constraint')
    # Leave a numerical margin when possible; never exclude a feasible anchor.
    target = max(base_volume, limit - min(1e-10, limit*1e-8))
    result = values.astype(float, copy=True)
    for start in range(0, len(result), 128):
        block = result[start:start+128]
        filtered = (matrix @ block.T).T
        bad = np.flatnonzero(volume(filtered) > limit + 1e-12)
        if not len(bad):
            continue
        delta = filtered[bad] - base
        lo, hi = np.zeros(len(bad)), np.ones(len(bad))
        for _ in range(48):
            mid = (lo+hi)*.5
            feasible = volume(base + mid[:, None]*delta) <= target
            lo = np.where(feasible, mid, lo)
            hi = np.where(feasible, hi, mid)
        block[bad] = origin + lo[:, None]*(block[bad]-origin)
        if np.any(volume((matrix @ block[bad].T).T) > limit + 1e-12):
            raise RuntimeError('P003 volume repair failed its final feasibility check')
    return result


def preset(name,config):
    if name=='uniform':
        return [inverse_project(config.volume_limit,config.projection_beta)]*config.dimension
    if name in ('all_rib','all_channel'):
        return [float(name=='all_channel')]*config.dimension
    if name!='parallel':raise ValueError('Unknown P003 seed')
    # Illustrative parallel seed, not digitized CAD from Figure 11.
    width=.0013;values=[]
    for j in range(config.design_ny+1):
        y=config.length_m*j/config.design_ny
        for i in range(config.design_nx+1):
            x=config.width_m*i/config.design_nx
            channel=(y<width or y>config.length_m-width or
                     min(abs(x-k*config.width_m/4) for k in range(5))<width/2)
            values.append(float(channel))
    rows,areas=stencil(config);filtered=[sum(values[n]*w for n,w in row) for row in rows]
    def fraction(scale):return sum(project(scale*v,config.projection_beta)*a for v,a in zip(filtered,areas))/sum(areas)
    if fraction(1)<config.volume_limit:raise ValueError('Seed cannot satisfy requested projected volume')
    lo,hi=0.,1.
    for _ in range(65):
        mid=(lo+hi)/2
        if fraction(mid)<config.volume_limit:lo=mid
        else:hi=mid
    return [v*(lo+hi)/2 for v in values]
