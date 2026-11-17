"""Cell-control density, conservative Helmholtz filter and paper tanh projection."""
from functools import lru_cache
import math


def grid(config):
    n = {'coarse':64, 'medium':96, 'fine':128}[config.mesh]
    x = [5*i/n for i in range(n+1)]
    # Both ports occupy exactly y = 2.425 .. 2.575. Resolve them at every level.
    y = [0.]
    for a,b,count in ((0.,2.425,n//2),(2.425,2.575,6*n//64),(2.575,5.,n//2)):
        y.extend(a+(b-a)*i/count for i in range(1,count+1))
    return x,y


@lru_cache(maxsize=8)
def operators(config):
    import numpy as np
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import splu
    x,y=map(np.asarray,grid(config));dx=np.diff(x);dy=np.diff(y)
    nx,ny=len(dx),len(dy);xc=(x[1:]+x[:-1])/2;yc=(y[1:]+y[:-1])/2
    areas=np.outer(dy,dx).ravel();rows=[];cols=[];data=[];diag=areas.copy()
    def edge(a,b,conductance):
        k=config.filter_radius**2*conductance
        rows.extend((a,b));cols.extend((b,a));data.extend((-k,-k));diag[a]+=k;diag[b]+=k
    for j in range(ny):
        for i in range(nx):
            c=j*nx+i
            if i+1<nx:edge(c,c+1,dy[j]/(xc[i+1]-xc[i]))
            if j+1<ny:edge(c,c+nx,dx[i]/(yc[j+1]-yc[j]))
    matrix=coo_matrix((data,(rows,cols)),shape=(nx*ny,nx*ny)).tocsc()+diags(diag)
    ix=np.minimum((xc/5*config.design_nx).astype(int),config.design_nx-1)
    iy=np.minimum((yc/5*config.design_ny).astype(int),config.design_ny-1)
    if config.mirror_y:iy=np.minimum(iy,config.design_ny-1-iy)
    mapping=(iy[:,None]*config.design_nx+ix[None,:]).ravel()
    return splu(matrix),areas,mapping


def projection(v,config):
    import numpy as np
    b,t=config.projection_beta,config.projection_threshold
    return (np.tanh(b*(v-t))+math.tanh(b*t))/(math.tanh(b*(1-t))+math.tanh(b*t))


def decode(values,config):
    import numpy as np
    solve,areas,mapping=operators(config)
    filtered=solve.solve(areas*np.asarray(values)[mapping])
    if np.min(filtered)<-1e-10 or np.max(filtered)>1+1e-10:
        raise ValueError('Helmholtz filter violated its maximum principle')
    gamma=projection(filtered,config)
    alpha=config.alpha_s*config.ramp_q*(1-gamma)/(config.ramp_q+gamma)
    return dict(normalized_design=list(values),filtered=filtered.tolist(),indicator=gamma.tolist(),
        brinkman=alpha.tolist(),cell_areas=areas.tolist(),volume_fraction=float(areas@gamma/25),
        flattening='x varies fastest; lower half controls reflected in y when mirror_y=true',
        design_shape=[config.design_ny//(2 if config.mirror_y else 1),config.design_nx])


def volume_gradient(values,config):
    import numpy as np
    solve,areas,mapping=operators(config);f=solve.solve(areas*np.asarray(values)[mapping])
    b,t=config.projection_beta,config.projection_threshold
    dp=b*(1-np.tanh(b*(f-t))**2)/(math.tanh(b*(1-t))+math.tanh(b*t))
    adj=solve.solve(areas*dp/25,trans='T')
    return np.bincount(mapping,weights=areas*adj,minlength=config.dimension).tolist()


def repair_candidates(candidates,config,*,anchor=None):
    import numpy as np
    a=np.asarray(candidates,dtype=float)
    if a.ndim!=2 or a.shape[1]!=config.dimension or not np.all(np.isfinite(a)) or np.any((a<0)|(a>1)):
        raise ValueError('Expected finite design rows within [0,1]')
    base=np.zeros(config.dimension) if anchor is None else np.asarray(anchor,dtype=float)
    if base.shape!=(config.dimension,) or not np.all(np.isfinite(base)) or np.any((base<0)|(base>1)) or np.any(a<base):
        raise ValueError('Repair anchor must be a componentwise lower bound in [0,1]')
    solve,areas,mapping=operators(config);f0=solve.solve(areas*base[mapping])
    def volume(f):return float(areas@projection(f,config)/25)
    if volume(f0)>config.volume_limit+1e-12:raise ValueError('Infeasible volume-repair anchor')
    result=a.copy()
    for i,row in enumerate(a):
        f=solve.solve(areas*row[mapping])
        if volume(f)<=config.volume_limit+1e-12:continue
        lo,hi=0.,1.
        for _ in range(48):
            mid=(lo+hi)/2
            if volume(f0+mid*(f-f0))<=config.volume_limit-1e-12:lo=mid
            else:hi=mid
        result[i]=base+lo*(row-base)
    return result

