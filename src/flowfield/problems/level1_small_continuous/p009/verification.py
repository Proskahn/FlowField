"""Recompute local continuity, pressure and plane moments from saved mesh/fields."""
from pathlib import Path
import re
from ...level4_high_dimensional_topology.p007.metrics import values, block


def mesh_list(path, dtype=float, components=1):
    import numpy as np
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',Path(path).read_text(),flags=re.S)
    match=re.search(r'\n\s*(\d+)\s*\((.*?)\)\s*$',text,re.S)
    if not match:raise ValueError(f'Malformed mesh list {path}')
    data=np.fromstring(match[2].replace('(',' ').replace(')',' '),sep=' ',dtype=dtype)
    count=int(match[1])
    if data.size != count*components:raise ValueError(f'Truncated mesh list {path}')
    return data.reshape(count,components) if components>1 else data


def verify_saved(case, config, step, cells, expected):
    import numpy as np
    case=Path(case);poly=case/'constant/polyMesh';saved=case/str(int(step))
    owner=mesh_list(poly/'owner',int);neighbour=mesh_list(poly/'neighbour',int);ni=len(neighbour)
    if min(owner)<0 or max(owner)>=cells or min(neighbour)<0 or max(neighbour)>=cells:
        raise ValueError('Invalid saved mesh addressing')
    raw=mesh_list(poly/'faces',int,5)
    if not np.all(raw[:,0]==4):raise ValueError('P009 requires quadrilateral mesh faces')
    points=mesh_list(poly/'points',float,3)
    faces=raw[:,1:]
    if faces.min()<0 or faces.max()>=len(points):raise ValueError('Invalid face points')
    xyz=points[faces]
    area_vectors=.5*np.sum(np.cross(xyz,np.roll(xyz,-1,axis=1)),axis=1)
    areas=np.linalg.norm(area_vectors,axis=1)
    ptext=(saved/'p').read_text();phitext=(saved/'phi').read_text()
    if not re.search(r'dimensions\s*\[0 2 -2 0 0 0 0\]',ptext):raise ValueError('P009 pressure must be kinematic')
    p=values(ptext,'internalField',cells)
    flux=np.empty(len(owner));flux[:ni]=values(phitext,'internalField',ni)
    boundary=(poly/'boundary').read_text();pressure={};patch_flux={}
    coverage=np.zeros(len(owner)-ni,dtype=int)
    for patch in ('inlet','outlet','walls'):
        b=block(boundary,patch)
        n=int(re.search(r'nFaces\s+(\d+)',b)[1]);start=int(re.search(r'startFace\s+(\d+)',b)[1])
        idx=np.arange(start,start+n)
        if start<ni or start+n>len(owner):raise ValueError('Invalid patch addressing')
        coverage[idx-ni]+=1
        flux[idx]=values(block(phitext,patch),'value',n)
        patch_flux[patch]=float(np.sum(abs(flux[idx])) if patch=='walls' else sum(flux[idx]))
        pb=block(ptext,patch)
        pv=p[owner[idx]] if 'zeroGradient' in pb else values(pb,'value',n)
        pressure[patch]=float(pv@areas[idx]/sum(areas[idx]))
    if not np.all(coverage==1):raise ValueError('Boundary faces not covered exactly once')
    net=np.bincount(owner,weights=flux,minlength=cells)-np.bincount(neighbour,weights=flux[:ni],minlength=cells)
    feed=abs(expected['q_inlet'])
    local=float(np.sum(abs(net)))/feed
    checks={'saved_local_continuity':local}
    for patch in ('inlet','outlet','walls'):
        checks['saved_'+patch+'_flux_history_error']=abs(patch_flux[patch]-expected['q_'+patch])/feed
    measured_dp=config.density_kg_m3*(pressure['inlet']-pressure['outlet'])
    checks['saved_pressure_history_error']=abs(measured_dp-expected['dp'])/abs(expected['dp'])
    # All mesh cells are straight extrusions. Their cross sections are the
    # horizontal end faces. This independently repeats the nickel-only slice.
    low=np.min(xyz[:,:,2],axis=1);high=np.max(xyz[:,:,2],axis=1)
    cell_low=np.full(cells,np.inf);cell_high=np.full(cells,-np.inf)
    np.minimum.at(cell_low,owner,low);np.minimum.at(cell_low,neighbour,low[:ni])
    np.maximum.at(cell_high,owner,high);np.maximum.at(cell_high,neighbour,high[:ni])
    horizontal=(high-low)<1e-12
    section=.5*(np.bincount(owner,weights=areas*horizontal,minlength=cells)
        +np.bincount(neighbour,weights=areas[:ni]*horizontal[:ni],minlength=cells))
    zone=block((poly/'cellZones').read_text(),'nickel')
    match=re.search(r'cellLabels\s+List<label>\s+(\d+)\s*\((.*?)\)',zone,re.S)
    if not match:raise ValueError('Missing nickel zone')
    ids=np.fromstring(match[2],sep=' ',dtype=int)
    if len(ids)!=int(match[1]) or ids.min()<0 or ids.max()>=cells:raise ValueError('Invalid nickel zone')
    select=np.zeros(cells,dtype=bool);select[ids]=True
    z=config.measurement_height_mm*.001
    select &= (cell_low<=z+1e-12)&(cell_high>z+1e-12)
    if not np.any(select):raise ValueError('Empty measurement plane')
    u=values((saved/'U').read_text(),'internalField',cells,3)
    speed=np.linalg.norm(u[select],axis=1);weights=section[select]
    mean=float(weights@speed/sum(weights));square=float(weights@(speed*speed)/sum(weights))
    checks['saved_plane_speed_history_error']=abs(mean-expected['speed'])/expected['speed']
    checks['saved_plane_speed2_history_error']=abs(square-expected['speed2'])/expected['speed2']
    if not all(np.isfinite(v) and 0<=v<=config.balance_tolerance for v in checks.values()):
        raise ValueError('Independent saved-field verification failed: '+str(checks))
    return dict(checks=checks,plane_area_m2=float(sum(weights)),pressure_drop_pa=measured_dp,
        mean_speed_m_s=mean,mean_speed_squared_m2_s2=square)
