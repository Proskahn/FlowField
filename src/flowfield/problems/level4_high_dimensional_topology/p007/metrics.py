"""Strict completion gates and independent checks of saved fields and port fluxes."""
import csv
import json
import math
from pathlib import Path
import re
from .design import grid

COLUMNS=('iteration,heat_transfer,dissipation,inlet_flow,outlet_flow,inlet_pressure,pumping_power,volume_error,'
    'local_continuity,power_error,momentum_residual,energy_balance,field_change,min_T,max_T,volume_fraction').split(',')
BALANCES=('volume_error','local_continuity','power_error','momentum_residual','energy_balance')


def values(text,key,count,components=1):
    import numpy as np
    match=re.search(r'\b'+re.escape(key)+r'\s+uniform\s+([^;]+);',text)
    if match:
        v=[float(x) for x in match[1].replace('(',' ').replace(')',' ').split()]
        if len(v)!=components:raise ValueError('Unexpected field components')
        result=np.tile(v,(count,1))
    else:
        match=re.search(r'\b'+re.escape(key)+r'\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
        if not match or int(match[1])!=count:raise ValueError(f'Expected {count} field values for {key}')
        result=np.array([float(x) for x in match[2].replace('(',' ').replace(')',' ').split()])
        if result.size!=count*components:raise ValueError('Truncated field')
        result=result.reshape(count,components)
    if not np.all(np.isfinite(result)):raise ValueError('Nonfinite field')
    return result[:,0] if components==1 else result


def block(text,name):
    m=re.search(r'\b'+re.escape(name)+r'\s*\{(.*?)\}',text,re.S)
    if not m:raise ValueError(f'Missing patch {name}')
    return m[1]


def scalarize(heat,dissipation,volume,config):
    if not all(math.isfinite(v) for v in (heat,dissipation,volume)) or min(heat,dissipation)<=0 or not 0<=volume<=1:
        raise ValueError('Invalid physical objectives')
    w1,w2=config.weights;raw=-w1*heat+w2*dissipation;g=volume-config.volume_limit
    return dict(objective=raw+config.volume_penalty*max(g,0),unpenalized_objective=raw,
        objectives=dict(negative_heat_transfer=-heat,dissipation=dissipation),
        constraints=dict(fluid_volume=g),feasible=g<=1e-12)


def extract_metrics(case,config):
    import numpy as np
    case=Path(case);manifest=json.loads((case/'case.json').read_text())
    status=json.loads((case/'postProcessing/p007/status.json').read_text())
    with (case/'postProcessing/p007/history.csv').open() as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=COLUMNS:raise ValueError('Unexpected P007 history columns')
        rows=[{k:float(v) for k,v in r.items()} for r in reader]
    if len(rows)<5 or any(r['iteration']!=i+1 or not all(math.isfinite(v) for v in r.values()) for i,r in enumerate(rows)):
        raise ValueError('Missing, restarted or nonfinite P007 history')
    last=rows[-1];log=(case/'log.flowfieldP007Foam').read_text()
    times=[p for p in case.iterdir() if p.is_dir() and re.fullmatch(r'\d+(?:\.\d+)?',p.name)]
    latest=max(times,key=lambda p:float(p.name))
    if (status.get('converged') is not True or status.get('processes')!=1 or status.get('cells')!=manifest['cells']
        or status.get('iteration')!=last['iteration'] or float(latest.name)!=last['iteration']
        or 'CONVERGED: P007 flow, heat and pumping power qualified.' not in log or not re.search(r'^End\s*$',log,re.M)):
        raise ValueError('P007 is unconverged, stale or missing final fields')
    for r in rows[-5:]:
        if r['iteration']<30 or not 0<=r['field_change']<config.outer_tolerance or any(not 0<=r[k]<config.balance_tolerance for k in BALANCES):
            raise ValueError('Last five P007 iterations fail qualification')
        if r['min_T']<-1e-8 or r['max_T']>1+1e-8:raise ValueError('Unbounded temperature')
    geometry=json.loads((case/'design.json').read_text());n=manifest['cells']
    fields={k:values((latest/k).read_text(),'internalField',n) for k in ('T','p','gamma','brinkman')}
    U=values((latest/'U').read_text(),'internalField',n,3)
    T,gamma,alpha=(fields[k] for k in ('T','gamma','brinkman'))
    if np.min(T)<-1e-8 or np.max(T)>1+1e-8:raise ValueError('Saved T outside [0,1]')
    for key,stored in (('gamma','indicator'),('brinkman','brinkman')):
        if not np.allclose(fields[key],geometry[stored],rtol=1e-11,atol=1e-12):raise ValueError('Saved material differs from design')
    areas=np.array(geometry['cell_areas']);volume=float(areas@gamma/25)
    heat=float(areas@(config.heat_coefficient*(1-gamma)*(1-T)))
    poly=(case/'constant/polyMesh/boundary').read_text();phi=(latest/'phi').read_text()
    flux={};ubound={};pbound={}
    for patch in ('inlet','outlet','walls'):
        count=int(re.search(r'nFaces\s+(\d+)',block(poly,patch))[1])
        flux[patch]=values(block(phi,patch),'value',count)
        if patch!='outlet':ubound[patch]=values(block((latest/'U').read_text(),patch),'value',count,3)
        if patch!='walls':pbound[patch]=values(block((latest/'p').read_text(),patch),'value',count)
    qin=-float(sum(flux['inlet']))/.1;qout=float(sum(flux['outlet']))/.1
    power=-float(pbound['inlet']@flux['inlet'])/.1
    if min(qin,qout,heat,power)<=0:raise ValueError('Reversed flow or nonpositive heating/power')
    # Reconstruct the Gauss-linear cell gradients independently from saved U.
    x,y=map(np.asarray,grid(config));dx=np.diff(x);dy=np.diff(y);nx,ny=len(dx),len(dy)
    u=U.reshape(ny,nx,3);gx=np.zeros((ny,nx+1,3));gy=np.zeros((ny+1,nx,3))
    gx[:,1:-1]=(u[:,:-1]*dx[1:][None,:,None]+u[:,1:]*dx[:-1][None,:,None])/(dx[1:]+dx[:-1])[None,:,None]
    gy[1:-1]=(u[:-1]*dy[1:,None,None]+u[1:]*dy[:-1,None,None])/(dy[1:]+dy[:-1])[:,None,None]
    port=np.flatnonzero(((y[1:]+y[:-1])/2>2.425)&((y[1:]+y[:-1])/2<2.575))
    gx[port,0]=ubound['inlet'];gx[port,-1]=u[port,-1]  # zeroGradient outlet
    grad=np.zeros((ny,nx,3,3))
    grad[:,:,0,:]=np.diff(gx,axis=1)/dx[None,:,None]
    grad[:,:,1,:]=np.diff(gy,axis=0)/dy[:,None,None]
    density=(np.sum(grad*(grad+grad.swapaxes(-1,-2)),axis=(-1,-2))/config.reynolds).ravel()+alpha*np.sum(U*U,axis=1)
    dissipation=float(areas@density)
    velocity_power=float(np.sum(np.linalg.norm(ubound['inlet'],axis=1)*dy[port]*pbound['inlet']))
    # Independent finite-volume thermal closure. gamma multiplies the cell
    # advection operator, so a plain outlet-minus-inlet heat flux is incorrect.
    internal_count=ny*(nx-1)+(ny-1)*nx
    f=values(phi,'internalField',internal_count)
    adv=np.zeros(n);diff=np.zeros(n);cursor=0
    for j in range(ny):
        for i in range(nx):
            c=j*nx+i
            for other,conductance in ((c+1,dy[j]*.1/((dx[i]+dx[i+1])/2)) if i+1<nx else (-1,0),
                                      (c+nx,dx[i]*.1/((dy[j]+dy[j+1])/2)) if j+1<ny else (-1,0)):
                if other<0:continue
                flow=f[cursor];cursor+=1
                transport=flow*(T[c] if flow>=0 else T[other]);adv[c]+=transport;adv[other]-=transport
                conduction=conductance*(T[other]-T[c]);diff[c]+=conduction;diff[other]-=conduction
    for k,j in enumerate(port):
        left=j*nx;right=left+nx-1
        adv[left]+=flux['inlet'][k]*(0 if flux['inlet'][k]<0 else T[left])
        adv[right]+=flux['outlet'][k]*T[right]
        diff[left]+=dy[j]*.1/(dx[0]/2)*(0-T[left])
    residual=config.reynolds*config.prandtl*gamma*adv-diff-config.heat_coefficient*(1-gamma)*(1-T)*areas*.1
    checks=dict(power_error=abs(power/config.pumping_power-1),volume_error=abs(qin-qout)/qin,
        inlet_velocity_power_error=abs(velocity_power-power)/power,
        saved_energy_balance=float(np.sum(np.abs(residual)))/(.1*heat),
        wall_flux=float(np.sum(np.abs(flux['walls'])))/(.1*qin),outlet_pressure=float(np.max(np.abs(pbound['outlet']))),
        heat_history_error=abs(heat-last['heat_transfer'])/heat,dissipation_history_error=abs(dissipation-last['dissipation'])/dissipation,
        power_history_error=abs(power-last['pumping_power'])/power,
        flow_history_error=abs(qin-last['inlet_flow'])/qin,
        temperature_history_error=max(abs(float(np.min(T))-last['min_T']),abs(float(np.max(T))-last['max_T'])),
        volume_history_error=abs(volume-last['volume_fraction']))
    if any(not math.isfinite(v) or v>config.balance_tolerance for v in checks.values()):
        raise ValueError(f'P007 saved-field verification failed: {checks}')
    result=scalarize(heat,dissipation,volume,config)
    result['metrics']=dict(heat_transfer=heat,dissipation=dissipation,inlet_flow=qin,outlet_flow=qout,
        inlet_pressure=float(np.mean(pbound['inlet'])),pumping_power=power,volume_fraction=volume,
        min_temperature=float(np.min(T)),max_temperature=float(np.max(T)),iteration=int(last['iteration']),
        cells=n,units='dimensionless; area integrals per unit depth',checks=checks)
    return result

