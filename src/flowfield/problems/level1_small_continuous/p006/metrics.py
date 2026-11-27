"""Accept only converged simulations; independently recompute the objective."""
import csv
import json
import math
from pathlib import Path
import re
from .geometry import build_geometry, PATCHES


def values(text,key,count):
    match=re.search(r'\b'+re.escape(key)+r'\s+uniform\s+([^;]+);',text)
    if match:
        data=[float(match.group(1))]*count
    else:
        match=re.search(r'\b'+re.escape(key)+r'\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
        if not match or int(match.group(1))!=count:raise ValueError(f'Expected {count} values for {key}')
        data=[float(v) for v in match.group(2).split()]
    if len(data)!=count or not all(math.isfinite(v) for v in data):raise ValueError('Truncated/nonfinite field')
    return data


def vectors(text,key,count):
    match=re.search(r'\b'+re.escape(key)+r'\s+uniform\s*\(([^)]+)\)\s*;',text)
    if match:
        data=[tuple(float(v) for v in match.group(1).split())]*count
    else:
        match=re.search(r'\b'+re.escape(key)+r'\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
        if not match or int(match.group(1))!=count:raise ValueError(f'Expected {count} vectors for {key}')
        data=[tuple(float(v) for v in row.split()) for row in re.findall(r'\(([^)]+)\)',match.group(2))]
    if len(data)!=count or any(len(row)!=3 or not all(math.isfinite(v) for v in row) for row in data):
        raise ValueError('Truncated/nonfinite vector field')
    return data


def verify_fields(case,manifest,config,iteration):
    m=build_geometry(manifest['design_geometry'],config)
    n=len(m['cells']);directory=case/str(int(iteration))
    fields={name:values((directory/{'Sl':'liquidSource','Sg':'gasSource'}.get(name,name)).read_text(),'internalField',n)
            for name in ('T','sg','p','pc','phiS','phiE','lambda','iVol','heatSource','Sl','Sg','hydrationSource')}
    velocity={name:vectors((directory/name).read_text(),'internalField',n) for name in ('Ul','Ug','qL','qG')}
    from .pressure import verify_pressure_forces
    force={phase:vectors((directory/('pressureForce'+phase)).read_text(),'internalField',n) for phase in ('L','G')}
    pressure_error=verify_pressure_forces(m,config,fields['p'],fields['sg'],force)
    if not all(273.15<v<373.15 for v in fields['T']):raise ValueError('Saved temperatures outside validity range')
    if not all(0<=v<=1 for v in fields['sg']):raise ValueError('Unbounded saved saturation')
    x,y,z=([v*.001 for v in m[axis]] for axis in ('x','y','z'))
    nx,ny=len(x)-1,len(y)-1
    samples=[];volumes=[];ia=ic=heat=oxygen=hydrogen=hydration=0.
    k=[];kinematic_error=0.
    velocity_scale=max(.01,max(abs(v) for name in ('qL','qG') for row in velocity[name] for v in row))
    for c,mat in enumerate(m['ids']):
        eps=1 if mat in (1,7) else config.porosity_ptl if mat in (2,6) else config.porosity_cl if mat in (3,5) else 0
        gas=fields['sg'][c]
        for q,u,alpha in (('qL','Ul',eps*(1-gas)),('qG','Ug',eps*gas)):
            kinematic_error=max(kinematic_error,max(abs(a-alpha*b) for a,b in zip(velocity[q][c],velocity[u][c]))/velocity_scale)
        if mat in (1,2,3,5,6,7) and min(101325+fields['p'][c],101325+fields['p'][c]+fields['pc'][c])<=0:
            raise ValueError('Nonpositive saved absolute phase pressure')
        kf=(1-gas)*config.liquid_conductivity+gas*(config.oxygen_conductivity if mat>=5 else config.hydrogen_conductivity)
        ks=config.membrane_conductivity if mat==4 else config.catalyst_conductivity if mat in (3,5) else config.solid_conductivity
        k.append(eps*kf+(1-eps)*ks)
    for c,((i,j,layer),mat) in enumerate(zip(m['cells'],m['ids'])):
        dx,dy,dz=x[i+1]-x[i],y[j+1]-y[j],z[layer+1]-z[layer]
        volume=dx*dy*dz;current=fields['iVol'][c]*volume
        volumes.append(volume)
        if mat==5:ia+=current;oxygen+=fields['Sg'][c]*volume*config.oxygen_density/.032
        if mat==3:ic-=current;hydrogen+=fields['Sg'][c]*volume*config.hydrogen_density/.002
        heat+=fields['heatSource'][c]*volume
        hydration+=fields['hydrationSource'][c]*volume
        neighbor=c+nx*ny
        if mat==4 and neighbor<n and m['ids'][neighbor]==5:
            ka=k[c]/(dz/2);kb=k[neighbor]/((z[layer+2]-z[layer+1])/2)
            samples.append((dx*dy,(ka*fields['T'][c]+kb*fields['T'][neighbor])/(ka+kb)))
    area=sum(a for a,_ in samples)
    if abs(area/450e-6-1)>1e-9:raise ValueError('Missing or incorrect membrane interface')
    mean=sum(a*t for a,t in samples)/area
    std=math.sqrt(sum(a*(t-mean)**2 for a,t in samples)/area)
    if kinematic_error>1e-8:raise ValueError('Saved superficial and intrinsic velocities are inconsistent')
    checks=dict(kinematic_velocity_consistency=kinematic_error,saved_pressure_force_consistency=pressure_error,
        charge_balance=abs(ia-ic)/max(abs(ia),1e-12),
        oxygen_faraday=abs(oxygen-ia/(4*96485.33212))/max(abs(oxygen),1e-12),
        hydrogen_faraday=abs(hydrogen-ic/(2*96485.33212))/max(abs(hydrogen),1e-12),
        heat_power_identity=abs(heat-ia*(config.cell_voltage-(1.229+.0009*298.15)))/max(abs(heat),.01),
        membrane_water_balance=abs(hydration)/(2*config.inlet_flow_ml_min*1e-6/60))
    checks.update(verify_phase_fluxes(case,directory,m['ids'],volumes,fields,config))
    if ia<=0 or ic<=0 or max(checks.values())>config.balance_tolerance:
        raise ValueError(f'Independent saved-field conservation failed: {checks}')
    return dict(temperature_std_K=std,temperature_mean_K=mean,
        temperature_min_K=min(t for _,t in samples),temperature_max_K=max(t for _,t in samples),
        current_anode_A=ia,current_cathode_A=ic,current_density_A_cm2=ia/4.5,
        heat_W=heat,checks=checks)


def verify_phase_fluxes(case,directory,materials,volumes,fields,config):
    """Independent signed face sums, including reconstructed MPI patch fluxes."""
    def labels(name):
        body=(case/'constant/polyMesh'/name).read_text().split('}',1)[1]
        match=re.search(r'(\d+)\s*\((.*?)\)',body,re.S)
        if not match:raise ValueError(f'Malformed mesh {name}')
        data=[int(v) for v in match.group(2).split()]
        if len(data)!=int(match.group(1)) or any(v<0 or v>=len(volumes) for v in data):
            raise ValueError(f'Invalid mesh {name}')
        return data
    owner,neighbor=labels('owner'),labels('neighbour')
    boundary=(case/'constant/polyMesh/boundary').read_text()
    checks={};q=config.inlet_flow_ml_min*1e-6/60
    for phase,source in (('L','Sl'),('G','Sg')):
        text=(directory/('phi'+phase)).read_text()
        flux=values(text,'internalField',len(neighbor))
        net=[0.]*len(volumes);ports=[0.,0.];sources=[0.,0.]
        for f,phi in enumerate(flux):net[owner[f]]+=phi;net[neighbor[f]]-=phi
        seen=set(range(len(neighbor)))
        for patch in PATCHES:
            block=re.search(r'\b'+patch+r'\s*\{(.*?)\}',boundary,re.S).group(1)
            count=int(re.search(r'nFaces\s+(\d+)',block).group(1))
            start=int(re.search(r'startFace\s+(\d+)',block).group(1))
            phi_block=re.search(r'\b'+patch+r'\s*\{(.*?)\}',text,re.S).group(1)
            face_flux=values(phi_block,'value',count)
            for j,phi in enumerate(face_flux):
                f=start+j
                if f in seen or f>=len(owner):raise ValueError('Invalid boundary face addressing')
                seen.add(f);net[owner[f]]+=phi
                if patch.endswith(('Inlet','Outlet')):ports[0 if patch.startswith('anode') else 1]+=phi
                elif abs(phi)>1e-16:raise ValueError('Phase leakage through an exterior wall')
        if len(seen)!=len(owner):raise ValueError('Incomplete saved boundary fluxes')
        residual=0.
        for c,mat in enumerate(materials):
            expected=fields[source][c]*volumes[c]
            if mat in (1,2,3,5,6,7):
                residual+=abs(net[c]-expected);sources[0 if mat>=5 else 1]+=expected
            elif abs(net[c])+abs(expected)>1e-16:raise ValueError('Phase transport in solid cells')
        norm=2*q if phase=='L' else max(sum(sources),2*q)
        checks['saved_local_'+phase+'_balance']=residual/norm
        checks['saved_global_'+phase+'_balance']=max(abs(a-b)/(q if phase=='L' else max(abs(b),q)) for a,b in zip(ports,sources))
    return checks


def extract_metrics(case,config):
    case=Path(case);manifest=json.loads((case/'case.json').read_text())
    status=json.loads((case/'postProcessing/p006/status.json').read_text())
    with (case/'postProcessing/p006/history.csv').open() as stream:
        rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    if len(rows)<5:raise ValueError('Need five converged iterations')
    for i,row in enumerate(rows):
        if row['iteration']!=i+1 or not all(math.isfinite(v) for v in row.values()):
            raise ValueError('History is mixed, truncated or nonfinite')
    log=(case/'log.flowfieldP006Foam').read_text()
    if (status.get('converged') is not True or status.get('iteration')!=rows[-1]['iteration']
        or status.get('cells')!=manifest['cells'] or status.get('processes')!=config.processes
        or 'CONVERGED: coupled fields and local/global balances passed.' not in log
        or not re.search(r'^End\s*$',log,re.M)):
        raise ValueError('A completed converged run is required')
    for row in rows[-5:]:
        if not (row['iteration']>=config.minimum_iterations and row['charge_residual']<1e-6
            and row.get('momentum_fraction')==1 and row.get('thermal_active')==1
            and status.get('momentum_fraction')==1 and status.get('thermal_active') is True
            and abs(row['applied_voltage_V']-config.cell_voltage)<1e-12
            and row['field_change']<config.field_tolerance and row['bounded_cells']==0
            and all(0<=row[key]<config.balance_tolerance for key in
                ('liquid_balance','gas_balance','local_liquid_balance','local_gas_balance','energy_balance',
                 'liquid_momentum_residual','gas_momentum_residual'))):
            raise ValueError('Coupled convergence or conservation limits were not met')
    fields=verify_fields(case,manifest,config,rows[-1]['iteration'])
    if abs(fields['temperature_std_K']-rows[-1]['T_std_K'])>1e-4:
        raise ValueError('Saved membrane objective differs from the solver history')
    constraints=manifest['design_geometry']['constraints']
    raw=fields['temperature_std_K']
    scalar=raw+1000*sum(max(0,v)**2 for v in constraints.values())
    return dict(objective=scalar,unpenalized_objective=raw,fidelity=config.fidelity,
        objectives=dict(temperature_std_K=raw),constraints=constraints,
        metrics=dict(rows[-1],**fields,cells=manifest['cells']))
