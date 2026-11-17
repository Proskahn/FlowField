"""Reject unqualified fields; independently reconstruct EWM indicators and balances."""
import csv
import json
import math
from pathlib import Path
import re
from .geometry import build_geometry,PATCHES,FLUID
from .ewm import INDICATORS,score
from ...level4_high_dimensional_topology.p007.metrics import values,block

BALANCES=('liquid_balance','gas_balance','momentum_residual','energy_balance')

def mesh_arrays(case,m):
    import numpy as np
    def labels(name):
        text=(case/'constant/polyMesh'/name).read_text().split('}',1)[1]
        match=re.fullmatch(r'\s*(\d+)\s*\((.*?)\)\s*',text,re.S)
        if not match:raise ValueError(f'Malformed {name}')
        data=np.array([int(v) for v in match[2].split()])
        if len(data)!=int(match[1]):raise ValueError('Truncated addressing')
        return data
    owner,neighbour=labels('owner'),labels('neighbour')
    if np.min(owner)<0 or np.max(owner)>=len(m['cells']) or np.min(neighbour)<0 or np.max(neighbour)>=len(m['cells']):raise ValueError('Invalid cell index')
    edges=[np.array(m[k])*1e-3 for k in ('x','y','z')]
    widths=[np.diff(v) for v in edges];centers=[(v[1:]+v[:-1])/2 for v in edges]
    xyz=np.array([[centers[d][idx] for d,idx in enumerate(cell)] for cell in m['cells']])
    size=np.array([[widths[d][idx] for d,idx in enumerate(cell)] for cell in m['cells']]);vol=np.prod(size,axis=1)
    normals=np.zeros((len(owner),3));do=np.zeros(len(owner));dn=do.copy();area=do.copy()
    for f,(o,n) in enumerate(zip(owner,neighbour)):
        delta=xyz[n]-xyz[o];axis=int(np.argmax(abs(delta)));normals[f,axis]=np.sign(delta[axis]);do[f]=size[o,axis]/2;dn[f]=size[n,axis]/2;area[f]=vol[o]/size[o,axis]
    patches={};boundary=(case/'constant/polyMesh/boundary').read_text()
    for patch in PATCHES:
        b=block(boundary,patch);count=int(re.search(r'nFaces\s+(\d+)',b)[1]);start=int(re.search(r'startFace\s+(\d+)',b)[1]);idx=np.arange(start,start+count);patches[patch]=idx
        for f in idx:
            o=owner[f];c=xyz[o];sz=size[o]
            if patch.endswith(('Inlet','Outlet')):axis=1;sign=-1 if patch.endswith('Inlet') else 1
            elif patch.endswith('Terminal'):axis=2;sign=-1 if patch=='cathodeTerminal' else 1
            else:
                candidates=[(d,s) for d in (0,1) for s in (-1,1) if math.isclose(c[d]+s*sz[d]/2,edges[d][0 if s<0 else -1],abs_tol=1e-12)]
                # A corner has two wall faces. The writer emits x faces before y.
                used=[int(np.argmax(abs(normals[ff]))) for ff in idx[idx<f] if owner[ff]==o]
                axis,sign=next((d,s) for d,s in candidates if d not in used)
            normals[f,axis]=sign;do[f]=sz[axis]/2;area[f]=vol[o]/sz[axis]
    return owner,neighbour,xyz,size,vol,normals,do,dn,area,patches


def verify_fields(case,manifest,config,step):
    import numpy as np
    m=build_geometry(manifest['design_geometry'],config);n=len(m['cells']);directory=case/str(int(step))
    fields={k:values((directory/k).read_text(),'internalField',n) for k in ('material','p','pc','T','sg','phiS','phiE','iVol','heatSource','liquidSource','gasSource')}
    U={k:values((directory/k).read_text(),'internalField',n,3) for k in ('U','Ul','Ug')}
    mat=np.array(m['ids']);s=fields['sg'];T=fields['T'];p=fields['p'];iv=fields['iVol'];heat=fields['heatSource'];fluid=np.isin(mat,list(FLUID));ch=np.isin(mat,[1,7])
    if not np.array_equal(mat,fields['material']):raise ValueError('Material topology differs from generated design')
    if np.min(T)<273.15 or np.max(T)>373.15 or np.min(s)<0 or np.max(s)>=config.maximum_packing:raise ValueError('Saved fields outside model validity range')
    if np.min(config.outlet_pressure+p[fluid])<=0:raise ValueError('Nonpositive absolute pressure')
    o,nb,xyz,size,v,normal,do,dn,area,patches=mesh_arrays(case,m);ni=len(nb)
    def surface(name):
        text=(directory/name).read_text();out=np.zeros(len(o));out[:ni]=values(text,'internalField',ni)
        for patch,idx in patches.items():out[idx]=values(block(text,patch),'value',len(idx))
        return out
    l,g=surface('phiL'),surface('phiG');feed=config.inlet_flow_ml_min*1e-6/60
    checks={};phase_nets={}
    for phase,flux,source in (('L',l,fields['liquidSource']),('G',g,fields['gasSource'])):
        net=np.bincount(o,weights=flux,minlength=n)-np.bincount(nb,weights=flux[:ni],minlength=n)
        phase_nets[phase]=net
        scale=2*feed if phase=='L' else max(float(abs(source)@v),1e-12)
        checks['saved_local_'+phase]=float(np.sum(abs(net-source*v)))/scale
        checks['saved_solid_'+phase]=float(np.sum(abs(net[~fluid])))/scale
        for patch,idx in patches.items():
            if patch.endswith('Inlet'):checks[patch+'_'+phase]=abs(float(sum(flux[idx]))+(feed if phase=='L' else 0))/feed
            elif not patch.endswith('Outlet'):checks[patch+'_'+phase]=float(np.sum(abs(flux[idx])))/feed
    F=96485.33212;ia=float(iv[mat==5]@v[mat==5]);ic=-float(iv[mat==3]@v[mat==3]);heatW=float(heat@v)
    if min(ia,ic)<=0:raise ValueError('Nonpositive electrolysis current')
    rhoG=config.outlet_pressure*np.where(mat>=5,.0319988,.00201588)/(8.314462618*config.inlet_temperature)
    expectedL=np.where(mat==5,iv*.01801528/(2*F*config.liquid_density),np.where(mat==3,iv*.01801528/(F*config.liquid_density),0))
    expectedG=np.where(mat==5,iv*.0319988/(4*F*rhoG),np.where(mat==3,-iv*.00201588/(2*F*rhoG),0))
    checks.update(charge_balance=abs(ia-ic)/ia,faraday_liquid=float(abs((expectedL-fields['liquidSource'])*v).sum())/(2*feed),
        faraday_gas=float(abs((expectedG-fields['gasSource'])*v).sum())/max(float(expectedG@v),1e-12),
        heat_power_identity=abs(heatW-ia*(config.cell_voltage-1.481))/max(abs(heatW),.01),
        homogeneous_channel_velocity=float(np.max(abs(U['Ul'][ch]-U['Ug'][ch])))/max(float(np.max(abs(U['U'][ch]))),.01))
    # Potentials independently satisfy both conservative conduction graphs.
    eps=np.where(ch,1,np.where(np.isin(mat,[2,6]),config.porosity_ptl,np.where(np.isin(mat,[3,5]),config.porosity_cl,0)))
    # Verify constitutive laws, not just divergence of the saved fluxes.
    porous=fluid&~ch;perm=np.where(np.isin(mat,[2,6]),config.permeability_ptl,config.permeability_cl)
    expected_pc=np.where(porous,config.surface_tension*math.cos(math.radians(config.contact_angle_deg))*np.sqrt(eps/perm)*(1.417*s-2.120*s*s+1.263*s*s*s),0)
    checks['leverett_pressure']=float(np.max(abs(expected_pc-fields['pc'])))/max(float(np.max(abs(expected_pc))),1.)
    impermeable=(~fluid[o[:ni]])|(~fluid[nb])
    checks['internal_wall_phase_leakage']=float(np.sum(abs(l[:ni][impermeable])+abs(g[:ni][impermeable])))/(2*feed)
    pf=np.flatnonzero(porous[o[:ni]]&porous[nb])
    for label,flux,gas in (('L',l,False),('G',g,True)):
        phase_p=p+(expected_pc if gas else 0);drive=phase_p[o[pf]]-phase_p[nb[pf]]
        donor=np.where(drive>=0,o[pf],nb[pf]);fraction=s[donor] if gas else 1-s[donor]
        mu=np.where(mat>=5,config.oxygen_viscosity,config.hydrogen_viscosity) if gas else np.full(n,config.liquid_viscosity)
        resistance=do[pf]*mu[o[pf]]/perm[o[pf]]+dn[pf]*mu[nb[pf]]/perm[nb[pf]]
        expected=area[pf]*drive*fraction**3/resistance
        checks['darcy_'+label]=float(np.sum(abs(expected-flux[pf])))/max(float(np.sum(abs(flux[pf]))),1e-12)
        reconstructed=np.zeros((n,3));np.add.at(reconstructed,o,flux[:,None]*do[:,None]*normal)
        np.add.at(reconstructed,nb,flux[:ni,None]*dn[:ni,None]*normal[:ni]);reconstructed/=v[:,None]
        from_velocity=U['Ug' if gas else 'Ul']*eps[:,None]*(s if gas else 1-s)[:,None]
        checks['porous_velocity_'+label]=float(np.max(abs(reconstructed[porous]-from_velocity[porous])))/max(float(np.max(abs(reconstructed[porous]))),1e-12)
    sigmaS=np.array([200000,0,1250,750,0,750,1250,0,200000])[mat]
    km=config.koh_molarity;koh=config.koh_conductivity_scale*(-2.04*km-.0027*km**2+.005332*km*T+207.2*km/T+.00105*km**3-.0000004*km**2*T*T)
    sigmaE=np.where(mat==4,(.524*config.membrane_water_content-.318)*np.exp(1270*(1/303.-1/T)),np.where(np.isin(mat,[3,5]),np.maximum(eps*(1-s),0)**1.5*koh,0))
    for name,sigma,source in (('phiS',sigmaS,iv),('phiE',sigmaE,-iv)):
        potential=fields[name];res=source*v;conduct=np.zeros(ni);active=(sigma[o[:ni]]>0)&(sigma[nb]>0)
        idx=np.flatnonzero(active);conduct[idx]=area[idx]/(do[idx]/sigma[o[idx]]+dn[idx]/sigma[nb[idx]])
        flux=conduct*(potential[o[:ni]]-potential[nb]);res=res+np.bincount(o[:ni],weights=flux,minlength=n)-np.bincount(nb,weights=flux,minlength=n)
        if name=='phiS':
            for patch,voltage in (('anodeTerminal',config.cell_voltage),('cathodeTerminal',0)):
                idx=patches[patch];bf=sigma[o[idx]]*area[idx]/do[idx]*(potential[o[idx]]-voltage);np.add.at(res,o[idx],bf)
        checks['saved_'+name+'_balance']=float(abs(res).sum())/ia
    # Independent saturation-dependent kinetics, without a hidden fitted current.
    alphaA=.0675+.00095*T;alphaC=.1175+.00095*T;beta=F/(8.314462618*T)
    eta=fields['phiS']-fields['phiE']-np.where(mat==5,1.481-.000846*T,0)
    cl=np.isin(mat,[3,5]);anode=mat==5
    pref=(1-s)*np.where(anode,config.active_area_anode*config.exchange_current_anode,config.active_area_cathode*config.exchange_current_cathode)*np.exp(np.where(anode,config.activation_energy_anode,config.activation_energy_cathode)/8.314462618*(1/config.kinetic_reference_temperature-1/T))
    bv=pref[cl]*(np.exp(alphaA[cl]*beta[cl]*eta[cl])-np.exp(-alphaC[cl]*beta[cl]*eta[cl]))
    checks['butler_volmer']=float(abs(bv-iv[cl])@v[cl])/(ia+ic)
    # Independent LTE energy residual, including physical convective wall traces.
    ks=np.array([200,0,3,1,.95,1,3,0,200])[mat]
    k=(1-eps)*ks+eps*((1-s)*config.liquid_conductivity+s*np.where(mat>=5,config.oxygen_conductivity,config.hydrogen_conductivity))
    if config.energy_form!='lte_phase_advection':raise ValueError('Printed-energy experiments are diagnostic only; EWM qualification requires the documented LTE form')
    energy=-heat*v
    conduct=area[:ni]/(do[:ni]/k[o[:ni]]+dn[:ni]/k[nb])
    flux=conduct*(T[o[:ni]]-T[nb]);np.add.at(energy,o[:ni],flux);np.add.at(energy,nb,-flux)
    adv=config.liquid_density*config.liquid_cp*l+rhoG[o]*np.where(mat[o]>=5,config.oxygen_cp,config.hydrogen_cp)*g
    upwind=np.where(adv[:ni]>=0,T[o[:ni]],T[nb])
    np.add.at(energy,o[:ni],adv[:ni]*(upwind-T[o[:ni]]));np.add.at(energy,nb,-adv[:ni]*(upwind-T[nb]))
    for patch,idx in patches.items():
        if patch.endswith('Inlet'):
            np.add.at(energy,o[idx],k[o[idx]]*area[idx]/do[idx]*(T[o[idx]]-config.inlet_temperature)+adv[idx]*(config.inlet_temperature-T[o[idx]]))
        elif not patch.endswith('Outlet'):
            np.add.at(energy,o[idx],area[idx]/(do[idx]/k[o[idx]]+1/config.external_heat_transfer)*(T[o[idx]]-config.ambient_temperature))
    checks['saved_energy_balance']=float(abs(energy).sum())/max(float(abs(heat)@v),1e-12)
    # Nine indicators, with explicitly defined volume averages.
    def mean(a,zone):return float(a[mat==zone]@v[mat==zone]/sum(v[mat==zone]))
    meanT=mean(T,4);uniformity=mean(abs(T-meanT),4)/meanT
    # Gauss-linear gradient, matching the declared FV interpolation (no smoothing).
    Tf=np.zeros(len(o));Tf[:ni]=(dn[:ni]*T[o[:ni]]+do[:ni]*T[nb])/(do[:ni]+dn[:ni]);Tf[ni:]=T[o[ni:]]
    for patch,idx in patches.items():
        if patch.endswith('Inlet'):Tf[idx]=config.inlet_temperature
        elif not patch.endswith('Outlet'):Tf[idx]=values(block((directory/'T').read_text(),patch),'value',len(idx))
    grad=np.zeros((n,3));np.add.at(grad,o,Tf[:,None]*area[:,None]*normal);np.add.at(grad,nb,-Tf[:ni,None]*area[:ni,None]*normal[:ni]);grad/=v[:,None]
    dp={}
    ptext=(directory/'p').read_text()
    for side in ('anode','cathode'):
        idx=patches[side+'Inlet'];pv=values(block(ptext,side+'Inlet'),'value',len(idx));dp[side]=float(pv@area[idx]/sum(area[idx]))
        idx=patches[side+'Outlet'];checks[side+'_outlet_pressure']=float(np.max(abs(values(block(ptext,side+'Outlet'),'value',len(idx)))))
        if dp[side]<=0:raise ValueError('Reversed pressure drop')
    metrics=dict(current_density_A_cm2=ia/4,membrane_temperature_uniformity=uniformity,temperature_gradient_K_m=float(np.linalg.norm(grad,axis=1)@v/sum(v)),
        cathode_mean_velocity_m_s=mean(np.linalg.norm(U['U'],axis=1),1),anode_mean_velocity_m_s=mean(np.linalg.norm(U['U'],axis=1),7),
        cathode_cl_gas_fraction=mean(s,3),anode_cl_gas_fraction=mean(s,5),cathode_pressure_drop_Pa=dp['cathode'],anode_pressure_drop_Pa=dp['anode'],
        current_anode_A=ia,current_cathode_A=ic,heat_W=heatW,min_temperature_K=float(min(T)),max_temperature_K=float(max(T)),checks=checks)
    if any(not math.isfinite(val) or val>config.balance_tolerance for val in checks.values()):raise ValueError(f'P008 saved-field verification failed: {checks}')
    return metrics

def extract_metrics(case,config):
    case=Path(case);manifest=json.loads((case/'case.json').read_text());status=json.loads((case/'postProcessing/p008/status.json').read_text())
    with (case/'postProcessing/p008/history.csv').open() as stream:rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    if len(rows)<5 or any(row['iteration']!=i+1 or not all(math.isfinite(v) for v in row.values()) for i,row in enumerate(rows)):raise ValueError('Missing, restarted or nonfinite P008 history')
    log=(case/'log.flowfieldP008Foam').read_text();last=rows[-1]
    times=[float(p.name) for p in case.iterdir() if p.is_dir() and re.fullmatch(r'\d+(?:\.\d+)?',p.name)]
    if (status.get('converged') is not True or status.get('iteration')!=last['iteration'] or max(times)!=last['iteration'] or status.get('cells')!=manifest['cells']
        or status.get('processes')!=1 or 'CONVERGED: P008 coupled flow, charge and heat qualified.' not in log or not re.search(r'^End\s*$',log,re.M)):
        raise ValueError('P008 is unconverged, stale, or missing complete final fields')
    for r in rows[-5:]:
        if r['iteration']<config.minimum_iterations or abs(r['applied_voltage_V']-config.cell_voltage)>1e-12 or r['charge_residual']>=1e-7 or not 0<=r['field_change']<config.field_tolerance or any(not 0<=r[k]<config.balance_tolerance for k in BALANCES):raise ValueError('Last five iterations fail P008 qualification')
    metrics=verify_fields(case,manifest,config,last['iteration']);metrics.update(iteration=int(last['iteration']),cells=manifest['cells'])
    for name,_ in INDICATORS:
        if not math.isclose(metrics[name],last[name],rel_tol=1e-5,abs_tol=1e-10):raise ValueError(f'Saved indicator differs from history: {name}')
    calibration=json.loads((case/'calibration.json').read_text())
    ewm=score(metrics,calibration) if calibration else None
    return dict(objective=-ewm['score'] if ewm else -metrics['current_density_A_cm2'],
        objectives=dict(negative_current_density=-metrics['current_density_A_cm2'],membrane_temperature_uniformity=metrics['membrane_temperature_uniformity'],
            pumping_power_W=(metrics['anode_pressure_drop_Pa']+metrics['cathode_pressure_drop_Pa'])*config.inlet_flow_ml_min*1e-6/60),
        constraints={},feasible=True,metrics=metrics,ewm=ewm,
        objective_definition='negative frozen EWM score' if ewm else 'negative current density for baseline diagnostics; EWM calibration required for optimization')
