"""Completion gate and independent finite-volume checks of saved PEMFC fields.

This audit establishes numerical consistency, not agreement with the article.
"""
import csv
import json
import math
from pathlib import Path
import re
from ...level4_high_dimensional_topology.p007.metrics import values,block
from .geometry import build_geometry,PATCHES
from .case import saturation_pressure,equilibrium_water

F=96485.33212
R=8.314462618
MW=.01801528
BALANCES=('gas_balance','species_balance','liquid_balance','membrane_balance','momentum_residual','energy_balance')

def mesh_arrays(case,g,config):
    import numpy as np
    m=build_geometry(g,config);poly=case/'constant/polyMesh'
    def body(name):
        text=(poly/name).read_text();match=re.search(r'\n\s*\d+\s*\n\((.*)\)\s*$',text,re.S)
        if not match:raise ValueError(f'Invalid polyMesh {name}')
        return match[1]
    points=np.fromstring(body('points').replace('(',' ').replace(')',' '),sep=' ').reshape(-1,3)
    faces=np.array([[int(v) for v in row.split()] for row in re.findall(r'4\(([^)]+)\)',body('faces'))])
    o=np.fromstring(body('owner'),sep=' ',dtype=int);ni=np.fromstring(body('neighbour'),sep=' ',dtype=int)
    if len(faces)!=len(o):raise ValueError('Face/owner mismatch')
    n=np.full(len(o),-1,dtype=int);n[:len(ni)]=ni
    xyz=points[faces];fc=xyz.mean(axis=1);sf=np.cross(xyz[:,1]-xyz[:,0],xyz[:,3]-xyz[:,0]);area=np.linalg.norm(sf,axis=1)
    idx=np.asarray(m['cells']);edges=[np.asarray(m[k])*1e-3 for k in ('x','y','z')]
    centers=np.column_stack([((e[1:]+e[:-1])/2)[idx[:,i]] for i,e in enumerate(edges)])
    volume=np.prod(np.column_stack([np.diff(e)[idx[:,i]] for i,e in enumerate(edges)]),axis=1)
    do=np.linalg.norm(fc-centers[o],axis=1);dn=np.zeros(len(o));dn[:len(ni)]=np.linalg.norm(fc[:len(ni)]-centers[ni],axis=1)
    patches={};boundary=(poly/'boundary').read_text()
    for name in PATCHES:
        text=block(boundary,name);start=int(re.search(r'startFace\s+(\d+)',text)[1]);count=int(re.search(r'nFaces\s+(\d+)',text)[1])
        patches[name]=np.arange(start,start+count)
    return dict(o=o,n=n,ni=len(ni),area=area,do=do,dn=dn,normal=sf/area[:,None],
                volumes=volume,centers=centers,material=np.asarray(m['ids']),patches=patches)

def saved_audit(case,config,last,latest):
    import numpy as np
    g=json.loads((case/'design.json').read_text());a=mesh_arrays(case,g,config)
    mat=a['material'];nc=len(mat);nf=len(a['o']);o,n=a['o'],a['n'];inter=n>=0
    gas=np.isin(mat,(1,2,3,5,6,7));ion=np.isin(mat,(3,4,5));cl=np.isin(mat,(3,5));an=mat>=5
    channel=np.isin(mat,(1,7));electronic=~np.isin(mat,(1,4,7));v=a['volumes'];checks={}
    names=('p','T','s','lambda','yr','yw','phiS','phiE','iVol','material','heatSource',
           'srcGas','srcReact','srcVapor','srcLiquid','srcMembrane','srcHeat')
    texts={k:(latest/k).read_text() for k in names}
    f={k:values(t,'internalField',nc) for k,t in texts.items()}
    utext=(latest/'U').read_text();values(utext,'internalField',nc,3)
    if not np.array_equal(f['material'],mat):raise ValueError('Saved material differs from prescribed geometry')
    T,s,lam,yr,yw,iv=(f[k] for k in ('T','s','lambda','yr','yw','iVol'))
    if (T.min()<273.15 or T.max()>373.15 or s.min()<-1e-10 or s.max()>=.98
        or min(yr[gas].min(),yw[gas].min())<-1e-8 or (yr[gas]+yw[gas]).max()>1.00001
        or lam[ion].min()<=.326/.5139):raise ValueError('Saved fields violate physical bounds')
    def surface(name):
        text=(latest/name).read_text();out=np.zeros(nf);out[:a['ni']]=values(text,'internalField',a['ni'])
        for patch,indices in a['patches'].items():out[indices]=values(block(text,patch),'value',len(indices))
        return out
    flux={k:surface('phi'+k) for k in ('Gas','React','Vapor','Liquid','Water','Heat')}
    def div(q):
        return np.bincount(o,weights=q,minlength=nc)-np.bincount(n[inter],weights=q[inter],minlength=nc)
    Ic=-float(v[mat==3]@iv[mat==3]);Ia=float(v[mat==5]@iv[mat==5]);water_scale=max(Ic/(2*F),1e-10)
    if min(Ic,Ia)<=0:raise ValueError('Fuel-cell current must be positive')
    ct=config.outlet_pressure/(R*T);eps=np.where(channel,1,np.where(cl,config.cl_porosity,g['gdl_porosity']))
    yw_in=g['humidity_percent']*.01*saturation_pressure(g['temperature_K'])/config.outlet_pressure
    feed=lambda anode: config.reference_current_density*g['active_area_m2']*(config.anode_stoichiometry if anode else g['cathode_stoichiometry'])/((2 if anode else 4)*F*(1-yw_in)*(1 if anode else .21))
    feed_scale=feed(True)+feed(False);heat_scale=max(float(v@np.abs(f['srcHeat'])),1e-10)
    for patch,idx in a['patches'].items():
        tblock=block(texts['T'],patch)
        # OpenFOAM omits redundant zeroGradient patch values on write.
        if patch.endswith('Outlet') and not re.search(r'\bvalue\s',tblock):
            if not re.search(r'\btype\s+zeroGradient\s*;',tblock):raise ValueError('Unexpected outlet temperature boundary')
            tb=T[o[idx]]
        else:tb=values(tblock,'value',len(idx))
        expected=T[o[idx]] if patch.endswith('Outlet') else g['temperature_K']
        checks[patch+'_temperature_bc']=float(np.max(np.abs(tb-expected))/g['temperature_K']) if len(idx) else 0.
        ub=values(block(utext,patch),'value',len(idx),3) if not patch.endswith('Outlet') else None
        if patch.endswith('Inlet'):
            anode=patch.startswith('anode');speed=feed(anode)/(config.outlet_pressure/(R*g['temperature_K']))/a['area'][idx].sum()
            checks[patch+'_velocity_bc']=float(np.max(np.linalg.norm(ub+a['normal'][idx]*speed,axis=1))/speed)
        elif not patch.endswith('Outlet'):
            checks[patch+'_wall_velocity']=float(np.max(np.linalg.norm(ub,axis=1))) if len(idx) else 0.
        if patch.endswith('Terminal'):
            pb=values(block(texts['phiS'],patch),'value',len(idx));target=config.cell_voltage if patch.startswith('cathode') else 0
            checks[patch+'_potential_bc']=float(np.max(np.abs(pb-target))) if len(idx) else 0.
    for key in flux:
        scale=heat_scale if key=='Heat' else feed_scale if key in ('Gas','React') else water_scale
        checks['saved_'+key.lower()+'_balance']=float(np.abs(div(flux[key])-v*f['src'+('Membrane' if key=='Water' else key)]).sum()/scale)
    expected_react=np.where(cl,-np.abs(iv)/(np.where(an,2,4)*F),0)
    production=np.where(mat==3,-iv/(2*F),0)
    checks['faraday_reactant']=float(v@np.abs(f['srcReact']-expected_react)/feed_scale)
    checks['total_molar_source']=float(v@np.abs(f['srcGas']-f['srcReact']-f['srcVapor'])/feed_scale)
    checks['total_water_source']=float(v@np.abs(f['srcVapor']+f['srcLiquid']+f['srcMembrane']-production)/water_scale)
    checks['latent_heat_source']=float(v@np.abs(f['srcHeat']-f['heatSource']-f['srcLiquid']*MW*config.latent_heat)/heat_scale)
    checks['current_continuity']=abs(Ia-Ic)/Ic
    checks['anode_composition']=float(np.max(np.abs(yr[gas&an]+yw[gas&an]-1)))
    def conductance(k,mask):
        out=np.zeros(nf);active=inter&mask[o]&mask[np.maximum(n,0)]
        out[active]=a['area'][active]/(a['do'][active]/k[o[active]]+a['dn'][active]/k[n[active]])
        return out
    def diff_flux(k,x,mask):
        q=np.zeros(nf);indices=np.flatnonzero(inter);q[indices]=conductance(k,mask)[indices]*(x[o[indices]]-x[n[indices]])
        return q
    def adv_flux(q,x,bc):
        result=q*x[np.where((q<0)&inter,n,o)]
        for patch,idx in a['patches'].items():
            if patch.endswith('Inlet'):result[idx]=q[idx]*bc[an[o[idx]].astype(int)]
            elif patch.endswith('Outlet'):result[idx]=q[idx]*np.where(q[idx]>=0,x[o[idx]],bc[an[o[idx]].astype(int)])
            else:result[idx]=0
        return result
    for key,x in (('React',yr),('Vapor',yw)):
        D=np.where(an,config.hydrogen_diffusivity,config.oxygen_diffusivity if key=='React' else config.vapor_diffusivity_air)
        k=ct*D*(T/333.15)**1.5*101325/config.outlet_pressure*(eps*(1-s))**1.5
        bc=np.array([.21,1])*(1-yw_in) if key=='React' else np.array([yw_in,yw_in])
        expected=diff_flux(k,x,gas)+adv_flux(flux['Gas'],x,bc)
        # L1 divergence of the constitutive defect checks local conservation;
        # max face defect guards cancellation of oppositely corrupted faces.
        scale=feed_scale if key=='React' else water_scale
        checks[key.lower()+'_constitutive_divergence']=float(np.abs(div(expected-flux[key])).sum()/scale)
        checks[key.lower()+'_constitutive_face']=float(np.max(np.abs(expected-flux[key]))/scale)
    sigmaS=np.where(np.isin(mat,(0,8)),config.electronic_conductivity_bp,np.where(cl,config.electronic_conductivity_cl,config.electronic_conductivity_gdl))
    sigmaE=np.where(ion,(.5139*lam-.326)*np.exp(1268*(1/303-1/T))*np.where(mat==4,1,config.ionomer_fraction**1.5),1)
    qs=diff_flux(sigmaS,f['phiS'],electronic);qe=diff_flux(sigmaE,f['phiE'],ion)
    joule=np.zeros(nc)
    for q,x in ((qs,f['phiS']),(qe,f['phiE'])):
        power=q[inter]*(x[o[inter]]-x[n[inter]])
        joule+=np.bincount(o[inter],weights=power/2,minlength=nc)+np.bincount(n[inter],weights=power/2,minlength=nc)
    for patch,potential in (('cathodeTerminal',config.cell_voltage),('anodeTerminal',0)):
        idx=a['patches'][patch];delta=f['phiS'][o[idx]]-potential
        qs[idx]=sigmaS[o[idx]]*a['area'][idx]/a['do'][idx]*delta
        joule+=np.bincount(o[idx],weights=qs[idx]*delta,minlength=nc)
    current_scale=max(Ic,1e-10)
    checks['electronic_charge_balance']=float(np.abs(div(qs)+iv*v).sum()/current_scale)
    checks['protonic_charge_balance']=float(np.abs(div(qe)-iv*v).sum()/current_scale)
    eta=f['phiS']-f['phiE']-np.where(an,0,config.open_circuit_voltage)
    expected_heat=np.where(cl,iv*eta,0)+joule/v
    checks['reaction_joule_heat']=float(v@np.abs(expected_heat-f['heatSource'])/heat_scale)
    cref=np.where(an,1,.21)*(1-.9*saturation_pressure(333.15)/101325)*101325/(R*333.15)
    pref=np.where(an,config.exchange_current_anode,config.exchange_current_cathode)*np.exp(np.where(an,config.activation_energy_anode,config.activation_energy_cathode)/R*(1/config.kinetic_reference_temperature-1/T))
    arg=np.where(an,2,4)*config.transfer_coefficient*F/(R*T)*eta
    if np.max(np.abs(arg[cl]))>=50:raise ValueError('Kinetic exponential limiter is active')
    predicted_iv=np.zeros(nc);predicted_iv[cl]=2*pref[cl]*(1-s[cl])**1.5*((1-s[cl])*eps[cl]*ct[cl]*yr[cl]/cref[cl])**config.concentration_exponent*np.sinh(arg[cl])
    checks['butler_volmer']=float(v@np.abs(predicted_iv-iv)/current_scale)
    cap=config.membrane_dry_density/config.membrane_equivalent_weight*np.where(mat==4,1,config.ionomer_fraction)
    k=cap*config.membrane_diffusivity*np.exp(2416*(1/303-1/T))
    expected=diff_flux(k,lam,ion);idx=np.flatnonzero(inter&ion[o]&ion[np.maximum(n,0)])
    expected[idx]+=2.5*(lam[o[idx]]+lam[n[idx]])/(2*22)*qe[idx]/F
    checks['membrane_transport']=float(np.abs(div(expected-flux['Water'])).sum()/water_scale)
    activity=ct*yw/(np.array([saturation_pressure(t) for t in T])/(R*T))+2*s
    equilibrium=np.array([equilibrium_water(z) for z in activity]);sorption=np.where(cl,cap*config.sorption_rate*(equilibrium-lam),0)
    checks['sorption_closure']=float(v@np.abs(production+sorption-f['srcMembrane'])/water_scale)
    excess=ct*yw-np.array([saturation_pressure(t) for t in T])/(R*T)
    phase=np.where(gas,config.phase_change_rate*eps*np.where(excess>=0,excess,excess*s/(s+1e-4)),0)
    checks['phase_change_closure']=float(v@np.abs(phase-f['srcLiquid'])/water_scale)
    solid_k=np.array([100,0,1.5,1,.2,1,1.5,0,100])[mat]
    conductivity=np.where(gas,(1-eps)*solid_k+eps*((1-s)*np.where(an,.19,.028)+s*config.liquid_conductivity),solid_k)
    carrier=flux['Gas'];donor=np.where((carrier<0)&inter,n,o)
    heat_adv=carrier*(29+5*yw[donor])+flux['Liquid']*MW*4180
    expected=diff_flux(conductivity,T,np.ones(nc,dtype=bool))+adv_flux(heat_adv,T-g['temperature_K'],np.zeros(2))
    for patch,idx in a['patches'].items():
        if not patch.endswith(('Inlet','Outlet')):
            expected[idx]+=conductivity[o[idx]]*a['area'][idx]/a['do'][idx]*(T[o[idx]]-g['temperature_K'])
    checks['heat_transport']=float(np.abs(div(expected-flux['Heat'])).sum()/heat_scale)
    in_idx=a['patches']['cathodeInlet'];out_idx=a['patches']['cathodeOutlet']
    pin=values(block(texts['p'],'cathodeInlet'),'value',len(in_idx));pout=values(block(texts['p'],'cathodeOutlet'),'value',len(out_idx))
    for patch in ('cathodeOutlet','anodeOutlet'):
        idx=a['patches'][patch];checks[patch+'_pressure_bc']=float(np.max(np.abs(values(block(texts['p'],patch),'value',len(idx)))))
    dp=float(np.average(pin,weights=a['area'][in_idx])-np.average(pout,weights=a['area'][out_idx]))
    Q=-float(flux['Gas'][in_idx].sum())/(config.outlet_pressure/(R*g['temperature_K']))
    pump=dp*Q;net=config.cell_voltage*Ic-pump
    for patch,anode in (('anodeInlet',True),('cathodeInlet',False)):
        idx=a['patches'][patch];checks[patch+'_flow']=abs(-float(flux['Gas'][idx].sum())/feed(anode)-1)
    if min(dp,Q,net)<=0:raise ValueError('Nonpositive net power, pressure drop, or inlet flow')
    for key,value in (('net_power_W',net),('current_cathode_A',Ic),('current_anode_A',Ia),('pressure_drop_Pa',dp),('inlet_flow_m3_s',Q),('pumping_power_W',pump)):
        checks[key+'_history']=abs(value-last[key])/max(abs(value),1e-12)
    if any(not math.isfinite(z) or z>config.balance_tolerance for z in checks.values()):
        raise ValueError('P010 saved-field verification failed: '+repr({k:z for k,z in checks.items() if not math.isfinite(z) or z>config.balance_tolerance}))
    return dict(net_power_W=net,current_cathode_A=Ic,current_anode_A=Ia,pressure_drop_Pa=dp,
        inlet_flow_m3_s=Q,pumping_power_W=pump,mean_membrane_water=float(np.average(lam[mat==4],weights=v[mat==4])),
        mean_cathode_liquid_saturation=float(np.average(s[(mat==2)|(mat==3)],weights=v[(mat==2)|(mat==3)])),
        max_temperature_K=float(T.max()),cells=nc,iteration=int(last['iteration']),checks=checks,
        audit_scope='Saved local balances, charge, kinetics, species diffusion/advection, membrane transport, source identities, heat production and objective; momentum and liquid constitutive convergence also require solver histories. No mesh independence or paper validation.')

def extract_metrics(case,config):
    case=Path(case);manifest=json.loads((case/'case.json').read_text());root=case/'postProcessing/p010'
    status=json.loads((root/'status.json').read_text())
    with (root/'history.csv').open() as stream:rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    if len(rows)<5 or any(row['iteration']!=i+1 or not all(math.isfinite(v) for v in row.values()) for i,row in enumerate(rows)):
        raise ValueError('Missing, restarted or nonfinite P010 history')
    last=rows[-1];log=(case/'log.flowfieldP010Foam').read_text();latest=case/str(int(last['iteration']))
    if (status.get('converged') is not True or status.get('cells')!=manifest['cells'] or status.get('processes')!=1
        or status.get('iteration')!=last['iteration'] or 'CONVERGED: P010 full coupled model' not in log
        or not re.search(r'^End\s*$',log,re.M) or 'FOAM FATAL' in log or not latest.is_dir()):
        raise ValueError('P010 has not completed with qualified final fields')
    for row in rows[-5:]:
        if (row['iteration']<config.minimum_iterations or abs(row['voltage_V']-config.cell_voltage)>1e-12
            or not 0<=row['field_change']<config.field_tolerance or not 0<=row['charge_residual']<1e-7
            or any(not 0<=row[k]<config.balance_tolerance for k in BALANCES)):
            raise ValueError('Last five P010 iterations fail qualification')
    metrics=saved_audit(case,config,last,latest)
    return dict(objective=-metrics['net_power_W'],unpenalized_objective=-metrics['net_power_W'],metrics=metrics,
                constraints={},feasible=True,paper_validated=False)
