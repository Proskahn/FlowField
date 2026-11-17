"""Independent saved-field and flux checks; dimensional raw objectives and explicit IOH penalty."""
import csv
import json
import math
from pathlib import Path
import re
from .config import current_density
from .design import stencil

COLUMNS=('iteration,mean_current_A_m2,std_current_A_m2,power_domain_W,inlet_flow_m3_s,outlet_flow_m3_s,'
         'volume_error,oxygen_error,water_error,field_change,min_oxygen,max_water,min_nitrogen,closure_net_mass_kg_s').split(',')


def values(text,key,count):
    match=re.search(r'\b'+re.escape(key)+r'\s+uniform\s+([^;]+);',text)
    if match:result=[float(match.group(1))]*count
    else:
        match=re.search(r'\b'+re.escape(key)+r'\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
        if not match or int(match.group(1))!=count:raise ValueError(f'Cannot read {key}: expected {count} values')
        result=[float(v) for v in match.group(2).split()]
    if len(result)!=count or not all(math.isfinite(v) for v in result):raise ValueError('Malformed or nonfinite field')
    return result


def scalarize(power,std,volume,config):
    if not all(math.isfinite(v) for v in (power,std,volume)) or power<=0 or std<0 or not 0<=volume<=1:
        raise ValueError('Invalid physical objectives')
    raw=config.uniformity_weight*std/config.current_std_scale_a_m2-config.power_weight*power/config.power_scale_w
    g=volume-config.volume_limit
    return dict(objective=raw+config.volume_penalty*max(g,0.),
                objectives={'negative_power_domain_W':-power,'current_std_A_m2':std},
                constraints={'channel_volume':g},unpenalized_objective=raw,feasible=g<=1e-12)


def history(case):
    files=list((case/'postProcessing/p003').glob('history_*.csv'))
    if len(files)!=1 or files[0].name!='history_0.csv':raise ValueError('Expected one fresh history')
    with files[0].open(newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=COLUMNS:raise ValueError('Unexpected history columns')
        rows=[]
        for row in reader:
            if set(row)!=set(COLUMNS):raise ValueError('Malformed history')
            row={k:float(v) for k,v in row.items()}
            if not all(math.isfinite(v) for v in row.values()) or row['iteration']!=len(rows)+1:
                raise ValueError('Nonfinite, restarted or incomplete history')
            rows.append(row)
    if len(rows)<5:raise ValueError('Insufficient convergence history')
    return rows


def extract_metrics(case,config,*,write_report=True):
    case=Path(case);manifest=json.loads((case/'case.json').read_text())
    status=json.loads((case/'postProcessing/p003/status.json').read_text());rows=history(case);last=rows[-1]
    log=(case/'log.flowfieldP003Foam').read_text()
    times=[d for d in case.iterdir() if d.is_dir() and re.fullmatch(r'\d+(?:\.\d+)?',d.name)]
    latest=max(times,key=lambda d:float(d.name))
    if (status.get('converged') is not True or status.get('processes')!=1 or status.get('cells')!=manifest['cells']
        or status.get('iteration')!=last['iteration'] or float(latest.name)!=last['iteration']
        or 'CONVERGED: P003 fields, volume and transported-species balances passed.' not in log
        or not re.search(r'^End\s*$',log,re.M)):
        raise ValueError('Unconverged, stale or non-serial result')
    for r in rows[-5:]:
        if (r['iteration']<30 or not 0<=r['field_change']<config.outer_tolerance
            or any(not 0<=r[k]<config.balance_tolerance for k in ('volume_error','oxygen_error','water_error'))):
            raise ValueError('Last five iterations fail convergence')
    _,areas=stencil(config);n=len(areas);area=sum(areas)
    fields={k:values((latest/k).read_text(),'internalField',n) for k in ('oxygen','water','nitrogen','currentDensity','porosity','diffusionFactor','brinkman')}
    geometry=json.loads((case/'design.json').read_text())
    for name,key in [('porosity','porosity'),('diffusionFactor','diffusivity_factor'),('brinkman','brinkman_pa_s_m2')]:
        if max(abs(a-b)/max(abs(b),1) for a,b in zip(fields[name],geometry[key]))>1e-10:
            raise ValueError('Saved material properties differ from the design')
    for name in ('oxygen','water','nitrogen'):
        if not all(0<=v<=1 for v in fields[name]):raise ValueError('Unphysical species values')
    if max(abs(o+w+n-1) for o,w,n in zip(fields['oxygen'],fields['water'],fields['nitrogen']))>1e-10:
        raise ValueError('Species fractions do not sum to one')
    j=fields['currentDensity'];expected=[current_density(o,config) for o in fields['oxygen']]
    kinetic_error=max(abs(a-b)/max(b,1e-12) for a,b in zip(j,expected))
    current=sum(a*v for a,v in zip(areas,j));mean=current/area
    std=math.sqrt(sum(a*(v-mean)**2 for a,v in zip(areas,j))/area);power=config.cell_voltage_v*current
    poly=case/'constant/polyMesh';boundary=(poly/'boundary').read_text()
    fluxes={}
    for name in ('phi','oxygenFlux','waterFlux'):
        text=(latest/name).read_text();fluxes[name]={}
        expected_dims='0 3 -1 0 0 0 0' if name=='phi' else '1 0 -1 0 0 0 0'
        if not re.search(r'dimensions\s*\['+expected_dims+r'\]',text):raise ValueError('Wrong saved flux units')
        for patch in ('inlet','outlet','walls','symmetry'):
            block=re.search(r'\b'+patch+r'\s*\{(.*?)\}',boundary,re.S).group(1)
            count=int(re.search(r'nFaces\s+(\d+)',block).group(1))
            fblock=re.search(r'\b'+patch+r'\s*\{(.*?)\}',text,re.S).group(1)
            fluxes[name][patch]=sum(values(fblock,'value',count))
    qin=-fluxes['phi']['inlet'];qout=fluxes['phi']['outlet']
    if qin<=0 or qout<=0:raise ValueError('Flow direction is reversed')
    pressure=(latest/'p').read_text()
    if not re.search(r'dimensions\s*\[1 -1 -2 0 0 0 0\]',pressure):
        raise ValueError('Pressure must be saved in Pa')
    port_pressure={}
    for patch in ('inlet','outlet'):
        block=re.search(r'\b'+patch+r'\s*\{(.*?)\}',boundary,re.S).group(1)
        count=int(re.search(r'nFaces\s+(\d+)',block).group(1))
        pblock=re.search(r'\b'+patch+r'\s*\{(.*?)\}',pressure,re.S).group(1)
        pv=values(pblock,'value',count)
        prescribed=config.inlet_pressure_pa if patch=='inlet' else 0.
        if max(abs(v-prescribed) for v in pv)>1e-10*config.inlet_pressure_pa:
            raise ValueError('Saved pressure boundary differs from the prescribed condition')
        port_pressure[patch]=sum(pv)/len(pv)
    consO=.032*current/(4*96485.3);prodW=2*.018*current/(4*96485.3)
    checks=dict(kinetic_relative_error=kinetic_error,volume_relative_error=abs(qin-qout)/qin,
        oxygen_relative_error=abs(sum(fluxes['oxygenFlux'].values())+consO)/consO,
        water_relative_error=abs(sum(fluxes['waterFlux'].values())-prodW)/prodW,
        impermeable_boundary_relative_error=max(abs(fluxes['phi'][k]) for k in ('walls','symmetry'))/qin,
        history_power_relative_error=abs(last['power_domain_W']/power-1),
        history_std_relative_error=abs(last['std_current_A_m2']-std)/max(std,1),
        history_mean_relative_error=abs(last['mean_current_A_m2']/mean-1))
    if not all(math.isfinite(v) and v<config.balance_tolerance for v in checks.values()):
        raise ValueError(f'Independent field checks failed: {checks}')
    result=scalarize(power,std,geometry['volume_fraction'],config)
    result['metrics']=dict(mean_current_A_m2=mean,current_std_A_m2=std,power_domain_W=power,
        power_mirrored_full_cell_W=2*power,inlet_flow_half_cell_m3_s=qin,outlet_flow_half_cell_m3_s=qout,
        prescribed_pressure_drop_Pa=config.inlet_pressure_pa,
        saved_pressure_drop_Pa=port_pressure['inlet']-port_pressure['outlet'],volume_fraction=geometry['volume_fraction'],
        min_oxygen=min(fields['oxygen']),max_water=max(fields['water']),min_nitrogen=min(fields['nitrogen']),
        reaction_net_mass_kg_s=prodW-consO,
        reaction_net_mass_relative_to_inlet=(prodW-consO)/(config.gas_density_kg_m3*qin),
        nitrogen_transport_equation_solved=False,checks=checks,iteration=last['iteration'],cells=n)
    if write_report:
        (case/'field_verification.json').write_text(json.dumps(dict(passed=True,**result['metrics']),indent=2,allow_nan=False)+'\n')
    return result
