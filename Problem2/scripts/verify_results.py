#!/usr/bin/env python3
"""Check saved fields independently of the solver's runtime diagnostics.

Checks membrane current, terminal electrical power, integrated heat sources,
Faraday stoichiometry and gas leaving the outlet patches. No third-party deps.
"""
import argparse
import importlib.util
import json
import math
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('generate',ROOT/'scripts/generate_cases.py')
gen=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
F=96485.33212
R=8.314462618


def values(text,key,count):
    match=re.search(r'\b'+re.escape(key)+r'\s+uniform\s+([^;]+);',text)
    if match:
        return [float(match.group(1))]*count
    match=re.search(r'\b'+re.escape(key)+r'\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
    if not match or int(match.group(1))!=count:
        raise ValueError(f'Cannot read {key}, expected {count} values')
    result=[float(v) for v in match.group(2).split()]
    if len(result)!=count:
        raise ValueError('Truncated OpenFOAM field')
    return result


def verify(case):
    metadata=json.loads((case/'case.json').read_text())
    m=gen.geometry(metadata['parameters']['geometry'],metadata['geometry'],metadata['mesh_level'])
    p=metadata['parameters']['model']
    time=max((d for d in case.iterdir() if d.is_dir() and re.fullmatch(r'[\d.]+',d.name)),key=lambda d: float(d.name))
    fields={name:values((time/name).read_text(),'internalField',len(m['cells']))
            for name in ('phiS','phiE','T','Cgas','iVol','heatSource','massSource','gasMolarSource')}
    x,y,z=([v*1e-3 for v in m[k]] for k in ('x','y','z'))
    nx,ny,nz=(len(v)-1 for v in (x,y,z))
    sigma_e=[6.94*math.exp(1268*(1/303-1/t)) for t in fields['T']]
    sigma_s=p['electronConductivity']
    volumes=[]
    electron={0,2,3,5,6,8}
    ion={3,4,5}
    joule=0.0
    membrane_planes={}
    current_anode=current_cathode=reaction_power=heat=thermo=net_mass=source_o=source_h=0.0
    voltage_integral=0.0
    for c,((i,j,k),zone) in enumerate(zip(m['cells'],m['ids'])):
        lengths=[x[i+1]-x[i],y[j+1]-y[j],z[k+1]-z[k]]
        volume=math.prod(lengths)
        volumes.append(volume)
        iv=fields['iVol'][c]
        current_anode+=iv*volume if zone==5 else 0
        current_cathode-=iv*volume if zone==3 else 0
        net_mass+=fields['massSource'][c]*volume
        heat+=fields['heatSource'][c]*volume
        if zone in (3,5):
            source_o+=fields['gasMolarSource'][c]*volume if zone==5 else 0
            source_h+=fields['gasMolarSource'][c]*volume if zone==3 else 0
            reaction_power+=iv*(fields['phiS'][c]-fields['phiE'][c])*volume
            anode=zone==5
            cref=p['referenceConcentrationO2' if anode else 'referenceConcentrationH2']
            ratio=max(fields['Cgas'][c]/cref,p['concentrationFloorRatio'])
            eq=1.229-.0009*(fields['T'][c]-298.15) if anode else 0
            if p['includeNernstConcentration']:
                eq+=(1/4 if anode else -1/2)*R*fields['T'][c]/F*math.log(ratio)
            thermo+=iv*eq*volume
            if anode:
                thermo-=p['reversibleHeatSign']*iv*fields['T'][c]*(-.0009)*volume
        area=lengths[0]*lengths[1]
        if k==0:
            joule+=sigma_s*area/(lengths[2]/2)*fields['phiS'][c]**2
        if k==nz-1:
            joule+=p['currentDensity']**2*area*(lengths[2]/2)/sigma_s
            voltage_integral+=area*(fields['phiS'][c]+p['currentDensity']*lengths[2]/2/sigma_s)
        for axis,(di,dj,dk) in enumerate(((1,0,0),(0,1,0),(0,0,1))):
            ii,jj,kk=i+di,j+dj,k+dk
            if ii==nx or jj==ny or kk==nz:
                continue
            n=(kk*ny+jj)*nx+ii
            nzone=m['ids'][n]
            coords=(x,y,z)[axis]
            index=(i,j,k)[axis]
            do=(coords[index+1]-coords[index])/2
            dn=(coords[index+2]-coords[index+1])/2
            face_area=volume/lengths[axis]
            if zone in electron and nzone in electron:
                joule+=sigma_s*face_area/(do+dn)*(fields['phiS'][c]-fields['phiS'][n])**2
            if zone in ion and nzone in ion:
                conductance=face_area/(do/sigma_e[c]+dn/sigma_e[n])
                delta=fields['phiE'][n]-fields['phiE'][c]
                joule+=conductance*delta**2
                if axis==2 and zone==nzone==4:
                    membrane_planes[k]=membrane_planes.get(k,0)+conductance*delta
    current=p['currentDensity']*m['area']
    voltage=voltage_integral/m['area']
    power=current*voltage
    # Recover patch owner cells and use their zero-gradient outlet concentration.
    poly=case/'constant/polyMesh'
    owner_text=(poly/'owner').read_text().split('}',1)[1]
    owner=[int(v) for v in re.search(r'\d+\s*\((.*?)\)',owner_text,re.S).group(1).split()]
    boundary=(poly/'boundary').read_text()
    phi=(time/'phi').read_text()
    gas_out={}
    for patch in ('anodeOutlet','cathodeOutlet'):
        block=re.search(r'\b'+patch+r'\s*\{(.*?)\}',boundary,re.S).group(1)
        count=int(re.search(r'nFaces\s+(\d+)',block).group(1))
        start=int(re.search(r'startFace\s+(\d+)',block).group(1))
        phi_block=re.search(r'\b'+patch+r'\s*\{(.*?)\}',phi,re.S).group(1)
        flux=values(phi_block,'value',count)
        gas_out[patch]=sum(f*fields['Cgas'][owner[start+n]] for n,f in enumerate(flux))
    checks={
        'anode_current_relative_error':abs(current_anode/current-1),
        'cathode_current_relative_error':abs(current_cathode/current-1),
        'membrane_plane_current_relative_error':max(abs(value/current-1) for value in membrane_planes.values()),
        'electrical_power_identity_relative_error':abs(power-joule-reaction_power)/power,
        'heat_source_identity_relative_error':abs(power-heat-thermo)/power,
        'gas_stoichiometry_relative_error':abs(source_h/source_o/2-1),
        'oxygen_out_relative_error':abs(gas_out['anodeOutlet']/(current/(4*F))-1),
        'hydrogen_out_relative_error':abs(gas_out['cathodeOutlet']/(current/(2*F))-1),
        'net_reaction_mass_relative_error':abs(net_mass)/(current*.00201588/(2*F))
    }
    result={'passed':max(checks.values())<1e-3,'iteration':float(time.name),
            'voltage_V':voltage,'electrical_power_W':power,'joule_heat_W':joule,
            'total_heat_source_W':heat,'membrane_currents_A':membrane_planes,'checks':checks}
    (case/'field_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path)
    args=parser.parse_args()
    result=verify(args.case.resolve())
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 2)
