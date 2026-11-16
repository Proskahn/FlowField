"""Independent structured-cell pressure-force audit (no OpenFOAM operators)."""
import math

from .geometry import FLUID


def pressure_forces(mesh, config, pressure, saturation):
    axes=[[v*.001 for v in mesh[a]] for a in ('x','y','z')]
    shape=[len(a)-1 for a in axes]
    strides=(1,shape[0],shape[0]*shape[1])
    widths=[[b-a for a,b in zip(axis,axis[1:])] for axis in axes]
    ids=mesh['ids'];n=len(ids)
    result={phase:[[0.,0.,0.] for _ in ids] for phase in ('L','G')}
    permeability=[];pc=[]
    for material,s in zip(ids,saturation):
        k=(config.permeability_channel if material in (1,7) else
           config.permeability_ptl if material in (2,6) else config.permeability_cl)
        permeability.append(k)
        cap=0.
        if material in (2,3,5,6):
            eps=config.porosity_ptl if material in (2,6) else config.porosity_cl
            cosine=math.cos(math.radians(config.contact_angle_deg))
            z=s if cosine>0 else 1-s
            cap=config.surface_tension*cosine*math.sqrt(eps/k)*(1.417*z-2.120*z*z+1.263*z**3)
        pc.append(cap)
    for phase in ('L','G'):
        p=[v+(pc[c] if phase=='G' else 0) for c,v in enumerate(pressure)]
        mobility=[]
        for c,material in enumerate(ids):
            mu=(config.liquid_viscosity if phase=='L' else
                config.oxygen_viscosity if material>=5 else config.hydrogen_viscosity)
            fraction=1-saturation[c] if phase=='L' else saturation[c]
            mobility.append(permeability[c]*max(fraction,1e-6)**3/mu)
        for c,(ijk,material) in enumerate(zip(mesh['cells'],ids)):
            if material not in FLUID:continue
            for axis in range(3):
                face_pressure=[]
                for direction in (-1,1):
                    coordinate=ijk[axis]+direction
                    face=p[c]
                    if 0<=coordinate<shape[axis]:
                        neighbour=c+direction*strides[axis]
                        if ids[neighbour] in FLUID:
                            own=widths[axis][ijk[axis]]/2
                            other=widths[axis][coordinate]/2
                            if ids[neighbour]!=material:
                                own/=mobility[c];other/=mobility[neighbour]
                            face=(other*p[c]+own*p[neighbour])/(own+other)
                    elif axis==0:
                        y=(mesh['y'][ijk[1]]+mesh['y'][ijk[1]+1])/2
                        outlet=(material==7 and direction==1 and y>25-mesh['manifold_width']) or (
                            material==1 and direction==-1 and y<mesh['manifold_width'])
                        if outlet:face=0.
                    face_pressure.append(face)
                result[phase][c][axis]=-(face_pressure[1]-face_pressure[0])/widths[axis][ijk[axis]]
    return result


def verify_pressure_forces(mesh,config,pressure,saturation,saved):
    expected=pressure_forces(mesh,config,pressure,saturation)
    worst=0.
    for phase in ('L','G'):
        for material in FLUID:
            error=reference=0.
            for c,m in enumerate(mesh['ids']):
                if m!=material:continue
                error+=sum((a-b)**2 for a,b in zip(saved[phase][c],expected[phase][c]))
                reference+=sum(a*a for a in expected[phase][c])
            worst=max(worst,math.sqrt(error)/max(math.sqrt(reference),1.))
    if not math.isfinite(worst) or worst>1e-6:
        raise ValueError(f'Saved physical pressure force is inconsistent: {worst}')
    return worst
