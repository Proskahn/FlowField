"""Exact stepped cross section, square floor gas inlet, and conformal hex blocks."""
import math


def decode(x, config):
    hb=(.5+1.5*x[0])*1e-3
    hd=(.5+3.5*x[1])*1e-3
    return dict(normalized_design=list(x),base_height_m=hb,degassing_height_m=hd,
        total_height_m=hb+hd,base_width_m=config.base_width_m,
        degassing_width_m=config.degassing_width_m,length_m=config.length_m,
        volume_m3=config.length_m*(hb*config.base_width_m+hd*config.degassing_width_m),
        inlet_area_m2=hb*config.base_width_m+hd*config.degassing_width_m,
        gas_inlet_area_m2=config.gas_inlet_side_m**2,
        gas_flow_m3_s=config.oxygen_velocity_m_s*config.gas_inlet_side_m**2,
        oxygen_velocity_m_s=config.oxygen_velocity_m_s)


def block_mesh(g,c):
    w,b,d=c.degassing_width_m,c.base_width_m,c.gas_inlet_side_m
    x=[-w/2,-b/2,-d/2,d/2,b/2,w/2]
    y=[0,g['base_height_m'],g['total_height_m']]
    z=[0,c.gas_inlet_z_m-d/2,c.gas_inlet_z_m+d/2,c.length_m]
    r=c.refinement
    nx=[max(1,math.ceil((x[i+1]-x[i])/(1e-4/r))) for i in range(5)]
    nx[2]=2*r
    ny=[max(2,math.ceil((y[i+1]-y[i])/(1e-4/r))) for i in range(2)]
    nz=[max(2,math.ceil((z[1]-z[0])/(1e-4/r))),2*r,max(2,math.ceil((z[3]-z[2])/(4e-4/r)))]
    blocks=[];verts=[];vmap={};faces={};cells=0
    # Conventional blockMesh vertex order; shared vertices and equal edge counts.
    corners=((0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1))
    sides=((0,4,7,3),(1,2,6,5),(0,1,5,4),(3,7,6,2),(0,3,2,1),(4,5,6,7))
    for k in range(3):
        for j in range(2):
            for i in (range(1,4) if j==0 else range(5)):
                ids=[]
                for a,b1,d1 in corners:
                    key=(i+a,j+b1,k+d1)
                    if key not in vmap: vmap[key]=len(verts);verts.append((x[key[0]],y[key[1]],z[key[2]]))
                    ids.append(vmap[key])
                blocks.append((ids,(nx[i],ny[j],nz[k])))
                cells+=nx[i]*ny[j]*nz[k]
                for side,cc in enumerate(sides):
                    face=tuple(ids[v] for v in cc);key=tuple(sorted(face))
                    if key in faces: del faces[key]
                    else:
                        patch='walls'
                        if side==4 and k==0:patch='waterInlet'
                        elif side==5 and k==2:patch='outlet'
                        elif side==2 and j==0:
                            patch='gasInlet' if i==2 and k==1 else 'electrode'
                        faces[key]=(face,patch)
    patches={name:[] for name in ('waterInlet','gasInlet','outlet','electrode','walls')}
    for face,patch in faces.values():patches[patch].append(face)
    fmt=lambda values:'('+' '.join(map(str,values))+')'
    body='convertToMeters 1;\nvertices\n(\n'+''.join(fmt(v)+'\n' for v in verts)+');\nblocks\n(\n'
    body+=''.join('hex '+fmt(ids)+' '+fmt(n)+' simpleGrading (1 1 1)\n' for ids,n in blocks)+');\nedges ();\nboundary\n(\n'
    for patch,ff in patches.items():
        typ='wall' if patch in ('electrode','walls') else 'patch'
        body+=patch+' { type '+typ+'; faces ('+' '.join(fmt(f) for f in ff)+'); }\n'
    body+=');\nmergePatchPairs ();\n'
    # Explicit capillary time scale supplements the advective Co constraint.
    hmin=min(
        [(x[i+1]-x[i])/nx[i] for i in range(5)]+[(y[i+1]-y[i])/ny[i] for i in range(2)]+[(z[i+1]-z[i])/nz[i] for i in range(3)])
    capillary_dt=.2*math.sqrt((c.water_density_kg_m3+c.oxygen_density_kg_m3)*hmin**3/(4*math.pi*c.surface_tension_n_m))
    return body,dict(cells=cells,blocks=len(blocks),minimum_cell_width_m=hmin,
        maximum_delta_t_s=min(c.max_delta_t_s,capillary_dt),gas_inlet_faces=nx[2]*nz[1],
        expected_volume_m3=g['volume_m3'],gas_inlet_area_m2=d*d)
