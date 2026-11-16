#!/usr/bin/env python3
"""Generate conformal full-cell meshes for the nine-channel counterflow cell. Standard library only.

All coordinates in geometry construction are mm; mesh output is SI.
Cells include the plates, channels, PTLs, CLs and membrane. No STL or CAD
dependency, and no optimization algorithm. Existing results are never deleted.
"""
import math
from collections import Counter
from pathlib import Path

ZONES = ('CBP', 'CCH', 'CPTL', 'CCL', 'PEM', 'ACL', 'APTL', 'ACH', 'ABP')
FLUID = {1, 2, 3, 5, 6, 7}
PATCHES = ('anodeInlet', 'anodeOutlet', 'cathodeInlet', 'cathodeOutlet',
           'anodeTerminal', 'cathodeTerminal', 'externalWalls')


def header(name, cls='dictionary'):
    return ('FoamFile\n{\n    version 2.0;\n    format ascii;\n'
            f'    class {cls};\n    object {name};\n}}\n\n')


def write(path, body, cls='dictionary'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header(path.name, cls) + body)


def subdivide(knots, spacing, *, max_ratio=None):
    """Keep every geometry boundary, optionally limiting adjacent cell widths."""
    knots = sorted(set(round(v, 10) for v in knots))
    out = [knots[0]]
    for a, b in zip(knots, knots[1:]):
        n = max(1, math.ceil((b-a)/spacing-1e-9))
        out.extend(a+(b-a)*i/n for i in range(1, n+1))
    if max_ratio is not None:
        if not math.isfinite(max_ratio) or max_ratio < 2:
            raise ValueError('max_ratio must be finite and at least 2')
        # A channel edge can lie close to a manifold edge. Refine its larger
        # neighbours instead of moving either physical boundary. Bisection
        # retains the maximum spacing and propagates grading outward as needed.
        while True:
            widths = [b-a for a, b in zip(out, out[1:])]
            split = {i for i, width in enumerate(widths)
                     if (i > 0 and width > max_ratio * widths[i-1])
                     or (i+1 < len(widths) and width > max_ratio * widths[i+1])}
            if not split:
                break
            refined = [out[0]]
            for i, (a, b) in enumerate(zip(out, out[1:])):
                if i in split:
                    refined.append((a+b)/2)
                refined.append(b)
            out = refined
    return out


def mesh(case, m):
    x, y, z = (m[v] for v in ('x', 'y', 'z'))
    nx, ny, nz = (len(v)-1 for v in (x, y, z))
    points, point_ids, internal = [], {}, []
    boundary = {p: [] for p in PATCHES}
    directions = [((-1,0,0), ((0,0,0),(0,0,1),(0,1,1),(0,1,0))),
                  ((1,0,0), ((1,0,0),(1,1,0),(1,1,1),(1,0,1))),
                  ((0,-1,0), ((0,0,0),(1,0,0),(1,0,1),(0,0,1))),
                  ((0,1,0), ((0,1,0),(0,1,1),(1,1,1),(1,1,0))),
                  ((0,0,-1), ((0,0,0),(0,1,0),(1,1,0),(1,0,0))),
                  ((0,0,1), ((0,0,1),(1,0,1),(1,1,1),(0,1,1)))]
    inlet_areas = Counter()
    inlet_face_areas = {p: [] for p in PATCHES if p.endswith('Inlet')}
    for c, (i, j, k) in enumerate(m['cells']):
        for side, ((di,dj,dk), corners) in enumerate(directions):
            ii, jj, kk = i+di, j+dj, k+dk
            nbr = (kk*ny+jj)*nx+ii if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz else -1
            if 0 <= nbr < c:
                continue
            face = []
            for a, b, d in corners:
                key = i+a, j+b, k+d
                if key not in point_ids:
                    point_ids[key] = len(points)
                    points.append(key)
                face.append(point_ids[key])
            if nbr >= 0:
                internal.append((face, c, nbr))
            else:
                p = 'externalWalls'
                if side == 4:
                    p = 'cathodeTerminal'
                elif side == 5:
                    p = 'anodeTerminal'
                elif m['ids'][c] in (1, 7):
                    electrode = 'anode' if m['ids'][c] == 7 else 'cathode'
                    cy = (y[j]+y[j+1])/2
                    if side == 0 and cy < m['manifold_width']:
                        p = electrode+('Inlet' if electrode == 'anode' else 'Outlet')
                    elif side == 1 and cy > m['length']-m['manifold_width']:
                        p = electrode+('Outlet' if electrode == 'anode' else 'Inlet')
                boundary[p].append((face, c, -1))
                if 'Inlet' in p:
                    area = ((x[i+1]-x[i]) if side in (2,3) else (y[j+1]-y[j]))*(z[k+1]-z[k])*1e-6
                    inlet_areas[p] += area
                    inlet_face_areas[p].append(area)
    poly = case/'constant/polyMesh'
    poly.mkdir(parents=True, exist_ok=True)
    write(poly/'points', str(len(points))+'\n(\n'+''.join(
        f'({x[i]*1e-3:.12g} {y[j]*1e-3:.12g} {z[k]*1e-3:.12g})\n' for i,j,k in points)+')\n', 'vectorField')
    all_faces = internal+[f for group in boundary.values() for f in group]
    write(poly/'faces', str(len(all_faces))+'\n(\n'+''.join('4('+ ' '.join(map(str,f))+')\n' for f,_,_ in all_faces)+')\n', 'faceList')
    write(poly/'owner', str(len(all_faces))+'\n(\n'+''.join(f'{o}\n' for _,o,_ in all_faces)+')\n', 'labelList')
    write(poly/'neighbour', str(len(internal))+'\n(\n'+''.join(f'{n}\n' for _,_,n in internal)+')\n', 'labelList')
    offset, btext = len(internal), str(len(PATCHES))+'\n(\n'
    for p, faces in boundary.items():
        typ = 'patch' if ('Inlet' in p or 'Outlet' in p) else 'wall'
        btext += f'{p}\n{{ type {typ}; nFaces {len(faces)}; startFace {offset}; }}\n'
        offset += len(faces)
    write(poly/'boundary', btext+')\n', 'polyBoundaryMesh')
    zonetext = '9\n(\n'
    for n, name in enumerate(ZONES):
        labels = [c for c, zone in enumerate(m['ids']) if zone == n]
        zonetext += f'{name}\n{{ type cellZone; cellLabels List<label> {len(labels)}\n(\n'+ '\n'.join(map(str,labels))+'\n); }\n'
    write(poly/'cellZones', zonetext+')\n', 'regIOobject')
    write(poly/'faceZones', '0\n(\n)\n', 'regIOobject')
    m['inlet_areas'] = dict(inlet_areas)
    m['inlet_face_areas'] = inlet_face_areas
    from ....mesh import PolyMesh
    return PolyMesh.from_case(case)


def field(case, name, dims, values, bcs, vector=False):
    if isinstance(values, list):
        typ = 'vector' if vector else 'scalar'
        value = f'nonuniform List<{typ}> {len(values)}\n(\n'+'\n'.join(map(str,values))+'\n)'
    else:
        value = f'uniform {values}'
    body = f'dimensions [{dims}];\ninternalField {value};\nboundaryField\n{{\n'
    body += ''.join(f'{p} {{ {bcs.get(p, "type zeroGradient;")} }}\n' for p in PATCHES)+'}\n'
    write(case/'0'/name, body, 'volVectorField' if vector else 'volScalarField')


def build_geometry(g, config):
    factor = {'coarse':1, 'medium':2, 'fine':3}[config.mesh]
    channels = [(c-w/2,c+w/2) for c,w in zip(g['centers_mm'],g['widths_mm'])]
    # Preserve topology throughout a width optimization: crossing 1 mm must
    # not suddenly add a cell and let the optimizer exploit remeshing noise.
    knots=[0]+[v for ch in channels for v in ch]+[18]
    x=[0.]
    for i,(a,b) in enumerate(zip(knots,knots[1:])):
        count=factor if i in (0,len(knots)-2) else 2*factor
        x.extend(a+(b-a)*j/count for j in range(1,count+1))
    w = config.manifold_width_mm
    y = subdivide([0,w,12.5,25-w,25], 1./factor, max_ratio=8)
    # 0.5 mm backing + 1 mm channel/rib on each side; 0.0178 mm PEM.
    edges = [0,.5,1.5,1.8,1.85,1.8678,1.9178,2.2178,3.2178,3.7178]
    counts = [1,2,2,1,2,1,2,2,1]
    z, layers = [0.], []
    for layer,(a,b),count in zip(range(9),zip(edges,edges[1:]),counts):
        count *= factor
        z.extend(a+(b-a)*i/count for i in range(1,count+1))
        layers.extend([layer]*count)
    cells, ids = [], []
    for k,layer in enumerate(layers):
        for j,(ya,yb) in enumerate(zip(y,y[1:])):
            for i,(xa,xb) in enumerate(zip(x,x[1:])):
                cx,cy=(xa+xb)/2,(ya+yb)/2
                channel=cy<w or cy>25-w or any(a<cx<b for a,b in channels)
                mat=layer
                if layer==1 and not channel: mat=0
                if layer==7 and not channel: mat=8
                cells.append((i,j,k));ids.append(mat)
    return dict(x=x,y=y,z=z,cells=cells,ids=ids,kind='parallel',width=18.,length=25.,
                area=450e-6,level=config.mesh,channels=channels,manifold_width=w)


def case_files(case, m, config):
    c=config
    vel,press,temp,sat={}, {}, {}, {}
    for patch in PATCHES:
        if 'Inlet' in patch:
            # Q is the liquid feed. With s_l=0.9, total volume flow is Q/s_l.
            speed=c.inlet_flow_ml_min*1e-6/60/c.inlet_liquid_saturation/m['inlet_areas'][patch]
            if patch.startswith('cathode'):speed=-speed
            vel[patch]=f'type fixedValue; value uniform ({speed:.12g} 0 0);'
            temp[patch]=f'type fixedValue; value uniform {c.inlet_temperature};'
            sat[patch]=f'type fixedValue; value uniform {1-c.inlet_liquid_saturation:.12g};'
        elif 'Outlet' in patch:
            vel[patch]='type zeroGradient;'
            press[patch]='type fixedValue; value uniform 0;'
        else:
            vel[patch]='type fixedValue; value uniform (0 0 0);'
    field(case,'Ul','0 1 -1 0 0 0 0','(0 0 0)',vel,True)
    field(case,'Ug','0 1 -1 0 0 0 0','(0 0 0)',vel,True)
    # Face fluxes are solver state as well as the cell velocities. Explicit
    # initial files allow a continuation to retain the conservative fluxes.
    for phase,ratio in (('L',1.),('G',(1-c.inlet_liquid_saturation)/c.inlet_liquid_saturation)):
        body='dimensions [0 3 -1 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n'
        for patch in PATCHES:
            value='uniform 0'
            if patch.endswith('Inlet'):
                flux=[-c.inlet_flow_ml_min*1e-6/60*ratio*a/m['inlet_areas'][patch]
                      for a in m['inlet_face_areas'][patch]]
                value=f'nonuniform List<scalar> {len(flux)}\n(\n'+'\n'.join(f'{v:.12g}' for v in flux)+'\n)'
            body+=f'{patch} {{ type calculated; value {value}; }}\n'
        write(case/'0'/('phi'+phase),body+'}\n','surfaceScalarField')
    field(case,'p','1 -1 -2 0 0 0 0',0,press)
    field(case,'T','0 0 0 1 0 0 0',c.inlet_temperature,temp)
    # Start porous layers liquid-filled to avoid imposing an artificial
    # capillary-pressure jump before gas is generated. The steady equations
    # and prescribed inlet saturation are unchanged.
    field(case,'sg','0 0 0 0 0 0 0',[1-c.inlet_liquid_saturation if i in (1,7)
          else 1e-6 if i in (2,3,5,6) else 0 for i in m['ids']],sat)
    field(case,'material','0 0 0 0 0 0 0',m['ids'],{})
    field(case,'lambda','0 0 0 0 0 0 0',c.membrane_water_boundary,{})
    field(case,'phiS','1 2 -3 0 0 -1 0',[c.cell_voltage if i in (5,6,8) else 0 for i in m['ids']],
          {'anodeTerminal':f'type fixedValue; value uniform {c.cell_voltage};',
           'cathodeTerminal':'type fixedValue; value uniform 0;'})
    field(case,'phiE','1 2 -3 0 0 -1 0',[.05 if i in (3,4,5) else 0 for i in m['ids']],{})
    props=c.to_dict();props.pop('mesh')
    props.update(reactionArea=m['area'],maxChargeNewton=40)
    write(case/'constant/modelProperties',''.join(f'{k} {v};\n' for k,v in props.items()))
    write(case/'system/controlDict',f'''application flowfieldP006Foam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {c.iterations};
deltaT 1;
writeControl timeStep;
writeInterval {c.write_interval};
purgeWrite 2;
writeFormat ascii;
writePrecision 12;
timeFormat general;
runTimeModifiable false;
''')
    write(case/'system/fvSchemes','''ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default Gauss upwind; }
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
fluxRequired { default no; p; sg; lambda; }
''')
    write(case/'system/fvSolution','''solvers
{
    p { solver PCG; preconditioner DIC; tolerance 1e-11; relTol 0; maxIter 6000; }
    "(Ul|Ug|qL|qG|sg|lambda|T)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-10; relTol 0; maxIter 4000; }
}
relaxationFactors { equations { qL 0.7; qG 0.7; } }
''')
    write(case/'system/decomposeParDict',f'numberOfSubdomains {c.processes};\nmethod scotch;\n')
    write(case/'system/meshQualityDict', """maxConcave 80;
minVol 1e-20;
minTetQuality 1e-30;
minArea -1;
minTwist 0.02;
minTriangleTwist -1;
minFaceFlatness -1;
nSmoothScale 4;
errorReduction 0.75;
// Thin, orthogonal 17.8 micron PEM: retain positive-volume/orthogonality checks.
minDeterminant 1e-8;
maxNonOrtho 1;
maxInternalSkewness 0.01;
maxBoundarySkewness 0.01;
minFaceWeight 0.05;
minVolRatio 0.01;
""")
    (case/'P006.foam').touch()
    return dict(cells=len(m['cells']),zone_cells=dict(Counter(ZONES[i] for i in m['ids'])),
                inlet_areas_m2=m['inlet_areas'],mesh_level=c.mesh)


