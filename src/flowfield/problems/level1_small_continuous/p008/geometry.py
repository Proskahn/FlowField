#!/usr/bin/env python3
"""Generate conformal full-cell meshes for the parallel-channel AEM cell. Standard library only.

All coordinates in geometry construction are mm; mesh output is SI.
Cells include the plates, channels, PTLs, CLs and membrane. No STL or CAD
dependency, and no optimization algorithm. Existing results are never deleted.
"""
import math
from collections import Counter
from pathlib import Path

ZONES = ('CBP', 'CCH', 'CPTL', 'CCL', 'AEM', 'ACL', 'APTL', 'ACH', 'ABP')
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
                    if side == 2: p = electrode+'Inlet'
                    elif side == 3: p = electrode+'Outlet'
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


def build_geometry(g,config):
    chx,ribx,ny,zcounts={'coarse':(2,1,12,(1,3,2,8,2,8,2,3,1)),
        'medium':(3,2,20,(2,4,3,12,3,12,3,4,2)),
        'fine':(4,3,28,(3,6,4,16,4,16,4,6,3))}[config.mesh]
    channels=[(c-g['width_mm']/2,c+g['width_mm']/2) for c in g['centers_mm']]
    knots=[0]+[v for a,b in channels for v in (a,b)]+[20.]
    x=[0.]
    for i,(a,b) in enumerate(zip(knots,knots[1:])):
        count=chx if i%2 else ribx
        x.extend(a+(b-a)*j/count for j in range(1,count+1))
    y=[20*j/ny for j in range(ny+1)]
    d=g['depth_mm'];widths=[2-d,d,.4,.01,.04,.01,.4,d,2-d]
    z=[0.];layers=[]
    for layer,(width,count) in enumerate(zip(widths,zcounts)):
        a=z[-1]
        z.extend(a+width*j/count for j in range(1,count+1));layers.extend([layer]*count)
    cells=[];ids=[]
    for k,layer in enumerate(layers):
        for j in range(len(y)-1):
            for i,(a,b) in enumerate(zip(x,x[1:])):
                channel=any(lo<(a+b)/2<hi for lo,hi in channels)
                mat=0 if layer==1 and not channel else 8 if layer==7 and not channel else layer
                cells.append((i,j,k));ids.append(mat)
    return dict(x=x,y=y,z=z,cells=cells,ids=ids,channels=channels,area=400e-6)


def case_files(case,m,c):
    vel={};press={};temp={};sat={}
    for patch in PATCHES:
        if 'Inlet' in patch:
            speed=c.inlet_flow_ml_min*1e-6/60/m['inlet_areas'][patch]
            vel[patch]=f'type fixedValue; value uniform (0 {speed:.14g} 0);'
            press[patch]='type calculated; value uniform 0;'
            temp[patch]=f'type fixedValue; value uniform {c.inlet_temperature};'
            sat[patch]='type fixedValue; value uniform 0;'
        elif 'Outlet' in patch:
            vel[patch]='type zeroGradient;';press[patch]='type fixedValue; value uniform 0;'
        else:
            vel[patch]='type fixedValue; value uniform (0 0 0);'
            temp[patch]=f'type calculated; value uniform {c.inlet_temperature};'
    for name in ('U','Ul','Ug'):field(case,name,'0 1 -1 0 0 0 0','(0 0 0)',vel,True)
    field(case,'p','1 -1 -2 0 0 0 0',0,press)
    field(case,'T','0 0 0 1 0 0 0',c.inlet_temperature,temp)
    field(case,'sg','0 0 0 0 0 0 0',[.02 if i in FLUID else 0 for i in m['ids']],sat)
    field(case,'material','0 0 0 0 0 0 0',m['ids'],{})
    field(case,'phiS','1 2 -3 0 0 -1 0',[c.cell_voltage if i in (5,6,8) else 0 for i in m['ids']],
        {'anodeTerminal':f'type fixedValue; value uniform {c.cell_voltage};','cathodeTerminal':'type fixedValue; value uniform 0;'})
    field(case,'phiE','1 2 -3 0 0 -1 0',[.05 if i in (3,4,5) else 0 for i in m['ids']],{})
    props={k:v for k,v in c.to_dict().items() if k not in ('mesh','ewm_calibration')}
    props.update(reactionArea=m['area'],maxChargeNewton=60)
    write(case/'constant/modelProperties',''.join(f'{k} {v};\n' for k,v in props.items()))
    write(case/'system/controlDict',f'''application flowfieldP008Foam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {c.iterations};
deltaT 1;
writeControl timeStep;
writeInterval {c.write_interval};
purgeWrite 2;
writeFormat ascii;
writePrecision 14;
runTimeModifiable false;
''')
    write(case/'system/fvSchemes','''ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default Gauss upwind; }
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
fluxRequired { default no; }
''')
    write(case/'system/fvSolution','solvers {}\n')
    write(case/'system/meshQualityDict','''maxNonOrtho 1;
maxBoundarySkewness .01;
maxInternalSkewness .01;
maxConcave 80;
minVol 1e-20;
minTetQuality 1e-30;
minArea -1;
minTwist .02;
minTriangleTwist -1;
minFaceWeight .001;
minVolRatio .001;
minDeterminant 1e-8;
''')
    (case/'P008.foam').touch()
    return dict(cells=len(m['cells']),zone_cells=dict(Counter(ZONES[i] for i in m['ids'])),
        inlet_areas_m2=m['inlet_areas'],mesh_shape=[len(m[k])-1 for k in ('x','y','z')])
