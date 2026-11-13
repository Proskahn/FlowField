#!/usr/bin/env python3
"""Generate orthogonal, conformal full-cell meshes. Standard library only.

All coordinates in geometry construction are mm; mesh output is SI.
Cells include the plates, channels, PTLs, CLs and membrane. No STL or CAD
dependency, and no optimization algorithm. Existing results are never deleted.
"""
import json
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


def subdivide(knots, spacing):
    knots = sorted(set(round(v, 10) for v in knots))
    out = [knots[0]]
    for a, b in zip(knots, knots[1:]):
        n = max(1, math.ceil((b-a)/spacing-1e-9))
        out.extend(a+(b-a)*i/n for i in range(1, n+1))
    return out


def geometry(g, kind, level):
    factor = {'coarse': 1, 'medium': 2, 'fine': 3}[level]
    if kind != 'parallel':
        raise ValueError('P002 uses the complete parallel cell only')
    widths = g['channel_widths_mm']
    ha, hc, b = (g[k] for k in ('anode_height_mm', 'cathode_height_mm', 'plate_total_thickness_mm'))
    pt, cl, pm = (g[k] for k in ('ptl_thickness_mm', 'cl_thickness_mm', 'membrane_thickness_mm'))
    if len(widths) != 5 or any(not math.isfinite(w) or not .7 <= w <= 1.9 for w in widths):
        raise ValueError('Five channel widths must lie in [0.7, 1.9] mm')
    if not (.5 <= ha <= 1.4 and .5 <= hc <= 1.4 and b > max(ha, hc)):
        raise ValueError('Channel heights must lie in [0.5, 1.4] mm with a conducting backing')
    W = L = g['parallel_size_mm']
    if W != 11 or g['parallel_channel_length_mm'] != 8 or min(pt, cl, pm) <= 0:
        raise ValueError('P002 requires 11 mm plates, 8 mm branches and positive layer thicknesses')
    channels = [(1.5+2*i-w/2, 1.5+2*i+w/2) for i,w in enumerate(widths)]
    x = subdivide([0, 1, 10, W]+[v for c in channels for v in c], .5/factor)
    y = subdivide([0, .5, 1.5, 9.5, 10.5, L], .5/factor)
    def in_channel(cx, cy):
        return ((.5 < cy < 1.5 and cx < 10)
                or (9.5 < cy < 10.5 and cx > 1)
                or (1.5 < cy < 9.5 and any(a < cx < d for a, d in channels)))
    # Bottom-to-top: cathode plate/channel, CPTL, CCL, membrane, ACL, APTL, anode.
    edges = [0, b-hc, b, b+pt, b+pt+cl, b+pt+cl+pm,
             b+pt+2*cl+pm, b+2*pt+2*cl+pm,
             b+2*pt+2*cl+pm+ha, 2*b+2*pt+2*cl+pm]
    spacings = [.25/factor, .25/factor, .1/factor, cl/(factor+1),
                pm/(2*factor), cl/(factor+1), .1/factor, .25/factor, .25/factor]
    z, layers = [0], []
    for k, (lo, hi) in enumerate(zip(edges, edges[1:])):
        vals = subdivide([lo, hi], spacings[k])
        z.extend(vals[1:])
        layers.extend([k]*(len(vals)-1))
    cells, ids = [], []
    for k, layer in enumerate(layers):
        for j in range(len(y)-1):
            for i in range(len(x)-1):
                cid = layer
                if layer == 1:
                    cid = 1 if in_channel((x[i]+x[i+1])/2, (y[j]+y[j+1])/2) else 0
                elif layer == 7:
                    cid = 7 if in_channel((x[i]+x[i+1])/2, (y[j]+y[j+1])/2) else 8
                cells.append((i, j, k))
                ids.append(cid)
    return dict(x=x, y=y, z=z, cells=cells, ids=ids, kind=kind, width=W, length=L,
                area=W*L*1e-6, level=level, channels=channels)


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
                    if (m['kind'] == 'single' and side == 2) or (m['kind'] == 'parallel' and side == 0):
                        p = electrode+'Inlet'
                    elif (m['kind'] == 'single' and side == 3) or (m['kind'] == 'parallel' and side == 1):
                        p = electrode+'Outlet'
                boundary[p].append((face, c, -1))
                if 'Inlet' in p:
                    inlet_areas[p] += ((x[i+1]-x[i]) if side in (2,3) else (y[j+1]-y[j]))*(z[k+1]-z[k])*1e-6
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


def case_files(case, m, config, iterations):
    p = dict(config['model'])
    temperature = p['inletTemperature']
    vel, press, temp, gas = {}, {}, {}, {}
    for patch in PATCHES:
        if 'Inlet' in patch:
            speed = p['inletFlow_mL_min']*1e-6/60/m['inlet_areas'][patch]
            v = f'(0 {speed:.12g} 0)' if m['kind'] == 'single' else f'({speed:.12g} 0 0)'
            vel[patch] = f'type fixedValue; value uniform {v};'
            gas[patch] = 'type fixedValue; value uniform 0;'
            temp[patch] = f'type fixedValue; value uniform {temperature};'
        elif 'Outlet' in patch:
            vel[patch] = 'type zeroGradient;'
            press[patch] = 'type fixedValue; value uniform 0;'
        else:
            vel[patch] = 'type fixedValue; value uniform (0 0 0);'
    field(case, 'U', '0 1 -1 0 0 0 0', '(0 0 0)', vel, True)
    field(case, 'p', '1 -1 -2 0 0 0 0', 0, press)
    field(case, 'T', '0 0 0 1 0 0 0', temperature, temp)
    field(case, 'Cgas', '0 -3 0 0 1 0 0', [5 if i in FLUID else 0 for i in m['ids']], gas)
    field(case, 'material', '0 0 0 0 0 0 0', m['ids'], {})
    field(case, 'phiS', '1 2 -3 0 0 -1 0', [1.8 if i in (5,6,8) else 0 for i in m['ids']], {})
    # Initial ionic potential only; the coupled Newton solver computes the actual field.
    field(case, 'phiE', '1 2 -3 0 0 -1 0', [.15 if i in (4,5) else 0 for i in m['ids']], {})
    p['reactionArea'] = m['area']
    p['catalystThickness'] = config['geometry']['cl_thickness_mm']*1e-3
    text = '// SI units, except explicitly named inletFlow_mL_min. See docs/model.md.\n'
    for key, value in p.items():
        if isinstance(value, bool):
            value = str(value).lower()
        text += f'{key} {value};\n'
    write(case/'constant/pemProperties', text)
    write(case/'system/controlDict', f'''application flowfieldP002Foam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {iterations};
deltaT 1;
writeControl timeStep;
writeInterval {config['write_interval']};
purgeWrite 2;
writeFormat ascii;
writePrecision 12;
timeFormat general;
runTimeModifiable false;
''')
    write(case/'system/fvSchemes', '''ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(massPhi,U) Gauss upwind;
    div(phi,Cgas) Gauss upwind;
    div(heatPhi,T) Gauss upwind;
}
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
fluxRequired { default no; p; }
''')
    write(case/'system/fvSolution', '''solvers
{
    p { solver PCG; preconditioner DIC; tolerance 1e-10; relTol 0.001; maxIter 4000; }
    U { solver PBiCGStab; preconditioner DILU; tolerance 1e-10; relTol 0.001; maxIter 2000; }
    Cgas { solver PBiCGStab; preconditioner DILU; tolerance 1e-11; relTol 0.0001; maxIter 3000; }
    T { solver PBiCGStab; preconditioner DILU; tolerance 1e-11; relTol 0.0001; maxIter 4000; }
}
relaxationFactors { equations { U 0.7; } }
''')
    write(case/'system/decomposeParDict', '''numberOfSubdomains 4;
method scotch;
''')
    (case/(case.name+'.foam')).touch()
    report = dict(geometry=m['kind'], mesh_level=m['level'], cells=len(m['ids']),
                  zone_cells={ZONES[i]: n for i, n in sorted(Counter(m['ids']).items())},
                  reaction_area_m2=m['area'], current_A=m['area']*p['currentDensity'],
                  inlet_areas_m2=m['inlet_areas'], parameters=config)
    return report

