"""Conformal 2D finite-volume mesh, physical ports fitted exactly; coordinates in SI."""
from .design import grid
PATCHES = ('inlet','outlet','symmetry','walls','frontBack')


def mesh_description(config):
    x,y = grid(config)
    return dict(x=x,y=y,z=[0.,config.depth_m],
                cells=[(i,j,0) for j in range(len(y)-1) for i in range(len(x)-1)],
                inlet_width=config.inlet_width_m,outlet_width=config.outlet_width_m)


def header(name, cls='dictionary'):
    return ('FoamFile\n{\n    version 2.0;\n    format ascii;\n'
            f'    class {cls};\n    object {name};\n}}\n\n')


def write(path, body, cls='dictionary'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header(path.name, cls) + body)

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
                p = 'walls'
                if side in (4,5):p = 'frontBack'
                elif side == 0:p = 'symmetry'
                elif side == 2 and (x[i]+x[i+1])/2 < m['inlet_width']:p = 'inlet'
                elif side == 3 and (x[i]+x[i+1])/2 > x[-1]-m['outlet_width']:p = 'outlet'
                boundary[p].append((face, c, -1))
    poly = case/'constant/polyMesh'
    poly.mkdir(parents=True, exist_ok=True)
    write(poly/'points', str(len(points))+'\n(\n'+''.join(
        f'({x[i]:.12g} {y[j]:.12g} {z[k]:.12g})\n' for i,j,k in points)+')\n', 'vectorField')
    all_faces = internal+[f for group in boundary.values() for f in group]
    write(poly/'faces', str(len(all_faces))+'\n(\n'+''.join('4('+ ' '.join(map(str,f))+')\n' for f,_,_ in all_faces)+')\n', 'faceList')
    write(poly/'owner', str(len(all_faces))+'\n(\n'+''.join(f'{o}\n' for _,o,_ in all_faces)+')\n', 'labelList')
    write(poly/'neighbour', str(len(internal))+'\n(\n'+''.join(f'{n}\n' for _,_,n in internal)+')\n', 'labelList')
    offset, btext = len(internal), str(len(PATCHES))+'\n(\n'
    for p, faces in boundary.items():
        typ = {'inlet':'patch','outlet':'patch','symmetry':'symmetryPlane','frontBack':'empty','walls':'wall'}[p]
        btext += f'{p}\n{{ type {typ}; nFaces {len(faces)}; startFace {offset}; }}\n'
        offset += len(faces)
    write(poly/'boundary', btext+')\n', 'polyBoundaryMesh')
    write(poly/'cellZones', '0\n(\n)\n', 'regIOobject')
    write(poly/'faceZones', '0\n(\n)\n', 'regIOobject')
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

