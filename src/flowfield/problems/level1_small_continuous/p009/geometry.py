"""Pinned standalone Gmsh mesher: porous disk, peripheral distributor, two stems.

Figure 26's angle is a central arc angle, not a conical diffuser angle.
The manifold width and stem length are unreported reconstruction parameters.
Coordinates and mesh output are SI. Shared extrusion faces are conformal.
"""
import json
import math
from pathlib import Path


def write_hex_mesh(gmsh, path):
    """Split each extruded triangle into three conformal hexahedra.

    This removes boundary prisms with only two internal faces, which fail
    OpenFOAM's cell-determinant check on the thin no-slip layer.
    """
    ids, xyz, _ = gmsh.model.mesh.getNodes()
    nodes = {int(tag): tuple(map(float, xyz[3*i:3*i+3])) for i,tag in enumerate(ids)}
    centers = {}
    next_id = max(nodes)+1
    def center(ids):
        nonlocal next_id
        key = tuple(sorted(ids))
        if key not in centers:
            tag = next_id
            next_id += 1
            nodes[tag] = tuple(sum(nodes[i][d] for i in key)/len(key) for d in range(3))
            centers[key] = tag
        return centers[key]
    faces, cells, names = [], [], []
    for dim, group in gmsh.model.getPhysicalGroups():
        names.append((dim,group,gmsh.model.getPhysicalName(dim,group)))
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim,group):
            types, _, arrays = gmsh.model.mesh.getElements(dim,entity)
            for kind, array in zip(types,arrays):
                if kind not in (2,3,6):
                    raise RuntimeError(f'Expected extruded triangular mesh; got element {kind}')
                size = {2:3,3:4,6:6}[kind]
                for i in range(0,len(array),size):
                    v = list(map(int,array[i:i+size]))
                    if kind == 2:
                        a,b,c = v; middle=center(v)
                        for a,b,c in ((a,b,c),(b,c,a),(c,a,b)):
                            faces.append((group,(a,center((a,b)),middle,center((a,c)))))
                    elif kind == 3:
                        if abs(nodes[v[0]][2]-nodes[v[1]][2]) > 1e-10:
                            v = v[1:]+v[:1]
                        a,b,c,d=v; ab=center((a,b)); cd=center((c,d))
                        faces.extend(((group,(a,ab,cd,d)),(group,(ab,b,c,cd))))
                    else:
                        a,b,c,d,e,f=v; lower=center((a,b,c)); upper=center((d,e,f))
                        for a,b,c,d,e,f in ((a,b,c,d,e,f),(b,c,a,e,f,d),(c,a,b,f,d,e)):
                            cells.append((group,(a,center((a,b)),lower,center((a,c)),
                                d,center((d,e)),upper,center((d,f)))))
    used = {node for _, element in faces+cells for node in element}
    nodes = {tag: point for tag, point in nodes.items() if tag in used}
    with Path(path).open('x') as stream:
        stream.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n'+str(len(names))+'\n')
        for dim,group,name in names:
            stream.write(f'{dim} {group} "{name}"\n')
        stream.write('$EndPhysicalNames\n$Nodes\n'+str(len(nodes))+'\n')
        for tag, xyz in nodes.items():
            stream.write(str(tag)+' '+' '.join(format(v,'.16g') for v in xyz)+'\n')
        stream.write('$EndNodes\n$Elements\n'+str(len(faces)+len(cells))+'\n')
        count=0
        for kind, elements in ((3,faces),(5,cells)):
            for group, ids in elements:
                count+=1
                stream.write(f'{count} {kind} 2 {group} {group} '+' '.join(map(str,ids))+'\n')
        stream.write('$EndElements\n')
    return len(cells)


def build_mesh(case):
    import gmsh
    if gmsh.__version__ != '4.13.1':
        raise RuntimeError('Requires gmsh==4.13.1')
    case = Path(case)
    data = json.loads((case/'case.json').read_text())
    g, c = data['geometry'], data['config']
    r = g['diameter_mm']*.0005
    w, h, height, band, length = [g[k]*.001 for k in
        ('inlet_width_mm', 'thickness_mm', 'inlet_height_mm', 'manifold_width_mm', 'port_length_mm')]
    # Keep the ports top-aligned as drawn in Figure 4.
    z0 = h-height
    step, layers = {'coarse': (.004, 8), 'medium': (.0025, 12), 'fine': (.0016, 18)}[c['mesh']]
    gmsh.initialize()
    try:
        gmsh.model.add('P009-Chen-reconstruction')
        occ = gmsh.model.occ
        disk = occ.addDisk(0, 0, z0, r, r)
        outlet_start = -math.sqrt(r*r-(w/2)**2)
        # A chord makes a conformal port opening without an arbitrarily thin
        # circular sliver at a tangent. The removed cap is recorded in docs.
        cutter = occ.addRectangle(-r-1, -r-1, z0, 2*r+2, 1+r+outlet_start)
        clipped, _ = occ.cut([(2,disk)],[(2,cutter)])
        disk = clipped[0][1]
        theta = math.radians(g['angle_deg'])/2
        center = occ.addPoint(0, 0, z0)
        def point(rad, t):
            return occ.addPoint(rad*math.sin(t), rad*math.cos(t), z0)
        a, b, d, e = point(r, -theta), point(r, theta), point(r+band, theta), point(r+band, -theta)
        curves = [occ.addCircleArc(a, center, b), occ.addLine(b, d),
                  occ.addCircleArc(d, center, e), occ.addLine(e, a)]
        ring = occ.addPlaneSurface([occ.addCurveLoop(curves)])
        inlet = occ.addRectangle(-w/2, r, z0, w, band+length)
        outlet = occ.addRectangle(-w/2, -r-band-length, z0, w, r+band+length+outlet_start)
        free, _ = occ.fuse([(2, ring)], [(2, inlet), (2, outlet)])
        surfaces, mapping = occ.fragment([(2, disk)], free)
        nickel_surfaces = set(mapping[0])
        free_surfaces = [s for s in surfaces if s not in nickel_surfaces]
        # Extrude each connected region in one operation so interfaces share faces.
        top = occ.extrude(surfaces, 0, 0, height, numElements=[layers], recombine=True)
        volumes = [s for s in top if s[0] == 3]
        occ.synchronize()
        nickel = []
        for dim, tag in volumes:
            faces = set(gmsh.model.getBoundary([(dim, tag)], oriented=False))
            if faces & nickel_surfaces:
                nickel.append(tag)
        if z0 > 1e-10:
            bottom = occ.extrude(list(nickel_surfaces), 0, 0, -z0,
                numElements=[max(2, math.ceil(layers*z0/height))], recombine=True)
            nickel += [tag for dim, tag in bottom if dim == 3]
            volumes += [s for s in bottom if s[0] == 3]
        elif z0 < -1e-10:
            raise ValueError('This reconstruction requires inlet height <= layer thickness')
        occ.synchronize()
        if not nickel:
            raise RuntimeError('Missing nickel volumes')
        gmsh.model.addPhysicalGroup(3, nickel, name='nickel')
        gmsh.model.addPhysicalGroup(3, [t for _, t in volumes if t not in nickel], name='manifold')
        patches = dict(inlet=[], outlet=[], walls=[])
        for dim, tag in gmsh.model.getBoundary(volumes, combined=True, oriented=False):
            box = gmsh.model.getBoundingBox(dim, tag)
            if abs(box[1]-(r+band+length)) < 1e-6 and abs(box[4]-(r+band+length)) < 1e-6:
                patches['inlet'].append(tag)
            elif abs(box[1]+r+band+length) < 1e-6 and abs(box[4]+r+band+length) < 1e-6:
                patches['outlet'].append(tag)
            else:
                patches['walls'].append(tag)
        for name, tags in patches.items():
            if not tags:
                raise RuntimeError(f'Missing {name} patch')
            gmsh.model.addPhysicalGroup(2, tags, name=name)
        for name, value in {'General.NumThreads': 1, 'Mesh.MshFileVersion': 2.2,
                'Mesh.Binary': 0, 'Mesh.MeshSizeMin': min(band/2, w/4), 'Mesh.MeshSizeMax': step,
                'Mesh.MeshSizeFromCurvature': 80, 'Mesh.Algorithm': 6,
                'Mesh.RandomSeed': 1}.items():
            gmsh.option.setNumber(name, value)
        gmsh.write(str(case/'geometry.brep'))
        gmsh.model.mesh.generate(3)
        cells = write_hex_mesh(gmsh, case/'mesh.msh')
        summary = dict(gmsh_version=gmsh.__version__, cells=cells,
            mesh=c['mesh'], maximum_inplane_size_m=step, inlet_height_layers=layers,
            nickel_volumes=nickel, patch_counts={k:len(v) for k,v in patches.items()},
            geometry_status='reconstructed; manifold width and stem length unreported')
        (case/'mesh-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    finally:
        gmsh.finalize()


if __name__ == '__main__':
    import sys
    build_mesh(sys.argv[1])
