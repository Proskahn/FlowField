"""Inspectable PPFC reconstruction and optional Gmsh 4.13.1 CAD/mesh builder.

No paper CAD was supplied. The rules below are exposed assumptions, not a
digitization of the authors' COMSOL geometry. All six variables move geometry.
"""

import json
import math
from pathlib import Path

from .config import MESHES, PPFCConfig


def unit_layout(geometry):
    """Fixed row populations across the whole six-dimensional box.

    Main spherical recess rows use the 30 mm longitudinal pitch. Protrusions
    are staggered halfway between rows and inherit the upstream lateral pitch.
    Counts use the largest allowed pitch, keeping edge units inside the circle.
    """
    p = geometry["spacings_m"]
    units = []
    for i in range(-2, 3):
        x = i*p["x1_1"]
        y = -math.sqrt((0.2-p["h1_1"])**2-x*x)
        units.append({"kind": "square" if i == 0 else "diamond", "x": x, "y": y, "row": "I-inlet"})
    for i in range(-3, 4):
        units.append({"kind": "diamond", "x": i*p["x1_2"], "y": -0.135-p["h1_2"], "row": "I-upper"})
    for j in range(-4, 7):
        y = j*0.030
        pitch = p["x2"] if y < -0.045 else p["x3"] if y < 0.045 else 0.030
        max_pitch = 0.032 if y < -0.045 else 0.031 if y < 0.045 else 0.030
        max_x = math.sqrt(0.191**2-y*y)
        for i in range(-int(max_x/max_pitch), int(max_x/max_pitch)+1):
            units.append({"kind": "concave", "x": i*pitch, "y": y, "row": j})
        if j < 6:
            cy = y + 0.015
            max_x = math.sqrt(0.191**2-cy*cy)
            for i in range(-7, 7):
                if abs((i+0.5)*max_pitch) <= max_x:
                    units.append({"kind": "convex", "x": (i+0.5)*pitch, "y": cy, "row": j+0.5})
    return units


def layout_svg(geometry):
    """CAD-independent plan view for checking design positions before meshing."""
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="-230 -245 460 495">',
        '<rect x="-230" y="-245" width="460" height="495" fill="white"/>',
        '<text x="-215" y="-226" font-size="10">P001: provisional reconstruction, dimensions in mm</text>',
        '<circle r="200" fill="#f1f5f9" stroke="#334155"/>']
    for y in (-135, -45, 45, 135):
        half = math.sqrt(200**2-y*y)
        parts.append(f'<path d="M {-half} {-y} H {half}" stroke="#94a3b8" stroke-dasharray="3 3"/>')
    for u in unit_layout(geometry):
        x, y = u['x']*1000, -u['y']*1000
        if u['kind'] in ('concave', 'convex'):
            fill = '#dbeafe' if u['kind'] == 'concave' else '#64748b'
            parts.append(f'<circle cx="{x}" cy="{y}" r="7.5" fill="{fill}" stroke="#334155" stroke-width=".5"/>')
        else:
            angle = 45 if u['kind'] == 'diamond' else 0
            parts.append(f'<rect x="{x-6.5}" y="{y-6.5}" width="13" height="13" transform="rotate({angle} {x} {y})" fill="#f59e0b" stroke="#92400e" stroke-width=".5"/>')
    parts.append('<text x="-212" y="231" font-size="9">Blue: recess; grey: protrusion; orange: full-height prism</text></svg>')
    return '\n'.join(parts)


def split_tetrahedra(nodes, triangles, tetrahedra):
    """Conforming barycentric split: 4 convex hexes/tet, 3 quads/triangle.

    Preserve the planar Gmsh boundary triangles. Reprojecting new vertices to
    curved CAD can invert cells at shallow spherical-cap rims. Shared edge and
    face centres ensure no cracks. Input tetrahedra must have positive volume.
    Triangles/tetrahedra are (physical_group, node_ids) records.
    """
    nodes = dict(nodes)
    centres = {}
    next_id = max(nodes)+1

    def centre(ids):
        nonlocal next_id
        key = tuple(sorted(ids))
        if key not in centres:
            centres[key] = next_id
            nodes[next_id] = tuple(sum(nodes[i][axis] for i in key)/len(key) for axis in range(3))
            next_id += 1
        return centres[key]

    quads, hexes = [], []
    for group, (a,b,c) in triangles:
        f = centre((a,b,c))
        for v,w,t in ((a,b,c),(b,c,a),(c,a,b)):
            quads.append((group,(v,centre((v,w)),f,centre((v,t)))))
    for group, (a,b,c,d) in tetrahedra:
        e1,e2,e3 = [tuple(nodes[q][i]-nodes[a][i] for i in range(3)) for q in (b,c,d)]
        determinant = (e1[0]*(e2[1]*e3[2]-e2[2]*e3[1])
                       -e1[1]*(e2[0]*e3[2]-e2[2]*e3[0])
                       +e1[2]*(e2[0]*e3[1]-e2[1]*e3[0]))
        if determinant <= 0:
            raise ValueError('Gmsh returned a nonpositive tetrahedron')
        middle = centre((a,b,c,d))
        for v,w,t,s in ((a,b,c,d),(b,c,a,d),(c,a,b,d),(d,b,a,c)):
            hexes.append((group, (v,centre((v,w)),centre((v,w,t)),centre((v,t)),
                centre((v,s)),centre((v,w,s)),middle,centre((v,t,s)))))
    return nodes, quads, hexes


def _write_hex_mesh(gmsh, path):
    ids, xyz, _ = gmsh.model.mesh.getNodes()
    nodes = {int(tag):tuple(map(float,xyz[3*i:3*i+3])) for i,tag in enumerate(ids)}
    triangles, tetrahedra = [], []
    names = []
    for dim, group in gmsh.model.getPhysicalGroups():
        names.append((dim,group,gmsh.model.getPhysicalName(dim,group)))
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim,group):
            types, _, arrays = gmsh.model.mesh.getElements(dim,entity)
            for kind, array in zip(types, arrays):
                expected, size, dest = (2,3,triangles) if dim == 2 else (4,4,tetrahedra)
                if kind != expected:
                    raise RuntimeError('Expected linear triangles/tetrahedra before barycentric subdivision')
                dest.extend((group,tuple(map(int,array[i:i+size]))) for i in range(0,len(array),size))
    nodes, quads, hexes = split_tetrahedra(nodes,triangles,tetrahedra)
    with Path(path).open('x') as stream:
        stream.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n'+str(len(names))+'\n')
        for dim,group,name in names:
            stream.write(f'{dim} {group} "{name}"\n')
        stream.write('$EndPhysicalNames\n$Nodes\n'+str(len(nodes))+'\n')
        for tag,xyz in sorted(nodes.items()):
            stream.write(str(tag)+' '+' '.join(format(v,'.16g') for v in xyz)+'\n')
        stream.write('$EndNodes\n$Elements\n'+str(len(quads)+len(hexes))+'\n')
        index = 0
        for kind, elements in ((3,quads),(5,hexes)):
            for group, ids in elements:
                index += 1
                stream.write(f'{index} {kind} 2 {group} {group} '+' '.join(map(str,ids))+'\n')
        stream.write('$EndElements\n')
    return len(hexes), len(tetrahedra)


def build_mesh(case):
    """Run in a subprocess: Gmsh globals are not thread-safe; errors are logged."""
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError('P001 meshing needs the optional dependency: pip install -e ".[p001]"') from exc
    if gmsh.__version__ != "4.13.1":
        raise RuntimeError(f"P001 mesh reproducibility requires gmsh==4.13.1; found {gmsh.__version__}")
    case = Path(case)
    manifest = json.loads((case/'case.json').read_text())
    config = PPFCConfig(**manifest['config'])
    geometry = manifest['geometry']
    gmsh.initialize()
    try:
        gmsh.model.add('P001-PPFC-reconstruction')
        occ = gmsh.model.occ
        height, radius = .0075, .2
        fluid = [(3, occ.addCylinder(0, 0, 0, 0, 0, height, radius))]
        w, length = config.port_width_m, config.port_length_m
        # Rectangular ports penetrate the circular wall by 1 mm to form a union.
        ports = [(3, occ.addBox(-w/2, -.2-length, 0, w, length+.001, height)),
                 (3, occ.addBox(-w/2, .199, 0, w, length+.001, height))]
        fluid, _ = occ.fuse(fluid, ports)
        a, h = .0075, config.sphere_cap_height_m
        sphere_radius = (a*a+h*h)/(2*h)
        recesses, solids = [], []
        for u in unit_layout(geometry):
            x, y = u['x'], u['y']
            if u['kind'] == 'concave':
                sphere = occ.addSphere(x, y, sphere_radius-h, sphere_radius)
                clip = occ.addBox(x-a-.001, y-a-.001, -h-.001, 2*a+.002, 2*a+.002, h+.001)
                cap, _ = occ.intersect([(3, sphere)], [(3, clip)])
                recesses.extend(cap)
            elif u['kind'] == 'convex':
                solids.append((3, occ.addSphere(x, y, h-sphere_radius, sphere_radius)))
            else:
                tag = occ.addBox(x-.0065, y-.0065, -.001, .013, .013, height+.002)
                if u['kind'] == 'diamond':
                    occ.rotate([(3, tag)], x, y, 0, 0, 0, 1, math.pi/4)
                solids.append((3, tag))
        fluid, _ = occ.fuse(fluid, recesses)
        fluid, _ = occ.cut(fluid, solids)
        occ.synchronize()
        if len(fluid) != 1:
            raise RuntimeError(f"Expected a single connected fluid volume, got {len(fluid)}")
        patches = {'inlet': [], 'outlet': [], 'walls': []}
        for dim, tag in gmsh.model.getBoundary(fluid, oriented=False):
            box = gmsh.model.getBoundingBox(dim, tag)
            if abs(box[1]+.2+length) < 1e-6 and abs(box[4]+.2+length) < 1e-6:
                patches['inlet'].append(tag)
            elif abs(box[1]-.2-length) < 1e-6 and abs(box[4]-.2-length) < 1e-6:
                patches['outlet'].append(tag)
            else:
                patches['walls'].append(tag)
        for name, tags in patches.items():
            if not tags or name != 'walls' and len(tags) != 1:
                raise RuntimeError(f"Unexpected {name} patch count: {len(tags)}")
            gmsh.model.addPhysicalGroup(2, tags, name=name)
        gmsh.model.addPhysicalGroup(3, [fluid[0][1]], name='fluid')
        small, large = MESHES[config.mesh]
        for name, value in {'Mesh.MeshSizeMin': small, 'Mesh.MeshSizeMax': large,
                            'Mesh.MeshSizeFromCurvature': 12, 'Mesh.Algorithm3D': 1,
                            'Mesh.Optimize': 1, 'Mesh.MshFileVersion': 2.2,
                            'General.NumThreads': 1, 'Mesh.Binary': 0}.items():
            gmsh.option.setNumber(name, value)
        gmsh.write(str(case/'geometry.brep'))
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize('Netgen')
        cells, tetrahedra = _write_hex_mesh(gmsh, case/'mesh.msh')
        (case/'mesh-summary.json').write_text(json.dumps({'gmsh_version': gmsh.__version__,
            'cells': cells, 'element_types': [5], 'mesh': config.mesh, 'source_tetrahedra':tetrahedra,
            'subdivision':'barycentric-four-hexes-per-tetrahedron-planar-boundary',
            'minimum_size_m': small, 'maximum_size_m': large, 'fluid_volumes': len(fluid),
            'patch_surface_counts': {k:len(v) for k,v in patches.items()}}, indent=2)+'\n')
    finally:
        gmsh.finalize()


if __name__ == '__main__':
    import sys
    build_mesh(sys.argv[1])
