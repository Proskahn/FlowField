"""Conformal orthogonal full-cell mesh of Li et al.'s tapered channels.

Geometry construction uses mm and OpenFOAM output uses SI. Both width and
height vary linearly along the 20 mm channel. The sloping surfaces are resolved
as a staircase, which permits the solver's orthogonal finite-volume operators.
Endpoint knots are retained unless separated by less than 0.001 mm; this
explicit mesh regularization prevents arbitrarily thin cells during search.
Every case records the resulting port-area and channel-volume discrepancies.
"""
import math
from bisect import bisect_right
from collections import Counter
from pathlib import Path

from ..p008.geometry import field, header, write

ZONES = ('CBP', 'CCH', 'CGDL', 'CCL', 'PEM', 'ACL', 'AGDL', 'ACH', 'ABP')
FLUID = {1, 2, 3, 5, 6, 7}
PATCHES = ('anodeInlet', 'anodeOutlet', 'cathodeInlet', 'cathodeOutlet',
           'anodeTerminal', 'cathodeTerminal', 'externalWalls')
LENGTH_MM = 20.0
PITCH_MM = 2.0
KNOT_MERGE_MM = 0.001
RESOLUTION = {
    'coarse': (20, .25, 3, 1, 2),
    'medium': (32, .18, 4, 1, 3),
    'fine': (48, .12, 5, 2, 4),
}


def section(g, electrode, position_mm):
    """Continuous design section (width, height) before mesh approximation."""
    fraction = position_mm / LENGTH_MM
    if not 0 <= fraction <= 1:
        raise ValueError('Section position must lie in [0, 20] mm')
    values = []
    for dimension in ('width', 'height'):
        first = g[f'{electrode}_inlet_{dimension}_mm']
        last = g[f'{electrode}_outlet_{dimension}_mm']
        values.append(first + fraction * (last-first))
    return tuple(values)


def analytic_channel_volume(g, electrode):
    """Integral of the product of two linear dimensions, returned in m3."""
    w0, h0 = section(g, electrode, 0)
    w1, h1 = section(g, electrode, LENGTH_MM)
    return LENGTH_MM * (2*w0*h0 + w0*h1 + w1*h0 + 2*w1*h1) / 6 * 1e-9


def _axis(knots, spacing):
    """Merge only nearby interior knots, then refine with adjacent ratio <=6."""
    unique = sorted(set(knots))
    groups = []
    for value in unique:
        if groups and value-groups[-1][0] < KNOT_MERGE_MM:
            groups[-1].append(value)
        else:
            groups.append([value])
    merged = [sum(group)/len(group) for group in groups]
    merged[0], merged[-1] = unique[0], unique[-1]
    out = [merged[0]]
    for first, last in zip(merged, merged[1:]):
        count = max(1, math.ceil((last-first)/spacing - 1e-12))
        out.extend(first+(last-first)*j/count for j in range(1, count+1))
    return _grade(out)


def _grade(out):
    """Retain physical interfaces while bounding face interpolation weights."""
    while True:
        widths = [b-a for a, b in zip(out, out[1:])]
        split = {i for i, width in enumerate(widths)
                 if (i and width > 6*widths[i-1])
                 or (i+1 < len(widths) and width > 6*widths[i+1])}
        if not split:
            return out
        refined = [out[0]]
        for i, (a, b) in enumerate(zip(out, out[1:])):
            if i in split:
                refined.append((a+b)/2)
            refined.append(b)
        out = refined


def _layer(z, layers, thickness, count, material, *, reverse=False, graded=False):
    weights = [1.5**i for i in range(count)] if graded else [1]*count
    if reverse:
        weights.reverse()
    total = sum(weights)
    for weight in weights:
        z.append(z[-1]+thickness*weight/total)
        layers.append(material)


def build_geometry(g, config):
    nx, spacing, ngdl, ncl, nmem = RESOLUTION[config.mesh]
    x = [LENGTH_MM*i/nx for i in range(nx+1)]
    # Mirror the radial mesh to preserve exact transverse symmetry.
    radial = _axis([0., PITCH_MM/2] + [g[f'{e}_{p}_width_mm']/2
        for e in ('cathode', 'anode') for p in ('inlet', 'outlet')], spacing)
    y = [PITCH_MM/2-v for v in reversed(radial)] + [PITCH_MM/2+v for v in radial[1:]]
    heights = {e: _axis([0., g[f'{e}_inlet_height_mm'], g[f'{e}_outlet_height_mm']], spacing)
               for e in ('cathode', 'anode')}
    plate = config.plate_thickness_mm
    z, layers = [0.], []
    _layer(z, layers, plate, max(1, math.ceil(plate/spacing)), 0)
    # On the cathode, depth is measured downwards from its GDL interface.
    cathode_top = plate+heights['cathode'][-1]
    z.extend(cathode_top-value for value in reversed(heights['cathode'][:-1]))
    layers.extend([1]*(len(heights['cathode'])-1))
    _layer(z, layers, .3, ngdl, 2, reverse=True, graded=True)
    _layer(z, layers, .0129, ncl, 3)
    _layer(z, layers, .108, nmem, 4)
    _layer(z, layers, .0129, ncl, 5)
    _layer(z, layers, .3, ngdl, 6, graded=True)
    anode_bottom = z[-1]
    z.extend(anode_bottom+value for value in heights['anode'][1:])
    layers.extend([7]*(len(heights['anode'])-1))
    _layer(z, layers, plate, max(1, math.ceil(plate/spacing)), 8)
    # Channel endpoint knots can approach a plate or GDL boundary closely.
    # Grade across those material interfaces as well as within each channel.
    original_z, original_layers = z, layers
    z = _grade(z)
    layers = [original_layers[bisect_right(original_z, (a+b)/2)-1]
              for a, b in zip(z, z[1:])]
    ny = len(y)-1
    samples = [(x[i]+x[i+1])/2 for i in range(nx)]
    # Exact end sections make imposed boundary geometry independent of dx.
    # This endpoint hold is a disclosed first/last-cell approximation.
    samples[0], samples[-1] = 0., LENGTH_MM
    sections = {e: [section(g, e, position) for position in samples]
                for e in ('cathode', 'anode')}
    cells, ids = [], []
    volume = Counter()
    areas = {e: [0.]*nx for e in ('cathode', 'anode')}
    for k, layer in enumerate(layers):
        zz = (z[k]+z[k+1])/2
        for j in range(ny):
            yy = (y[j]+y[j+1])/2
            area = (y[j+1]-y[j])*(z[k+1]-z[k])*1e-6
            for i in range(nx):
                material = layer
                if layer in (1, 7):
                    electrode = 'cathode' if layer == 1 else 'anode'
                    width, height = sections[electrode][i]
                    depth = cathode_top-zz if layer == 1 else zz-anode_bottom
                    if abs(yy-PITCH_MM/2) > width/2 or depth > height:
                        material = 0 if layer == 1 else 8
                    else:
                        areas[electrode][i] += area
                        volume[electrode] += area*(x[i+1]-x[i])*1e-3
                cells.append((i, j, k))
                ids.append(material)
    port_areas = {e+port: areas[e][0 if port == 'Inlet' else -1]
                  for e in areas for port in ('Inlet', 'Outlet')}
    exact_volume = {e: analytic_channel_volume(g, e) for e in areas}
    approximation = dict(
        representation='orthogonal staircase of linearly tapered width and height',
        axial_sampling='cell centers; exact endpoint sections held in first and last cells',
        knot_merge_threshold_mm=KNOT_MERGE_MM,
        cells_per_axis=[nx, ny, len(z)-1],
        minimum_spacing_mm={axis: min(b-a for a, b in zip(values, values[1:]))
                            for axis, values in (('x', x), ('y', y), ('z', z))},
        maximum_spacing_mm={axis: max(b-a for a, b in zip(values, values[1:]))
                            for axis, values in (('x', x), ('y', y), ('z', z))},
        target_port_area_m2=1e-6,
        port_areas_m2=port_areas,
        port_area_relative_errors={name: abs(value/1e-6-1) for name, value in port_areas.items()},
        analytical_channel_volume_m3=exact_volume,
        meshed_channel_volume_m3=dict(volume),
        channel_volume_relative_errors={e: abs(volume[e]/exact_volume[e]-1) for e in areas},
        maximum_section_area_relative_errors={e: max(abs(a/(w*h*1e-6)-1)
            for a, (w, h) in zip(areas[e], sections[e])) for e in areas},
        plate_thickness_mm=plate,
        active_area_m2=LENGTH_MM*PITCH_MM*1e-6,
    )
    if max(approximation['port_area_relative_errors'].values()) > .005:
        raise ValueError('P011 mesh port-area error exceeds 0.5%; refine geometry regularization')
    return dict(x=x, y=y, z=z, cells=cells, ids=ids, area=LENGTH_MM*PITCH_MM*1e-6,
                approximation=approximation, port_areas=port_areas,
                channel_sections_m2=areas, layer_materials=layers)


def boundary_patch(m, cell, side):
    material = m['ids'][cell]
    if material in (1, 7) and side in (0, 1):
        return ('cathode' if material == 1 else 'anode') + ('Inlet' if side == 0 else 'Outlet')
    return 'cathodeTerminal' if side == 4 else 'anodeTerminal' if side == 5 else 'externalWalls'


def mesh(case, m):
    """Write an owner-oriented conformal hexahedral OpenFOAM polyMesh."""
    case = Path(case)
    x, y, z = (m[name] for name in ('x', 'y', 'z'))
    nx, ny, nz = (len(axis)-1 for axis in (x, y, z))
    points, point_ids, internal = [], {}, []
    boundary = {name: [] for name in PATCHES}
    directions = [((-1,0,0), ((0,0,0),(0,0,1),(0,1,1),(0,1,0))),
                  ((1,0,0), ((1,0,0),(1,1,0),(1,1,1),(1,0,1))),
                  ((0,-1,0), ((0,0,0),(1,0,0),(1,0,1),(0,0,1))),
                  ((0,1,0), ((0,1,0),(0,1,1),(1,1,1),(1,1,0))),
                  ((0,0,-1), ((0,0,0),(0,1,0),(1,1,0),(1,0,0))),
                  ((0,0,1), ((0,0,1),(1,0,1),(1,1,1),(0,1,1)))]
    inlet_areas = Counter()
    inlet_face_areas = {name: [] for name in PATCHES if name.endswith('Inlet')}
    for cell, (i, j, k) in enumerate(m['cells']):
        for side, ((di, dj, dk), corners) in enumerate(directions):
            ii, jj, kk = i+di, j+dj, k+dk
            neighbour = (kk*ny+jj)*nx+ii if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz else -1
            if 0 <= neighbour < cell:
                continue
            face = []
            for a, b, d in corners:
                key = i+a, j+b, k+d
                if key not in point_ids:
                    point_ids[key] = len(points)
                    points.append(key)
                face.append(point_ids[key])
            if neighbour >= 0:
                internal.append((face, cell, neighbour))
            else:
                patch = boundary_patch(m, cell, side)
                boundary[patch].append((face, cell, -1))
                if patch.endswith('Inlet'):
                    area = (y[j+1]-y[j])*(z[k+1]-z[k])*1e-6
                    inlet_areas[patch] += area
                    inlet_face_areas[patch].append(area)
    poly = case/'constant/polyMesh'
    poly.mkdir(parents=True, exist_ok=True)
    write(poly/'points', str(len(points))+'\n(\n'+''.join(
        f'({x[i]*1e-3:.14g} {y[j]*1e-3:.14g} {z[k]*1e-3:.14g})\n' for i, j, k in points)+')\n', 'vectorField')
    all_faces = internal+[face for group in boundary.values() for face in group]
    write(poly/'faces', str(len(all_faces))+'\n(\n'+''.join(
        '4('+' '.join(map(str, face))+')\n' for face, _, _ in all_faces)+')\n', 'faceList')
    write(poly/'owner', str(len(all_faces))+'\n(\n'+''.join(
        f'{owner}\n' for _, owner, _ in all_faces)+')\n', 'labelList')
    write(poly/'neighbour', str(len(internal))+'\n(\n'+''.join(
        f'{neighbour}\n' for _, _, neighbour in internal)+')\n', 'labelList')
    offset, btext = len(internal), str(len(PATCHES))+'\n(\n'
    for patch, faces in boundary.items():
        kind = 'patch' if patch.endswith(('Inlet', 'Outlet')) else 'wall'
        btext += f'{patch}\n{{ type {kind}; nFaces {len(faces)}; startFace {offset}; }}\n'
        offset += len(faces)
    write(poly/'boundary', btext+')\n', 'polyBoundaryMesh')
    zones = '9\n(\n'
    for material, name in enumerate(ZONES):
        labels = [cell for cell, value in enumerate(m['ids']) if value == material]
        zones += f'{name}\n{{ type cellZone; cellLabels List<label> {len(labels)}\n(\n'+'\n'.join(map(str, labels))+'\n); }\n'
    write(poly/'cellZones', zones+')\n', 'regIOobject')
    write(poly/'faceZones', '0\n(\n)\n', 'regIOobject')
    m['inlet_areas'] = dict(inlet_areas)
    m['inlet_face_areas'] = inlet_face_areas
    from ....mesh import PolyMesh
    return PolyMesh.from_case(case)
