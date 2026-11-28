"""Physical geometry, independent mesh orientation and taper approximation."""
import math
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

from flowfield.problems.level1_small_continuous.p011.geometry import (
    analytic_channel_volume, boundary_patch, build_geometry, mesh, section,
)


def geometry(widths=(1., 1., 1., 1.)):
    result = {}
    for key, width in zip(('cathode_inlet', 'cathode_outlet', 'anode_inlet', 'anode_outlet'), widths):
        result[key+'_width_mm'] = width
        result[key+'_height_mm'] = 1/width
    return result


def config(resolution='coarse'):
    return SimpleNamespace(mesh=resolution, plate_thickness_mm=.5)


def scalar_list(path):
    return [int(value) for value in re.search(r'\n\(\n(.*?)\n\)', path.read_text(), re.S)[1].split()]


class TaperedGeometryTests(unittest.TestCase):
    def test_continuous_section_and_exact_volume(self):
        g = geometry((.6, 1.7, 1.329, 1.612))
        for electrode in ('anode', 'cathode'):
            first, last = section(g, electrode, 0), section(g, electrode, 20)
            middle = section(g, electrode, 10)
            self.assertAlmostEqual(math.prod(first), 1.)
            self.assertAlmostEqual(math.prod(last), 1.)
            self.assertAlmostEqual(middle[0], (first[0]+last[0])/2)
            self.assertAlmostEqual(middle[1], (first[1]+last[1])/2)
            # Simpson's rule is exact for this quadratic cross-sectional area.
            expected = 20/6*(math.prod(first)+4*math.prod(middle)+math.prod(last))*1e-9
            self.assertAlmostEqual(analytic_channel_volume(g, electrode), expected, places=18)
            self.assertGreater(expected, 20e-9)

    def test_rectangular_baseline_remains_exact_under_refinement(self):
        sizes = []
        for resolution in ('coarse', 'medium', 'fine'):
            m = build_geometry(geometry(), config(resolution))
            sizes.append(len(m['ids']))
            self.assertEqual(set(m['ids']), set(range(9)))
            self.assertAlmostEqual(m['z'][-1], 3.7338)
            self.assertAlmostEqual(m['area'], 40e-6)
            for error in m['approximation']['channel_volume_relative_errors'].values():
                self.assertLess(error, 1e-11)
            for value in m['port_areas'].values():
                self.assertAlmostEqual(value, 1e-6, places=16)
        self.assertLess(sizes[0], sizes[1])
        self.assertLess(sizes[1], sizes[2])

    def test_extreme_and_close_designs_preserve_ports_without_slivers(self):
        for widths in ((1/1.8, 1.8, 1.8, 1/1.8), (1.329, 1.612, 1., 1.4),
                       (1., 1.001, 1.0005, 1.002), (1.8, 1.799999999, .6, .600000001)):
            m = build_geometry(geometry(widths), config())
            for axis in ('x', 'y', 'z'):
                spacing = [b-a for a, b in zip(m[axis], m[axis][1:])]
                self.assertGreater(min(spacing), 1e-5)
                self.assertTrue(all(max(a/b, b/a) <= 6+1e-9 for a, b in zip(spacing, spacing[1:])))
            self.assertLess(max(m['approximation']['port_area_relative_errors'].values()), .005)
            nx, ny, nz = (len(m[axis])-1 for axis in ('x', 'y', 'z'))
            for material in (1, 7):
                remaining = {i for i, value in enumerate(m['ids']) if value == material}
                visited, queue = set(), [next(iter(remaining))]
                while queue:
                    cell = queue.pop()
                    if cell in visited:
                        continue
                    visited.add(cell)
                    i, j, k = m['cells'][cell]
                    for di, dj, dk in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)):
                        ii, jj, kk = i+di, j+dj, k+dk
                        if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                            neighbor = (kk*ny+jj)*nx+ii
                            if neighbor in remaining and neighbor not in visited:
                                queue.append(neighbor)
                self.assertEqual(visited, remaining)

    def test_written_mesh_face_orientation_and_closed_cells(self):
        m = build_geometry(geometry(), config())
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp)
            mesh(case, m)
            poly = case/'constant/polyMesh'
            points = [tuple(map(float, line[1:-1].split())) for line in (poly/'points').read_text().splitlines()
                      if re.fullmatch(r'\([^()]+ [^()]+ [^()]+\)', line)]
            faces = [tuple(map(int, line[2:-1].split())) for line in (poly/'faces').read_text().splitlines()
                     if line.startswith('4(')]
            owners, neighbors = scalar_list(poly/'owner'), scalar_list(poly/'neighbour')
            centers = [tuple((m[axis][index]+m[axis][index+1])*.0005
                             for axis, index in zip(('x', 'y', 'z'), cell)) for cell in m['cells']]
            closure = [[0., 0., 0.] for _ in centers]
            volume = [0.]*len(centers)
            for number, (face, owner) in enumerate(zip(faces, owners)):
                vertices = [points[p] for p in face]
                first = [vertices[1][d]-vertices[0][d] for d in range(3)]
                second = [vertices[3][d]-vertices[0][d] for d in range(3)]
                normal = [first[1]*second[2]-first[2]*second[1],
                          first[2]*second[0]-first[0]*second[2],
                          first[0]*second[1]-first[1]*second[0]]
                center = [sum(v[d] for v in vertices)/4 for d in range(3)]
                target = centers[neighbors[number]] if number < len(neighbors) else center
                self.assertGreater(sum(normal[d]*(target[d]-centers[owner][d]) for d in range(3)), 0.)
                for d in range(3):
                    closure[owner][d] += normal[d]
                volume[owner] += sum(normal[d]*center[d] for d in range(3))/3
                if number < len(neighbors):
                    neighbor = neighbors[number]
                    for d in range(3):
                        closure[neighbor][d] -= normal[d]
                    volume[neighbor] -= sum(normal[d]*center[d] for d in range(3))/3
            self.assertLess(max(abs(value) for cell in closure for value in cell), 1e-17)
            self.assertTrue(all(value > 0 for value in volume))
            self.assertAlmostEqual(sum(volume), .02*.002*.0037338, places=17)
            self.assertAlmostEqual(m['inlet_areas']['cathodeInlet'], 1e-6, places=16)
            # Only channels on the longitudinal ends are flow boundaries.
            for cell, value in enumerate(m['ids']):
                if value in (1, 7):
                    self.assertTrue(boundary_patch(m, cell, 0).endswith('Inlet'))
                    self.assertTrue(boundary_patch(m, cell, 1).endswith('Outlet'))
                    self.assertEqual(boundary_patch(m, cell, 2), 'externalWalls')


if __name__ == '__main__':
    unittest.main()
