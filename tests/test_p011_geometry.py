"""Physical geometry, independent mesh orientation and taper approximation."""
import math
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

from flowfield.problems.level1_small_continuous.p011.geometry import (
    KNOT_MERGE_MM, RESOLUTION, _axis, analytic_channel_volume, boundary_patch,
    build_geometry, mesh, section,
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
    def assert_continuous_mesh_pair(self, base, changed, delta):
        self.assertEqual(base['approximation']['cells_per_axis'],
                         changed['approximation']['cells_per_axis'])
        self.assertTrue(base['ids'] == changed['ids'],
                        'An infinitesimal perturbation changed material topology')
        for axis in ('x', 'y', 'z'):
            self.assertEqual(len(base[axis]), len(changed[axis]))
            # Reciprocal heights amplify width changes, and the z coordinate
            # includes successive layers. This bound allows that physical
            # movement while rejecting a repartition of whole grid intervals.
            displacement = max(abs(a-b) for a, b in zip(base[axis], changed[axis]))
            self.assertLess(displacement, 8*abs(delta))

    def test_endpoint_refinement_preserves_the_common_background_lattice(self):
        for level, (_, spacing, _, _, _) in RESOLUTION.items():
            for knots in ((0., .5, 1.), (0., .5-1e-8, .5, 1.),
                          (0., .211, .347, .537, .883, 1.),
                          (0., .5004, .5008, .7498, 1.)):
                with self.subTest(level=level, knots=knots):
                    values = _axis(knots, spacing)
                    for index in range(round(1/spacing)+1):
                        anchor = index*spacing
                        if min(abs(anchor-knot) for knot in knots) > 2*KNOT_MERGE_MM:
                            self.assertLess(min(abs(value-anchor) for value in values), 1e-13)
                    self.assertLessEqual(max(b-a for a, b in zip(values, values[1:])),
                                         spacing+2*KNOT_MERGE_MM)

    def test_infinitesimal_baseline_changes_do_not_change_resolution(self):
        widths = (1., 1., 1., 1.)
        for level in RESOLUTION:
            base = build_geometry(geometry(widths), config(level))
            for coordinate in range(4):
                for delta in (-1e-8, 1e-8):
                    with self.subTest(level=level, coordinate=coordinate, delta=delta):
                        perturbed = list(widths)
                        perturbed[coordinate] += delta
                        changed = build_geometry(geometry(perturbed), config(level))
                        self.assert_continuous_mesh_pair(base, changed, delta)

    def test_infinitesimal_taper_changes_preserve_local_mesh_continuity(self):
        # The paper IGA point is away from voxel-center classification
        # crossings. Global smoothness of a staircase is not assumed.
        widths = (1.329, 1.612, .779, 1.013)
        for level in RESOLUTION:
            base = build_geometry(geometry(widths), config(level))
            for coordinate in range(4):
                for delta in (-1e-8, 1e-8):
                    with self.subTest(level=level, coordinate=coordinate, delta=delta):
                        perturbed = list(widths)
                        perturbed[coordinate] += delta
                        changed = build_geometry(geometry(perturbed), config(level))
                        self.assert_continuous_mesh_pair(base, changed, delta)

    def test_merge_boundary_keeps_background_resolution_and_port_quality(self):
        spacing = RESOLUTION['coarse'][1]
        # A half-width separation of 0.001 mm reaches the declared local
        # merge boundary. Local topology may change here; resolution away
        # from those interfaces must remain anchored on both sides.
        for width in (.998-1e-8, .998+1e-8, 1.002-1e-8, 1.002+1e-8):
            with self.subTest(cathode_outlet_width_mm=width):
                m = build_geometry(geometry((1., width, 1., 1.)), config())
                for anchor in (.125, .25, .375, .625, .75, .875):
                    self.assertLess(min(abs(value-anchor) for value in m['y']), 1e-13)
                    self.assertLess(min(abs(value-(2-anchor)) for value in m['y']), 1e-13)
                for axis in ('x', 'y', 'z'):
                    intervals = [b-a for a, b in zip(m[axis], m[axis][1:])]
                    self.assertTrue(all(math.isfinite(value) and value > 1e-5 for value in intervals))
                    self.assertTrue(all(max(a/b, b/a) <= 6+1e-9
                                        for a, b in zip(intervals, intervals[1:])))
                self.assertLessEqual(max(b-a for a, b in zip(m['y'], m['y'][1:])),
                                     spacing+2*KNOT_MERGE_MM)
                self.assertLess(max(m['approximation']['port_area_relative_errors'].values()), .005)

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
