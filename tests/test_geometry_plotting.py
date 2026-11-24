"""Offline P003 reconstruction, physical sampling, and exact raster exports."""

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield.benchmarking import plot_geometry
from flowfield.benchmarking.plotting import main
from flowfield.problems.level4_high_dimensional_topology.p003.config import TopologyConfig
from flowfield.problems.level4_high_dimensional_topology.p003.design import grid, preset


@unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'optional matplotlib unavailable')
class GeometryPlottingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'suite' / 'P003'
        self.run.mkdir(parents=True)
        (self.run / 'experiment.json').write_text('{}')
        self.config = TopologyConfig()
        self.saved = self.run / 'best-feasible.json'
        self.save(preset('uniform', self.config))

    def save(self, values, config=None):
        config = config or self.config
        record = dict(evaluation=196, valid=True, feasible=True, x=values,
                      objective=-.227, result=dict(problem='P003', config=config.to_dict(),
                      case_directory='/unavailable/server/case'),
                      proposal=dict(proposed_x=[1.]*config.dimension))
        self.saved.write_text(json.dumps(record))
        return record

    def test_reconstructs_evaluated_geometry_from_standalone_json_without_cfd(self):
        # Only the small saved record is needed; no manifest, journal or mesh.
        (self.run / 'experiment.json').unlink()
        with patch('flowfield.runtime.Runtime.resolve', side_effect=AssertionError('No CFD')):
            ax = plot_geometry(self.saved)
        pixels = ax.images[0].get_array()
        self.assertEqual(pixels.shape, (65, 33))
        self.assertTrue(abs(pixels - .5).max() < 1e-12)
        self.assertEqual(ax.images[0].origin, 'lower')
        self.assertEqual(ax.images[0].get_clim(), (0, 1))
        self.assertEqual(ax.images[0].get_interpolation(), 'nearest')
        for actual, expected in zip(ax.images[0].get_extent(), (0, 11.18, 0, 22.36)):
            self.assertAlmostEqual(actual, expected)
        self.assertIn('196', ax.get_title())

    def test_saved_filter_projection_and_port_fitted_mesh_are_used(self):
        # A smaller nodal grid and a finer CFD mesh expose accidental reshapes,
        # transposition, use of defaults, or projection without radial filtering.
        config = TopologyConfig(design_nx=4, design_ny=8, mesh='medium',
                                projection_beta=3., filter_radius_m=.003)
        values = [(i+2*j)/20 for j in range(9) for i in range(5)]
        self.save(values, config)
        pixels = plot_geometry(self.saved).images[0].get_array()
        self.assertEqual(pixels.shape, (9, 5))
        x_edges, y_edges = grid(config)
        for row, column in ((0, 0), (1, 3), (7, 1), (8, 4)):
            px, py = (column+.5)*config.width_m/5, (row+.5)*config.length_m/9
            xa, xb = next((a, b) for a, b in zip(x_edges, x_edges[1:]) if a <= px < b)
            ya, yb = next((a, b) for a, b in zip(y_edges, y_edges[1:]) if a <= py < b)
            cx, cy = (xa+xb)/2, (ya+yb)/2
            weights, weighted = [], []
            for j in range(9):
                for i in range(5):
                    distance = math.hypot(i*config.width_m/4-cx, j*config.length_m/8-cy)
                    weight = max(0, 1-distance/config.filter_radius_m)
                    weights.append(weight)
                    weighted.append(weight*values[j*5+i])
            filtered = sum(weighted)/sum(weights)
            expected = 1-math.exp(-3*filtered)+filtered*math.exp(-3)
            self.assertAlmostEqual(pixels[row, column], expected)

    def test_raw_nodes_and_pixel_png_preserve_order_orientation_and_dimensions(self):
        import numpy as np
        from PIL import Image

        values = [i/(self.config.dimension-1) for i in range(self.config.dimension)]
        self.save(values)
        path = self.root / 'pixels.png'
        ax = plot_geometry(self.saved, field='design', pixels_output=path)
        data = ax.images[0].get_array()
        np.testing.assert_array_equal(data, np.array(values).reshape(65, 33))
        with Image.open(path) as image:
            self.assertEqual(image.size, (33, 65))
            raster = np.array(image.convert('L'))/255
        np.testing.assert_allclose(raster, np.flipud(data), atol=1/255)
        self.assertEqual(raster[0, -1], 1.)
        self.assertEqual(raster[-1, 0], 0.)

    def test_binary_view_thresholds_indicator_not_raw_nodes(self):
        import numpy as np
        values = [0. if i < self.config.dimension//2 else .1 for i in range(self.config.dimension)]
        self.save(values)
        continuous = plot_geometry(self.saved).images[0].get_array()
        binary_ax = plot_geometry(self.saved, threshold=.5)
        binary = binary_ax.images[0].get_array()
        np.testing.assert_array_equal(binary, continuous >= .5)
        self.assertEqual(set(binary.ravel()), {0., 1.})
        self.assertIn('display only', binary_ax.get_title())

    def test_cli_discovers_current_run_ignores_resume_snapshots_and_keeps_inputs(self):
        from PIL import Image
        snapshot = self.run / 'resume-sessions' / '000001'
        snapshot.mkdir(parents=True)
        (snapshot / 'best-feasible.json').write_text('null')
        (snapshot / 'previous-experiment.json').write_text('{}')
        before = {p: p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        figure, pixels = self.root / 'figure.svg', self.root / 'raster.png'
        with patch('flowfield.runtime.Runtime.resolve', side_effect=AssertionError('No CFD')):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(self.run.parent), '--kind', 'geometry',
                                       '--output', str(figure), '--pixels-output', str(pixels)]), 0)
        self.assertIn('<svg', figure.read_text())
        with Image.open(pixels) as image:
            self.assertEqual(image.size, (33, 65))
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_rejects_missing_invalid_infeasible_or_wrong_problem_results(self):
        record = json.loads(self.saved.read_text())
        invalid = [None, {}, record | {'valid': False}, record | {'feasible': False},
                   record | {'result': dict(problem='P002')},
                   record | {'result': dict(problem='P003', config={})},
                   record | {'result': record['result'] | {'model_revision': 'unknown'}},
                   record | {'x': [0.]},
                   record | {'x': [True]*self.config.dimension},
                   record | {'x': [float('nan')]*self.config.dimension},
                   record | {'x': [1.1]*self.config.dimension}]
        for value in invalid:
            with self.subTest(value=str(value)[:100]):
                self.saved.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    plot_geometry(self.saved)
        self.saved.unlink()
        with self.assertRaisesRegex(ValueError, 'No saved best-feasible'):
            plot_geometry(self.run)

    def test_rejects_ambiguous_runs_and_invalid_plot_options(self):
        second = self.run.parent / 'another-run'
        second.mkdir()
        (second / 'experiment.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            plot_geometry(self.run.parent)
        for options in ({'field': 'unknown'}, {'threshold': float('nan')},
                        {'threshold': True}, {'threshold': -.1}, {'threshold': 1.1},
                        {'field': 'design', 'threshold': .5}, {'pixels_output': 'pixels.svg'},
                        {'output': 'same.png', 'pixels_output': 'same.png'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                plot_geometry(self.saved, **options)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            main([str(self.run), '--field', 'design', '--output', str(self.root/'out.png')])
        self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
