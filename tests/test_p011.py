"""Tapered PEMFC design, provenance, qualification and optimizer contracts.

Synthetic history fixtures exercise acceptance plumbing only; they are never
reported as CFD validation or used to fit the published performance.
"""
import contextlib
import csv
from dataclasses import replace
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield import CFDProblem, create_problem, problem_info
from flowfield.evaluation import CaseEvaluator
from flowfield.problems.level1_small_continuous.p011 import P011Problem, TaperedConfig
from flowfield.problems.level1_small_continuous.p011.config import (
    BASELINE, BOUNDS, DOI, PARAMETERS, REFERENCE_WIDTHS, normalized)
from flowfield.stability.profiles import experiments, limits, observables


class P011Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_area_constraint_and_reference_designs(self):
        problem = P011Problem()
        designs = list(itertools.product((0., 1.), repeat=4))
        designs += [problem.initial_design(), [.13, .27, .59, .91]]
        for design in designs:
            with self.subTest(design=design):
                geometry = problem.decode(design)
                self.assertEqual(geometry['normalized_design'], list(design))
                for key, (lower, upper) in zip(PARAMETERS, BOUNDS):
                    width, height = geometry[key], geometry[key.replace('width', 'height')]
                    self.assertGreaterEqual(width, lower)
                    self.assertLessEqual(width, upper)
                    self.assertGreaterEqual(height, lower)
                    self.assertLessEqual(height, upper)
                    self.assertAlmostEqual(width*height, 1., places=14)
                self.assertEqual(geometry['constraints'], {})
                self.assertAlmostEqual(geometry['active_area_m2'], .020*.002)
        for label, widths in REFERENCE_WIDTHS.items():
            geometry = problem.decode(normalized(widths))
            for key, width in zip(PARAMETERS, widths):
                self.assertAlmostEqual(geometry[key], width, places=14)
        self.assertEqual(problem.initial_design(), normalized(BASELINE))
        iga = problem.decode(normalized(REFERENCE_WIDTHS['IGA']))
        self.assertAlmostEqual(iga['cathode_taper_angle_deg'], .38, delta=.01)
        self.assertAlmostEqual(iga['anode_taper_angle_deg'], .84, delta=.02)

    def test_invalid_configuration_and_design_fail_before_case_creation(self):
        invalid = (
            {'processes': 2}, {'mesh': 'unknown'}, {'iterations': 2},
            {'temperature_K': 400}, {'humidity_percent': 101},
            {'outlet_pressure': 10}, {'gdl_porosity': 1},
            {'balance_tolerance': .01}, {'field_tolerance': .001},
            {'cell_voltage': 1.2}, {'reference_current_density': math.nan},
            {'objective_scale': 0}, {'iterations': True},
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                TaperedConfig(**options)
        problem = P011Problem(output_root=self.root/'cases')
        for design in ([0]*3, [0]*5, [-.01]*4, [1.01]*4,
                       [True, 0, 0, 0], [math.nan, 0, 0, 0]):
            with self.subTest(design=design), self.assertRaises(ValueError):
                problem.decode(design)
        for widths in ([1]*3, [.5]*4, [1.9]*4, [math.inf]*4, [True]*4):
            with self.subTest(widths=widths), self.assertRaises(ValueError):
                normalized(widths)
        self.assertFalse((self.root/'cases').exists())

    def test_registry_identity_and_stability_integration(self):
        problem = create_problem('P011')
        self.assertIsInstance(problem, CFDProblem)
        self.assertEqual(problem.dimension, 4)
        self.assertEqual(problem.lower_bounds, [0.]*4)
        self.assertEqual(problem.upper_bounds, [1.]*4)
        self.assertTrue(problem.requires_qualified_baseline)
        self.assertEqual(problem_info('P011').reference, DOI)
        self.assertEqual(problem_info('P011').extra, 'p011')
        self.assertEqual(problem_info('P011').level, 1)
        changed = create_problem('P011', config={'objective_scale': 1e4})
        self.assertNotEqual(changed.ioh_name, problem.ioh_name)
        self.assertIn('membrane_balance', limits('P011', problem.config))
        jobs, unsupported = experiments('P011', problem.config, ['mesh', 'numerics', 'parallel'])
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len(unsupported), 1)
        metrics = dict(current_cathode_A=2., pumping_power_W=.02,
                       net_power_W=.78, pressure_drop_Pa=10.)
        self.assertEqual(observables('P011', {'metrics': metrics}), metrics)

    def test_generated_case_preserves_immutable_mesh_and_solver_sources(self):
        problem = P011Problem()
        case = problem.write_case(problem.initial_design(), self.root/'case')
        evaluator = CaseEvaluator(problem.case_definition)
        manifest, config = evaluator._load_case(case)
        self.assertEqual(config, problem.config)
        self.assertEqual(manifest['problem'], 'P011')
        self.assertFalse(manifest['paper_validated'])
        self.assertGreater(manifest['cells'], 0)
        self.assertAlmostEqual(manifest['inlet_areas_m2']['cathodeInlet'], 1e-6, places=14)
        with self.assertRaises(FileExistsError):
            problem.write_case(problem.initial_design(), case)
        points = case/'constant/polyMesh/points'
        original = points.read_bytes()
        points.write_bytes(original+b'\n// changed input\n')
        with self.assertRaisesRegex(ValueError, 'Case input changed'):
            evaluator._load_case(case)
        points.write_bytes(original)
        # Even a matching edited manifest must not permit an unpinned solver.
        source = case/'solver/Gas.H'
        source.write_text(source.read_text()+'\n// locally changed solver\n')
        manifest['input_sha256']['solver/Gas.H'] = hashlib.sha256(source.read_bytes()).hexdigest()
        (case/'case.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Solver sources changed'):
            evaluator._load_case(case)

    def _history_fixture(self, directory='fixture'):
        from flowfield.problems.level1_small_continuous.p011.metrics import BALANCES
        config = TaperedConfig(iterations=10, minimum_iterations=1, voltage_ramp_iterations=1)
        case = self.root/directory
        output = case/'postProcessing/p011'
        output.mkdir(parents=True)
        (case/'5').mkdir()
        (case/'log.checkMesh').write_text('cells: 12\nMesh OK.\nEnd\n')
        (case/'log.flowfieldP011Foam').write_text('CONVERGED: P011 full coupled model\nEnd\n')
        (case/'case.json').write_text(json.dumps({'cells': 12}))
        status = dict(converged=True, iteration=5, cells=12, processes=1, voltage_V=.4)
        (output/'status.json').write_text(json.dumps(status))
        rows = [dict(iteration=i, voltage_V=.4, field_change=1e-7, charge_residual=1e-10,
                     **{key: 1e-7 for key in BALANCES}) for i in range(1, 6)]
        self._write_rows(output/'history.csv', rows)
        metrics = dict(current_cathode_A=2., current_anode_A=2., pumping_power_W=.02,
                       net_power_W=.78, pressure_drop_Pa=10., inlet_flow_m3_s=.002)
        return case, config, status, rows, metrics

    @staticmethod
    def _write_rows(path, rows):
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_objective_uses_gross_output_and_retains_raw_pareto_objectives(self):
        from flowfield.problems.level1_small_continuous.p011.metrics import extract_metrics
        case, config, _, _, metrics = self._history_fixture()
        # The conservation/field audit is independently exercised by CFD tests;
        # this fixture isolates the physical objective and unit conversion.
        with patch('flowfield.problems.level1_small_continuous.p011.metrics.saved_audit', return_value=metrics):
            result = extract_metrics(case, config)
        expected = 1e5*.02/(.4*2.)
        self.assertEqual(result['objective'], expected)
        self.assertNotAlmostEqual(result['objective'], 1e5*.02/.78)
        self.assertEqual(result['objectives'], {'pumping_power_W': .02, 'negative_output_power_W': -.8})
        self.assertAlmostEqual(result['metrics']['current_density_A_cm2'], 5.)
        self.assertAlmostEqual(result['metrics']['power_density_mW_cm2'], 2000.)
        self.assertFalse(result['paper_validated'])

    def test_unqualified_and_unknown_completion_states_fail_closed(self):
        from flowfield.problems.level1_small_continuous.p011.metrics import extract_metrics
        case, config, status, _, _ = self._history_fixture()
        for changed in ({'converged': False}, {'converged': 'true'},
                        {'converged': None}, {'cells': 13},
                        {'processes': 2}, {'iteration': 4}, {'voltage_V': .5}):
            with self.subTest(status=changed):
                (case/'postProcessing/p011/status.json').write_text(json.dumps(dict(status, **changed)))
                with patch('flowfield.problems.level1_small_continuous.p011.metrics.saved_audit') as audit:
                    with self.assertRaises(ValueError):
                        extract_metrics(case, config)
                    audit.assert_not_called()

    def test_status_rounding_does_not_override_precise_history_voltage(self):
        from flowfield.problems.level1_small_continuous.p011.metrics import extract_metrics
        case, config, status, rows, metrics = self._history_fixture()
        config = replace(config, cell_voltage=.41234567)
        # C++ writes this status marker at default six-significant-digit
        # precision; history is the authoritative full-precision voltage.
        status['voltage_V'] = .412346
        (case/'postProcessing/p011/status.json').write_text(json.dumps(status))
        for row in rows:
            row['voltage_V'] = config.cell_voltage
        self._write_rows(case/'postProcessing/p011/history.csv', rows)
        with patch('flowfield.problems.level1_small_continuous.p011.metrics.saved_audit', return_value=metrics):
            result = extract_metrics(case, config)
        self.assertAlmostEqual(result['objective'], 1e5*.02/(config.cell_voltage*2.))
        rows[-1]['voltage_V'] = status['voltage_V']
        self._write_rows(case/'postProcessing/p011/history.csv', rows)
        with self.assertRaisesRegex(ValueError, 'Last five'):
            extract_metrics(case, config)

    def test_bad_history_and_mesh_never_reach_physical_objective(self):
        from flowfield.problems.level1_small_continuous.p011.metrics import extract_metrics
        case, config, _, rows, _ = self._history_fixture()
        bad_rows = []
        for key, value in (('iteration', 4.), ('field_change', .1), ('gas_balance', .1),
                           ('voltage_V', .5), ('charge_residual', math.nan)):
            altered = [dict(row) for row in rows]
            altered[-1][key] = value
            bad_rows.append(altered)
        bad_rows.append(rows[:4])
        for altered in bad_rows:
            with self.subTest(last=altered[-1]):
                self._write_rows(case/'postProcessing/p011/history.csv', altered)
                with patch('flowfield.problems.level1_small_continuous.p011.metrics.saved_audit') as audit:
                    with self.assertRaises(ValueError):
                        extract_metrics(case, config)
                    audit.assert_not_called()
        self._write_rows(case/'postProcessing/p011/history.csv', rows)
        (case/'log.checkMesh').write_text('Mesh OK.\nFailed 1 mesh checks\n')
        with self.assertRaisesRegex(ValueError, 'mesh is not qualified'):
            extract_metrics(case, config)

    def test_cli_reference_is_read_only_and_rejects_immutable_overrides(self):
        from flowfield.problems.level1_small_continuous.p011.cli import main
        stdout = io.StringIO()
        with patch.object(P011Problem, 'evaluate_raw', side_effect=AssertionError('CFD forbidden')):
            with patch.object(P011Problem, 'write_case', side_effect=AssertionError('Write forbidden')):
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(main(['reference']), 0)
        reference = json.loads(stdout.getvalue())
        self.assertFalse(reference['cfd_executed'])
        self.assertFalse(reference['paper_validated'])
        for action in ('run', 'extract'):
            for override in (['--x', '.5', '.5', '.5', '.5'],
                             ['--widths-mm', '1', '1', '1', '1'], ['--design', 'IGA'],
                             ['--mesh', 'fine'], ['--config', 'custom.json'], ['--voltage', '.5']):
                with self.subTest(action=action, override=override):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main([action, '--case', str(self.root/'case'), *override])


if __name__ == '__main__':
    unittest.main()
