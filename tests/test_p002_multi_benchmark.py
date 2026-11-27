"""P002 campaign integration: real optimizers/IOH, explicitly synthetic CFD."""

from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield.benchmarking.inspector import analyze_runs
from flowfield.benchmarking.runner import main
from flowfield.problems.level2_medium_parametric.p002 import BASELINE_X, P002Problem
from flowfield.problems.level2_medium_parametric.p002.config import scalar_objective
from flowfield.runtime import SimulationError


METHODS = ('random', 'nsga2', 'qlognehvi', 'qlognparego')
OBJECTIVES = ['pressure_sum_Pa', 'voltage_V']
EVALUATE = 'flowfield.problems.level2_medium_parametric.p002.evaluation.evaluate'


def synthetic_evaluator(calls, *, failures=()):
    """Replace only the CFD boundary; exercise P002 decode/config/result metadata."""
    def evaluate(geometry, config, output_root, runtime):
        x = geometry['normalized_design']
        calls[str(output_root)].append(x)
        attempt = len(calls[str(output_root)])
        if attempt in failures:
            raise SimulationError('Synthetic rejected mesh')
        voltage = 2.2 + .05*x[0] + .01*x[5]
        pressure = 120. + 300.*(1.-x[0]) + 20.*x[6]
        case = Path(output_root)/str(attempt)
        case.mkdir(parents=True)
        return dict(status='ok', objective=scalar_objective(voltage, pressure, config),
                    objectives=dict(voltage_V=voltage, pressure_sum_Pa=pressure), constraints={},
                    case_directory=str(case), runtime={'backend': 'synthetic-test'},
                    metrics={'synthetic_test_only': True})
    return evaluate


def physical_hypervolume(points, reference):
    """Independent 2D rectangle union, including dominated/duplicate observations."""
    high_v, high_p = reference['voltage_V'], reference['pressure_sum_Pa']
    area = 0.
    for voltage, pressure in sorted(points):
        if voltage < high_v and pressure < high_p:
            area += (high_v-voltage)*(high_p-pressure)
            high_p = pressure
    return area


class P002CampaignTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def invoke(self, *args):
        stream = io.StringIO()
        with redirect_stdout(stream), redirect_stderr(io.StringIO()):
            code = main(list(args))
        self.assertEqual(code, 0)
        return json.loads(stream.getvalue())

    def test_plan_preserves_seven_variables_and_routes_population_and_failure_policy(self):
        output = self.root/'plan'
        with patch.object(P002Problem, 'preflight', side_effect=AssertionError('No CFD')), \
             patch.object(P002Problem, 'evaluate_raw', side_effect=AssertionError('No evaluations')):
            plan = self.invoke('--problem', 'P002', '--regime', 'multi', '--optimizers', *METHODS,
                '--seeds', '42', '43', '--budget', '20', '--population', '8',
                '--failure-penalty', '1000000', '--output', str(output), '--dry-run')
        self.assertFalse(output.exists())
        self.assertEqual((plan['run_count'], plan['maximum_attempts']), (8, 160))
        self.assertEqual(plan['instances'][0]['dimension'], 7)
        self.assertEqual(plan['instances'][0]['initial_design'], list(BASELINE_X))
        self.assertEqual(plan['optimizer_regimes'], dict.fromkeys(METHODS, 'multi'))
        for method, settings in plan['optimizer_options'].items():
            self.assertEqual(settings['failure_penalty'], 1000000.)
            self.assertEqual(settings.get('population'), 8 if method == 'nsga2' else None)

    def test_invalid_controls_and_regimes_fail_without_output(self):
        for method, options in [('nsga2', ['--population', '3']),
                                ('nsga2', ['--failure-penalty', 'nan']),
                                ('nsga2', ['--failure-penalty', '-1']),
                                ('random', ['--population', '8']),
                                ('cmaes', ['--regime', 'multi'])]:
            with self.subTest(method=method, options=options), \
                 patch.object(P002Problem, 'preflight', side_effect=AssertionError('No CFD')), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as error:
                main(['--problem', 'P002', '--optimizers', method, '--seeds', '42',
                      '--output', str(self.root/'bad'), '--dry-run', *options])
            self.assertNotEqual(error.exception.code, 0)
            self.assertFalse((self.root/'bad').exists())

    def test_no_failure_penalty_still_stops_at_first_failed_cfd(self):
        calls = defaultdict(list)
        folder = self.root/'stop'
        with patch.object(P002Problem, 'preflight'), \
             patch(EVALUATE, synthetic_evaluator(calls, failures=(2,))), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit):
            main(['--problem', 'P002', '--regime', 'multi', '--optimizers', 'nsga2',
                  '--population', '4', '--budget', '6', '--seeds', '42',
                  '--no-ioh', '--output', str(folder)])
        self.assertEqual(sum(map(len, calls.values())), 2)
        manifest = json.loads((folder/'benchmark.json').read_text())
        self.assertEqual(manifest['status'], 'failed')

    def test_nsga2_uses_same_pareto_objectives_for_each_p002_scalarization(self):
        histories = []
        for scalarization in ('voltage_v', 'pressure_sum_pa', 'total_power_w'):
            config = self.root/f'{scalarization}.json'
            config.write_text(json.dumps(dict(scalarization=scalarization)))
            folder = self.root/scalarization
            with patch.object(P002Problem, 'preflight'), \
                 patch(EVALUATE, synthetic_evaluator(defaultdict(list))):
                self.invoke('--problem', 'P002', '--regime', 'multi', '--optimizers', 'nsga2',
                    '--population', '4', '--budget', '12', '--seeds', '42', '--no-ioh',
                    '--config', str(config), '--output', str(folder))
            rows = [json.loads(line) for line in (folder/'P002/nsga2/seed-42/evaluations.jsonl').read_text().splitlines()]
            histories.append(rows)
        for rows in histories[1:]:
            self.assertEqual([r['x'] for r in rows], [r['x'] for r in histories[0]])
            self.assertEqual([r['result']['objectives'] for r in rows],
                             [r['result']['objectives'] for r in histories[0]])
            self.assertNotEqual([r['objective'] for r in rows], [r['objective'] for r in histories[0]])

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in
        ('numpy', 'scipy', 'botorch', 'ioh', 'iohinspector', 'matplotlib')), 'optional benchmark dependencies unavailable')
    def test_all_four_optimizers_log_failures_and_match_independent_hypervolume(self):
        calls = defaultdict(list)
        folder = self.root/'campaign'
        with patch.object(P002Problem, 'preflight'), \
             patch(EVALUATE, synthetic_evaluator(calls, failures=(2,))):
            manifest = self.invoke('--problem', 'P002', '--regime', 'multi', '--optimizers', *METHODS,
                '--seeds', '42', '--budget', '6', '--population', '4', '--failure-penalty', '1000000',
                '--bo-initial-points', '3', '--bo-candidates', '32', '--mobo-mc-samples', '16',
                '--output', str(folder))
        self.assertEqual(manifest['status'], 'completed_with_failures')
        self.assertEqual(sum(map(len, calls.values())), 24)
        self.assertTrue(all(len(xs) == 6 and xs[0] == list(BASELINE_X) for xs in calls.values()))
        sources = list(folder.rglob('evaluations.jsonl')) + list(folder.rglob('experiment.json'))
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        with patch.object(P002Problem, 'preflight', side_effect=AssertionError('Analysis cannot run CFD')), \
             patch(EVALUATE, side_effect=AssertionError('Analysis cannot evaluate')):
            report = analyze_runs(folder, output=self.root/'analysis', regime='multi')
        self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        self.assertEqual(report['objectives'], OBJECTIVES)
        self.assertEqual(report['identity']['dimension'], 7)
        self.assertEqual(report['settings']['normalization'], 'pooled_feasible_min_max')
        reports = {Path(run['directory']): run for run in report['runs']}
        for entry in manifest['runs']:
            path = folder/entry['directory']
            run = reports[path.resolve()]
            self.assertEqual(entry['objective_names'], OBJECTIVES)
            self.assertEqual(entry['scalar_score_role'], 'diagnostic only')
            self.assertTrue(list((path/'ioh').rglob('*.json')))
            experiment = json.loads((path/'experiment.json').read_text())
            self.assertEqual(experiment['failure_penalty'], 1000000.)
            self.assertEqual(experiment['population'], 4 if entry['optimizer'] == 'nsga2' else None)
            rows = [json.loads(line) for line in (path/'evaluations.jsonl').read_text().splitlines()]
            self.assertEqual((run['attempts'], run['failed_evaluations']), (6, 1))
            self.assertFalse(rows[1]['valid'])
            self.assertEqual(rows[1]['result']['objectives'], {})
            self.assertEqual(run['values'][0], run['values'][1])
            points = []
            scale = report['settings']['scale']
            for row, actual in zip(rows, run['values']):
                if row['feasible']:
                    raw = row['result']['objectives']
                    points.append((raw['voltage_V'], raw['pressure_sum_Pa']))
                expected = physical_hypervolume(points, report['settings']['reference_point'])
                self.assertAlmostEqual(actual, expected/(scale['voltage_V']*scale['pressure_sum_Pa']), places=10)
            archive = json.loads((path/'pareto.json').read_text())
            expected = {r['evaluation'] for r in rows if r['feasible'] and not any(
                all(s['result']['objectives'][k] <= r['result']['objectives'][k] for k in OBJECTIVES)
                and any(s['result']['objectives'][k] < r['result']['objectives'][k] for k in OBJECTIVES)
                for s in rows if s['feasible'])}
            self.assertEqual({r['evaluation'] for r in archive['solutions']}, expected)


if __name__ == '__main__':
    unittest.main()
