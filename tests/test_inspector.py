"""Real IOHinspector imports, known hypervolumes and strict regime boundaries."""

from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield.problem import FlowFieldProblem
from flowfield.optimization import optimize
from flowfield.optimization.cli import main as optimize_main
from flowfield.optimization.registry import resolve_regime
from flowfield.optimization.state import ResumeError
from flowfield.benchmarking.planning import plan_benchmark
from flowfield.benchmarking.runner import run_benchmark, main as benchmark_main
from flowfield.benchmarking.inspector import analyze_runs, main as analyze_main


class Tradeoff(FlowFieldProblem):
    def __init__(self, **kwargs):
        super().__init__('InspectorTest', 2, [0., 0.], [1., 1.])
        self.calls = []

    def decode(self, x):
        return self.validate_design(x)

    def evaluate_raw(self, x):
        self.calls.append(x)
        return dict(status='ok', objective=sum(x), objectives={'a': x[0], 'b': 1-x[0]+x[1]},
                    constraints={'volume': x[1]-.8})


class RegimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_explicit_compatibility_before_preflight_or_output(self):
        for method, regime in (('turbo', 'multi'), ('cmaes', 'multi'), ('nsga2', 'single'),
                               ('qlognehvi', 'single'), ('random', 'typo')):
            p = Tradeoff()
            with self.subTest(method=method, regime=regime), \
                 patch.object(p, 'preflight', side_effect=AssertionError('No preflight')), \
                 self.assertRaises(ValueError):
                optimize(p, output=self.root/'bad', optimizer=method, regime=regime)
            self.assertFalse((self.root/'bad').exists())
            self.assertEqual(p.calls, [])
        self.assertEqual(resolve_regime('qlognehvi'), 'multi')
        self.assertEqual(resolve_regime('random'), 'single')

    def test_random_multi_archive_and_resume_retain_regime(self):
        p = Tradeoff()
        path = self.root/'random'
        summary = optimize(p, output=path, optimizer='random', regime='multi', budget=3, with_ioh=False)
        self.assertEqual(summary['regime'], 'multi')
        self.assertGreater(summary['pareto_size'], 0)
        before = len(p.calls)
        result = optimize(p, output=path, optimizer='random', resume=True, extend_budget=5, with_ioh=False)
        self.assertEqual(len(p.calls)-before, 2)
        self.assertEqual(result['regime'], 'multi')
        with self.assertRaisesRegex(ResumeError, 'regime'):
            optimize(p, output=path, optimizer='random', regime='single', budget=5, resume=True, with_ioh=False)

    def test_explicit_multi_rejects_scalar_only_problem(self):
        p = Tradeoff()
        with patch.object(p, 'evaluate_raw', return_value=dict(objective=1., objectives={'only': 1.}, constraints={})), \
             self.assertRaisesRegex(ValueError, 'at least two'):
            optimize(p, output=self.root/'scalar', optimizer='random', regime='multi', budget=2, with_ioh=False)

    def test_cli_resume_restores_multi_random_and_legacy_inference(self):
        p = Tradeoff()
        p.name = 'P000'
        for regime in ([], ['--regime', 'multi']):
            folder = self.root/('explicit' if regime else 'legacy')
            with patch('flowfield.optimization.cli._make_problem', return_value=p), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(optimize_main(['--problem', 'P000', '--optimizer', 'random',
                    '--budget', '3', '--output', str(folder), *regime]), 0)
                self.assertEqual(optimize_main(['--resume', '--extend-budget', '5', '--output', str(folder)]), 0)
            summary = json.loads((folder/'P000/summary.json').read_text())
            self.assertEqual(summary['regime'], 'multi' if regime else 'single')
            self.assertEqual(summary['new_attempts'], 2)

    def test_cli_plans_and_benchmark_random_multi(self):
        for main, flags in ((optimize_main, ['--optimizer', 'random']),
                            (benchmark_main, ['--optimizers', 'random', 'qlognehvi', '--seeds', '42'])):
            stream = io.StringIO()
            with redirect_stdout(stream):
                main(['--problem', 'P003', '--dimension', '500', '--regime', 'multi',
                      '--output', str(self.root/'plan'), '--dry-run', *flags])
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan.get('regime', plan['problems'][0].get('regime') if isinstance(plan['problems'][0], dict) else None), 'multi')
        self.assertFalse((self.root/'plan').exists())
        with patch('flowfield.benchmarking.planning.create_problem', side_effect=lambda *a, **k: Tradeoff()), \
             patch('flowfield.benchmarking.runner.create_problem', side_effect=lambda *a, **k: Tradeoff()):
            # P000 catalog metadata is used only to avoid registering a test ID.
            manifest = run_benchmark(problems=['P000'], optimizers=['random', 'nsga2'], seeds=[42],
                                     budget=4, regime='multi', output=self.root/'matrix', with_ioh=False)
        self.assertEqual(manifest['regime'], 'multi')
        self.assertTrue(all(r['pareto_size'] > 0 for r in manifest['runs']))
        with self.assertRaisesRegex(ValueError, 'single-objective'):
            plan_benchmark(problems=['P003'], optimizers=['turbo'], seeds=[42], budget=2, regime='multi')


@unittest.skipUnless(importlib.util.find_spec('iohinspector'), 'optional IOHinspector unavailable')
class InspectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make_run(self, name='run', seed=42, objectives=None, **changes):
        folder = self.root/name
        folder.mkdir()
        exp = dict(algorithm='Test', optimizer='random', problem='Test', problem_name='test_config',
                   dimension=2, budget=6, seed=seed, config={'mesh': 'fixed'}, failure_penalty=None,
                   objective_direction='minimize', **changes)
        (folder/'experiment.json').write_text(json.dumps(exp))
        # HV with reference (4, 4): 3, 5, 5, 5, 5, 9. Last point dominates
        # earlier points; final-front-only logging would lose their early progress.
        vectors = objectives or [(1, 3), (2, 2), (0, 0), None, (3, 3), (1, 1)]
        rows = []
        for i, vector in enumerate(vectors, 1):
            valid = vector is not None
            feasible = valid and i != 3
            raw = (dict(status='ok', objective=float(i), objectives=dict(zip(('a', 'b'), vector)),
                        constraints={'volume': -1. if feasible else 1.}) if valid else {})
            rows.append(dict(evaluation=i, x=[.2, .3], phase='baseline' if i == 1 else 'candidate',
                             generation=0, valid=valid, feasible=feasible, result=raw,
                             objective=float(i) if valid else None, wall_time_s=.5))
        self.write_rows(folder, rows)
        return folder

    def write_rows(self, folder, rows):
        (folder/'evaluations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def hashes(self, folder):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}

    def test_known_hypervolume_uses_full_vectors_and_counts_failures(self):
        import iohinspector
        folder = self.make_run()
        before = self.hashes(folder)
        out = self.root/'analysis'
        with patch('flowfield.registry.create_problem', side_effect=AssertionError('No CFD')), \
             patch.object(iohinspector, 'add_indicator', wraps=iohinspector.add_indicator) as indicator:
            report = analyze_runs([folder], output=out, reference_point={'a': 4., 'b': 4.})
        indicator.assert_called_once()
        self.assertEqual(report['runs'][0]['values'], [3., 5., 5., 5., 5., 9.])
        self.assertEqual(report['runs'][0]['failed_evaluations'], 1)
        self.assertEqual(report['runs'][0]['cumulative_evaluation_wall_time_s'], [.5, 1., 1.5, 2., 2.5, 3.])
        self.assertEqual(self.hashes(folder), before)
        manager = iohinspector.DataManager()
        manager.add_folder(out/'evaluations.csv', mapping=out/'mapping.json')
        data = manager.load(monotonic=False, include_meta_data=True)
        self.assertEqual(len(data), 6)
        self.assertEqual(data['objective_0'][2], 0.)  # Infeasible raw value retained.
        self.assertGreater((out/'convergence.png').stat().st_size, 1000)
        from iohinspector.verify import verify
        checked = verify(out/'evaluations.csv', out/'iohinspector')
        self.assertTrue(checked.ok, checked.to_dict())

    def test_auto_normalization_is_pooled_and_frozen_settings_reproduce(self):
        a = self.make_run('a')
        b = self.make_run('b', seed=43, objectives=[(2, 4), (3, 3), (0, 0), None, (4, 4), (2, 2)])
        report = analyze_runs([a, b], output=self.root/'auto')
        self.assertEqual(report['settings']['offset'], {'a': 1, 'b': 1})
        self.assertEqual(report['settings']['scale'], {'a': 3, 'b': 3})
        again = analyze_runs([a], output=self.root/'reuse', settings=self.root/'auto/analysis-settings.json')
        self.assertEqual(again['runs'][0]['values'], report['runs'][0]['values'])
        for run in report['runs']:
            self.assertEqual(run['values'], sorted(run['values']))

    def test_empty_feasible_archive_zero_with_explicit_reference(self):
        folder = self.make_run(objectives=[None]*6)
        with self.assertRaisesRegex(ValueError, 'schema'):
            analyze_runs(folder, output=self.root/'bad')
        report = analyze_runs(folder, output=self.root/'empty', reference_point={'a': 4, 'b': 4})
        self.assertEqual(report['runs'][0]['values'], [0.]*6)

    def test_recovery_and_unfinished_attempt_are_counted_once(self):
        folder = self.make_run()
        rows = [json.loads(line) for line in (folder/'evaluations.jsonl').read_text().splitlines()]
        starts = [{k: r[k] for k in ('evaluation', 'x', 'phase', 'generation')} | {'event': 'started'} for r in rows]
        (folder/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in starts))
        recovered = dict(rows[3], objective=1e6, result=dict(status='penalized', objective=1e6, objectives={}, constraints={}))
        (folder/'recoveries.jsonl').write_text(json.dumps(recovered)+'\n')
        self.write_rows(folder, rows[:-1])  # Sixth started, but no finished result.
        report = analyze_runs(folder, output=self.root/'recover', reference_point={'a': 4, 'b': 4})
        self.assertEqual(report['runs'][0]['values'], [3., 5., 5., 5., 5., 5.])
        self.assertIsNone(report['runs'][0]['cumulative_evaluation_wall_time_s'][-1])

    def test_reject_mixed_instances_duplicate_seeds_schema_and_bad_settings(self):
        a = self.make_run('a')
        b = self.make_run('b', seed=43)
        c = self.make_run('c')
        with self.assertRaisesRegex(ValueError, 'Duplicate algorithm/seed'):
            analyze_runs([a, c], output=self.root/'duplicate')
        exp = json.loads((b/'experiment.json').read_text())
        exp['dimension'] = 3
        (b/'experiment.json').write_text(json.dumps(exp))
        with self.assertRaisesRegex(ValueError, 'same problem'):
            analyze_runs([a, b], output=self.root/'mixed')
        with self.assertRaisesRegex(ValueError, 'Reference point'):
            analyze_runs(a, output=self.root/'badref', reference_point={'a': 4})
        rows = [json.loads(line) for line in (a/'evaluations.jsonl').read_text().splitlines()]
        rows[1]['result']['objectives']['c'] = rows[1]['result']['objectives'].pop('b')
        self.write_rows(a, rows)
        with self.assertRaisesRegex(ValueError, 'names differ'):
            analyze_runs(a, output=self.root/'schema')

    def test_single_regime_and_cli_do_not_change_sources(self):
        folder = self.make_run()
        report = analyze_runs(folder, output=self.root/'single', regime='single')
        self.assertEqual(report['indicator'], 'best_feasible_objective')
        self.assertEqual(report['runs'][0]['values'], [1.]*6)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(analyze_main(['--input', str(folder/'evaluations.jsonl'), '--output', str(self.root/'cli'),
                                         '--reference-point', '{"a":4,"b":4}']), 0)
        self.assertTrue((self.root/'cli/indicators.csv').is_file())

    @unittest.skipUnless(importlib.util.find_spec('botorch'), 'optional BoTorch unavailable')
    def test_p003_poc_pipeline_with_real_optimizers_and_synthetic_cfd(self):
        from flowfield import P003Problem
        calls = []
        def synthetic(problem, x):
            volume = problem.decode(x)['volume_fraction']
            self.assertLessEqual(volume, problem.config.volume_limit+1e-12)
            calls.append(x)
            power_loss = sum((v-.01)**2 for v in x)
            uniformity = sum((v-.02)**2 for v in x)
            return dict(status='ok', objective=power_loss,
                        objectives={'negative_power_domain_W': power_loss, 'current_std_A_m2': uniformity},
                        constraints={'channel_volume': volume-problem.config.volume_limit})
        folder = self.root/'poc'
        with patch.object(P003Problem, 'preflight'), patch.object(P003Problem, 'evaluate_raw', synthetic), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(benchmark_main(['--problem', 'P003', '--dimension', '500', '--regime', 'multi',
                '--optimizers', 'random', 'qlognehvi', 'qlognparego', '--seeds', '42', '--budget', '6',
                '--bo-initial-points', '3', '--bo-candidates', '64', '--mobo-mc-samples', '32',
                '--backend', 'docker', '--timeout', '14400', '--output', str(folder)]), 0)
        self.assertEqual(len(calls), 18)
        self.assertTrue(all(len(x) == 500 for x in calls))
        report = analyze_runs(folder, output=self.root/'poc-analysis')
        self.assertEqual(len(report['runs']), 3)
        self.assertTrue(all(r['attempts'] == 6 for r in report['runs']))
        self.assertTrue(all(r['values'] == sorted(r['values']) for r in report['runs']))


if __name__ == '__main__':
    unittest.main()
