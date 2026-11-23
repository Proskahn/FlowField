"""GP/EI behavior, scalar observations, budgets and deterministic journal replay."""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield.optimization import BayesianOptimizationSampler, optimize, main, list_optimizers
from flowfield.optimization.bo import expected_improvement
from flowfield.optimization.state import ResumeError
from flowfield.problem import FlowFieldProblem
from flowfield.runtime import SimulationError

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('numpy', 'scipy', 'sklearn'))


class Quadratic(FlowFieldProblem):
    def __init__(self, failures=(), interrupt=None):
        super().__init__('BOTest', 1, [-2.], [4.])
        self.calls = []
        self.failures, self.interrupt = failures, interrupt

    def decode(self, x):
        return self.validate_design(x)

    def evaluate_raw(self, x):
        self.calls.append(list(x))
        if len(self.calls) == self.interrupt:
            raise KeyboardInterrupt()
        if len(self.calls) in self.failures:
            raise SimulationError('deliberate test failure')
        value = (x[0] - .3)**2
        return dict(status='ok', objective=value, objectives={'opposite': -value},
                    constraints={}, metadata={'valid_cfd': True})


@unittest.skipUnless(AVAILABLE, 'optional BO dependencies unavailable')
class BOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_problem(self, problem=None, name='run', **options):
        return optimize(problem or Quadratic(), optimizer='bo', output=self.root/name,
                        **(dict(budget=8, bo_initial_points=3, bo_candidates=64, with_ioh=False) | options))

    def rows(self, name='run'):
        return [json.loads(line) for line in (self.root/name/'evaluations.jsonl').read_text().splitlines()]

    def test_ei_minimization_and_zero_variance(self):
        import numpy as np
        ei = expected_improvement([0., -2., 2., 1e6], [1., 0., 0., 1.], 0., xi=0.)
        np.testing.assert_allclose(ei, [1/math.sqrt(2*math.pi), 2., 0., 0.])

    def test_protocol_bounds_constant_observations_and_private_rng(self):
        import numpy as np
        state = np.random.get_state()
        problem = Quadratic()
        sampler = BayesianOptimizationSampler(problem, seed=0, initial=[1.],
                                               bo_initial_points=3, bo_candidates=32)
        with self.assertRaises(RuntimeError):
            sampler.ask()
        sampler.initialize([1.], dict(objective=1e300))
        with self.assertRaises(ValueError):
            sampler.initialize([1.], dict(objective=1.))
        seen = {(1.,)}
        for _ in range(5):
            designs = sampler.ask()
            problem.validate_design(designs[0])
            self.assertNotIn(tuple(designs[0]), seen)
            seen.add(tuple(designs[0]))
            with self.assertRaises(RuntimeError):
                sampler.ask()
            with self.assertRaises(ValueError):
                sampler.tell(designs, [])
            with self.assertRaises(ValueError):
                sampler.tell(designs, [dict(objective=math.nan)])
            sampler.tell(designs, [dict(objective=1e300)])
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_budget_no_unused_model_fit_and_improvement(self):
        original = BayesianOptimizationSampler._fit
        for budget in (1, 3, 10):
            fits = []
            def fit(sampler):
                fits.append(len(sampler.x))
                original(sampler)
            with patch.object(BayesianOptimizationSampler, '_fit', fit):
                problem = Quadratic()
                summary = self.run_problem(problem, name=str(budget), budget=budget)
            self.assertEqual(len(problem.calls), budget)
            self.assertEqual(len(fits), max(0, budget-3))
            self.assertEqual(summary['stop_reason'], 'evaluation_budget')
            self.assertEqual(summary['complete_generations'], 0)
            self.assertEqual(summary['algorithm'], 'Vanilla BO')
            if budget == 10:
                rows = self.rows('10')
                initial_best = min(r['objective'] for r in rows[:3])
                self.assertLess(summary['best_objective'], initial_best*.1)
                self.assertLess(summary['best_objective'], 1e-2)
                self.assertEqual(summary['best_objective'], min(r['objective'] for r in rows))
                self.assertFalse((self.root/'10'/'pareto.json').exists())

    def test_seed_reproducibility_and_variation(self):
        for name, seed in (('one', 0), ('two', 0), ('three', 1)):
            self.run_problem(name=name, seed=seed, budget=5)
        designs = lambda name: [r['x'] for r in self.rows(name)]
        self.assertEqual(designs('one'), designs('two'))
        self.assertNotEqual(designs('one'), designs('three'))

    def test_resume_initialization_and_gp_phase_without_repeat_evaluations(self):
        reference = Quadratic()
        self.run_problem(reference, name='reference', budget=7)
        for stop in (2, 5):
            name = str(stop)
            before, after = Quadratic(), Quadratic()
            def progress(problem, index, budget, status):
                if index == stop and status == 'ok':
                    raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                self.run_problem(before, name=name, budget=7, progress=progress)
            prefix = (self.root/name/'evaluations.jsonl').read_bytes()
            summary = self.run_problem(after, name=name, budget=7, resume=True)
            self.assertEqual(before.calls + after.calls, reference.calls)
            self.assertEqual(summary['new_attempts'], 7-stop)
            self.assertTrue((self.root/name/'evaluations.jsonl').read_bytes().startswith(prefix))

    def test_failed_baseline_and_recovered_attempt_are_penalty_observations(self):
        reference = Quadratic(failures=(1, 4))
        self.run_problem(reference, name='reference', budget=6, failure_penalty=100.)
        before, after = Quadratic(failures=(1,), interrupt=4), Quadratic()
        with self.assertRaises(KeyboardInterrupt):
            self.run_problem(before, budget=6, failure_penalty=100.)
        summary = self.run_problem(after, budget=6, failure_penalty=100., resume=True)
        self.assertEqual(before.calls + after.calls, reference.calls)
        self.assertEqual(summary['failed_evaluations'], 2)
        self.assertEqual(summary['successful_evaluations'], 4)
        self.assertLess(summary['best_objective'], 100.)

    def test_all_failures_and_default_error_policy(self):
        summary = self.run_problem(Quadratic(failures=range(1, 6)), budget=5, failure_penalty=100.)
        self.assertEqual(summary['attempts'], 5)
        self.assertEqual(summary['failed_evaluations'], 5)
        self.assertIsNone(summary['best_feasible_objective'])
        with self.assertRaises(SimulationError):
            self.run_problem(Quadratic(failures=(1,)), name='raises')
        self.assertEqual(len(self.rows('raises')), 1)

    def test_resume_rejects_settings_and_dependency_changes_before_cfd(self):
        self.run_problem(budget=3)
        problem = Quadratic()
        for options in (dict(bo_initial_points=4), dict(bo_candidates=32)):
            with self.assertRaisesRegex(ResumeError, 'optimizer_settings'):
                self.run_problem(problem, budget=3, resume=True, **options)
        manifest = self.root/'run'/'experiment.json'
        saved = json.loads(manifest.read_text())
        self.assertIsNone(saved['population'])
        self.assertIsNone(saved['sigma_normalized'])
        for dependency in ('numpy', 'scipy', 'scikit-learn'):
            changed = json.loads(json.dumps(saved))
            changed['versions'][dependency] = 'different'
            manifest.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ResumeError, dependency):
                self.run_problem(problem, budget=3, resume=True)
        self.assertFalse(problem.calls)

    def test_cli_registration_validation_and_resume(self):
        self.assertIn('bo', list_optimizers())
        for options in (dict(bo_initial_points=1), dict(bo_candidates=0)):
            with self.assertRaises(ValueError):
                self.run_problem(**options)
        self.assertFalse((self.root/'run').exists())
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            main(['--problem', 'P002', '--optimizer', 'bo', '--output', str(self.root/'plan'), '--dry-run'])
        self.assertEqual(json.loads(capture.getvalue())['problems'][0]['algorithm'], 'Vanilla BO')
        self.assertFalse((self.root/'plan').exists())
        problem = Quadratic()
        problem.name = 'P000'
        with patch('flowfield.optimization.cli._make_problem', return_value=problem), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--problem', 'P000', '--optimizer', 'bo', '--budget', '4',
                '--bo-initial-points', '3', '--bo-candidates', '32', '--output', str(self.root/'cli')]), 0)
            self.assertEqual(main(['--resume', '--output', str(self.root/'cli')]), 0)
        self.assertEqual(len(problem.calls), 4)

    def test_benchmark_matrix_and_integer_rejection(self):
        from flowfield.benchmarking.runner import run_benchmark
        from flowfield.benchmarking.analysis import compare_runs
        from flowfield.optimization.registry import resolve_optimizer
        with patch('flowfield.benchmarking.runner.create_problem', side_effect=lambda *a, **k: Quadratic()):
            result = run_benchmark(problems=['P000'], optimizers=['bo'], seeds=[0, 1],
                budget=4, output=self.root/'benchmark', with_ioh=False,
                optimizer_options={'bo': dict(bo_initial_points=3, bo_candidates=32)})
        self.assertEqual(result['status'], 'completed')
        folders = [self.root/'benchmark'/r['directory'] for r in result['runs']]
        report = compare_runs(folders)
        self.assertEqual(len(report['algorithms']['Vanilla BO']['runs']), 2)
        problem = Quadratic()
        problem.variable_type = 'integer'
        with self.assertRaisesRegex(ValueError, 'does not support integer'):
            resolve_optimizer(problem, 'bo')

    def test_high_dimension_uses_all_variables_without_local_search(self):
        problem = Quadratic()
        problem.dimension = 2145
        problem.lower_bounds, problem.upper_bounds = [0.]*2145, [1.]*2145
        initial = problem.initial_design()
        sampler = BayesianOptimizationSampler(problem, seed=42, initial=initial,
                                               bo_initial_points=2, bo_candidates=16)
        sampler.initialize(initial, dict(objective=1.))
        designs = sampler.ask()
        sampler.tell(designs, [dict(objective=2.)])
        with patch.object(sampler, 'minimize', side_effect=AssertionError('high-dimensional local search')):
            designs = sampler.ask()
        self.assertEqual(len(designs[0]), 2145)
        problem.validate_design(designs[0])

    def test_ioh_logs_every_attempt(self):
        if importlib.util.find_spec('ioh') is None:
            self.skipTest('optional IOH unavailable')
        problem = Quadratic(failures=(2,))
        self.run_problem(problem, budget=5, failure_penalty=100., with_ioh=True)
        path = next((self.root/'run'/'ioh').rglob('*.dat'))
        self.assertEqual([int(line.split()[0]) for line in path.read_text().splitlines()[1:]], list(range(1, 6)))
        self.assertEqual(len(problem.calls), 5)


if __name__ == '__main__':
    unittest.main()
