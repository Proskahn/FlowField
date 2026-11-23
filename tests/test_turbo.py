"""TuRBO adaptation, correlated posterior draws, restarts and durable replay."""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield.optimization import TuRBOSampler, optimize, main, list_optimizers
from flowfield.optimization.turbo import TrustRegionState, posterior_sample
from flowfield.optimization.state import ResumeError
from test_bo import AVAILABLE, Quadratic


class TrustRegionTests(unittest.TestCase):
    def test_adaptation_threshold_reset_and_restart(self):
        state = TrustRegionState(2)
        state.observe(-10., adapt=False)
        state.observe(-10.001, adapt=True)  # Too little improvement to count as a success.
        self.assertEqual((state.successes, state.failures), (0, 1))
        for value in (-11., -12., -13.):
            state.observe(value, adapt=True)
        self.assertEqual((state.length, state.successes, state.failures), (1.6, 0, 0))
        for _ in range(4):
            state.observe(-13., adapt=True)
        self.assertEqual((state.length, state.failures), (.8, 0))
        for _ in range(7*4):
            state.observe(-13., adapt=True)
        self.assertTrue(state.restart_required)
        self.assertLess(state.length, 2.**-7)
        state = TrustRegionState(7)
        state.observe(0., adapt=False)
        for _ in range(6):
            state.observe(0., adapt=True)
        self.assertEqual(state.length, .8)
        state.observe(0., adapt=True)
        self.assertEqual(state.length, .4)


class FlatProblem(Quadratic):
    def evaluate_raw(self, x):
        self.calls.append(list(x))
        return dict(objective=1., status='ok')


@unittest.skipUnless(AVAILABLE, 'optional Bayesian optimization dependencies unavailable')
class TuRBOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_problem(self, problem=None, name='run', **options):
        return optimize(problem or Quadratic(), optimizer='turbo', output=self.root/name,
                        **(dict(budget=8, turbo_initial_points=3, turbo_candidates=32, with_ioh=False) | options))

    def rows(self, name='run'):
        return [json.loads(line) for line in (self.root/name/'evaluations.jsonl').read_text().splitlines()]

    def test_posterior_draws_preserve_candidate_correlations(self):
        import numpy as np
        class Model:
            def predict(self, candidates, *, return_cov):
                self.return_cov = return_cov
                return np.array([2., 3.]), np.array([[1., .8], [.8, 1.]])
        model = Model()
        rng = np.random.RandomState(0)
        draws = np.asarray([posterior_sample(model, [[0.], [1.]], rng) for _ in range(2000)])
        self.assertTrue(model.return_cov)
        np.testing.assert_allclose(draws.mean(axis=0), [2., 3.], atol=.07)
        np.testing.assert_allclose(np.cov(draws.T), [[1., .8], [.8, 1.]], atol=.08)

    def test_bounds_protocol_and_private_randomness(self):
        import numpy as np
        problem = Quadratic()
        state = np.random.get_state()
        sampler = TuRBOSampler(problem, seed=0, initial=[1.], turbo_initial_points=2, turbo_candidates=16)
        with self.assertRaises(RuntimeError):
            sampler.ask()
        sampler.initialize([1.], dict(objective=1.))
        with self.assertRaises(ValueError):
            sampler.initialize([1.], dict(objective=1.))
        seen = {(1.,)}
        for _ in range(6):
            designs = sampler.ask()
            problem.validate_design(designs[0])
            self.assertNotIn(tuple(designs[0]), seen)
            seen.add(tuple(designs[0]))
            if sampler.pending_adapt:
                center = sampler.x[int(np.argmin(sampler.y))][0]
                normalized = (designs[0][0]+2)/6
                self.assertLessEqual(abs(normalized-center), sampler.state.length/2 + 1e-12)
            with self.assertRaises(RuntimeError):
                sampler.ask()
            with self.assertRaises(ValueError):
                sampler.tell(designs, [])
            with self.assertRaises(ValueError):
                sampler.tell(designs, [dict(objective=math.nan)])
            sampler.tell(designs, [dict(objective=(designs[0][0]-.3)**2)])
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_ard_bounds_and_sparse_full_dimension_perturbations(self):
        import numpy as np
        from types import SimpleNamespace
        problem = Quadratic()
        problem.dimension = 2145
        problem.lower_bounds, problem.upper_bounds = [0.]*2145, [1.]*2145
        sampler = TuRBOSampler(problem, seed=42, initial=problem.initial_design(), turbo_candidates=64)
        sampler.initialize(problem.initial_design(), dict(objective=1.))
        lengths = np.linspace(.1, 1., problem.dimension)
        sampler.model = SimpleNamespace(kernel_=SimpleNamespace(k2=SimpleNamespace(length_scale=lengths)))
        pool = sampler._candidates()
        half_widths = .4*lengths / np.exp(np.log(lengths).mean())
        self.assertEqual(pool.shape, (64, 2145))
        self.assertTrue(np.all(np.abs(pool-.5) <= half_widths+1e-12))
        changed = np.count_nonzero(pool != .5, axis=1)
        self.assertTrue(np.all(changed >= 1))
        self.assertLess(changed.mean(), 35.)
        self.assertGreater(changed.mean(), 10.)

    def test_budget_learning_and_no_unused_model_fit(self):
        original = TuRBOSampler._fit
        for budget in (1, 3, 12):
            fits = []
            def fit(sampler):
                fits.append(len(sampler.y))
                original(sampler)
            problem = Quadratic()
            with patch.object(TuRBOSampler, '_fit', fit):
                summary = self.run_problem(problem, name=str(budget), budget=budget, turbo_candidates=64)
            self.assertEqual((len(problem.calls), summary['attempts']), (budget, budget))
            self.assertEqual(len(fits), max(0, budget-3))
            self.assertEqual(summary['algorithm'], 'TuRBO-1')
            self.assertEqual(summary['stop_reason'], 'evaluation_budget')
            if budget == 12:
                initial_best = min(r['objective'] for r in self.rows('12')[:3])
                self.assertLess(summary['best_objective'], initial_best*.1)
                self.assertLess(summary['best_objective'], .01)

    def test_seed_reproducibility_and_variation(self):
        for name, seed in (('one', 0), ('two', 0), ('three', 1)):
            self.run_problem(name=name, seed=seed, budget=5)
        designs = lambda name: [r['x'] for r in self.rows(name)]
        self.assertEqual(designs('one'), designs('two'))
        self.assertNotEqual(designs('one'), designs('three'))

    def test_natural_restart_and_resume_keep_budget_global_best_and_journal(self):
        reference = FlatProblem()
        expected = self.run_problem(reference, name='reference', budget=34,
                                    turbo_initial_points=2, turbo_candidates=16)
        self.assertEqual(expected['turbo_state']['restarts'], 1)
        self.assertEqual(expected['turbo_state']['local_evaluations'], 4)
        self.assertEqual(expected['turbo_state']['length'], .8)
        self.assertEqual(expected['best_objective'], 1.)
        for stop in (1, 5, 30, 31):
            before, after = FlatProblem(), FlatProblem()
            def progress(problem, index, budget, status):
                if index == stop and status == 'ok':
                    raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                self.run_problem(before, name=str(stop), budget=34, turbo_initial_points=2,
                                 turbo_candidates=16, progress=progress)
            path = self.root/str(stop)/'evaluations.jsonl'
            prefix = path.read_bytes()
            result = self.run_problem(after, name=str(stop), budget=34, turbo_initial_points=2,
                                      turbo_candidates=16, resume=True)
            self.assertEqual(before.calls+after.calls, reference.calls)
            self.assertEqual(result['new_attempts'], 34-stop)
            self.assertEqual(result['turbo_state'], expected['turbo_state'])
            self.assertTrue(path.read_bytes().startswith(prefix))

    def test_restart_fits_only_new_region_data(self):
        problem = Quadratic()
        sampler = TuRBOSampler(problem, seed=1, initial=[1.], turbo_initial_points=2, turbo_candidates=16)
        sampler.initialize([1.], dict(objective=-100.))
        sampler.tell(sampler.ask(), [dict(objective=-99.)])
        sampler.state.restart_required = True
        for _ in range(2):
            sampler.tell(sampler.ask(), [dict(objective=10.)])
        self.assertEqual(sampler.y, [10., 10.])
        self.assertEqual(len(sampler.seen), 4)
        self.assertEqual(sampler.state.best, 10.)
        sampler.ask()
        self.assertEqual(len(sampler.model.X_train_), 2)

    def test_failure_recovery_and_all_failures(self):
        reference = Quadratic(failures=(1, 4))
        self.run_problem(reference, name='reference', budget=6, failure_penalty=100.)
        before, after = Quadratic(failures=(1,), interrupt=4), Quadratic()
        with self.assertRaises(KeyboardInterrupt):
            self.run_problem(before, budget=6, failure_penalty=100.)
        result = self.run_problem(after, budget=6, failure_penalty=100., resume=True)
        self.assertEqual(before.calls+after.calls, reference.calls)
        self.assertEqual(result['failed_evaluations'], 2)
        self.assertEqual(result['successful_evaluations'], 4)
        failed = self.run_problem(Quadratic(failures=range(1, 6)), name='failed', budget=5, failure_penalty=1e300)
        self.assertEqual(failed['failed_evaluations'], 5)
        self.assertIsNone(failed['best_feasible_objective'])

    def test_resume_rejects_setting_or_dependency_changes_before_cfd(self):
        self.run_problem(budget=3)
        problem = Quadratic()
        for changed in (dict(turbo_initial_points=4), dict(turbo_candidates=16)):
            with self.assertRaisesRegex(ResumeError, 'optimizer_settings'):
                self.run_problem(problem, budget=3, resume=True, **changed)
        path = self.root/'run'/'experiment.json'
        saved = json.loads(path.read_text())
        self.assertIsNone(saved['population'])
        self.assertIsNone(saved['sigma_normalized'])
        for dependency in ('numpy', 'scipy', 'scikit-learn'):
            changed = json.loads(json.dumps(saved))
            changed['versions'][dependency] = 'different'
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ResumeError, dependency):
                self.run_problem(problem, budget=3, resume=True)
        self.assertFalse(problem.calls)

    def test_cli_settings_validation_resume_and_integer_rejection(self):
        from flowfield.optimization.registry import resolve_optimizer
        self.assertIn('turbo', list_optimizers())
        for changed in (dict(turbo_initial_points=1), dict(turbo_candidates=True)):
            with self.assertRaises(ValueError):
                self.run_problem(**changed)
        self.assertFalse((self.root/'run').exists())
        problem = Quadratic()
        problem.variable_type = 'integer'
        with self.assertRaisesRegex(ValueError, 'does not support integer'):
            resolve_optimizer(problem, 'turbo')
        problem.variable_type, problem.name = 'real', 'P000'
        with patch('flowfield.optimization.cli._make_problem', return_value=problem), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--problem', 'P000', '--optimizer', 'turbo', '--budget', '4',
                '--turbo-initial-points', '3', '--turbo-candidates', '16', '--output', str(self.root/'cli')]), 0)
            self.assertEqual(main(['--resume', '--output', str(self.root/'cli')]), 0)
        self.assertEqual(len(problem.calls), 4)

    def test_benchmark_matrix_and_plot_data(self):
        from flowfield.benchmarking.runner import run_benchmark
        from flowfield.benchmarking.analysis import compare_runs
        with patch('flowfield.benchmarking.runner.create_problem', side_effect=lambda *a, **k: Quadratic()):
            manifest = run_benchmark(problems=['P000'], optimizers=['turbo'], seeds=[0, 1],
                budget=4, output=self.root/'benchmark', with_ioh=False,
                optimizer_options={'turbo': dict(turbo_initial_points=3, turbo_candidates=16)})
        self.assertEqual(manifest['status'], 'completed')
        folders = [self.root/'benchmark'/r['directory'] for r in manifest['runs']]
        report = compare_runs(folders)
        self.assertEqual(len(report['algorithms']['TuRBO-1']['runs']), 2)

    def test_ioh_counts_penalties_without_extra_evaluations(self):
        if importlib.util.find_spec('ioh') is None:
            self.skipTest('optional IOH unavailable')
        problem = Quadratic(failures=(2,))
        self.run_problem(problem, budget=5, failure_penalty=100., with_ioh=True)
        path = next((self.root/'run'/'ioh').rglob('*.dat'))
        self.assertEqual([int(line.split()[0]) for line in path.read_text().splitlines()[1:]], list(range(1, 6)))
        self.assertEqual(len(problem.calls), 5)


if __name__ == '__main__':
    unittest.main()
