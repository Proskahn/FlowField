"""Every P003 adapter repairs before evaluation; objectives here are synthetic."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from flowfield import P003Problem, TopologyConfig
from flowfield.optimization import create_sampler, list_optimizers, main, optimize
from flowfield.optimization.registry import optimizer_definition, optimizer_revision
from flowfield.optimization.mobo import MOBO_METHODS
from flowfield.optimization.state import ResumeError, load_history
from flowfield.runtime import SimulationError
from test_optimization import CountingProblem


NEW_ADAPTERS = ('cmaes', 'lqcmaes', 'nsga2', 'cobyla')
AVAILABLE = all(importlib.util.find_spec(name) for name in ('numpy', 'scipy', 'cma', 'sklearn'))
MOBO_AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'botorch', 'gpytorch'))


class RepairProblem(P003Problem):
    def __init__(self, *, config=None, failures=(), interrupt=None):
        super().__init__(config=config or TopologyConfig(design_nx=4, design_ny=4, filter_radius_m=.004))
        self.calls = []
        self.failures, self.interrupt = failures, interrupt

    def evaluate_raw(self, x):
        self.calls.append(list(x))
        volume = self.decode(x)['volume_fraction']
        if volume > self.config.volume_limit+1e-12:
            raise AssertionError('Volume-infeasible design reached evaluation')
        if len(self.calls) == self.interrupt:
            raise KeyboardInterrupt()
        if len(self.calls) in self.failures:
            raise SimulationError('Synthetic evaluation failure')
        loss = sum((v-.01)**2 for v in x)
        return dict(status='ok', objective=loss, unpenalized_objective=loss,
                    objectives={'loss': loss,'other_loss':sum((v-.02)**2 for v in x)},
                    constraints={'channel_volume': volume-self.config.volume_limit})


@unittest.skipUnless(AVAILABLE, 'optional optimizer dependencies unavailable')
class P003OptimizerRepairTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def run_problem(self, method, problem, folder=None, **options):
        return optimize(problem, optimizer=method, output=self.root/(folder or method),
            **(dict(budget=9, seed=42, population=4, sigma=.1, failure_penalty=1e6, with_ioh=False,
                    bo_initial_points=2, bo_candidates=16, turbo_initial_points=2, turbo_candidates=16) | options))

    def rows(self, folder):
        return [json.loads(line) for line in (self.root/folder/'evaluations.jsonl').read_text().splitlines()]

    def check_rows(self, problem, rows):
        self.assertEqual(rows[0]['x'], problem.initial_design())
        for row in rows[1:]:
            detail = row['proposal']
            expected = problem.repair_candidates([detail['proposed_x']], anchor=detail['anchor_x'])[0]
            self.np.testing.assert_array_equal(expected, row['x'])
            self.assertLessEqual(problem.decode(row['x'])['volume_fraction'], problem.config.volume_limit+1e-12)
            if detail['applied']:
                self.assertGreater(problem.decode(detail['proposed_x'])['volume_fraction'], problem.config.volume_limit)

    def test_all_registered_optimizers_use_feasible_designs_and_correct_row_metadata(self):
        for method in list_optimizers():
            if method in MOBO_METHODS and not MOBO_AVAILABLE:continue
            with self.subTest(method=method):
                problem = RepairProblem()
                summary = self.run_problem(method, problem)
                rows = self.rows(method)
                self.assertEqual((summary['attempts'], len(problem.calls), len(rows)), (9, 9, 9))
                self.assertTrue(all(r['valid'] and r['feasible'] for r in rows))
                self.assertTrue(any(r.get('proposal', {}).get('applied') for r in rows))
                self.check_rows(problem, rows)
                saved = json.loads((self.root/method/'experiment.json').read_text())
                self.assertEqual(saved['optimizer_settings']['candidate_repair'], problem.candidate_repair)
                if method in NEW_ADAPTERS:
                    self.assertIn('+candidate-repair-1', saved['optimizer_revision'])
                    self.assertEqual(saved['optimizer_settings']['repair_stage'], 'before CFD')

    def test_new_adapters_preserve_all_2145_variables(self):
        for method in NEW_ADAPTERS:
            with self.subTest(method=method):
                problem = RepairProblem(config=TopologyConfig())
                self.run_problem(method, problem, budget=2)
                self.assertEqual(len(problem.calls), 2)
                self.assertTrue(all(len(x) == 2145 for x in problem.calls))
                self.check_rows(problem, self.rows(method))

    def test_cma_and_lq_match_independent_pycma_on_composed_objective(self):
        import cma
        from cma.fitness_models import SurrogatePopulation
        np = self.np
        for method in ('cmaes', 'lqcmaes'):
            with self.subTest(method=method):
                reference = RepairProblem()
                initial = reference.initial_design()
                first = reference.evaluate_raw(initial)
                es = cma.CMAEvolutionStrategy(initial, .1, dict(bounds=[0, 1], popsize=4,
                    CMA_diagonal=False, CMA_mirrors=0, randn=np.random.RandomState(42).randn,
                    seed=float('nan'), verbose=-9, verb_log=0, verb_disp=0))
                def objective(z):
                    return reference.evaluate_raw(reference.repair_candidates([z])[0].tolist())['objective']
                surrogate = SurrogatePopulation(objective)
                surrogate.settings.logging_push = False
                surrogate.model.add_data_row(initial, first['objective'])
                surrogate.evaluations = 1
                if method == 'lqcmaes':
                    es.countevals = 1
                for _ in range(5):
                    population = es.ask()
                    scores = surrogate(population) if method == 'lqcmaes' else [objective(z) for z in population]
                    es.tell(population, scores)
                    if method == 'lqcmaes':
                        es.countevals = surrogate.evaluations
                        es.inject([surrogate.model.xopt])
                problem = RepairProblem()
                sampler = create_sampler(method, problem, seed=42, initial=initial, population=4, sigma=.1)
                sampler.initialize(initial, problem.evaluate_raw(initial))
                while sampler.es.countiter < 5:
                    batch = sampler.ask()
                    sampler.tell(batch, [problem.evaluate_raw(x) for x in batch])
                self.assertEqual(problem.calls, reference.calls)
                np.testing.assert_array_equal(sampler.es.mean, es.mean)
                if method == 'lqcmaes':
                    np.testing.assert_array_equal(sampler.surrogate.model.X, surrogate.model.X)
                    self.assertEqual(sampler.es.countevals, surrogate.evaluations)

    def test_nsga_population_and_offspring_use_repaired_coordinates(self):
        problem = RepairProblem()
        initial = problem.initial_design()
        sampler = create_sampler('nsga2', problem, seed=42, initial=initial, population=4)
        sampler.initialize(initial, problem.evaluate_raw(initial))
        for _ in range(3):
            batch = sampler.ask()
            sampler.tell(batch, [problem.evaluate_raw(x) for x in batch])
            for x, fitness in sampler.population:
                self.assertIn(x, problem.calls)
                self.assertEqual(fitness.violation, 0.)
                self.assertLessEqual(problem.decode(x)['volume_fraction'], .5+1e-12)

    def test_cobyla_reuses_results_when_different_search_points_repair_to_same_design(self):
        problem = RepairProblem()
        initial = problem.initial_design()
        sampler = create_sampler('cobyla', problem, seed=42, initial=initial)
        def minimize(fun, x0, **kwargs):
            fun(x0)
            fun([.5]*problem.dimension)
            fun([.6]*problem.dimension)
            return SimpleNamespace(message='test exhausted')
        sampler.minimize = minimize
        sampler.initialize(initial, problem.evaluate_raw(initial))
        batch = sampler.ask()
        self.assertTrue(sampler.proposal_metadata['applied'])
        sampler.tell(batch, [problem.evaluate_raw(batch[0])])
        self.assertEqual(sampler.ask(), [])
        self.assertEqual(len(problem.calls), 2)

    def test_partial_batch_resume_and_pending_failure_reproduce_proposals(self):
        for method in NEW_ADAPTERS:
            for pending in (False, True):
                folder = method+str(pending)
                reference = RepairProblem(failures=(3,) if pending else ())
                expected = self.run_problem(method, reference, folder+'-whole')
                before, after = RepairProblem(interrupt=3 if pending else None), RepairProblem()
                def progress(name, index, budget, status):
                    if not pending and index == 3 and status == 'ok':
                        raise KeyboardInterrupt()
                with self.assertRaises(KeyboardInterrupt):
                    self.run_problem(method, before, folder, progress=progress)
                prefix = (self.root/folder/'evaluations.jsonl').read_bytes()
                result = self.run_problem(method, after, folder, resume=True)
                self.assertEqual(before.calls+after.calls, reference.calls)
                self.assertEqual(result['new_attempts'], 6)
                self.assertEqual(result['failed_evaluations'], int(pending))
                self.assertEqual(result['best_objective'], expected['best_objective'])
                self.assertTrue((self.root/folder/'evaluations.jsonl').read_bytes().startswith(prefix))
                rows, recoveries = load_history(self.root/folder, 9, 1e6)
                self.assertFalse(recoveries)
                self.check_rows(after, rows)
                for actual, whole in zip(rows, self.rows(folder+'-whole')):
                    self.assertEqual(actual.get('proposal'), whole.get('proposal'))

    def test_old_unrepaired_runs_and_changed_policy_or_dependencies_refuse_before_cfd(self):
        for method in NEW_ADAPTERS:
            self.run_problem(method, RepairProblem(), budget=2)
            path = self.root/method/'experiment.json'
            original = json.loads(path.read_text())
            for change in ('revision', 'policy', 'scipy'):
                saved = json.loads(json.dumps(original))
                if change == 'revision':
                    saved['optimizer_revision'] = optimizer_definition(method).revision
                elif change == 'policy':
                    saved['optimizer_settings']['candidate_repair']['revision'] = 'other'
                else:
                    saved['versions']['scipy'] = 'other'
                path.write_text(json.dumps(saved))
                problem = RepairProblem()
                with self.assertRaises(ResumeError):
                    self.run_problem(method, problem, budget=2, resume=True)
                self.assertEqual(problem.calls, [])

    def test_previous_bo_turbo_random_runs_remain_resumable(self):
        for method in ('bo', 'turbo', 'random'):
            before, after = RepairProblem(), RepairProblem()
            before.repair_before_evaluation = False  # The pre-existing adapters already repaired.
            def progress(name, index, budget, status):
                if index == 3 and status == 'ok':
                    raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                self.run_problem(method, before, progress=progress)
            self.assertEqual(optimizer_revision(method, after), optimizer_definition(method).revision)
            result = self.run_problem(method, after, resume=True)
            reference = RepairProblem()
            self.run_problem(method, reference, method+'-whole')
            self.assertEqual(before.calls+after.calls, reference.calls)
            self.assertEqual(result['new_attempts'], 6)

    def test_bad_baseline_is_rejected_before_cfd_for_every_optimizer(self):
        for method in list_optimizers():
            if method in MOBO_METHODS and not MOBO_AVAILABLE:continue
            problem = RepairProblem()
            with self.assertRaisesRegex(ValueError, 'Initial design violates'):
                self.run_problem(method, problem, initial=[1.]*problem.dimension)
            self.assertFalse(problem.calls)

    def test_cli_reports_repair_for_all_optimizers_without_running_cfd(self):
        for method in list_optimizers():
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(main(['--problem', 'P003', '--optimizer', method, '--budget', '100',
                                       '--dry-run', '--output', str(self.root/method)]), 0)
            entry = json.loads(stream.getvalue())['problems'][0]
            self.assertEqual(entry['candidate_repair']['limit'], .5)
            self.assertFalse((self.root/method).exists())

    def test_other_problems_do_not_silently_gain_new_repair_policy(self):
        for method in NEW_ADAPTERS:
            problem = CountingProblem()
            problem.candidate_repair = {'method': 'unused'}
            def repair(*args, **kwargs):
                raise AssertionError('Other problems did not opt in to evaluation repair')
            problem.repair_candidates = repair
            sampler = create_sampler(method, problem, seed=42, initial=problem.initial_design(), population=4)
            self.assertIsNone(sampler.evaluation_repair)
            self.assertEqual(optimizer_revision(method, problem), optimizer_definition(method).revision)

    def test_ioh_records_actual_evaluations_with_failure_penalties(self):
        if importlib.util.find_spec('ioh') is None:
            self.skipTest('optional IOH unavailable')
        for method in NEW_ADAPTERS:
            problem = RepairProblem(failures=(3,))
            summary = self.run_problem(method, problem, with_ioh=True)
            dat = next(Path(summary['ioh_directory']).rglob('*.dat'))
            rows = dat.read_text().splitlines()[1:]
            self.assertEqual([int(r.split()[0]) for r in rows], list(range(1, 10)))
            self.assertEqual(summary['failed_evaluations'], 1)
            self.assertEqual(len(problem.calls), 9)


if __name__ == '__main__':
    unittest.main()
