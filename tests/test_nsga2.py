"""NSGA-II selection, real operators, Pareto persistence and deterministic replay."""
import contextlib
import io
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from flowfield.problem import FlowFieldProblem
from flowfield.optimization import NSGAIISampler, optimize, main, list_optimizers
from flowfield.optimization.nsga2 import Fitness, ObjectiveSpace, fronts, crowding
from flowfield.runtime import SimulationError


class Tradeoff(FlowFieldProblem):
    def __init__(self, failures=(), interrupt=None):
        super().__init__('Tradeoff', 2, [-2., 0.], [4., 2.])
        self.calls = []
        self.failures, self.interrupt = failures, interrupt

    def decode(self, x):
        return self.validate_design(x)

    def evaluate_raw(self, x):
        self.calls.append(list(x))
        if len(self.calls) == self.interrupt:
            raise KeyboardInterrupt()
        if len(self.calls) in self.failures:
            raise SimulationError('test failure')
        a, b = x
        return dict(status='ok', objective=a*a+b*b,
                    objectives={'left': a*a+b*b, 'right': (a-2)**2+b*b},
                    constraints={'limit': b-1.5}, metadata={'valid_cfd': True})


class NSGAIITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_problem(self, problem, folder='run', **options):
        return optimize(problem, output=self.root/folder, optimizer='nsga2',
                        with_ioh=False, **options)

    def test_sorting_crowding_and_constrained_dominance(self):
        values = [Fitness(v) for v in ((0, 3), (1, 2), (2, 1), (3, 0), (3, 3))]
        values += [Fitness((-10, -10), .1), Fitness((-20, -20), 1), Fitness((), valid=False)]
        self.assertEqual(fronts(values), [[0, 1, 2, 3], [4], [5], [6], [7]])
        distances = crowding(values, [0, 1, 2, 3])
        self.assertEqual(distances[0], math.inf)
        self.assertEqual(distances[3], math.inf)
        self.assertAlmostEqual(distances[1], 4/3)
        self.assertEqual(crowding([Fitness((1, 1))]*3, [0, 1, 2]), {0: 0., 1: 0., 2: 0.})

    def test_schema_validation_and_initial_failures(self):
        space = ObjectiveSpace()
        self.assertFalse(space.fitness(dict(status='penalized', objective=-100)).valid)
        self.assertIsNone(space.names)
        result = dict(objective=99, objectives={'b': 2., 'a': 1.}, constraints={'g': -.1})
        self.assertEqual(space.fitness(result).objectives, (1., 2.))
        for changed in (dict(objectives={'a': 1}), dict(constraints={}),
                        dict(objectives={'a': math.nan, 'b': 2}),
                        dict(constraints={'g': math.inf}), dict(objectives={'a': True, 'b': 2})):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                space.fitness(result | changed)

    def test_elitist_survival_uses_vector_not_scalar(self):
        problem = Tradeoff()
        sampler = NSGAIISampler(problem, seed=3, initial=problem.initial_design(), population=4)
        def result(a, b):
            return dict(objective=-a-b, objectives={'a': a, 'b': b}, constraints={})
        sampler.initialize(problem.initial_design(), result(0, 3))
        batch = sampler.ask()
        sampler.tell(batch, [result(1, 2), result(2, 1), result(3, 0)])
        batch = sampler.ask()
        sampler.tell(batch, [result(100, 100)]*4)
        # Offspring look better to scalar minimization, but are vector-dominated.
        self.assertEqual({f.objectives for _, f in sampler.population}, {(0, 3), (1, 2), (2, 1), (3, 0)})

    def test_bounds_odd_population_protocol_and_private_rng(self):
        problem = Tradeoff()
        state = random.getstate()
        sampler = NSGAIISampler(problem, seed=0, initial=problem.initial_design(), population=5)
        with self.assertRaises(RuntimeError):
            sampler.ask()
        sampler.initialize(problem.initial_design(), problem.evaluate_raw(problem.initial_design()))
        for generation in range(15):
            batch = sampler.ask()
            self.assertEqual(len(batch), 4 if generation == 0 else 5)
            for x in batch:
                problem.validate_design(x)
            with self.assertRaises(RuntimeError):
                sampler.ask()
            with self.assertRaises(ValueError):
                sampler.tell(batch, [])
            sampler.tell(batch, [problem.evaluate_raw(x) for x in batch])
        self.assertEqual(state, random.getstate())

    def test_budget_archive_and_failed_baseline(self):
        for budget in (1, 2, 4, 5, 9, 20):
            p = Tradeoff(failures=(1, 3))
            summary = self.run_problem(p, str(budget), budget=budget, failure_penalty=.001)
            self.assertEqual(len(p.calls), budget)
            self.assertEqual(summary['complete_generations'], budget//4)
            self.assertEqual(summary['partial_generation_evaluations'],
                             max(0, budget-1) if budget < 4 else budget % 4)
            data = json.loads((self.root/str(budget)/'pareto.json').read_text())
            rows = [json.loads(s) for s in (self.root/str(budget)/'evaluations.jsonl').read_text().splitlines()]
            # Independent exhaustive oracle over all observations, not just the last generation.
            feasible = [r for r in rows if r['feasible']]
            def dominates(a, b):
                a, b = a['result']['objectives'], b['result']['objectives']
                return all(a[k] <= b[k] for k in a) and any(a[k] < b[k] for k in a)
            nondominated = [r for r in feasible if not any(dominates(other, r) for other in feasible)]
            unique = {}
            for r in nondominated:
                unique.setdefault((tuple(r['x']), tuple(sorted(r['result']['objectives'].items()))), r)
            expected = [r['evaluation'] for r in unique.values()]
            self.assertEqual([r['evaluation'] for r in data['solutions']], expected)
            self.assertEqual(summary['pareto_size'], len(expected))
            self.assertTrue(all(r['valid'] for r in data['solutions']))
        all_failed = self.run_problem(Tradeoff(failures=range(1, 10)), 'failed', budget=9, failure_penalty=1)
        self.assertEqual((all_failed['attempts'], all_failed['pareto_size']), (9, 0))

    def test_resume_replays_identical_population_and_archive(self):
        reference = Tradeoff()
        self.run_problem(reference, 'reference', budget=13)
        for stop in (1, 3, 4, 5, 8, 13):
            before, after = Tradeoff(), Tradeoff()
            def progress(name, index, budget, status):
                if index == stop and status == 'ok':
                    raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                self.run_problem(before, str(stop), budget=13, progress=progress)
            self.run_problem(after, str(stop), budget=13, resume=True)
            self.assertEqual(before.calls+after.calls, reference.calls)
            self.assertEqual(len(after.calls), 13-stop)
            def archive(folder):
                return [(r['x'], r['result']['objectives']) for r in
                        json.loads((self.root/folder/'pareto.json').read_text())['solutions']]
            self.assertEqual(archive(str(stop)), archive('reference'))

    def test_interrupted_evaluation_recovery_matches_penalized_run(self):
        before, after = Tradeoff(interrupt=3), Tradeoff()
        with self.assertRaises(KeyboardInterrupt):
            self.run_problem(before, budget=12, failure_penalty=1e6)
        summary = self.run_problem(after, budget=12, failure_penalty=1e6, resume=True)
        reference = Tradeoff(failures=(3,))
        self.run_problem(reference, 'reference', budget=12, failure_penalty=1e6)
        self.assertEqual(before.calls+after.calls, reference.calls)
        self.assertEqual(summary['failed_evaluations'], 1)

    def test_cli_registration_and_discrete_rejection(self):
        self.assertIn('nsga2', list_optimizers())
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            self.assertEqual(main(['--problem', 'P002', '--optimizer', 'nsga2', '--budget', '20',
                                   '--population', '8', '--dry-run', '--output', str(self.root/'cli')]), 0)
        self.assertEqual(json.loads(capture.getvalue())['problems'][0]['algorithm'], 'NSGA-II')
        self.assertFalse((self.root/'cli').exists())
        p = Tradeoff()
        p.variable_type = 'integer'
        with self.assertRaisesRegex(ValueError, 'does not support integer'):
            self.run_problem(p)
        self.assertEqual(p.calls, [])

    def test_zdt1_moves_toward_known_front(self):
        problem = FlowFieldZDT1()
        sampler = NSGAIISampler(problem, seed=42, initial=problem.initial_design(), population=24)
        sampler.initialize(problem.initial_design(), problem.evaluate_raw(problem.initial_design()))
        errors = []
        for _ in range(40):
            batch = sampler.ask()
            sampler.tell(batch, [problem.evaluate_raw(x) for x in batch])
            errors.append(sum(f.objectives[1]-(1-math.sqrt(f.objectives[0]))
                              for _, f in sampler.population)/24)
        self.assertLess(errors[-1], .15*errors[0])
        self.assertGreater(max(f.objectives[0] for _, f in sampler.population), .8)
        self.assertLess(min(f.objectives[0] for _, f in sampler.population), .1)

    @unittest.skipUnless(importlib.util.find_spec('ioh'), 'optional IOH unavailable')
    def test_ioh_logs_scalar_without_extra_evaluations(self):
        p = Tradeoff(failures=(2,))
        summary = optimize(p, output=self.root/'ioh-run', optimizer='nsga2',
                           budget=8, with_ioh=True, failure_penalty=1e6)
        self.assertEqual(len(p.calls), 8)
        self.assertEqual(summary['objective_names'], ['left', 'right'])
        logs = list((self.root/'ioh-run'/'ioh').rglob('*.dat'))
        self.assertTrue(logs)
        self.assertEqual(sum(len(p.read_text().splitlines())-1 for p in logs), 8)
        manifest = json.loads((self.root/'ioh-run'/'experiment.json').read_text())
        self.assertIn('scalar diagnostic only', manifest['ioh_trace'])

    def test_benchmark_matrix_runs_registered_nsga2(self):
        from flowfield.benchmarking import run_benchmark
        with patch('flowfield.benchmarking.runner.create_problem', side_effect=lambda *a, **k: Tradeoff()):
            result = run_benchmark(problems=['P002'], optimizers=['nsga2'], seeds=[0, 1],
                                   budget=6, output=self.root/'matrix', with_ioh=False)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(all(r['attempts'] == 6 for r in result['runs']))
        self.assertEqual(len(list((self.root/'matrix').rglob('pareto.json'))), 2)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'optional matplotlib unavailable')
    def test_pareto_plot_matches_complete_and_interrupted_nsga2_archives(self):
        from flowfield.benchmarking import plot_pareto
        for interrupted in (False, True):
            folder = self.root / str(interrupted)
            def progress(name, index, budget, status):
                if interrupted and index == 6:
                    raise KeyboardInterrupt()
            if interrupted:
                with self.assertRaises(KeyboardInterrupt):
                    self.run_problem(Tradeoff(failures=(2,)), str(interrupted), budget=13,
                                     failure_penalty=1e6, progress=progress)
            else:
                self.run_problem(Tradeoff(failures=(2,)), str(interrupted), budget=13,
                                 failure_penalty=1e6)
            archive = json.loads((folder / 'pareto.json').read_text())
            expected = sorted((row['result']['objectives']['left'], row['result']['objectives']['right'])
                              for row in archive['solutions'])
            self.assertTrue(expected)
            before = {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}
            with patch.object(Tradeoff, 'evaluate_raw', side_effect=AssertionError('No reevaluation')):
                ax = plot_pareto(folder)
            actual = list(zip(ax.lines[0].get_xdata(), ax.lines[0].get_ydata()))
            self.assertEqual(actual, expected)
            self.assertEqual(before, {p: p.read_bytes() for p in before})


class FlowFieldZDT1(FlowFieldProblem):
    def __init__(self):
        super().__init__('ZDT1', 6, [0.]*6, [1.]*6)

    def decode(self, x):
        return self.validate_design(x)

    def evaluate_raw(self, x):
        g = 1+9*sum(x[1:])/5
        return dict(objective=x[0], objectives={'f1': x[0], 'f2': g*(1-math.sqrt(x[0]/g))}, constraints={})
