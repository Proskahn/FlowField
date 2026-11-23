"""Real pycma lq evolution, true-call accounting, and deterministic recovery."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from flowfield.optimization import LQCMAESSampler, list_optimizers, main, optimize
from flowfield.optimization.state import ResumeError
from flowfield.runtime import SimulationError
from test_optimization import CountingProblem, DiscreteProblem, cma


@unittest.skipIf(cma is None, 'optional pycma unavailable')
class LQCMAESTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_problem(self, name, problem=None, **options):
        problem = problem or CountingProblem()
        settings = dict(method='lqcmaes', seed=0, population=8, budget=40, with_ioh=False)
        settings.update(options)
        summary = optimize(problem, output=self.root/name, **settings)
        rows = [json.loads(line) for line in (self.root/name/'evaluations.jsonl').read_text().splitlines()]
        return problem, summary, rows

    def test_matches_direct_pycma_surrogate_population_evolution(self):
        """An independent upstream ask/tell loop must choose the same true calls."""
        import numpy as np
        from cma.fitness_models import SurrogatePopulation
        p = CountingProblem(dimension=3)
        x0 = p.initial_design()
        lower, span = np.array(p.lower_bounds), np.subtract(p.upper_bounds, p.lower_bounds)
        normalize = lambda x:(np.array(x)-lower)/span
        observed = []

        def objective(z):
            x = (lower+span*z).tolist()
            observed.append(x)
            return sum(v*v for v in x)

        es = cma.CMAEvolutionStrategy(normalize(x0).tolist(), .1, dict(
            bounds=[0,1], popsize=8, CMA_diagonal=False, CMA_mirrors=0,
            randn=np.random.RandomState(0).randn, seed=float('nan'),
            verbose=-9, verb_log=0, verb_disp=0))
        surrogate = SurrogatePopulation(objective)
        surrogate.model.add_data_row(normalize(x0), sum(v*v for v in x0))
        surrogate.evaluations = es.countevals = 1
        for _ in range(12):
            population = es.ask()
            es.tell(population, surrogate(population))
            es.countevals = surrogate.evaluations
            es.inject([surrogate.model.xopt])
        sampler = LQCMAESSampler(p, seed=0, initial=x0, population=8, sigma=.1)
        sampler.initialize(x0, p.evaluate_raw(x0))
        actual = []
        while sampler.es.countiter < 12:
            batch = sampler.ask()
            actual.extend(batch)
            sampler.tell(batch, [p.evaluate_raw(x) for x in batch])
        self.assertEqual(actual, observed)
        np.testing.assert_array_equal(sampler.es.mean, es.mean)
        self.assertEqual(sampler.es.countevals, surrogate.evaluations)
        self.assertEqual(sampler.surrogate.model.types, surrogate.model.types)
        self.assertGreater(sampler.diagnostics['lq_surrogate_only_evaluations'], 0)

    def test_hard_budget_true_only_logs_and_generation_accounting(self):
        for budget in (1,2,4,8,9,13,40):
            with self.subTest(budget=budget):
                p, s, rows = self.run_problem(str(budget), budget=budget)
                self.assertEqual((len(p.calls), len(rows), s['attempts']), (budget,)*3)
                self.assertEqual(s['lq_true_evaluations'], budget)
                self.assertEqual(rows[0]['phase'], 'baseline')
                self.assertTrue(all(-2 <= v <= 4 for row in rows for v in row['x']))
                self.assertEqual(s['best_objective'], min(r['objective'] for r in rows if r['valid']))
                self.assertEqual(s['lq_completed_population_members'], 8*s['complete_generations'])
                self.assertEqual(s['lq_completed_population_members'],
                    budget-1-s['partial_generation_evaluations']+s['lq_surrogate_only_evaluations'])
                self.assertTrue(all(r['proposal']['source']=='pycma_lq_true_evaluation' for r in rows[1:]))
        self.assertGreater(s['lq_surrogate_only_evaluations'], 0)
        self.assertLess(s['best_objective'], s['baseline_objective'])

    def test_seed_zero_determinism_and_numpy_state_isolation(self):
        import numpy as np
        np.random.seed(73)
        state = np.random.get_state()
        first, _, _ = self.run_problem('first')
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        second, _, _ = self.run_problem('second')
        self.assertEqual(first.calls, second.calls)

    def test_resume_matches_real_call_stream_within_adaptive_populations(self):
        reference, expected, _ = self.run_problem('reference')
        for stop in (1,2,5,13,29,40):
            with self.subTest(stop=stop):
                name = 'resume-'+str(stop)
                before, after = CountingProblem(), CountingProblem()
                def progress(name, index, budget, status):
                    if index == stop and status == 'ok':
                        raise KeyboardInterrupt
                with self.assertRaises(KeyboardInterrupt):
                    self.run_problem(name, before, progress=progress)
                prefix = (self.root/name/'evaluations.jsonl').read_bytes()
                _, resumed, _ = self.run_problem(name, after, resume=True)
                self.assertEqual(before.calls+after.calls, reference.calls)
                self.assertEqual(resumed['new_attempts'], 40-stop)
                for key in ('best_objective', 'complete_generations', 'partial_generation_evaluations',
                            'lq_surrogate_only_evaluations'):
                    self.assertEqual(resumed[key], expected[key])
                self.assertTrue((self.root/name/'evaluations.jsonl').read_bytes().startswith(prefix))

    def test_interrupted_evaluation_consumes_budget_and_resumes_as_penalty(self):
        before, after = CountingProblem(interrupt=5), CountingProblem()
        with self.assertRaises(KeyboardInterrupt):
            self.run_problem('pending', before, failure_penalty=1e6)
        _, resumed, _ = self.run_problem('pending', after, failure_penalty=1e6, resume=True)
        reference, expected, _ = self.run_problem('penalty', CountingProblem(failures=(5,)), failure_penalty=1e6)
        self.assertEqual(before.calls+after.calls, reference.calls)
        self.assertEqual((resumed['attempts'], resumed['new_attempts'], resumed['failed_evaluations']), (40,35,1))
        self.assertEqual(resumed['best_objective'], expected['best_objective'])

    def test_failure_penalties_and_ioh_never_log_predictions_as_true_values(self):
        try:
            import ioh
        except ImportError:
            self.skipTest('optional IOH unavailable')
        p, s, rows = self.run_problem('ioh', CountingProblem(failures=(3,)),
            failure_penalty=1e6, with_ioh=True)
        self.assertEqual(s['failed_evaluations'], 1)
        self.assertFalse(rows[2]['valid'])
        self.assertEqual(rows[2]['objective'], 1e6)
        self.assertEqual(s['best_objective'], min(r['objective'] for r in rows if r['valid']))
        self.assertGreater(s['lq_surrogate_only_evaluations'], 0)
        dat = next((self.root/'ioh/ioh').rglob('*.dat'))
        self.assertEqual([int(line.split()[0]) for line in dat.read_text().splitlines()[1:]], list(range(1,len(p.calls)+1)))

    def test_failed_qualified_baseline_stops_before_candidates(self):
        p = CountingProblem(failures=(1,))
        p.requires_qualified_baseline = True
        with self.assertRaisesRegex(SimulationError, 'baseline did not qualify'):
            self.run_problem('baseline', p, failure_penalty=1e6)
        self.assertEqual(len(p.calls), 1)

    def test_soft_limit_does_not_start_another_true_evaluation(self):
        clock = iter([0]*6+[100]*30)
        with patch('flowfield.optimization.runner.time.monotonic', side_effect=lambda:next(clock,100)):
            p, summary, _ = self.run_problem('time-limit', max_seconds=1)
        self.assertLessEqual(len(p.calls), 1)
        self.assertEqual(summary['stop_reason'], 'soft_wall_time_limit')

    def test_separable_covariance_keeps_all_design_coordinates(self):
        p, summary, rows = self.run_problem('separable', CountingProblem(dimension=101), budget=5)
        self.assertEqual(summary['algorithm'], 'Sep-lq-CMA-ES')
        self.assertTrue(all(len(r['x'])==101 for r in rows))
        self.assertEqual(len(p.calls), 5)
        self.assertTrue(json.loads((self.root/'separable/experiment.json').read_text())['diagonal_covariance'])

    def test_cli_resume_restores_optimizer_and_rebuilds_true_only_ioh_trace(self):
        try:
            import ioh
        except ImportError:
            self.skipTest('optional IOH unavailable')
        folder = self.root/'cli'
        before = CountingProblem(interrupt=10); before.name = 'P000'
        with patch('flowfield.optimization.cli._make_problem', return_value=before), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--problem','P000','--optimizer','lqcmaes','--population','8',
                '--budget','24','--seed','0','--failure-penalty','1000000','--ioh',
                '--output',str(folder)]), 130)
        old_ioh = {p:p.read_bytes() for p in (folder/'P000/ioh').rglob('*') if p.is_file()}
        after = CountingProblem(); after.name = 'P000'
        with patch('flowfield.optimization.cli._make_problem', return_value=after), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--resume','--output',str(folder)]), 1)
        self.assertEqual(len(before.calls)+len(after.calls), 24)
        self.assertEqual(old_ioh, {p:p.read_bytes() for p in old_ioh})
        summary = json.loads((folder/'P000/summary.json').read_text())
        self.assertEqual((summary['algorithm'],summary['new_attempts']), ('lq-CMA-ES',14))
        dat = next(Path(summary['ioh_directory']).rglob('*.dat'))
        self.assertEqual([int(row.split()[0]) for row in dat.read_text().splitlines()[1:]], list(range(1,25)))

    def test_settings_and_optional_scipy_version_are_pinned_for_resume(self):
        self.run_problem('settings', budget=9)
        path = self.root/'settings/experiment.json'
        original = json.loads(path.read_text())
        self.assertEqual(original['optimizer'], 'lqcmaes')
        self.assertTrue(original['optimizer_settings']['baseline_in_model'])
        for change in ('settings','scipy'):
            saved = json.loads(json.dumps(original))
            if change == 'settings':
                saved['optimizer_settings']['tau_truth_threshold'] = .1
            else:
                saved['versions']['scipy'] = 'different'
            path.write_text(json.dumps(saved))
            p = CountingProblem()
            with self.assertRaises(ResumeError):
                self.run_problem('settings', p, budget=9, resume=True)
            self.assertEqual(p.calls, [])

    def test_process_exit_recovery_does_not_depend_on_live_surrogate_state(self):
        name = 'hard-exit'
        code = '''import os,sys
from flowfield.optimization import optimize
from test_optimization import CountingProblem
def progress(name,index,budget,status):
    if index==13 and status=='ok': os._exit(17)
optimize(CountingProblem(), output=sys.argv[1], method='lqcmaes', budget=40,
         population=8, seed=0, with_ioh=False, progress=progress)
'''
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        child = subprocess.run([sys.executable, '-c', code, str(self.root/name)], env=env,
                               capture_output=True, text=True)
        self.assertEqual(child.returncode, 17, child.stderr)
        after, resumed, rows = self.run_problem(name, resume=True)
        reference, _, _ = self.run_problem('reference')
        self.assertEqual([r['x'] for r in rows], reference.calls)
        self.assertEqual((len(after.calls), resumed['new_attempts']), (27,27))

    def test_cli_registration_dry_run_and_unsupported_discrete_problem(self):
        self.assertIn('lqcmaes', list_optimizers())
        output = self.root/'plan'
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            self.assertEqual(main(['--problem','P009','--optimizer','lqcmaes', '--population','8',
                '--budget','100','--sigma','.12','--output',str(output),'--dry-run']), 0)
        plan = json.loads(capture.getvalue())
        self.assertEqual(plan['maximum_cfd_attempts'], 100)
        self.assertEqual(plan['problems'][0]['algorithm'], 'lq-CMA-ES')
        self.assertEqual(plan['problems'][0]['sigma_normalized'], .12)
        self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            self.run_problem('discrete', DiscreteProblem())
        self.assertFalse((self.root/'discrete').exists())


if __name__ == '__main__':
    unittest.main()
