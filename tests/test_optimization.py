import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield.problem import AbstractFlowFieldProblem
from flowfield.optimization import optimize, main
from flowfield.runtime import SimulationError

try:
    import cma
except ImportError:
    cma = None


class CountingProblem(AbstractFlowFieldProblem):
    def __init__(self, *, dimension=2, failures=(), interrupt=None, constrained=False):
        super().__init__('OptimizationTest',dimension,[-2.]*dimension,[4.]*dimension)
        self.calls=[]
        self.failures=failures
        self.interrupt=interrupt
        self.constrained=constrained

    def decode(self,x):
        return self.validate_design(x)

    def evaluate_raw(self,x):
        self.calls.append(list(x))
        n=len(self.calls)
        if n == self.interrupt:
            raise KeyboardInterrupt()
        if n in self.failures:
            raise SimulationError('deliberate test failure',case_directory='/test/failed-case')
        return dict(status='ok',objective=-float(n) if self.constrained else sum(v*v for v in x),
                    objectives={},constraints={'g':-1. if n==1 else 1.} if self.constrained else {},
                    metadata={'valid_cfd':True})


class DiscreteProblem(AbstractFlowFieldProblem):
    variable_type='integer'

    def __init__(self):
        super().__init__('DiscreteOptimizationTest',2,[0,0],[1,1])
        self.calls=[]

    def initial_design(self):
        return [0,0]

    def sample_design(self,rng):
        k=rng.randrange(2)
        return [k,k]

    def decode(self,x):
        if tuple(x) not in ((0,0),(1,1)):
            raise ValueError('Invalid discrete design')
        return list(x)

    def evaluate_raw(self,x):
        self.decode(x)
        self.calls.append(list(x))
        return dict(status='ok',objective=float(sum(x)),constraints={})


@unittest.skipIf(cma is None,'optional cma package unavailable')
class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_problem(self,problem=None,**kwargs):
        problem=problem or CountingProblem()
        output=self.root/str(len(list(self.root.iterdir())))
        summary=optimize(problem,output=output,with_ioh=False,**kwargs)
        rows=[json.loads(line) for line in (output/'evaluations.jsonl').read_text().splitlines()]
        return problem,summary,rows,output

    def test_hard_budget_includes_baseline_and_partial_generation(self):
        for budget in (1,2,4,5,6,10,20):
            p,s,rows,_=self.run_problem(budget=budget)
            self.assertEqual((len(p.calls),s['attempts'],len(rows)),(budget,)*3)
            self.assertEqual(s['complete_generations'],(budget-1)//4)
            self.assertEqual(s['partial_generation_evaluations'],(budget-1)%4)
            self.assertEqual(rows[0]['phase'],'baseline')
            self.assertEqual(s['stop_reason'],'evaluation_budget')
            self.assertTrue(all(-2<=v<=4 for row in rows for v in row['x']))

    def test_seed_zero_is_reproducible_and_does_not_reseed_numpy(self):
        import numpy as np
        np.random.seed(73)
        state=np.random.get_state()
        p1,_,_,_=self.run_problem(seed=0,budget=10)
        after=np.random.get_state()
        self.assertEqual(state[0],after[0])
        np.testing.assert_array_equal(state[1],after[1])
        self.assertEqual(state[2:],after[2:])
        p2,_,_,_=self.run_problem(seed=0,budget=10)
        self.assertEqual(p1.calls,p2.calls)

    def test_diagonal_full_dimension_not_a_reduced_problem(self):
        p,s,rows,out=self.run_problem(CountingProblem(dimension=2145),budget=5)
        self.assertEqual(s['algorithm'],'Sep-CMA-ES')
        self.assertEqual(s['complete_generations'],1)
        self.assertTrue(all(len(row['x'])==2145 for row in rows))
        self.assertTrue(json.loads((out/'experiment.json').read_text())['diagonal_covariance'])

    def test_failed_cfd_counts_and_is_never_best(self):
        p,s,rows,_=self.run_problem(CountingProblem(failures=(2,3)),budget=6,failure_penalty=1e6)
        self.assertEqual(len(p.calls),6)
        self.assertEqual(s['failed_evaluations'],2)
        self.assertEqual(s['successful_evaluations'],4)
        self.assertEqual(s['status'],'completed_with_failures')
        self.assertFalse(rows[1]['valid'])
        self.assertEqual(rows[1]['objective'],1e6)
        self.assertEqual(s['best_objective'],min(r['objective'] for r in rows if r['valid']))

    def test_unpenalized_error_preserves_spent_budget(self):
        p=CountingProblem(failures=(2,))
        out=self.root/'failure'
        with self.assertRaises(SimulationError):
            optimize(p,budget=20,output=out,with_ioh=False)
        s=json.loads((out/'summary.json').read_text())
        self.assertEqual(s['attempts'],2)
        self.assertEqual(s['failed_evaluations'],1)
        self.assertEqual(s['status'],'failed')
        self.assertEqual(len((out/'evaluations.jsonl').read_text().splitlines()),2)

    def test_interrupt_is_counted_and_preserved(self):
        out=self.root/'interrupt'
        with self.assertRaises(KeyboardInterrupt):
            optimize(CountingProblem(interrupt=2),budget=20,output=out,with_ioh=False)
        s=json.loads((out/'summary.json').read_text())
        self.assertEqual((s['status'],s['attempts']),('interrupted',2))
        events=[json.loads(v) for v in (out/'events.jsonl').read_text().splitlines()]
        self.assertEqual(events[-1]['event'],'started')

    def test_tracks_best_feasible_separately_from_scalar_best(self):
        p,s,_,out=self.run_problem(CountingProblem(constrained=True),budget=5)
        self.assertEqual(s['best_objective'],-5)
        self.assertEqual(s['best_feasible_objective'],-1)
        self.assertFalse(json.loads((out/'best.json').read_text())['feasible'])
        self.assertTrue(json.loads((out/'best-feasible.json').read_text())['feasible'])

    def test_discrete_candidates_stay_feasible_without_repeat_calls(self):
        p,s,rows,_=self.run_problem(DiscreteProblem(),budget=20)
        self.assertEqual(s['attempts'],2)
        self.assertEqual(len(set(tuple(x) for x in p.calls)),2)
        self.assertEqual(s['stop_reason'],'no_new_design_found')
        self.assertTrue(all(type(v) is int for row in rows for v in row['x']))
        with self.assertRaises(ValueError):
            optimize(DiscreteProblem(),method='cmaes',output=self.root/'bad',with_ioh=False)
        self.assertFalse((self.root/'bad').exists())

    def test_validation_and_no_overwrite(self):
        for kwargs in ({'budget':0},{'budget':True},{'population':3},{'sigma':float('nan')},
                       {'seed':-1},{'failure_penalty':-5},{'max_seconds':0}):
            with self.assertRaises(ValueError):
                optimize(CountingProblem(),output=self.root/'bad',with_ioh=False,**kwargs)
        self.assertFalse((self.root/'bad').exists())
        p,_,_,out=self.run_problem(budget=1)
        with self.assertRaises(FileExistsError):
            optimize(p,output=out,with_ioh=False)
        self.assertEqual(len(p.calls),1)

    def test_soft_wall_limit_does_not_start_extra_cfd(self):
        # A deterministic clock: initialization=0, first start=0, then cap reached.
        clock=iter([0,0,0,0,0,0,100,100,100,100,100,100,100,100,100])
        with patch('flowfield.optimization.runner.time.monotonic',side_effect=lambda:next(clock,100)):
            p,s,_,_=self.run_problem(budget=20,max_seconds=1)
        self.assertLessEqual(len(p.calls),1)
        self.assertEqual(s['stop_reason'],'soft_wall_time_limit')

    def test_dry_run_has_160_cfd_attempts_and_skips_missing_inputs(self):
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):
            result=main(['--problems','all','--budget','20','--output',str(self.root/'plan'),'--dry-run'])
        plan=json.loads(capture.getvalue())
        self.assertEqual(result,0)
        self.assertEqual(plan['maximum_cfd_attempts'],160)
        self.assertEqual(next(p for p in plan['problems'] if p['problem']=='P003')['sigma_normalized'],.005)
        self.assertFalse((self.root/'plan').exists())
        self.assertEqual(next(p for p in plan['problems'] if p['problem']=='P004')['status'],'skipped')
        skipped = next(p for p in plan['problems'] if p['problem']=='P008')
        self.assertEqual(skipped['status'], 'skipped')
        self.assertIn('frozen EWM calibration', skipped['reason'])
        timeouts={p['problem']:p['command_timeout_s'] for p in plan['problems'] if p['status']=='ready'}
        self.assertEqual(timeouts.pop('P006'),14400)
        self.assertTrue(all(value==1800 for value in timeouts.values()))

    def test_explicit_timeout_overrides_defaults_in_a_mixed_suite(self):
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):
            result=main(['--problems','P000','P006','--timeout','7200',
                         '--output',str(self.root/'plan'),'--dry-run'])
        self.assertEqual(result,0)
        self.assertEqual([p['command_timeout_s'] for p in json.loads(capture.getvalue())['problems']],
                         [7200,7200])
        self.assertFalse((self.root/'plan').exists())

    def test_explicit_p008_requires_calibration_before_creating_output(self):
        error=io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
            main(['--problem','P008','--output',str(self.root/'uncalibrated'),'--dry-run'])
        self.assertIn('frozen EWM calibration',error.getvalue())
        self.assertFalse((self.root/'uncalibrated').exists())

    def test_native_ioh_counts_the_same_calls_including_penalties(self):
        try:
            import ioh
        except ImportError:
            self.skipTest('optional ioh unavailable')
        p=CountingProblem(failures=(3,))
        out=self.root/'ioh'
        s=optimize(p,budget=6,output=out,with_ioh=True,failure_penalty=1e6)
        self.assertEqual(len(p.calls),6)
        dat=next((out/'ioh').rglob('*.dat'))
        lines=dat.read_text().splitlines()
        self.assertEqual(len(lines),7)
        self.assertEqual([int(row.split()[0]) for row in lines[1:]],list(range(1,7)))


if __name__ == '__main__':
    unittest.main()
