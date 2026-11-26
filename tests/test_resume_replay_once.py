"""Replay preserves proposals while fitting each historical GP only once."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield.optimization import main, optimize
from flowfield.optimization import cli
from flowfield.optimization.bo import BayesianOptimizationSampler
from flowfield.optimization.turbo import TuRBOSampler
from flowfield.optimization.runner import _verify_history
from flowfield.optimization.state import load_history
from test_optimization import CountingProblem


AVAILABLE = all(importlib.util.find_spec(name) for name in ('numpy','scipy','sklearn'))


@unittest.skipUnless(AVAILABLE, 'optional Bayesian optimization dependencies unavailable')
class SingleReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def run_api(self, method, folder, problem, **options):
        return optimize(problem, output=folder, method=method,
            **(dict(budget=6, with_ioh=False, bo_initial_points=3, bo_candidates=16,
                    turbo_initial_points=3, turbo_candidates=16) | options))

    def run_cli(self, method, folder, problem, *args):
        problem.name = 'P000'
        out, err = io.StringIO(), io.StringIO()
        with patch.object(cli,'_make_problem',return_value=problem), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = main(['--output',str(folder),'--optimizer',method,
                          '--bo-initial-points','3','--bo-candidates','16',
                          '--turbo-initial-points','3','--turbo-candidates','16',*args])
        return status, out.getvalue(), err.getvalue()

    def test_api_and_cli_reconstruct_each_gp_once_and_preserve_trajectory(self):
        for method, cls in (('bo',BayesianOptimizationSampler),('turbo',TuRBOSampler)):
            for interface in ('api','cli'):
                with self.subTest(method=method, interface=interface):
                    folder = self.root/(method+'-'+interface)
                    first, after, reference = CountingProblem(), CountingProblem(), CountingProblem()
                    if interface=='api':
                        self.run_api(method,folder,first)
                    else:
                        self.run_cli(method,folder,first,'--budget','6')
                        reference.name='P000'
                    fits = []
                    original = cls._fit
                    def fit(sampler):
                        fits.append(len(sampler.x))
                        return original(sampler)
                    with patch.object(cls,'_fit',fit):
                        if interface=='api':
                            self.run_api(method,folder,after,resume=True,extend_budget=13)
                        else:
                            code, _, log = self.run_cli(method,folder,after,'--resume','--extend-budget','13')
                            self.assertEqual(code,0)
                            self.assertIn('rebuilding optimizer from 6 saved evaluations',log)
                            self.assertEqual(log.count('verifying saved proposal'),6)
                            self.assertIn('P000: 7/13 proposing',log)
                            self.assertIn('P000: 7/13 started',log)
                    self.assertEqual(fits,list(range(3,13)))
                    self.run_api(method,self.root/(folder.name+'-reference'),reference,budget=13)
                    self.assertEqual(first.calls+after.calls,reference.calls)
                    self.assertEqual(len(after.calls),7)

    def test_dry_run_stops_at_saved_proposal_and_completed_run_needs_no_fits(self):
        folder = self.root/'dry-run'
        self.run_cli('turbo',folder,CountingProblem(),'--budget','6')
        before = {p.relative_to(folder):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
        fits = []
        original = TuRBOSampler._fit
        def fit(sampler):
            fits.append(len(sampler.x))
            return original(sampler)
        after = CountingProblem()
        with patch.object(TuRBOSampler,'_fit',fit):
            code, output, log = self.run_cli('turbo',folder,after,'--resume','--extend-budget','13','--dry-run')
        self.assertEqual(code,0)
        self.assertEqual(json.loads(output)['remaining_attempts'],7)
        self.assertEqual(fits,[3,4,5])
        self.assertIn('verified saved proposal 6/6',log)
        self.assertEqual(after.calls,[])
        self.assertEqual(before,{p.relative_to(folder):p.read_bytes() for p in folder.rglob('*') if p.is_file()})
        fits.clear()
        with patch.object(TuRBOSampler,'_fit',fit):
            self.run_cli('turbo',folder,after,'--resume','--extend-budget','6')
        self.assertEqual(fits,[])

    def test_progress_announces_gp_fit_before_it_starts(self):
        folder = self.root/'progress'
        original_problem = CountingProblem()
        self.run_api('turbo',folder,original_problem)
        history, _ = load_history(folder,6,None)
        stages = []
        original = TuRBOSampler._fit
        def fit(sampler):
            self.assertEqual(stages[-1],(len(sampler.x)+1,'verifying'))
            return original(sampler)
        with patch.object(TuRBOSampler,'_fit',fit):
            replay = _verify_history(CountingProblem(),original_problem.initial_design(),'turbo',4,.1,42,history,
                budget=13,turbo_initial_points=3,turbo_candidates=16,
                progress=lambda name,index,total,stage:stages.append((index,stage)))
        self.addCleanup(replay.close)
        self.assertEqual(stages[0],(0,'verification_started'))
        self.assertEqual(stages[-1],(6,'verification_finished'))
        self.assertIsNotNone(replay.proposals.gi_frame)

    def test_changed_journal_between_validation_and_execution_refuses_without_cfd(self):
        folder = self.root/'changed'
        self.run_cli('turbo',folder,CountingProblem(),'--budget','6')
        original_execute = cli._execute_suite
        captured = []
        def change_then_execute(args,*rest):
            captured.extend(args._verified_replays.values())
            path = folder/'P000/evaluations.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['objective'] += .01
            rows[0]['result']['objective'] = rows[0]['objective']
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            return original_execute(args,*rest)
        after = CountingProblem()
        with patch.object(cli,'_execute_suite',change_then_execute), self.assertRaises(SystemExit):
            self.run_cli('turbo',folder,after,'--resume','--extend-budget','13')
        self.assertEqual(after.calls,[])
        self.assertTrue(captured)
        self.assertTrue(all(replay.proposals.gi_frame is None for replay in captured))


if __name__ == '__main__':
    unittest.main()
