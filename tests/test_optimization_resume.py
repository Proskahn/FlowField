"""Recovery tests use real pycma/IOH and process exits, without CFD costs."""
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

from flowfield.optimization import main, optimize
from flowfield.optimization.state import ResumeError, check_legacy_pid, run_lock, validate_experiment
from test_optimization import CountingProblem, DiscreteProblem, cma


@unittest.skipIf(cma is None, 'optional cma package unavailable')
class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def rows(self, folder, name='evaluations.jsonl'):
        return [json.loads(line) for line in (folder/name).read_text().splitlines()]

    def stop_after(self, n):
        def progress(name, index, budget, status):
            if index == n and status == 'ok':
                raise KeyboardInterrupt()
        return progress

    def test_resume_matches_uninterrupted_at_partial_and_full_generations(self):
        reference = CountingProblem()
        optimize(reference, output=self.root/'reference', budget=20, with_ioh=False)
        for n in (1,2,4,5,6,9,19,20):
            with self.subTest(n=n):
                folder = self.root/str(n)
                before, after = CountingProblem(), CountingProblem()
                with self.assertRaises(KeyboardInterrupt):
                    optimize(before,output=folder,budget=20,with_ioh=False,progress=self.stop_after(n))
                prefix = (folder/'evaluations.jsonl').read_bytes()
                result = optimize(after,output=folder,budget=20,with_ioh=False,resume=True)
                self.assertEqual(before.calls+after.calls,reference.calls)
                self.assertEqual(len(after.calls),20-n)
                self.assertTrue((folder/'evaluations.jsonl').read_bytes().startswith(prefix))
                self.assertEqual(result['attempts'],20)
                self.assertEqual(result['new_attempts'],20-n)
                self.assertEqual(result['complete_generations'],4)
                self.assertEqual(result['partial_generation_evaluations'],3)

    def test_interrupted_attempt_keeps_budget_and_receives_original_penalty(self):
        folder = self.root/'pending'
        before = CountingProblem(interrupt=3)
        with self.assertRaises(KeyboardInterrupt):
            optimize(before,output=folder,budget=10,with_ioh=False,failure_penalty=1e6)
        after = CountingProblem()
        result = optimize(after,output=folder,budget=10,with_ioh=False,failure_penalty=1e6,resume=True)
        reference = CountingProblem(failures=(3,))
        optimize(reference,output=self.root/'reference',budget=10,with_ioh=False,failure_penalty=1e6)
        self.assertEqual(before.calls+after.calls,reference.calls)
        self.assertEqual((result['attempts'],result['failed_evaluations'],len(after.calls)),(10,1,7))
        self.assertIsNone(self.rows(folder)[2]['objective'])  # Original evidence untouched.
        recovery = self.rows(folder,'recoveries.jsonl')
        self.assertEqual(len(recovery),1)
        self.assertEqual(recovery[0]['evaluation'],3)
        again = CountingProblem()
        optimize(again,output=folder,budget=10,with_ioh=False,failure_penalty=1e6,resume=True)
        self.assertEqual(again.calls,[])
        self.assertEqual(self.rows(folder,'recoveries.jsonl'),recovery)

    def test_pending_attempt_without_penalty_refuses_without_new_calls(self):
        folder = self.root/'pending'
        with self.assertRaises(KeyboardInterrupt):
            optimize(CountingProblem(interrupt=2),output=folder,budget=10,with_ioh=False)
        after = CountingProblem()
        original = (folder/'summary.json').read_bytes()
        with self.assertRaisesRegex(ResumeError,'failure-penalty'):
            optimize(after,output=folder,budget=10,with_ioh=False,resume=True)
        self.assertFalse(after.calls)
        self.assertEqual((folder/'summary.json').read_bytes(),original)

    def test_changed_settings_or_versions_refuse(self):
        folder = self.root/'settings'
        optimize(CountingProblem(),output=folder,budget=5,with_ioh=False)
        for changes in ({'budget':6},{'seed':7},{'sigma':.2},{'population':6},{'failure_penalty':1e6}):
            options = dict(budget=5,with_ioh=False,**{k:v for k,v in changes.items() if k!='budget'})
            options['budget'] = changes.get('budget',5)
            p = CountingProblem()
            with self.assertRaises(ResumeError):
                optimize(p,output=folder,resume=True,**options)
            self.assertEqual(p.calls,[])
        path = folder/'experiment.json'
        saved = json.loads(path.read_text());saved['versions']['cma']='wrong'
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ResumeError,'cma version'):
            optimize(CountingProblem(),output=folder,resume=True,budget=5,with_ioh=False)

    def test_tuple_configuration_survives_json_round_trip(self):
        current = dict(config={'cells':(50,8,8)},population=None)
        validate_experiment(json.loads(json.dumps(current)),current)

    def test_initialization_crash_can_resume_without_spending_budget(self):
        folder = self.root/'initializing';folder.mkdir()
        (folder/'.optimizer.lock').write_text('stale lock metadata')
        p = CountingProblem()
        result = optimize(p,output=folder,budget=5,with_ioh=False,resume=True)
        self.assertEqual((len(p.calls),result['attempts']),(5,5))

    def test_corruption_and_proposal_divergence_refuse_before_mutation(self):
        folder = self.root/'corrupt'
        with self.assertRaises(KeyboardInterrupt):
            optimize(CountingProblem(),output=folder,budget=10,with_ioh=False,progress=self.stop_after(3))
        original = (folder/'events.jsonl').read_bytes()
        (folder/'events.jsonl').write_bytes(original+b'{"event":')
        with self.assertRaisesRegex(ResumeError,'Truncated'):
            optimize(CountingProblem(),output=folder,budget=10,with_ioh=False,resume=True)
        (folder/'events.jsonl').write_bytes(original)
        events, rows = self.rows(folder,'events.jsonl'),self.rows(folder)
        events[2]['x'][0] += .01;rows[1]['x'][0] += .01
        for name,values in (('events.jsonl',events),('evaluations.jsonl',rows)):
            (folder/name).write_text(''.join(json.dumps(v)+'\n' for v in values))
        before = (folder/'summary.json').read_bytes()
        p = CountingProblem()
        with self.assertRaisesRegex(ResumeError,'proposal 2'):
            optimize(p,output=folder,budget=10,with_ioh=False,resume=True)
        self.assertFalse(p.calls)
        self.assertEqual((folder/'summary.json').read_bytes(),before)

    def test_separable_2145_dimensions_and_discrete_pool_resume(self):
        for cls in (lambda:CountingProblem(dimension=2145),DiscreteProblem):
            folder = self.root/cls().__class__.__name__
            before, after, reference = cls(),cls(),cls()
            with self.assertRaises(KeyboardInterrupt):
                optimize(before,output=folder,budget=6,with_ioh=False,progress=self.stop_after(1))
            resumed = optimize(after,output=folder,budget=6,with_ioh=False,resume=True)
            expected = optimize(reference,output=self.root/(folder.name+'reference'),budget=6,with_ioh=False)
            self.assertEqual(before.calls+after.calls,reference.calls)
            self.assertEqual(resumed['best_objective'],expected['best_objective'])

    def test_ioh_resume_preserves_old_trace_and_builds_one_complete_new_trace(self):
        try:
            import ioh
        except ImportError:
            self.skipTest('optional ioh unavailable')
        folder = self.root/'ioh'
        with self.assertRaises(KeyboardInterrupt):
            optimize(CountingProblem(),output=folder,budget=10,with_ioh=True,progress=self.stop_after(3))
        original = {p.relative_to(folder):p.read_bytes() for p in (folder/'ioh').rglob('*') if p.is_file()}
        p = CountingProblem()
        result = optimize(p,output=folder,budget=10,with_ioh=True,resume=True)
        self.assertEqual(len(p.calls),7)
        self.assertEqual(original,{path:(folder/path).read_bytes() for path in original})
        dat = next(Path(result['ioh_directory']).rglob('*.dat'))
        rows = dat.read_text().splitlines()[1:]
        self.assertEqual([int(r.split()[0]) for r in rows],list(range(1,11)))

    def test_process_exit_without_finally_can_resume_and_releases_lock(self):
        folder = self.root/'hard-exit'
        code = '''import os
from flowfield.optimization import optimize
from test_optimization import CountingProblem
def progress(name,index,budget,status):
    if index==3 and status=='ok': os._exit(17)
optimize(CountingProblem(),output=__import__('sys').argv[1],budget=10,with_ioh=False,progress=progress)
'''
        env = dict(os.environ,PYTHONPATH=os.pathsep.join(sys.path))
        child = subprocess.run([sys.executable,'-c',code,str(folder)],env=env,capture_output=True,text=True)
        self.assertEqual(child.returncode,17,child.stderr)
        p = CountingProblem()
        result = optimize(p,output=folder,budget=10,with_ioh=False,resume=True)
        self.assertEqual(len(p.calls),7)
        self.assertEqual(result['attempts'],10)

    def test_exclusive_lock_blocks_other_process(self):
        path = self.root/'lock'
        code = 'from flowfield.optimization.state import run_lock\nwith run_lock(__import__("sys").argv[1]): pass'
        with run_lock(path):
            child = subprocess.run([sys.executable,'-c',code,str(path)],capture_output=True,text=True)
            self.assertNotEqual(child.returncode,0)
            self.assertIn('already active',child.stderr)
        with run_lock(path):
            pass

    def test_live_legacy_pid_refuses(self):
        path = self.root/'suite'
        Path(str(path)+'.pid').write_text(str(os.getppid()))
        with self.assertRaisesRegex(ResumeError,'still exists'):
            check_legacy_pid(path)

    def test_cli_restores_options_skips_complete_and_preserves_total_budget(self):
        folder = self.root/'cli'
        first = CountingProblem(interrupt=3);first.name='P000'
        second = CountingProblem();second.name='P001'
        second.initial_design = lambda:[1.,1.]
        def make(identifier,args):
            return first if identifier=='P000' else second
        with patch('flowfield.optimization.cli._make_problem',side_effect=make), patch('flowfield.optimization.cli.Runtime.resolve',return_value={'backend':'docker'}):
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                result = main(['--problems','P001','P000','--budget','6','--seed','0','--failure-penalty','1000',
                               '--output',str(folder)])
        self.assertEqual(result,130)
        after = CountingProblem();after.name='P000'
        other = CountingProblem();other.name='P001'
        other.initial_design = lambda:[1.,1.]
        with patch('flowfield.optimization.cli._make_problem',side_effect=lambda identifier,args:after if identifier=='P000' else other), patch('flowfield.optimization.cli.Runtime.resolve',return_value={'backend':'docker'}):
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                result = main(['--resume','--output',str(folder)])
        self.assertEqual(result,1)  # Interruption is counted as a penalized failure.
        self.assertEqual(len(after.calls),3)
        self.assertEqual(other.calls,[])
        summary = json.loads((folder/'suite-summary.json').read_text())
        self.assertEqual(summary['total_attempts'],12)
        self.assertEqual(summary['budget_per_problem'],6)

    def test_legacy_cli_infers_settings_and_dry_run_is_read_only(self):
        folder = self.root/'legacy';p=CountingProblem();p.name='P000'
        with patch('flowfield.optimization.cli._make_problem',return_value=p), patch('flowfield.optimization.cli.Runtime.resolve',return_value={'backend':'docker'}):
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                main(['--problems','P000','--budget','6','--seed','0','--output',str(folder)])
        (folder/'run-options.json').unlink()
        before = {p.relative_to(folder):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
        out = io.StringIO()
        with patch('flowfield.optimization.cli._make_problem',return_value=p),contextlib.redirect_stdout(out):
            self.assertEqual(main(['--resume','--output',str(folder),'--dry-run']),0)
        plan = json.loads(out.getvalue())
        self.assertEqual((plan['budget_per_problem'],plan['seed'],plan['remaining_attempts']),(6,0,0))
        self.assertEqual(before,{p.relative_to(folder):p.read_bytes() for p in folder.rglob('*') if p.is_file()})
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            main(['--resume','--output',str(folder),'--budget','20'])


if __name__ == '__main__':
    unittest.main()
