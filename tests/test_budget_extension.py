"""Budget increases replay actual optimizer/repair code with synthetic objectives."""
import contextlib
import errno
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield.optimization import main, optimize, list_optimizers
from flowfield.optimization.state import ResumeError, run_lock, save_json
from flowfield.benchmarking.plotting import _comparison
from test_optimization import CountingProblem, DiscreteProblem
from test_p003_optimizer_repair import RepairProblem


AVAILABLE = all(importlib.util.find_spec(name) for name in ('cma', 'numpy', 'scipy', 'sklearn'))


class TradeoffProblem(RepairProblem):
    def evaluate_raw(self, x):
        result = super().evaluate_raw(x)
        result['objectives']['negative_sum'] = -sum(x)
        return result


@unittest.skipUnless(AVAILABLE, 'optional optimizer dependencies unavailable')
class BudgetExtensionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def read(self, path):
        return json.loads(path.read_text())

    def rows(self, folder):
        return [json.loads(line) for line in (folder/'evaluations.jsonl').read_text().splitlines()]

    def snapshot(self, folder):
        return {p.relative_to(folder): p.read_bytes() for p in folder.rglob('*')
                if p.is_file() and p.suffix != '.lock'}

    def run_problem(self, problem, folder, **options):
        return optimize(problem, output=folder, **(dict(
            budget=6, seed=42, population=4, with_ioh=False, failure_penalty=1e6,
            bo_initial_points=3, bo_candidates=16, turbo_initial_points=3, turbo_candidates=16) | options))

    def cli(self, folder, problem, *args):
        problem.name = 'P000'
        output, error = io.StringIO(), io.StringIO()
        with patch('flowfield.optimization.runner._make_problem', return_value=problem), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = main(['--output', str(folder), *args])
        return code, output.getvalue(), error.getvalue()

    def test_all_optimizers_extend_repaired_history_identically_to_uninterrupted(self):
        for method in list_optimizers():
            with self.subTest(method=method):
                folder, reference_folder = self.root/method, self.root/(method+'-reference')
                first, after, reference = TradeoffProblem(), TradeoffProblem(), TradeoffProblem()
                self.run_problem(first, folder, method=method)
                prefix = (folder/'evaluations.jsonl').read_bytes()
                event_prefix = (folder/'events.jsonl').read_bytes()
                summary = self.run_problem(after, folder, method=method, resume=True, extend_budget=13)
                expected = self.run_problem(reference, reference_folder, method=method, budget=13)
                self.assertEqual(first.calls+after.calls, reference.calls)
                self.assertEqual((summary['attempts'],summary['new_attempts']), (13,7))
                self.assertEqual(summary['best_objective'], expected['best_objective'])
                self.assertTrue((folder/'evaluations.jsonl').read_bytes().startswith(prefix))
                self.assertTrue((folder/'events.jsonl').read_bytes().startswith(event_prefix))
                for row, ref in zip(self.rows(folder), self.rows(reference_folder)):
                    self.assertEqual({k:v for k,v in row.items() if k!='wall_time_s'},
                                     {k:v for k,v in ref.items() if k!='wall_time_s'})
                manifest = self.read(folder/'experiment.json')
                self.assertEqual((manifest['original_budget'],manifest['budget']), (6,13))
                self.assertEqual(manifest['budget_extensions'][0]['previous_budget'], 6)
                self.assertEqual(self.read(folder/'resume-sessions/000001/previous-experiment.json')['budget'], 6)
                if method == 'nsga2':
                    actual = self.read(folder/'pareto.json')
                    expected = self.read(reference_folder/'pareto.json')
                    self.assertEqual(actual['objective_names'], ['loss','negative_sum'])
                    self.assertEqual([r['x'] for r in actual['solutions']], [r['x'] for r in expected['solutions']])
                # Backups must not be discovered as duplicate independent runs.
                report = _comparison(folder)
                self.assertEqual(report['budget'], 13)
                self.assertEqual(len(next(iter(report['algorithms'].values()))['runs']), 1)

    def test_separable_100_to_200_preserves_the_exact_trajectory(self):
        folder = self.root/'separable'
        first, after, reference = (CountingProblem(dimension=2145) for _ in range(3))
        self.run_problem(first, folder, budget=100, population=8)
        summary = self.run_problem(after, folder, budget=100, population=8, resume=True, extend_budget=200)
        self.run_problem(reference, self.root/'reference', budget=200, population=8)
        self.assertEqual(first.calls+after.calls, reference.calls)
        self.assertEqual((len(after.calls),summary['attempts'],summary['replayed_evaluations']), (100,200,100))
        self.assertTrue(self.read(folder/'experiment.json')['diagonal_covariance'])

    def test_multiple_extensions_and_same_target_are_idempotent(self):
        folder = self.root/'chain'
        self.run_problem(CountingProblem(),folder)
        for target in (9,13):
            self.run_problem(CountingProblem(),folder,resume=True,extend_budget=target)
        before = (folder/'experiment.json').read_bytes()
        after = CountingProblem()
        summary = self.run_problem(after,folder,resume=True,extend_budget=13)
        self.assertEqual(after.calls, [])
        self.assertEqual(before, (folder/'experiment.json').read_bytes())
        self.assertEqual(summary['original_budget'], 6)
        self.assertEqual([r['budget'] for r in summary['budget_extensions']], [9,13])
        self.run_problem(after,folder,budget=13,resume=True)
        self.assertEqual(after.calls, [])

    def test_invalid_extension_and_changed_settings_leave_history_untouched(self):
        folder = self.root/'invalid'
        self.run_problem(CountingProblem(),folder)
        before = self.snapshot(folder)
        for options in ({'extend_budget':0}, {'extend_budget':-1}, {'extend_budget':5},
                        {'extend_budget':True}, {'extend_budget':13,'seed':43},
                        {'extend_budget':13,'population':8}, {'budget':13}):
            with self.subTest(options=options), self.assertRaises((ResumeError,ValueError)):
                self.run_problem(CountingProblem(),folder,resume=True,**options)
            self.assertEqual(before,self.snapshot(folder))
        with self.assertRaisesRegex(ValueError,'resume=True'):
            self.run_problem(CountingProblem(),self.root/'new',extend_budget=13)

    def test_early_optimizer_termination_is_not_forced_to_restart(self):
        folder = self.root/'finite'
        self.run_problem(DiscreteProblem(),folder)
        after = DiscreteProblem()
        summary = self.run_problem(after,folder,resume=True,extend_budget=13)
        self.assertEqual(after.calls, [])
        self.assertEqual(summary['attempts'], 2)
        self.assertEqual(summary['budget'], 13)
        self.assertEqual(summary['stop_reason'], 'no_new_design_found')

    def test_cli_dry_run_extension_and_later_plain_resume(self):
        folder = self.root/'cli'
        first = CountingProblem()
        self.cli(folder,first,'--budget','6','--population','8')
        before = self.snapshot(folder)
        after = CountingProblem()
        code, text, _ = self.cli(folder,after,'--resume','--extend-budget','13','--dry-run')
        self.assertEqual(code,0)
        plan = json.loads(text)
        self.assertEqual((plan['budget_per_problem'],plan['remaining_attempts']), (13,7))
        self.assertEqual(plan['problems'][0]['saved_attempts'],6)
        self.assertEqual(after.calls, [])
        self.assertEqual(before,self.snapshot(folder))
        code, _, _ = self.cli(folder,after,'--resume','--extend-budget','13')
        self.assertEqual((code,len(after.calls)), (0,7))
        for filename, key in (('run-options.json','budget'), ('plan.json','budget_per_problem'),
                              ('P000/experiment.json','budget'), ('P000/summary.json','budget')):
            self.assertEqual(self.read(folder/filename)[key],13)
        for args in (('--resume',), ('--resume','--extend-budget','13')):
            again = CountingProblem()
            code, _, _ = self.cli(folder,again,*args)
            self.assertEqual((code,again.calls), (0,[]))
        self.assertEqual(len(self.read(folder/'run-options.json')['budget_extensions']),1)

    def test_cli_legacy_options_are_upgraded_and_ordinary_budget_change_still_refuses(self):
        folder = self.root/'legacy'
        self.cli(folder,CountingProblem(),'--budget','6')
        (folder/'run-options.json').unlink()
        for args in (('--resume','--budget','13'), ('--resume','--extend-budget','5'),
                     ('--extend-budget','13'), ('--resume','--extend-budget','13','--seed','43')):
            before = self.snapshot(folder)
            with self.assertRaises(SystemExit):
                self.cli(folder,CountingProblem(),*args)
            self.assertEqual(before,self.snapshot(folder))
        after = CountingProblem()
        code, _, _ = self.cli(folder,after,'--resume','--extend-budget','13')
        self.assertEqual((code,len(after.calls)), (0,7))
        self.assertEqual(self.read(folder/'run-options.json')['budget'],13)

    def test_cli_quota_failure_after_authorization_recovers_without_repeating_cfd(self):
        folder = self.root/'quota'
        self.cli(folder,CountingProblem(),'--budget','6')
        prefix = (folder/'P000/evaluations.jsonl').read_bytes()
        def quota(path, value):
            if Path(path) == folder/'plan.json':
                raise OSError(errno.EDQUOT,'simulated quota failure')
            save_json(path,value)
        after = CountingProblem()
        with patch('flowfield.optimization.runner._save',side_effect=quota), self.assertRaises(SystemExit):
            self.cli(folder,after,'--resume','--extend-budget','13')
        self.assertEqual(after.calls, [])
        self.assertEqual(self.read(folder/'run-options.json')['budget'],13)
        self.assertEqual(self.read(folder/'plan.json')['budget_per_problem'],6)
        self.assertEqual(self.read(folder/'P000/experiment.json')['budget'],6)
        code, _, _ = self.cli(folder,after,'--resume')
        self.assertEqual((code,len(after.calls)), (0,7))
        self.assertTrue((folder/'P000/evaluations.jsonl').read_bytes().startswith(prefix))
        self.assertEqual(self.read(folder/'plan.json')['budget_per_problem'],13)

    def test_cli_failure_after_manifest_commit_does_not_report_old_success(self):
        folder = self.root/'manifest-failure'
        self.cli(folder,CountingProblem(),'--budget','6')
        def fail_after_commit(path,value):
            save_json(path,value)
            if Path(path) == folder/'P000/experiment.json':
                raise OSError(errno.EDQUOT,'simulated crash after manifest commit')
        after = CountingProblem()
        with patch('flowfield.optimization.runner._save',side_effect=fail_after_commit):
            code, _, _ = self.cli(folder,after,'--resume','--extend-budget','13')
        self.assertEqual((code,after.calls), (1,[]))
        code, _, _ = self.cli(folder,after,'--resume')
        self.assertEqual((code,len(after.calls)), (0,7))

    def test_cli_interrupted_extension_resumes_the_new_total(self):
        folder = self.root/'interrupted'
        self.cli(folder,CountingProblem(),'--budget','6','--failure-penalty','1000000')
        after = CountingProblem(interrupt=3)
        code, _, _ = self.cli(folder,after,'--resume','--extend-budget','13')
        self.assertEqual((code,len(after.calls)), (130,3))
        last = CountingProblem()
        code, _, _ = self.cli(folder,last,'--resume')
        self.assertEqual((code,len(last.calls)), (1,4))
        summary = self.read(folder/'P000/summary.json')
        self.assertEqual((summary['attempts'],summary['failed_evaluations']), (13,1))
        self.assertEqual(len(self.rows(folder/'P000')),13)

    def test_cli_rejects_incompatible_or_corrupt_history_before_authorization(self):
        for corruption in ('revision','version','proposal','truncated'):
            with self.subTest(corruption=corruption):
                folder = self.root/corruption
                self.cli(folder,CountingProblem(),'--budget','6')
                manifest = folder/'P000/experiment.json'
                if corruption in ('revision','version'):
                    data = self.read(manifest)
                    if corruption == 'revision':
                        data['optimizer_revision'] = 'incompatible'
                    else:
                        data['versions']['cma'] = 'incompatible'
                    save_json(manifest,data)
                else:
                    events = folder/'P000/events.jsonl'
                    if corruption == 'truncated':
                        events.write_bytes(events.read_bytes()+b'{"event":')
                    else:
                        for name in ('events.jsonl','evaluations.jsonl'):
                            path = folder/'P000'/name
                            rows = [json.loads(line) for line in path.read_text().splitlines()]
                            for row in rows:
                                if row.get('evaluation') == 2 and 'x' in row:
                                    row['x'][0] += .01
                            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                before = self.snapshot(folder)
                after = CountingProblem()
                with self.assertRaises(SystemExit):
                    self.cli(folder,after,'--resume','--extend-budget','13')
                self.assertEqual(after.calls,[])
                self.assertEqual(before,self.snapshot(folder))

    def test_multi_problem_partial_commit_resumes_only_unfinished_problem(self):
        folder = self.root/'suite'
        def run(*args):
            problems = {name: CountingProblem() for name in ('P000','P002')}
            for name, problem in problems.items():
                problem.name = name
            with patch('flowfield.optimization.runner._make_problem',side_effect=lambda name,args:problems[name]), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main(['--output',str(folder),*args])
            return code, problems
        run('--problem','P000','P002','--budget','6')
        def fail_second(path,value):
            if Path(path) == folder/'P002/experiment.json':
                raise OSError(errno.EDQUOT,'simulated second-manifest failure')
            save_json(path,value)
        with patch('flowfield.optimization.runner._save',side_effect=fail_second):
            code, problems = run('--resume','--extend-budget','13')
        self.assertEqual((code,len(problems['P000'].calls),len(problems['P002'].calls)),(1,7,0))
        code, problems = run('--resume')
        self.assertEqual((code,len(problems['P000'].calls),len(problems['P002'].calls)),(0,0,7))
        self.assertEqual(self.read(folder/'suite-summary.json')['total_attempts'],26)

    def test_extension_respects_suite_and_optimizer_locks(self):
        folder = self.root/'locked'
        self.cli(folder,CountingProblem(),'--budget','6')
        before = self.snapshot(folder)
        with run_lock(folder/'.suite.lock'), self.assertRaises(SystemExit):
            self.cli(folder,CountingProblem(),'--resume','--extend-budget','13')
        self.assertEqual(before,self.snapshot(folder))
        with run_lock(folder/'P000/.optimizer.lock'), self.assertRaisesRegex(ResumeError,'already active'):
            problem = CountingProblem();problem.name='P000'
            optimize(problem,output=folder/'P000',budget=6,with_ioh=False,resume=True,extend_budget=13)
        self.assertEqual(before,self.snapshot(folder))
        with run_lock(folder/'P000/.optimizer.lock'), self.assertRaises(SystemExit):
            self.cli(folder,CountingProblem(),'--resume','--extend-budget','13')
        self.assertEqual(before,self.snapshot(folder))

    def test_ioh_extension_preserves_original_and_logs_each_evaluation_once(self):
        if importlib.util.find_spec('ioh') is None:
            self.skipTest('optional IOH unavailable')
        folder = self.root/'ioh'
        self.run_problem(CountingProblem(),folder,with_ioh=True)
        old_ioh = self.snapshot(folder/'ioh')
        after = CountingProblem()
        summary = self.run_problem(after,folder,with_ioh=True,resume=True,extend_budget=13)
        self.assertEqual(len(after.calls),7)
        self.assertEqual(old_ioh,self.snapshot(folder/'ioh'))
        dat = next(Path(summary['ioh_directory']).rglob('*.dat'))
        self.assertEqual([int(line.split()[0]) for line in dat.read_text().splitlines()[1:]], list(range(1,14)))


if __name__ == '__main__':
    unittest.main()
