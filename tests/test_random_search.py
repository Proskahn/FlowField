"""Random baseline, P003 geometric feasibility and journal recovery; no CFD."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from flowfield import P003Problem, TopologyConfig
from flowfield.optimization import RandomSampler, main, optimize
from flowfield.optimization.state import ResumeError, load_history
from flowfield.runtime import SimulationError
from test_optimization import CountingProblem, DiscreteProblem


REPAIR_AVAILABLE = all(importlib.util.find_spec(name) for name in ('numpy', 'scipy'))


class VolumeProblem(P003Problem):
    """Actual P003 geometry with an explicitly synthetic objective for tests."""
    def __init__(self, *, failures=(), interrupt=None):
        super().__init__(config=TopologyConfig(design_nx=4, design_ny=4, filter_radius_m=.004))
        self.calls = []
        self.failures, self.interrupt = failures, interrupt

    def evaluate_raw(self, x):
        self.calls.append(list(x))
        volume = self.decode(x)['volume_fraction']
        if volume > self.config.volume_limit+1e-12:
            raise AssertionError('Infeasible design reached evaluation')
        if len(self.calls) == self.interrupt:
            raise KeyboardInterrupt()
        if len(self.calls) in self.failures:
            raise SimulationError('Synthetic evaluation failure')
        return dict(status='ok', objective=sum(v*v for v in x),
                    constraints={'channel_volume': volume-self.config.volume_limit})


class RandomSearchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def run_problem(self, problem, folder='run', **options):
        return optimize(problem, optimizer='random', output=self.root/folder,
                        **(dict(budget=7, seed=42, with_ioh=False, failure_penalty=1e6) | options))

    def rows(self, folder='run'):
        return [json.loads(line) for line in (self.root/folder/'evaluations.jsonl').read_text().splitlines()]

    def test_budget_includes_baseline_and_failures_and_feedback_does_not_adapt(self):
        for budget in (1, 2, 7):
            problem = CountingProblem(failures=(2, 4))
            summary = self.run_problem(problem, str(budget), budget=budget)
            rng = random.Random(42)
            expected = [problem.initial_design()] + [[rng.uniform(-2, 4) for _ in range(2)]
                                                      for _ in range(budget-1)]
            self.assertEqual(problem.calls, expected)
            self.assertEqual(summary['attempts'], budget)
            self.assertEqual(summary['failed_evaluations'], sum(i <= budget for i in (2, 4)))
            self.assertEqual(summary['random_draws'], budget-1)
            self.assertEqual(summary['complete_generations'], 0)
            self.assertEqual(summary['stop_reason'], 'evaluation_budget')
            self.assertEqual(self.rows(str(budget))[0]['phase'], 'baseline')
        other = CountingProblem(constrained=True)
        self.run_problem(other, 'other')
        self.assertEqual(other.calls, problem.calls)

    def test_private_rng_seed_zero_and_different_seeds(self):
        original = random.getstate()
        problems = [CountingProblem() for _ in range(3)]
        for index, (problem, seed) in enumerate(zip(problems, (0, 0, 1))):
            self.run_problem(problem, str(index), seed=seed)
        self.assertEqual(random.getstate(), original)
        self.assertEqual(problems[0].calls, problems[1].calls)
        self.assertNotEqual(problems[0].calls[1:], problems[2].calls[1:])

    def test_unconstrained_random_runs_without_optional_dependencies(self):
        code = '''
import sys
class BlockOptional:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('numpy', 'scipy', 'cma', 'sklearn', 'ioh'):
            raise ImportError('Optional import blocked: '+fullname)
sys.meta_path.insert(0, BlockOptional())
from flowfield import P000Problem
from flowfield.optimization import RandomSampler
p = P000Problem()
s = RandomSampler(p, seed=42, initial=p.initial_design())
assert len(s.ask()[0]) == 2
'''
        child = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_discrete_hook_and_exhaustion_preserve_unique_feasible_paths(self):
        problem = DiscreteProblem()
        summary = self.run_problem(problem)
        self.assertEqual(problem.calls, [[0, 0], [1, 1]])
        self.assertEqual(summary['attempts'], 2)
        self.assertEqual(summary['stop_reason'], 'no_new_design_found')
        self.assertGreaterEqual(summary['random_duplicate_draws'], 1000)

    def test_ioh_counts_only_budgeted_evaluations_including_penalties(self):
        if importlib.util.find_spec('ioh') is None:
            self.skipTest('optional ioh unavailable')
        problem = CountingProblem(failures=(3,))
        summary = self.run_problem(problem, with_ioh=True)
        dat = next(Path(summary['ioh_directory']).rglob('*.dat'))
        rows = dat.read_text().splitlines()[1:]
        self.assertEqual([int(row.split()[0]) for row in rows], list(range(1, 8)))
        self.assertEqual(len(problem.calls), 7)

    def test_cli_dry_run_declares_p003_repair_without_creating_output(self):
        output = self.root/'plan'
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(main(['--problem', 'P003', '--optimizer', 'random', '--budget', '100',
                                   '--output', str(output), '--dry-run']), 0)
        plan = json.loads(stream.getvalue())
        self.assertEqual(plan['maximum_cfd_attempts'], 100)
        entry = plan['problems'][0]
        self.assertEqual(entry['dimension'], 2145)
        self.assertEqual(entry['candidate_repair']['limit'], .5)
        self.assertFalse(entry['optimizer_settings']['objective_adaptation'])
        self.assertFalse(output.exists())

    @unittest.skipUnless(REPAIR_AVAILABLE, 'optional volume repair dependencies unavailable')
    def test_full_p003_dimension_is_feasible_and_raw_proposals_are_logged(self):
        import numpy as np
        problem = P003Problem()
        sampler = RandomSampler(problem, seed=42, initial=problem.initial_design())
        rng = random.Random(42)
        for _ in range(5):
            batch = sampler.ask()
            design = batch[0]
            raw = [rng.uniform(0, 1) for _ in range(2145)]
            self.assertEqual(sampler.proposal_metadata['proposed_x'], raw)
            self.assertTrue(sampler.proposal_metadata['applied'])
            self.assertGreater(problem.decode(raw)['volume_fraction'], .9)
            self.assertEqual(len(design), 2145)
            self.assertLessEqual(problem.decode(design)['volume_fraction'], .5+1e-12)
            np.testing.assert_array_equal(problem.repair_candidates([raw])[0], design)
            sampler.tell(batch, [dict(objective=1.)])

    @unittest.skipUnless(REPAIR_AVAILABLE, 'optional volume repair dependencies unavailable')
    def test_duplicates_are_checked_after_repair(self):
        import numpy as np
        problem = CountingProblem()
        problem.candidate_repair = dict(method='test_cap', revision='1')
        problem.repair_candidates = lambda rows, **kwargs: np.minimum(rows, problem.initial_design())
        problem.sample_design = lambda rng: [4., 4.]
        summary = self.run_problem(problem)
        self.assertEqual(problem.calls, [problem.initial_design()])
        self.assertEqual(summary['random_repaired_draws'], 1000)
        self.assertEqual(summary['random_duplicate_draws'], 1000)
        self.assertEqual(summary['stop_reason'], 'no_new_design_found')

    @unittest.skipUnless(REPAIR_AVAILABLE, 'optional volume repair dependencies unavailable')
    def test_infeasible_initial_design_rejected_before_evaluation(self):
        problem = VolumeProblem()
        problem.initial_design = lambda: [1.]*problem.dimension
        with self.assertRaisesRegex(ValueError, 'Initial design violates'):
            self.run_problem(problem)
        self.assertEqual(problem.calls, [])

    @unittest.skipUnless(REPAIR_AVAILABLE, 'optional volume repair dependencies unavailable')
    def test_repaired_journals_resume_interrupted_attempt_without_repeating_it(self):
        reference = VolumeProblem(failures=(3,))
        self.run_problem(reference, 'reference')
        before, after = VolumeProblem(interrupt=3), VolumeProblem()
        with self.assertRaises(KeyboardInterrupt):
            self.run_problem(before)
        prefix = (self.root/'run/evaluations.jsonl').read_bytes()
        summary = self.run_problem(after, resume=True)
        self.assertEqual(before.calls+after.calls, reference.calls)
        self.assertEqual(len(after.calls), 4)
        self.assertEqual(summary['attempts'], 7)
        self.assertEqual(summary['failed_evaluations'], 1)
        self.assertTrue((self.root/'run/evaluations.jsonl').read_bytes().startswith(prefix))
        recovered, pending = load_history(self.root/'run', 7, 1e6)
        self.assertEqual(pending, [])
        expected = self.rows('reference')
        for row, ref in zip(recovered, expected):
            self.assertEqual(row['x'], ref['x'])
            self.assertEqual(row['objective'], ref['objective'])
            self.assertEqual(row.get('proposal'), ref.get('proposal'))
        self.assertEqual(summary['random_draws'], 6)
        self.assertTrue(all(row['feasible'] for row in recovered if row['valid']))

    @unittest.skipUnless(REPAIR_AVAILABLE, 'optional volume repair dependencies unavailable')
    def test_changed_repair_revision_or_dependencies_refuses_before_new_calls(self):
        self.run_problem(VolumeProblem())
        path = self.root/'run/experiment.json'
        original = json.loads(path.read_text())
        variants = []
        for key in ('optimizer_revision', 'settings', 'numpy', 'scipy', 'python'):
            saved = json.loads(json.dumps(original))
            if key == 'optimizer_revision':
                saved[key] = '1'
            elif key == 'settings':
                saved['optimizer_settings']['candidate_repair']['revision'] = 'old'
            else:
                saved['versions'][key] = 'wrong'
            variants.append(saved)
        for saved in variants:
            path.write_text(json.dumps(saved))
            problem = VolumeProblem()
            with self.assertRaises(ResumeError):
                self.run_problem(problem, resume=True)
            self.assertEqual(problem.calls, [])

    def test_missing_repair_dependency_has_actionable_install_hint(self):
        with patch('flowfield.optimization.repair.CandidateRepair', side_effect=ImportError('numpy')):
            with self.assertRaisesRegex(RuntimeError, r'\[random\]'):
                RandomSampler(P003Problem(), seed=42, initial=P003Problem().initial_design())


if __name__ == '__main__':
    unittest.main()
