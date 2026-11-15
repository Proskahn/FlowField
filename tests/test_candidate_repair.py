"""P003 volume feasibility and model/CFD coordinate consistency; no CFD solves."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield import P003Problem, TopologyConfig
from flowfield.optimization import BayesianOptimizationSampler, TuRBOSampler, optimize
from flowfield.optimization.state import ResumeError
from flowfield.optimization.turbo import posterior_sample
from flowfield.runtime import SimulationError
from test_bo import AVAILABLE


class GeometricProblem(P003Problem):
    """Real P003 geometry with a cheap, explicitly non-CFD test objective."""
    def __init__(self, *, config=None, failures=(), interrupt=None):
        super().__init__(config=config or TopologyConfig(design_nx=4, design_ny=4, filter_radius_m=.004))
        self.calls = []
        self.failures, self.interrupt = failures, interrupt

    def evaluate_raw(self, x):
        self.calls.append(list(x))
        if len(self.calls) == self.interrupt:
            raise KeyboardInterrupt()
        if len(self.calls) in self.failures:
            raise SimulationError('test interruption recovery')
        volume = self.decode(x)['volume_fraction']
        if volume > self.config.volume_limit+1e-12:
            raise AssertionError('An infeasible proposal reached evaluation')
        return dict(objective=sum((v-.01)**2 for v in x), status='ok',
                    constraints={'channel_volume': volume-self.config.volume_limit})


@unittest.skipUnless(AVAILABLE, 'optional Bayesian optimization dependencies unavailable')
class CandidateRepairTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_real_projected_volume_preserves_feasible_designs_and_is_idempotent(self):
        np = self.np
        problem = P003Problem()
        original = np.vstack([np.zeros(problem.dimension), problem.initial_design(),
                              problem.preset('parallel'), np.ones(problem.dimension),
                              np.random.RandomState(42).random_sample((12, problem.dimension))])
        snapshot = original.copy()
        repaired = problem.repair_candidates(original)
        np.testing.assert_array_equal(original, snapshot)
        np.testing.assert_array_equal(repaired[:3], original[:3])
        self.assertTrue(np.all(repaired <= original))
        for row in repaired:
            self.assertLessEqual(problem.decode(row)['volume_fraction'], .5+1e-12)
        self.assertAlmostEqual(repaired[3, 0], problem.initial_design()[0], places=10)
        np.testing.assert_array_equal(problem.repair_candidates(repaired), repaired)
        self.assertEqual(len(np.unique(repaired, axis=0)), len(repaired))

    def test_configurable_projection_mesh_and_limit(self):
        for beta, limit, mesh in ((0., .3, 'coarse'), (1., .7, 'medium'), (80., .35, 'coarse'), (50., 1., 'coarse')):
            problem = P003Problem(config=replace(TopologyConfig(), projection_beta=beta, volume_limit=limit, mesh=mesh))
            original = self.np.random.RandomState(1).random_sample((3, problem.dimension))
            repaired = problem.repair_candidates(original)
            for row in repaired:
                self.assertLessEqual(problem.decode(row)['volume_fraction'], limit+1e-12)

    def test_trust_region_anchor_preserves_box_and_rejects_invalid_requests(self):
        np = self.np
        problem = P003Problem()
        center = np.asarray(problem.initial_design())
        lower, upper = center-.0001, center+.0001
        original = np.vstack([upper, center, lower])
        repaired = problem.repair_candidates(original, anchor=lower)
        self.assertTrue(np.all(repaired >= lower))
        self.assertTrue(np.all(repaired <= original))
        for row in repaired:
            self.assertLessEqual(problem.decode(row)['volume_fraction'], .5+1e-12)
        for values, anchor in ((original, upper), (original, [0.]),
                               (np.ones_like(original), np.ones(problem.dimension)),
                               (np.full_like(original, np.nan), None)):
            with self.assertRaises(ValueError):
                problem.repair_candidates(values, anchor=anchor)

    def test_acquisition_and_training_use_repaired_full_dimension_inputs(self):
        np = self.np
        for kind in ('bo', 'turbo'):
            problem = GeometricProblem(config=TopologyConfig())
            initial = problem.initial_design()
            cls = BayesianOptimizationSampler if kind == 'bo' else TuRBOSampler
            sampler = cls(problem, seed=42, initial=initial,
                          **{f'{kind}_initial_points': 2, f'{kind}_candidates': 16})
            sampler.initialize(initial, problem.evaluate_raw(initial))
            first = sampler.ask()
            self.assertTrue(sampler.proposal_metadata['applied'])
            self.assertGreater(problem.decode(sampler.proposal_metadata['proposed_x'])['volume_fraction'], .9)
            sampler.tell(first, [problem.evaluate_raw(first[0])])
            evaluated = np.asarray(problem.calls)
            checked = []
            def verify(pool):
                checked.append(len(pool))
                for row in pool:
                    self.assertEqual(len(row), 2145)
                    self.assertLessEqual(problem.decode(row)['volume_fraction'], .5+1e-12)
            if kind == 'bo':
                original = sampler._acquisition
                def acquisition(pool):
                    verify(pool)
                    return original(pool)
                with patch.object(sampler, '_acquisition', acquisition):
                    batch = sampler.ask()
            else:
                sampler.state.length = .001  # A narrow region with a nonzero lower corner.
                def sample(model, pool, rng):
                    verify(pool)
                    self.assertTrue(np.all(pool >= sampler.candidate_anchor))
                    return posterior_sample(model, pool, rng)
                with patch('flowfield.optimization.turbo.posterior_sample', sample):
                    batch = sampler.ask()
            self.assertTrue(checked)
            np.testing.assert_array_equal(sampler.model.X_train_, evaluated)
            sampler.tell(batch, [problem.evaluate_raw(batch[0])])
            np.testing.assert_array_equal(sampler.x, problem.calls)

    def test_turbo_restart_warmup_is_also_repaired(self):
        problem = GeometricProblem()
        sampler = TuRBOSampler(problem, seed=0, initial=problem.initial_design(),
                               turbo_initial_points=2, turbo_candidates=16)
        sampler.initialize(problem.initial_design(), problem.evaluate_raw(problem.initial_design()))
        sampler.tell(sampler.ask(), [dict(objective=1.)])
        sampler.state.restart_required = True
        for _ in range(2):
            batch = sampler.ask()
            self.assertTrue(sampler.proposal_metadata['applied'])
            sampler.tell(batch, [problem.evaluate_raw(batch[0])])
        self.assertEqual(sampler.restarts, 1)
        self.assertEqual(len(sampler.y), 2)

    def run_problem(self, kind, problem, folder, **options):
        return optimize(problem, optimizer=kind, output=self.root/folder, with_ioh=False,
                        **(dict(budget=5, failure_penalty=100.,
                               **{f'{kind}_initial_points': 2, f'{kind}_candidates': 16}) | options))

    def test_journals_replay_repair_and_recover_interrupted_candidates(self):
        from flowfield.optimization.state import load_history
        for kind in ('bo', 'turbo'):
            reference = GeometricProblem(failures=(3,))
            self.run_problem(kind, reference, kind+'-reference')
            before, after = GeometricProblem(interrupt=3), GeometricProblem()
            with self.assertRaises(KeyboardInterrupt):
                self.run_problem(kind, before, kind)
            prefix = (self.root/kind/'evaluations.jsonl').read_bytes()
            summary = self.run_problem(kind, after, kind, resume=True)
            self.assertEqual(before.calls+after.calls, reference.calls)
            self.assertEqual(summary['new_attempts'], 2)
            self.assertEqual(summary['failed_evaluations'], 1)
            self.assertTrue((self.root/kind/'evaluations.jsonl').read_bytes().startswith(prefix))
            history, pending = load_history(self.root/kind, 5, 100.)
            self.assertFalse(pending)
            for record in history[1:]:
                detail = record['proposal']
                repaired = after.repair_candidates([detail['proposed_x']], anchor=detail['anchor_x'])
                self.np.testing.assert_array_equal(repaired[0], record['x'])
            manifest = json.loads((self.root/kind/'experiment.json').read_text())
            self.assertEqual(manifest['optimizer_revision'], '2')
            self.assertEqual(manifest['optimizer_settings']['candidate_repair'], after.candidate_repair)

    def test_legacy_revisions_and_modified_proposal_evidence_refuse_before_evaluation(self):
        for kind in ('bo', 'turbo'):
            self.run_problem(kind, GeometricProblem(), kind, budget=2)
            path = self.root/kind/'experiment.json'
            manifest = json.loads(path.read_text())
            path.write_text(json.dumps(manifest | dict(optimizer_revision='1')))
            after = GeometricProblem()
            with self.assertRaisesRegex(ResumeError, 'optimizer_revision'):
                self.run_problem(kind, after, kind, budget=2, resume=True)
            path.write_text(json.dumps(manifest))
            for filename in ('events.jsonl', 'evaluations.jsonl'):
                path = self.root/kind/filename
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                for row in rows:
                    if row.get('proposal'):
                        row['proposal']['proposed_x'][0] = 0.
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ResumeError, 'Cannot reproduce saved proposal'):
                self.run_problem(kind, after, kind, budget=2, resume=True)
            self.assertFalse(after.calls)

    def test_infeasible_custom_baseline_is_rejected_before_evaluation(self):
        for kind in ('bo', 'turbo'):
            problem = GeometricProblem()
            with self.assertRaisesRegex(ValueError, 'feasible baseline'):
                self.run_problem(kind, problem, kind, initial=[1.]*problem.dimension)
            self.assertFalse(problem.calls)


if __name__ == '__main__':
    unittest.main()
