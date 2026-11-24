"""Actual GCMMA subproblems, FD accuracy, continuation and durable CFD replay."""

import contextlib
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

from flowfield.optimization.gcmma import GCMMA, GCMMAOptions
from flowfield.optimization.state import ResumeError
from flowfield.problems.level4_high_dimensional_topology.p003.config import TopologyConfig
from flowfield.problems.level4_high_dimensional_topology.p003.design import decode, preset
from flowfield.problems.level4_high_dimensional_topology.p003.metrics import scalarize
from flowfield.problems.level4_high_dimensional_topology.p003.gcmma import (
    ContinuationOptions, EvaluationBudgetReached, EvaluationStore,
    finite_difference_gradient, plan, run_continuation)


AVAILABLE = importlib.util.find_spec('mmapy') and importlib.util.find_spec('numpy')


@unittest.skipUnless(AVAILABLE, 'optional GCMMA dependencies unavailable')
class GCMMATests(unittest.TestCase):
    def test_constrained_quadratic_reaches_known_optimum(self):
        import numpy as np
        optimizer = GCMMA([.3, .3], options=GCMMAOptions(move=.1))
        for _ in range(30):
            x = optimizer.x.ravel()
            optimizer.propose(np.sum((x-[.8, .6])**2), 2*(x-[.8, .6]), [sum(x)-1], [[1, 1]])
            while True:
                trial = optimizer.trial.ravel()
                if optimizer.observe(np.sum((trial-[.8, .6])**2), [sum(trial)-1]):
                    break
        np.testing.assert_allclose(optimizer.x.ravel(), [.6, .4], atol=1e-5)
        self.assertLessEqual(float(optimizer.x.sum()), 1.+1e-8)

    def test_nonconservative_trial_is_rejected_and_regularization_increases(self):
        optimizer = GCMMA([.2], options=GCMMAOptions(move=.1))
        optimizer.propose(.36, [-1.2], [-.6], [[1.]])
        old_raa = float(optimizer.raa0.item())
        self.assertFalse(optimizer.observe(optimizer.f_approx+1., [-.5]))
        self.assertEqual(optimizer.inner, 1)
        self.assertGreater(float(optimizer.raa0.item()), old_raa)
        self.assertAlmostEqual(float(optimizer.x.item()), .2)
        with self.assertRaisesRegex(RuntimeError, 'pending'):
            optimizer.propose(.36, [-1.2], [-.6], [[1.]])

    def test_inner_limit_does_not_accept_a_bad_step(self):
        optimizer = GCMMA([.2], options=GCMMAOptions(max_inner=1))
        optimizer.propose(.36, [-1.2], [-.6], [[1.]])
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            optimizer.observe(optimizer.f_approx+1., [-.5])
        self.assertAlmostEqual(float(optimizer.x.item()), .2)

    def test_finite_differences_include_bounds_and_never_repair_probes(self):
        import numpy as np
        x = np.array([0., .5, 1.])
        seen = []
        def objective(v):
            return float(np.dot(v, v)+np.dot([1., 2., 3.], v))
        def batch(requests):
            values = []
            for key, v in requests:
                v = np.asarray(v)
                self.assertTrue(np.all(v >= 0) and np.all(v <= 1))
                self.assertEqual(np.count_nonzero(v != x), 1)
                seen.append(v)
                values.append(objective(v))
            return values
        for scheme, count, tolerance in (('forward', 3, 1.01e-4), ('central', 6, 1e-10)):
            seen.clear()
            gradient = finite_difference_gradient(x, objective(x), batch, scheme=scheme)
            np.testing.assert_allclose(gradient, 2*x+[1., 2., 3.], atol=tolerance, rtol=0)
            self.assertEqual(len(seen), count)


class ToyCFD:
    """Analytic stand-in for CFD; real P003 material mapping and constraints."""
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, x, config, case_root):
        with self.lock:
            self.calls.append((config.to_dict(), x))
        geometry = decode(x, config)
        area = geometry['cell_areas_m2']
        # Deliberately depends on the changing material interpolation.
        power = .1+.2*sum(a*d for a, d in zip(area, geometry['diffusivity_factor']))/sum(area)
        std = 100.
        result = scalarize(power, std, geometry['volume_fraction'], config)
        result.update(status='ok', problem='P003', config=config.to_dict(),
                      config_sha256=config.fingerprint, metadata=dict(valid_cfd=True),
                      metrics=dict(power_domain_W=power, current_std_A_m2=std))
        return result


@unittest.skipUnless(AVAILABLE, 'optional GCMMA dependencies unavailable')
class GCMMAContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = TopologyConfig(design_nx=2, design_ny=2, filter_radius_m=.02)
        self.options = ContinuationOptions(iterations_per_stage=2, move=.002)

    def test_stage_transitions_recompute_physics_and_freeze_initial_scales(self):
        toy = ToyCFD()
        output = self.root/'run'
        result = run_continuation(output, config=self.config, options=self.options, evaluate=toy)
        self.assertEqual(result['status'], 'completed_schedule')
        self.assertEqual([s['stage'] for s in result['completed_stages']], [1, 2, 3])
        self.assertEqual(result['attempts'], len(toy.calls))
        self.assertEqual(len(toy.calls), 1+3+3*2*(9+1))
        scales = {c['power_scale_w'] for c, x in toy.calls[1:]}
        self.assertEqual(len(scales), 1)
        self.assertNotEqual(next(iter(scales)), 1.)
        previous = None
        hashes = set()
        for stage in (1, 2, 3):
            folder = output/f'stage-{stage}'
            experiment = json.loads((folder/'experiment.json').read_text())
            hashes.add(experiment['config_sha256'])
            self.assertEqual(experiment['config']['stage'], stage)
            baseline = json.loads((folder/'evaluations/baseline.json').read_text())
            if previous is not None:
                self.assertEqual(baseline['x'], previous['x'])
                self.assertNotEqual(baseline['objective'], previous['objective'])
            previous = json.loads((folder/'current.json').read_text())
            self.assertTrue(previous['feasible'])
            self.assertTrue((folder/'best-feasible.json').is_file())

    def test_resume_in_middle_of_gradient_reuses_every_completed_solve(self):
        output = self.root/'resumed'
        first = ToyCFD()
        def interrupt(attempt, phase, iteration, record):
            if attempt == 7:
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            run_continuation(output, config=self.config, options=self.options,
                             evaluate=first, progress=interrupt)
        self.assertEqual(len(first.calls), 7)
        resumed = ToyCFD()
        run_continuation(output, config=self.config, options=self.options, evaluate=resumed, resume=True)
        whole = ToyCFD()
        run_continuation(self.root/'whole', config=self.config, options=self.options, evaluate=whole)
        self.assertEqual(first.calls+resumed.calls, whole.calls)
        self.assertEqual(json.loads((output/'summary.json').read_text())['attempts'], len(whole.calls))
        again = ToyCFD()
        run_continuation(output, config=self.config, options=self.options, evaluate=again, resume=True)
        self.assertEqual(again.calls, [])

    def test_budget_exhaustion_and_explicit_extension(self):
        output = self.root/'budget'
        toy = ToyCFD()
        result = run_continuation(output, config=self.config, options=self.options,
                                  evaluate=toy, max_evaluations=7)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(len(toy.calls), 7)
        result = run_continuation(output, config=self.config, options=self.options,
                                  evaluate=toy, max_evaluations=200, resume=True)
        self.assertEqual(result['status'], 'completed_schedule')
        self.assertEqual(result['attempts'], len(toy.calls))
        saved = json.loads((output/'gcmma-run.json').read_text())
        self.assertEqual(saved['budget_extensions'], [dict(previous=7, current=200)])

    def test_parallel_probes_honor_budget_and_replay(self):
        output = self.root/'parallel'
        toy = ToyCFD()
        result = run_continuation(output, config=self.config, options=self.options,
                                  evaluate=toy, max_evaluations=9, workers=3)
        self.assertEqual(result['attempts'], 9)
        self.assertEqual(len(toy.calls), 9)
        result = run_continuation(output, config=self.config, options=self.options,
                                  evaluate=toy, max_evaluations=200, workers=2, resume=True)
        self.assertEqual(result['status'], 'completed_schedule')
        self.assertEqual(len(toy.calls), 64)

    def test_unfinished_probe_needs_explicit_retry_and_counts_both_attempts(self):
        output = self.root/'retry'
        calls = []
        toy = ToyCFD()
        def interrupted(x, config, folder):
            calls.append(x)
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            run_continuation(output, config=self.config, options=self.options, evaluate=interrupted)
        with self.assertRaisesRegex(ResumeError, 'retry-interrupted'):
            run_continuation(output, config=self.config, options=self.options, evaluate=toy, resume=True)
        self.assertEqual(toy.calls, [])
        result = run_continuation(output, config=self.config, options=self.options,
                                  evaluate=toy, resume=True, retry_interrupted=True)
        self.assertEqual(result['attempts'], 1+len(toy.calls))
        self.assertEqual(len(list((output/'attempts').glob('*.json'))), result['attempts'])

    def test_failed_cfd_is_not_differentiated_or_converted_to_a_penalty(self):
        output = self.root/'failed'
        toy = ToyCFD()
        def failure(x, config, folder):
            result = toy(x, config, folder)
            if len(toy.calls) == 4:
                result.update(status='penalized', objective=1000000.)
            return result
        with self.assertRaisesRegex(RuntimeError, 'successful CFD'):
            run_continuation(output, config=self.config, options=self.options, evaluate=failure)
        self.assertEqual(len(toy.calls), 4)
        self.assertEqual(json.loads((output/'summary.json').read_text())['status'], 'failed')
        with self.assertRaisesRegex(ResumeError, 'Saved CFD failed'):
            run_continuation(output, config=self.config, options=self.options, evaluate=toy, resume=True)
        self.assertEqual(len(toy.calls), 4)
        result = run_continuation(output, config=self.config, options=self.options,
                                  evaluate=toy, resume=True, retry_failed=True)
        self.assertEqual(result['status'], 'completed_schedule')
        self.assertEqual(result['attempts'], 65)  # 64 successes plus the original failure.
        saved_failure = json.loads((output/'failed-attempts/00000004.json').read_text())
        self.assertFalse(saved_failure['valid'])

    def test_probe_case_cleanup_preserves_results_trials_and_replay(self):
        output = self.root/'cleanup'
        toy = ToyCFD()
        def with_case(x, config, folder):
            result = toy(x, config, folder)
            case = folder/f'P003-{len(toy.calls)}'
            case.mkdir(parents=True)
            (case/'field').write_text('CFD data')
            result['case_directory'] = str(case)
            return result
        run_continuation(output, config=self.config, options=self.options,
                         evaluate=with_case, discard_probe_cases=True)
        records = [json.loads(p.read_text()) for p in output.glob('*/evaluations/*.json')]
        for record in records:
            self.assertEqual(Path(record['result']['case_directory']).is_dir(),
                             record['phase'] != 'gradient')
        before = len(toy.calls)
        run_continuation(output, config=self.config, options=self.options,
                         evaluate=with_case, discard_probe_cases=True, resume=True)
        self.assertEqual(len(toy.calls), before)
        self.assertTrue((output/'stage-3/best-feasible.json').exists())

    def test_resume_refuses_changed_schedule_model_or_dependencies(self):
        output = self.root/'identity'
        toy = ToyCFD()
        run_continuation(output, config=self.config, options=self.options, evaluate=toy, max_evaluations=2)
        before = len(toy.calls)
        for config, options in ((replace(self.config, projection_beta=30), self.options),
                                (self.config, replace(self.options, fd_step=.002)),
                                (self.config, replace(self.options, stages=(1, 3)))):
            with self.assertRaises(ResumeError):
                run_continuation(output, config=config, options=options, evaluate=toy, resume=True)
        self.assertEqual(len(toy.calls), before)
        path = output/'gcmma-run.json'
        manifest = json.loads(path.read_text())
        manifest['versions']['mmapy'] = '0.0.0'
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ResumeError, 'versions'):
            run_continuation(output, config=self.config, options=self.options, evaluate=toy, resume=True)

    def test_directional_gradient_check_and_actual_geometry_output(self):
        toy = ToyCFD()
        output = self.root/'checked'
        options = replace(self.options, iterations_per_stage=1, fd_scheme='central', check_gradients=True)
        run_continuation(output, config=self.config, options=options, evaluate=toy)
        for stage in options.stages:
            check = json.loads((output/f'stage-{stage}/gradient-check.json').read_text())
            self.assertTrue(check['passed'])
        if importlib.util.find_spec('matplotlib'):
            from flowfield.benchmarking import plot_geometry
            ax = plot_geometry(output/'stage-3')
            self.assertEqual(ax.images[0].get_array().shape, (3, 3))

    def test_default_plan_counts_all_2145_coordinates_and_inner_trials(self):
        report = plan(TopologyConfig(), ContinuationOptions(check_gradients=True))
        self.assertEqual(report['design_variables'], 2145)
        self.assertEqual(report['probes_per_gradient'], 2145)
        self.assertEqual(report['outer_iterations'], 180)
        self.assertEqual(report['nominal_evaluations'], 386320)
        self.assertEqual(report['maximum_evaluations'], 388840)

    def test_cli_dry_run_does_not_create_files_or_require_openfoam(self):
        from flowfield.problems.level4_high_dimensional_topology.p003.cli import main
        output = self.root/'dry'
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(main(['gcmma', '--output', str(output), '--dry-run', '--check-gradients']), 0)
        self.assertEqual(json.loads(stream.getvalue())['nominal_evaluations'], 386320)
        self.assertFalse(output.exists())

    def test_cli_resume_extends_total_budget_and_keeps_retention_policy(self):
        from flowfield.problems.level4_high_dimensional_topology.p003.cli import main
        output = self.root/'cli-budget'
        run_continuation(output, config=self.config, options=self.options, evaluate=ToyCFD(),
                         max_evaluations=2, discard_probe_cases=True)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(main(['gcmma', '--output', str(output), '--resume',
                                   '--extend-budget', '20', '--dry-run']), 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report['max_evaluations'], 22)
        self.assertTrue(report['discard_probe_cases'])


if __name__ == '__main__':
    unittest.main()
