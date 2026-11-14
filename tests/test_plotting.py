"""Plot statistics, saved-run integrity, and headless PNG/SVG/PDF rendering."""

import importlib.util
import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from flowfield.benchmarking import compare_runs, plot_benchmark, plot_convergence, plot_ert, plot_ecdf, plot_fixed_budget
from flowfield.benchmarking.plotting import main


@unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'optional matplotlib unavailable')
class PlottingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / 'runs'
        self.a = self.make_run('A', 1, [(10, True, True), (-100, False, False),
                                      (-50, True, False), (4, True, True)])
        self.b = self.make_run('A', 2, [(12, True, True), (9, True, True),
                                      (8, True, True), (7, True, True)])
        self.c = self.make_run('B', 1, [(-100, False, False)] * 4)
        (self.c / 'summary.json').write_text(json.dumps(dict(
            status='failed', stop_reason='evaluation_budget', attempts=4)))

    def tearDown(self):
        self.temp.cleanup()

    def make_run(self, algorithm, seed, values):
        path = self.runs / algorithm / str(seed)
        path.mkdir(parents=True)
        experiment = dict(problem_name='PlotFixture', dimension=1, budget=4,
                          algorithm=algorithm, seed=seed, failure_penalty=-100)
        (path / 'experiment.json').write_text(json.dumps(experiment))
        (path / 'summary.json').write_text(json.dumps(dict(status='completed_with_failures', attempts=len(values))))
        events, rows = [], []
        for i, (value, valid, feasible) in enumerate(values, start=1):
            common = dict(evaluation=i, x=[i / 4], phase='candidate', generation=None)
            events.append(dict(common, event='started'))
            rows.append(dict(common, objective=value, valid=valid, feasible=feasible,
                             result={'objective': value}))
        for name, data in (('events', events), ('evaluations', rows)):
            (path / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in data))
        return path

    def test_failure_and_infeasibility_do_not_improve_convergence(self):
        ax = plot_convergence(self.runs)
        self.assertEqual(list(ax.lines[0].get_ydata()), [10, 10, 10, 4])
        self.assertEqual(list(ax.lines[1].get_ydata()), [12, 9, 8, 7])
        self.assertTrue(all(math.isnan(y) for y in ax.lines[2].get_ydata()))
        self.assertEqual(list(ax.lines[0].get_xdata()), [1, 2, 3, 4])

    def test_ert_and_ecdf_keep_unsuccessful_runs_in_the_statistics(self):
        ert = plot_ert(self.runs, targets=[5, 10])
        self.assertEqual(list(ert.lines[0].get_ydata()), [8., 1.5])
        self.assertTrue(all(math.isnan(y) for y in ert.lines[1].get_ydata()))
        self.assertTrue(any('ERT = ∞' in label for label in ert.get_legend_handles_labels()[1]))
        with_missing_target = plot_ert(self.runs, targets=[1, 5, 10])
        self.assertLess(with_missing_target.get_xlim()[0], 1)
        self.assertGreater(with_missing_target.get_xlim()[1], 10)
        ecdf = plot_ecdf(self.runs, targets=[5, 10])
        self.assertEqual(list(ecdf.lines[0].get_xdata()), [0, 1, 2, 4])
        self.assertEqual(list(ecdf.lines[0].get_ydata()), [0, .25, .5, .75])
        self.assertEqual(list(ecdf.lines[1].get_ydata()), [0, 0])

    def test_fixed_budget_shows_feasibility_counts_and_early_termination(self):
        ax = plot_fixed_budget(self.runs)
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ['A\n2/2 feasible', 'B\n0/1 feasible'])
        self.assertIn('No feasible result', [text.get_text() for text in ax.texts])
        short = self.make_run('C', 1, [(6, True, True)])
        ax = plot_convergence(short)
        self.assertEqual(list(ax.lines[0].get_xdata()), [1])
        self.assertEqual(compare_runs([short])['algorithms']['C']['runs'][0]['best_feasible'], 6)

    def test_recovery_is_counted_once(self):
        path = self.a / 'evaluations.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        recovered = rows[1].copy()
        rows[1]['objective'] = None
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (self.a / 'recoveries.jsonl').write_text(json.dumps(recovered) + '\n')
        ax = plot_convergence(self.a)
        self.assertEqual(list(ax.lines[0].get_ydata()), [10, 10, 10, 4])

    def test_rendering_does_not_change_inputs_or_call_cfd(self):
        before = {p: p.read_bytes() for p in self.runs.rglob('*') if p.is_file()}
        with patch('flowfield.runtime.Runtime.resolve', side_effect=AssertionError('No CFD')):
            for suffix, signature in (('png', b'\x89PNG'), ('pdf', b'%PDF'), ('svg', b'<?xml')):
                output = self.root / 'figures' / ('comparison.' + suffix)
                figure = plot_benchmark(self.runs, targets=[5, 10], output=output)
                self.assertEqual(len(figure.axes), 4)
                self.assertTrue(output.read_bytes().startswith(signature))
            figure = plot_benchmark(self.a)
            self.assertEqual(len(figure.axes), 2)
            self.assertIs(plot_convergence(self.a, ax=figure.axes[0]), figure.axes[0])
        self.assertEqual(before, {p: p.read_bytes() for p in self.runs.rglob('*') if p.is_file()})

    def test_invalid_comparisons_are_rejected(self):
        for targets in ([], [5, 5], [math.nan], [True]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                plot_ecdf(self.a, targets=targets)
        with self.assertRaisesRegex(ValueError, 'Duplicate seeds'):
            plot_convergence([self.a, self.a])
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            plot_convergence(self.root / 'runs' / 'A' / '1' / 'missing')
        empty = self.root / 'empty'
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, 'No experiment'):
            plot_convergence(empty)

    def test_cli_and_mixed_objectives(self):
        output = self.root / 'cli.png'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.a), '--targets', '5', '10', '--output', str(output)]), 0)
        self.assertTrue(output.is_file())
        path = self.b / 'experiment.json'
        data = json.loads(path.read_text())
        data['problem_name'] = 'DifferentPhysicalObjective'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "problem_name: 'PlotFixture' != 'DifferentPhysicalObjective'"):
            plot_benchmark(self.runs)

    def test_mismatch_error_identifies_budget_and_both_runs(self):
        path = self.b / 'experiment.json'
        data = json.loads(path.read_text())
        data['budget'] = 20
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError) as caught:
            plot_fixed_budget([self.a, self.b])
        message = str(caught.exception)
        self.assertIn('budget: 4 != 20', message)
        self.assertIn(str(self.a), message)
        self.assertIn(str(self.b), message)

    def test_convergence_allows_unequal_budgets_without_extrapolating(self):
        longer = self.make_run('LongRun', 1, [(12-i, True, True) for i in range(8)])
        path = longer / 'experiment.json'
        data = json.loads(path.read_text())
        data['budget'] = 8
        path.write_text(json.dumps(data))
        ax = plot_convergence([self.a, longer])
        self.assertEqual(list(ax.lines[0].get_xdata()), [1, 2, 3, 4])
        self.assertEqual(list(ax.lines[1].get_xdata()), list(range(1, 9)))
        self.assertGreater(ax.get_xlim()[1], 8)
        self.assertIn('budget 4', ax.lines[0].get_label())
        self.assertIn('budget 8', ax.lines[1].get_label())
        with self.assertRaisesRegex(ValueError, 'budget: 4 != 8'):
            plot_benchmark([self.a, longer])
        with self.assertRaisesRegex(ValueError, 'without targets'):
            compare_runs([self.a, longer], targets=[5], allow_unequal_budgets=True)
        data['problem_name'] = 'DifferentPhysicalObjective'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'problem_name'):
            plot_convergence([self.a, longer])

    def test_convergence_labels_failure_policies_and_keeps_model_identity(self):
        base = 'P003_depth-averaged-cathode-v1_98a7221a4e37_failure_raise'
        for folder, penalty, suffix in ((self.a, None, 'raise'), (self.b, 1000000, '1000000')):
            path = folder / 'experiment.json'
            data = json.loads(path.read_text())
            data.update(problem_name=base+'_optimizer_failure_'+suffix, failure_penalty=penalty)
            path.write_text(json.dumps(data))
        before = {p: p.read_bytes() for d in (self.a, self.b) for p in d.iterdir()}
        ax = plot_convergence([self.a, self.b])
        self.assertIn('failure: stop', ax.lines[0].get_label())
        self.assertIn('failure: penalty 1000000', ax.lines[1].get_label())
        self.assertEqual(list(ax.lines[0].get_ydata()), [10, 10, 10, 4])
        self.assertEqual(list(ax.lines[1].get_ydata()), [12, 9, 8, 7])
        output = self.root/'policies.png'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.a), str(self.b), '--kind', 'convergence',
                                   '--output', str(output)]), 0)
        self.assertTrue(output.exists())
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        with self.assertRaisesRegex(ValueError, 'problem_name'):
            plot_fixed_budget([self.a, self.b])
        with self.assertRaisesRegex(ValueError, 'without targets'):
            compare_runs([self.a, self.b], targets=[5], allow_different_failure_policies=True)
        path = self.b/'experiment.json'
        data = json.loads(path.read_text())
        for changed in (base.replace('98a7221a4e37', 'differentconfig')+'_optimizer_failure_1000000',
                        base.replace('failure_raise', 'failure_42')+'_optimizer_failure_1000000',
                        base+'_optimizer_failure_42'):
            with self.subTest(problem_name=changed):
                path.write_text(json.dumps(data | {'problem_name': changed}))
                with self.assertRaisesRegex(ValueError, 'problem_name'):
                    plot_convergence([self.a, self.b])

    def test_same_seed_different_policies_are_separate_traces(self):
        path = self.b/'experiment.json'
        data = json.loads(path.read_text())
        data.update(seed=1, failure_penalty=1000000)
        path.write_text(json.dumps(data))
        ax = plot_convergence([self.a, self.b])
        self.assertEqual(len(ax.lines), 2)
        self.assertNotEqual(ax.lines[0].get_label(), ax.lines[1].get_label())


class OptionalPlottingTests(unittest.TestCase):
    def test_public_import_does_not_load_matplotlib(self):
        subprocess.run([sys.executable, '-c',
            'import sys; from flowfield.benchmarking import plot_benchmark; '
            'assert "matplotlib" not in sys.modules'], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
