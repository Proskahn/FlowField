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

from flowfield.benchmarking import compare_runs, plot_benchmark, plot_convergence, plot_mean_convergence, plot_evaluations, plot_objectives, plot_pareto, plot_ert, plot_ecdf, plot_fixed_budget
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

    def test_mean_convergence_averages_incumbents_and_uses_sample_deviation(self):
        ax = plot_mean_convergence([self.a, self.b])
        self.assertEqual(list(ax.lines[0].get_ydata()), [11, 9.5, 9, 5.5])
        self.assertIn('2 seeds', ax.lines[0].get_label())
        vertices = ax.collections[0].get_paths()[0].vertices
        first_band = [y for x, y in vertices if x == 1]
        self.assertAlmostEqual(min(first_band), 11-math.sqrt(2))
        self.assertAlmostEqual(max(first_band), 11+math.sqrt(2))

    def test_mean_convergence_keeps_all_seeds_and_stops_at_shortest_run(self):
        first = self.make_run('Short', 1, [(10, True, True)] * 4)
        second = self.make_run('Short', 2, [(12, True, True), (8, True, True)])
        ax = plot_mean_convergence([first, second])
        self.assertEqual(list(ax.lines[0].get_xdata()), [1, 2])
        self.assertEqual(list(ax.lines[0].get_ydata()), [11, 9])

    def test_mean_convergence_does_not_drop_seeds_without_feasible_results(self):
        first = self.make_run('Gaps', 1, [(-100, False, False), (8, True, True),
                                         (6, True, True), (4, True, True)])
        second = self.make_run('Gaps', 2, [(-100, False, False), (-50, True, False),
                                          (10, True, True), (9, True, True)])
        ax = plot_mean_convergence([first, second])
        values = list(ax.lines[0].get_ydata())
        self.assertTrue(all(math.isnan(value) for value in values[:2]))
        self.assertEqual(values[2:], [8, 6.5])
        never = self.make_run('B', 2, [(10, True, True)] * 4)
        ax = plot_mean_convergence([self.c, never])
        self.assertTrue(all(math.isnan(value) for value in ax.lines[0].get_ydata()))
        self.assertIn('no shared feasible result', ax.lines[0].get_label())

    def test_mean_convergence_cli_five_seeds_is_read_only(self):
        for seed in (3, 4, 5):
            self.make_run('A', seed, [(8+2*seed, True, True)] * 4)
        before = {p: p.read_bytes() for p in self.runs.rglob('*') if p.is_file()}
        output = self.root/'mean.png'
        with patch('flowfield.runtime.Runtime.resolve', side_effect=AssertionError('No CFD')):
            ax = plot_mean_convergence(self.a.parent)
            self.assertEqual(list(ax.lines[0].get_ydata()), [14, 13.4, 13.2, 11.8])
            self.assertIn('5 seeds', ax.lines[0].get_label())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(self.a.parent), '--kind', 'mean-convergence',
                                       '--output', str(output)]), 0)
        self.assertTrue(output.read_bytes().startswith(b'\x89PNG'))
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_mean_convergence_requires_repeats_completed_runs_and_matching_objective(self):
        with self.assertRaisesRegex(ValueError, 'at least two seeds'):
            plot_mean_convergence(self.a)
        with self.assertRaisesRegex(ValueError, 'Duplicate seeds'):
            plot_mean_convergence([self.a, self.a])
        path = self.b/'summary.json'
        data = json.loads(path.read_text())
        path.write_text(json.dumps(data | {'status': 'running'}))
        with self.assertRaisesRegex(ValueError, 'Incomplete run'):
            plot_mean_convergence([self.a, self.b])
        path.write_text(json.dumps(data))
        path = self.b/'experiment.json'
        data = json.loads(path.read_text())
        path.write_text(json.dumps(data | {'problem_name': 'DifferentPhysicalObjective'}))
        with self.assertRaisesRegex(ValueError, 'problem_name'):
            plot_mean_convergence([self.a, self.b])

    def test_mean_convergence_five_seeds_with_100_and_200_evaluation_budgets(self):
        folders = []
        for algorithm, budget in (('TuRBO', 100), ('BO', 200)):
            for seed in range(42, 47):
                folder = self.make_run(algorithm, seed,
                                       [(300+seed-i, True, True) for i in range(budget)])
                path = folder/'experiment.json'
                path.write_text(json.dumps(json.loads(path.read_text()) | {'budget': budget}))
                folders.append(folder)
        before = {p: p.read_bytes() for folder in folders for p in folder.iterdir()}
        with patch('flowfield.runtime.Runtime.resolve', side_effect=AssertionError('No CFD')):
            ax = plot_mean_convergence(folders)
            self.assertEqual(list(ax.lines[0].get_xdata()), list(range(1, 101)))
            self.assertEqual(list(ax.lines[1].get_xdata()), list(range(1, 201)))
            self.assertEqual(list(ax.lines[0].get_ydata()), list(range(344, 244, -1)))
            self.assertIn('5 seeds', ax.lines[0].get_label())
            self.assertIn('budget 100', ax.lines[0].get_label())
            self.assertIn('budget 200', ax.lines[1].get_label())
            self.assertGreater(ax.get_xlim()[1], 200)
            self.assertEqual(max(ax.collections[0].get_paths()[0].vertices[:, 0]), 100)
            output = self.root/'unequal-mean.png'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(folder) for folder in folders] +
                                      ['--kind', 'mean-convergence', '--output', str(output)]), 0)
        self.assertTrue(output.read_bytes().startswith(b'\x89PNG'))
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_mean_convergence_mixed_budgets_within_optimizer_keeps_all_seeds(self):
        longer = self.make_run('A', 3, [(12-i, True, True) for i in range(8)])
        path = longer/'experiment.json'
        path.write_text(json.dumps(json.loads(path.read_text()) | {'budget': 8}))
        ax = plot_mean_convergence([self.a, longer])
        self.assertEqual(list(ax.lines[0].get_xdata()), [1, 2, 3, 4])
        self.assertEqual(list(ax.lines[0].get_ydata()), [11, 10.5, 10, 6.5])
        self.assertIn('2 seeds', ax.lines[0].get_label())
        self.assertIn('budget 4, 8', ax.lines[0].get_label())

    def test_individual_evaluations_keep_variation_and_exclude_failed_cfd(self):
        ax = plot_evaluations(self.a)
        values = list(ax.lines[0].get_ydata())
        self.assertEqual((values[0], values[2], values[3]), (10, -50, 4))
        self.assertTrue(math.isnan(values[1]))
        self.assertEqual(list(ax.lines[0].get_xdata()), [1, 2, 3, 4])
        self.assertEqual(list(ax.collections[0].get_offsets()[:, 0]), [1, 4])
        self.assertEqual(list(ax.collections[1].get_offsets()[:, 0]), [3])
        self.assertIn('1 failed CFD (omitted)', ax.lines[0].get_label())
        output = self.root/'evaluations.png'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.runs), '--kind', 'evaluations', '--output', str(output)]), 0)
        self.assertTrue(output.exists())

    def set_objectives(self, folder, objectives):
        path = folder / 'evaluations.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row, values in zip(rows, objectives):
            row['result']['objectives'] = values
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_objectives_keep_raw_tradeoffs_and_scalar_on_independent_scales(self):
        self.set_objectives(self.a, [
            {'voltage_V': .6, 'pressure_sum_Pa': 300},
            {'voltage_V': -999, 'pressure_sum_Pa': -999, 'failed_only': -999},
            {'pressure_sum_Pa': 100, 'voltage_V': .8},
            {'voltage_V': .7, 'pressure_sum_Pa': 200},
        ])
        scalar, pressure, voltage = plot_objectives(self.a).axes
        self.assertEqual([ax.get_ylabel() for ax in (scalar, pressure, voltage)],
                         ['Scalar objective ↓', 'pressure_sum_Pa ↓', 'voltage_V ↓'])
        for ax, expected in ((scalar, [10, -50, 4]), (pressure, [300, 100, 200]),
                             (voltage, [.6, .8, .7])):
            values = list(ax.lines[0].get_ydata())
            self.assertEqual([values[i] for i in (0, 2, 3)], expected)
            self.assertTrue(math.isnan(values[1]))
            self.assertEqual(list(ax.lines[0].get_xdata()), [1, 2, 3, 4])
            self.assertEqual(list(ax.collections[0].get_offsets()[:, 0]), [1, 4])
            self.assertEqual(list(ax.collections[1].get_offsets()[:, 0]), [3])
            self.assertIn('1 failed CFD (omitted)', ax.lines[0].get_label())
        self.assertTrue(pressure.get_shared_x_axes().joined(pressure, voltage))
        self.assertFalse(pressure.get_shared_y_axes().joined(pressure, voltage))

    def test_objectives_support_scalar_only_named_single_objective_and_all_failures(self):
        for folder in (self.a, self.c):
            figure = plot_objectives(folder)
            self.assertEqual(len(figure.axes), 1)
            expected = plot_evaluations(folder).lines[0].get_ydata()
            actual = figure.axes[0].lines[0].get_ydata()
            for a, b in zip(actual, expected):
                self.assertTrue(a == b or (math.isnan(a) and math.isnan(b)))
        self.set_objectives(self.a, [{'temperature_std_K': value} for value in (5, 0, 2, 3)])
        self.assertEqual(len(plot_objectives(self.a).axes), 2)

    def test_objectives_missing_components_are_gaps_without_scalar_fallback(self):
        self.set_objectives(self.a, [
            {'power': -10}, {}, {'power': -20, 'uniformity': .5}, {'uniformity': .3}])
        _, power, uniformity = plot_objectives(self.a).axes
        self.assertEqual(list(power.lines[0].get_ydata())[::2], [-10, -20])
        self.assertTrue(math.isnan(power.lines[0].get_ydata()[3]))
        self.assertTrue(math.isnan(uniformity.lines[0].get_ydata()[0]))
        self.assertEqual(list(uniformity.lines[0].get_ydata())[2:], [.5, .3])
        self.assertIn('1 missing values (gaps)', power.lines[0].get_label())
        self.assertIn('1 missing values (gaps)', uniformity.lines[0].get_label())

    def test_objectives_overlay_different_budgets_and_policies_without_extrapolation(self):
        longer = self.make_run('LongRun', 2, [(12-i, True, True) for i in range(8)])
        path = longer / 'experiment.json'
        path.write_text(json.dumps(json.loads(path.read_text()) | {'budget': 8, 'failure_penalty': 1000}))
        self.set_objectives(longer, [{'physical': 20-i} for i in range(8)])
        self.set_objectives(self.a, [{'physical': i} for i in range(4)])
        for ax in plot_objectives([self.a, longer]).axes:
            self.assertEqual(list(ax.lines[0].get_xdata()), [1, 2, 3, 4])
            self.assertEqual(list(ax.lines[1].get_xdata()), list(range(1, 9)))
            self.assertIn('seed 1; budget 4', ax.lines[0].get_label())
            self.assertIn('seed 2; budget 8', ax.lines[1].get_label())
            self.assertIn('on error: penalty 1000', ax.lines[1].get_label())

    def test_objectives_use_recovered_values_once(self):
        self.set_objectives(self.a, [{'physical': 8}, {}, {'physical': 2}, {'physical': 4}])
        path = self.a / 'evaluations.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        recovered = rows[2].copy()
        rows[2]['objective'] = None
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (self.a / 'recoveries.jsonl').write_text(json.dumps(recovered) + '\n')
        raw = plot_objectives(self.a).axes[1].lines[0]
        self.assertEqual(list(raw.get_xdata()), [1, 2, 3, 4])
        self.assertEqual([raw.get_ydata()[i] for i in (0, 2, 3)], [8, 2, 4])

    def test_objectives_distinguish_seeds_consistently_in_every_panel(self):
        for folder in (self.a, self.b):
            self.set_objectives(folder, [{'objective': value} for value in (5, 0, 2, 3)])
        scalar, raw = plot_objectives(self.a.parent).axes
        self.assertNotEqual(scalar.lines[0].get_color(), scalar.lines[1].get_color())
        self.assertEqual([line.get_color() for line in scalar.lines],
                         [line.get_color() for line in raw.lines])
        self.assertEqual(scalar.lines[0].get_ydata()[0], 10)
        self.assertEqual(raw.lines[0].get_ydata()[0], 5)
        self.assertTrue(all(tick.is_integer() for tick in raw.get_xticks()))

    def test_objectives_validate_raw_values_and_comparisons(self):
        for values in ([], None, {'': 1}, {'a': True}, {'a': math.nan},
                       {'a': math.inf}, {'a': '1'}, {'a': None}):
            with self.subTest(values=values):
                self.set_objectives(self.a, [values] * 4)
                with self.assertRaisesRegex(ValueError, 'Raw objectives'):
                    plot_objectives(self.a)
        self.set_objectives(self.a, [{}] * 4)
        with self.assertRaisesRegex(ValueError, 'Duplicate seeds'):
            plot_objectives([self.a, self.a])
        path = self.b / 'experiment.json'
        path.write_text(json.dumps(json.loads(path.read_text()) | {'problem_name': 'Different'}))
        with self.assertRaisesRegex(ValueError, 'problem_name'):
            plot_objectives([self.a, self.b])
        path = self.a / 'summary.json'
        path.write_text(json.dumps(json.loads(path.read_text()) | {'status': 'running'}))
        with self.assertRaisesRegex(ValueError, 'Incomplete run'):
            plot_objectives(self.a)

    def test_objectives_cli_renders_formats_without_modifying_runs_or_calling_cfd(self):
        self.set_objectives(self.a, [{'physical': value} for value in (5, 0, 2, 3)])
        before = {p: p.read_bytes() for p in self.runs.rglob('*') if p.is_file()}
        with patch('flowfield.runtime.Runtime.resolve', side_effect=AssertionError('No CFD')):
            for suffix, signature in (('png', b'\x89PNG'), ('pdf', b'%PDF'), ('svg', b'<?xml')):
                output = self.root / 'figures' / ('objectives.' + suffix)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main([str(self.a), '--kind', 'objectives',
                                           '--output', str(output)]), 0)
                self.assertTrue(output.read_bytes().startswith(signature))
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def make_pareto(self, folder, points, names=('pressure_sum_Pa', 'voltage_V')):
        archive = dict(objective_names=list(names), objective_direction='minimize', solutions=[
            dict(evaluation=i, x=[i], valid=True, feasible=True,
                 result=dict(objective=-999, objectives=dict(zip(names, point))))
            for i, point in enumerate(points, 1)])
        path = folder / 'pareto.json'
        path.write_text(json.dumps(archive))
        return path

    def test_pareto_uses_saved_raw_front_sorted_by_selected_axis(self):
        archive = self.make_pareto(self.a, [(300, .6), (100, .8), (200, .7)])
        ax = plot_pareto(archive)
        self.assertEqual(list(ax.lines[0].get_xdata()), [100, 200, 300])
        self.assertEqual(list(ax.lines[0].get_ydata()), [.8, .7, .6])
        self.assertEqual((ax.get_xlabel(), ax.get_ylabel()), ('pressure_sum_Pa ↓', 'voltage_V ↓'))
        self.assertIn('3 designs', ax.lines[0].get_label())
        reversed_ax = plot_pareto(self.a, objectives=('voltage_V', 'pressure_sum_Pa'))
        self.assertEqual(list(reversed_ax.lines[0].get_xdata()), [.6, .7, .8])
        self.assertEqual(list(reversed_ax.lines[0].get_ydata()), [300, 200, 100])
        self.assertIs(plot_pareto(self.a, ax=ax), ax)

    def test_pareto_projection_keeps_full_dimensional_front_membership(self):
        self.make_pareto(self.a, [(0, 0, 2), (1, 1, 0)], names=('a', 'b', 'c'))
        with self.assertRaisesRegex(ValueError, 'Select two objective names'):
            plot_pareto(self.a)
        ax = plot_pareto(self.a, objectives=('a', 'b'))
        self.assertEqual(list(ax.lines[0].get_xdata()), [0, 1])
        self.assertEqual(list(ax.lines[0].get_ydata()), [0, 1])
        self.assertEqual(ax.lines[0].get_linestyle(), 'None')
        self.assertIn('projection from 3 objectives', ax.get_title())
        for names in (('a', 'a'), ('a', 'unknown'), ('a',), 'ab', 42):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, 'two distinct'):
                plot_pareto(self.a, objectives=names)

    def test_pareto_overlays_archives_and_labels_budgets_and_empty_fronts(self):
        self.make_pareto(self.a, [(100, .8)])
        self.make_pareto(self.b, [])
        path = self.b / 'experiment.json'
        original = json.loads(path.read_text())
        path.write_text(json.dumps(original | {'budget': 8}))
        ax = plot_pareto(self.a.parent)
        self.assertEqual(len(ax.lines), 2)
        self.assertNotEqual(ax.lines[0].get_color(), ax.lines[1].get_color())
        self.assertIn('seed 1; budget 4', ax.lines[0].get_label())
        self.assertIn('seed 2; budget 8; no feasible Pareto points', ax.lines[1].get_label())
        empty = plot_pareto(self.b)
        self.assertIn('No feasible Pareto points', empty.texts[0].get_text())
        with self.assertRaisesRegex(ValueError, 'Duplicate seeds'):
            plot_pareto([self.a, self.a / 'pareto.json'])
        for change in ({'problem_name': 'different'}, {'dimension': 2}, {'failure_penalty': 1000}):
            path.write_text(json.dumps(original | change))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'same objective/configuration'):
                plot_pareto([self.a, self.b])
        path.write_text(json.dumps(original))
        self.make_pareto(self.b, [], names=('a', 'b'))
        with self.assertRaisesRegex(ValueError, 'different objective names'):
            plot_pareto([self.a, self.b])

    def test_pareto_rejects_missing_or_malformed_archives(self):
        with self.assertRaisesRegex(ValueError, 'No pareto.json'):
            plot_pareto(self.a)
        with self.assertRaisesRegex(ValueError, 'at least one saved'):
            plot_pareto([])
        path = self.make_pareto(self.a, [(100, .8)])
        original = json.loads(path.read_text())
        for change in ({'objective_direction': 'maximize'}, {'objective_names': []},
                       {'objective_names': ['a']}, {'objective_names': ['a', 'a']},
                       {'objective_names': ['', 'b']}, {'solutions': {}},
                       {'solutions': [dict(valid=False, feasible=False)]}):
            path.write_text(json.dumps(original | change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                plot_pareto(self.a)
        for values in ({'pressure_sum_Pa': 100}, {'pressure_sum_Pa': True, 'voltage_V': .8},
                       {'pressure_sum_Pa': 100, 'voltage_V': math.inf}):
            archive = json.loads(json.dumps(original))
            archive['solutions'][0]['result']['objectives'] = values
            path.write_text(json.dumps(archive))
            with self.subTest(values=values), self.assertRaises(ValueError):
                plot_pareto(self.a)

    def test_pareto_cli_renders_interrupted_snapshots_read_only_without_cfd(self):
        archive = self.make_pareto(self.a, [(100, .8), (200, .6)])
        path = self.a / 'summary.json'
        path.write_text(json.dumps(dict(status='interrupted', attempts=4)))
        before = {p: p.read_bytes() for p in self.runs.rglob('*') if p.is_file()}
        with patch('flowfield.runtime.Runtime.resolve', side_effect=AssertionError('No CFD')), \
                patch('flowfield.optimization.state.load_history', side_effect=AssertionError('Use saved archive')):
            for suffix, signature in (('png', b'\x89PNG'), ('pdf', b'%PDF'), ('svg', b'<?xml')):
                output = self.root / 'figures' / ('pareto.' + suffix)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main([str(archive), '--kind', 'pareto',
                                           '--objectives', 'voltage_V', 'pressure_sum_Pa',
                                           '--output', str(output)]), 0)
                self.assertTrue(output.read_bytes().startswith(signature))
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main([str(self.a), '--kind', 'convergence', '--objectives', 'a', 'b',
                  '--output', str(self.root / 'invalid.png')])
        self.assertEqual(caught.exception.code, 2)

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
        self.assertIn('on error: stop', ax.lines[0].get_label())
        self.assertIn('on error: penalty 1000000', ax.lines[1].get_label())
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
            'import sys; from flowfield.benchmarking import plot_benchmark, plot_mean_convergence, plot_objectives, plot_pareto; '
            'assert "matplotlib" not in sys.modules'], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
