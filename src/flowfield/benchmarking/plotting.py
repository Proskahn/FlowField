"""Plot saved optimization runs without importing an optimizer or running CFD.

Matplotlib is optional and loaded only when creating a figure. Functions accept
one run directory, a parent directory containing runs, or a list of either.
Runs must share objective/configuration and dimension. Evaluation and convergence
plots permit different budgets and optimizer failure policies, labeled in the legend.
Fixed-budget and target-based comparisons require matching budgets and policies.
Geometry plots read one P003 best-feasible record, including standalone JSON.
"""

import argparse
from bisect import bisect_right
from collections import Counter
import json
import math
from pathlib import Path
from statistics import fmean, stdev
import textwrap

from .analysis import compare_runs


def _run_directories(directories):
    if isinstance(directories, (str, Path)):
        directories = [directories]
    runs = []
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f'Run directory does not exist: {directory}')
        manifests = ([directory / 'experiment.json'] if (directory / 'experiment.json').is_file()
                     else sorted(directory.rglob('experiment.json')))
        if not manifests:
            raise ValueError(f'No experiment.json files found in {directory}')
        runs.extend(path.parent for path in manifests)
    return runs


def _comparison(directories, targets=(), *, allow_unequal_budgets=False,
                allow_different_failure_policies=False):
    return compare_runs(_run_directories(directories), targets=targets,
                        allow_unequal_budgets=allow_unequal_budgets,
                        allow_different_failure_policies=allow_different_failure_policies)


def _figure(figsize):
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as exc:
        raise RuntimeError('Install plotting with: python -m pip install -e ".[plot]"') from exc
    figure = Figure(figsize=figsize, layout='constrained')
    FigureCanvasAgg(figure)  # Works headlessly; does not change the caller's backend.
    return figure


def _save(figure, output):
    if output is not None:
        path = Path(output)
        if path.suffix.lower() not in ('.png', '.svg', '.pdf'):
            raise ValueError('Plot output must have a .png, .svg, or .pdf extension')
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)


def _style(ax, title, xlabel, ylabel):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.2)


def _convergence(report, ax):
    for index, (algorithm, group) in enumerate(report['algorithms'].items()):
        runs = group['runs']
        label = f'{algorithm} ({len(runs)} {"run" if len(runs) == 1 else "runs"})'
        if len(report['budgets']) > 1:
            label += '; budget ' + ', '.join(str(b) for b in sorted({r['budget'] for r in runs}))
        for number, run in enumerate(runs):
            values = [float('nan') if value is None else value for value in run['best_so_far']]
            ax.step(range(1, run['attempts'] + 1), values, where='post', color=f'C{index % 10}',
                    linestyle=('-', '--', '-.', ':')[index % 4],
                    alpha=.8 if len(runs) == 1 else .55, marker='.' if run['attempts'] == 1 else None,
                    label=label if number == 0 else None)
    _style(ax, 'Convergence · individual runs', 'Evaluation attempts', 'Best feasible objective ↓')
    ax.set_xlim(.5, report['budget'] + .5)
    ax.legend(fontsize='small')


def _mean_convergence(report, ax):
    for index, (algorithm, group) in enumerate(report['algorithms'].items()):
        runs = group['runs']
        if len(runs) < 2:
            raise ValueError(f'Mean convergence requires at least two seeds for {algorithm}')
        # zip stops at the shortest run: the seed cohort never shrinks, and no
        # values are extrapolated after an optimizer has stopped evaluating.
        means, deviations = [], []
        for values in zip(*(run['best_so_far'] for run in runs)):
            if any(value is None for value in values):
                # A successful-seeds-only mean would hide missing incumbents.
                mean = deviation = float('nan')
            else:
                mean, deviation = fmean(values), stdev(values)
            means.append(mean)
            deviations.append(deviation)
        xs = range(1, len(means) + 1)
        color = f'C{index % 10}'
        label = f'{algorithm} ({len(runs)} seeds)'
        if len(report['budgets']) > 1:
            label += '; budget ' + ', '.join(str(b) for b in sorted({r['budget'] for r in runs}))
        if not any(math.isfinite(value) for value in means):
            label += '; no shared feasible result'
        ax.step(xs, means, where='post', color=color,
                linestyle=('-', '--', '-.', ':')[index % 4],
                marker='.' if len(means) == 1 else None, label=label)
        ax.fill_between(xs, [m-s for m, s in zip(means, deviations)],
                        [m+s for m, s in zip(means, deviations)],
                        step='post', color=color, alpha=.18, linewidth=0)
    _style(ax, 'Mean convergence · shaded ±1 sample standard deviation',
           'Evaluation attempts', 'Mean best feasible objective ↓')
    ax.set_xlim(.5, report['budget'] + .5)
    ax.legend(fontsize='small')


def _evaluations(report, ax, *, objective=None):
    series = 0
    for algorithm, group in report['algorithms'].items():
        for run in group['runs']:
            color = f'C{series % 10}'
            rows = run['evaluations']
            values = [r['objective'] if objective is None else r['objectives'].get(objective)
                      for r in rows]
            failed = sum(r['objective'] is None for r in rows)
            label = f'{algorithm}; seed {run["seed"]}'
            if len(report['budgets']) > 1:
                label += f'; budget {run["budget"]}'
            if failed:
                label += f'; {failed} failed CFD (omitted)'
            missing = sum(value is None and row['objective'] is not None
                          for value, row in zip(values, rows))
            if missing:
                label += f'; {missing} missing values (gaps)'
            ax.plot(range(1, run['attempts']+1),
                    [float('nan') if value is None else value for value in values], color=color, alpha=.7,
                    linestyle=('-', '--', '-.', ':')[(series // 10) % 4], label=label)
            for feasible, marker in ((True, 'o'), (False, 'x')):
                points = [(i, value) for i, (value, row) in enumerate(zip(values, rows), 1)
                          if value is not None and row['feasible'] == feasible]
                if points:
                    xs, ys = zip(*points)
                    ax.scatter(xs, ys, marker=marker, color=color, s=28, zorder=3)
            series += 1
    _style(ax, 'Individual evaluations · circles: feasible; crosses: infeasible',
           'Evaluation attempts', 'Scalar objective ↓' if objective is None else f'{objective} ↓')
    ax.set_xlim(.5, report['budget']+.5)
    ax.legend(fontsize='small')


def _fixed_budget(report, ax):
    labels = []
    for index, (algorithm, group) in enumerate(report['algorithms'].items(), start=1):
        runs = group['runs']
        values = [run['best_feasible'] for run in runs if run['best_feasible'] is not None]
        labels.append(f'{algorithm}\n{len(values)}/{len(runs)} feasible')
        if values:
            box = ax.boxplot([values], positions=[index], widths=.45, patch_artist=True,
                             showfliers=False, medianprops={'color': '#17202a'})
            box['boxes'][0].set(facecolor=f'C{(index - 1) % 10}', alpha=.25)
            offsets = [0.] if len(values) == 1 else [(.24 * i / (len(values) - 1)) - .12 for i in range(len(values))]
            ax.scatter([index + offset for offset in offsets], values,
                       color=f'C{(index - 1) % 10}', s=25, zorder=3)
        else:
            ax.text(index, .5, 'No feasible result', ha='center', va='center', rotation=90,
                    transform=ax.get_xaxis_transform(), fontsize='small')
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_xlim(.5, len(labels) + .5)
    _style(ax, f'Fixed budget · {report["budget"]} attempts', 'Optimizer', 'Final feasible objective ↓')


def _ert(report, ax):
    for index, (algorithm, group) in enumerate(report['algorithms'].items()):
        rows = sorted(group['targets'], key=lambda row: row['target'])
        ax.plot([row['target'] for row in rows],
                [float('nan') if row['ert'] is None else row['ert'] for row in rows],
                marker='o', color=f'C{index % 10}', label=algorithm)
        missing = [row['target'] for row in rows if row['ert'] is None]
        if missing:
            ax.scatter(missing, [.96 - index * .06] * len(missing), marker='x',
                       color=f'C{index % 10}', transform=ax.get_xaxis_transform(),
                       label=f'{algorithm}: ERT = ∞ (no hits)')
    _style(ax, 'Expected running time', 'Objective target (≤ target succeeds)', 'Evaluation attempts ↓')
    # NaN ERT values do not contribute to autoscaling; include their markers too.
    targets = [row['target'] for group in report['algorithms'].values() for row in group['targets']]
    lower, upper = min(targets), max(targets)
    padding = .05 * (upper - lower) if upper > lower else max(1., abs(lower) * .05)
    ax.set_xlim(lower - padding, upper + padding)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize='small')


def _ecdf(report, ax):
    for index, (algorithm, group) in enumerate(report['algorithms'].items()):
        hits = [hit for row in group['targets'] for hit in row['first_hits']]
        counts = Counter(hit for hit in hits if hit is not None)
        xs, ys, reached = [0], [0.], 0
        for evaluation, count in sorted(counts.items()):
            reached += count
            xs.append(evaluation)
            ys.append(reached / len(hits))  # Misses remain in the denominator.
        if xs[-1] != report['budget']:
            xs.append(report['budget'])
            ys.append(ys[-1])
        ax.step(xs, ys, where='post', color=f'C{index % 10}', label=algorithm)
    _style(ax, 'ECDF · all run–target pairs', 'Evaluation attempts', 'Fraction of targets reached ↑')
    ax.set(xlim=(0, report['budget']), ylim=(0, 1.02))
    ax.legend(fontsize='small')


def _plot(directories, draw, *, targets=(), ax=None, output=None, allow_unequal_budgets=False,
          allow_different_failure_policies=False):
    report = _comparison(directories, targets, allow_unequal_budgets=allow_unequal_budgets,
                         allow_different_failure_policies=allow_different_failure_policies)
    if ax is None:
        ax = _figure((7, 4.5)).subplots()
    draw(report, ax)
    _save(ax.figure, output)
    return ax


def plot_convergence(directories, *, ax=None, output=None):
    """Best feasible objective per attempt, one line per seed; return an Axes.

    Failed/infeasible attempts consume x-axis budget but cannot improve the best.
    Lines stop at the last actual attempt; no observations are extrapolated.
    Different declared budgets and optimizer failure policies are allowed and
    shown in the legend. Physical objective/configuration must still match.
    """
    return _plot(directories, _convergence, ax=ax, output=output, allow_unequal_budgets=True,
                 allow_different_failure_policies=True)


def plot_mean_convergence(directories, *, ax=None, output=None):
    """Mean best-so-far across seeds, shaded by ±1 sample standard deviation.

    Requires completed runs with matching objective and failure policy, and at
    least two seeds per optimizer. Different budgets are labeled in the legend.
    Each mean stops at that optimizer's shortest run, without extrapolation or
    dropping shorter seeds. A point is missing until every seed has a feasible incumbent;
    failed seeds are never silently dropped. The band describes seed variation,
    not a confidence interval. Return an Axes without changing saved runs.
    """
    return _plot(directories, _mean_convergence, ax=ax, output=output, allow_unequal_budgets=True)


def plot_evaluations(directories, *, ax=None, output=None):
    """Plot each valid candidate's scalar objective, including constraint penalties.

    Circles indicate feasible candidates; crosses indicate constraint violations.
    Failed CFD is omitted as a gap, with counts in the legend. Different budgets
    and optimizer failure policies are labeled as in plot_convergence.
    """
    return _plot(directories, _evaluations, ax=ax, output=output, allow_unequal_budgets=True,
                 allow_different_failure_policies=True)


def plot_objectives(directories, *, output=None):
    """Return a Figure with the scalar and each named raw objective over attempts.

    Each panel has its own y scale and one trace per run, using saved candidate
    values, not best-so-far values. The first panel shows the scalar (including
    constraint penalties); subsequent panels show result['objectives'] in sorted
    name order and their original units/signs. Scalar-only journals still work.
    Iterations are evaluation attempts, including the baseline and failures,
    rather than optimizer generations. Failed or missing values remain gaps;
    circles mark feasible candidates and crosses mark constraint violations.
    Completed runs must share objective/configuration and dimension. Different
    budgets and failure policies are allowed and labeled as in plot_evaluations.
    """
    report = _comparison(directories, allow_unequal_budgets=True,
                         allow_different_failure_policies=True)
    names = set()
    for algorithm, group in report['algorithms'].items():
        for run in group['runs']:
            for index, row in enumerate(run['evaluations'], 1):
                objectives = row['objectives']
                context = f'{algorithm}; seed {run["seed"]}; evaluation {index}'
                if not isinstance(objectives, dict) or any(
                        not isinstance(name, str) or not name for name in objectives):
                    raise ValueError(f'Raw objectives must be a dictionary of named numbers: {context}')
                if any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in objectives.values()):
                    raise ValueError(f'Raw objectives must be finite numbers: {context}')
                names.update(objectives)
    objectives = [None, *sorted(names)]
    figure = _figure((9, 3.5 * len(objectives)))
    from matplotlib.ticker import MaxNLocator
    axes = figure.subplots(len(objectives), 1, sharex=True, squeeze=False).flat
    for objective, ax in zip(objectives, axes):
        _evaluations(report, ax, objective=objective)
        ax.set_title('Scalar objective' if objective is None else f'Raw objective · {objective}')
        ax.label_outer()
    figure.axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    title = textwrap.fill(f'{report["problem_name"]} · objective history', width=105)
    figure.suptitle(title + '\nCircles: feasible; crosses: infeasible; gaps: failed or missing values',
                   fontsize=11)
    _save(figure, output)
    return figure


def _pareto_archives(directories):
    if isinstance(directories, (str, Path)):
        directories = [directories]
    directories = [Path(d).parent if Path(d).is_file() and Path(d).name == 'pareto.json'
                   else Path(d) for d in directories]
    runs, seeds = [], set()
    identity = names = reference = None
    for folder in _run_directories(directories):
        path = folder / 'pareto.json'
        if not path.is_file():
            raise ValueError(f'No pareto.json found in {folder}; use a saved multi-objective run')
        experiment = json.loads((folder / 'experiment.json').read_text())
        archive = json.loads(path.read_text())
        current = (experiment['problem_name'], experiment['dimension'], experiment.get('failure_penalty'))
        if identity is not None and current != identity:
            raise ValueError('Pareto comparisons require the same objective/configuration, dimension and failure policy. '
                             f'Mismatch between {reference} and {folder}')
        identity, reference = current, folder
        if not isinstance(archive, dict) or archive.get('objective_direction') != 'minimize':
            raise ValueError(f'Pareto archive must declare objective_direction="minimize": {path}')
        current_names = archive.get('objective_names')
        if (not isinstance(current_names, list) or len(current_names) < 2
                or any(not isinstance(name, str) or not name for name in current_names)
                or len(set(current_names)) != len(current_names)):
            raise ValueError(f'Pareto plotting requires at least two distinct named objectives: {path}')
        current_names = tuple(sorted(current_names))
        if names is not None and current_names != names:
            raise ValueError(f'Pareto archives have different objective names: {path}')
        names = current_names
        solutions = archive.get('solutions')
        if not isinstance(solutions, list):
            raise ValueError(f'Pareto archive solutions must be a list: {path}')
        points = []
        for record in solutions:
            if not isinstance(record, dict) or record.get('valid') is not True or record.get('feasible') is not True:
                raise ValueError(f'Pareto archive contains an invalid or infeasible solution: {path}')
            result = record.get('result')
            values = result.get('objectives') if isinstance(result, dict) else None
            if not isinstance(values, dict) or set(values) != set(names):
                raise ValueError(f'Pareto solution objective names do not match the archive: {path}')
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in values.values()):
                raise ValueError(f'Pareto objectives must be finite numbers: {path}')
            points.append(tuple(values[name] for name in names))
        key = (experiment['algorithm'], experiment['seed'])
        if key in seeds:
            raise ValueError(f'Duplicate seeds for {key[0]}; do not plot a resumed archive as a new run')
        seeds.add(key)
        runs.append(dict(algorithm=key[0], seed=key[1], budget=experiment['budget'], points=points))
    if not runs:
        raise ValueError('Provide at least one saved NSGA-II run')
    return identity[0], names, runs


def plot_pareto(directories, *, objectives=None, ax=None, output=None):
    """Plot saved NSGA-II Pareto archives; return an Axes without running CFD.

    Accept run directories, parents containing runs, or pareto.json paths with
    adjacent experiment.json manifests. Saved snapshots of interrupted runs are
    supported: this shows the archive, not a claim that optimization completed.
    Each archive contains all observed feasible nondominated designs, including
    partial generations. Scalar diagnostics are never used as coordinates.

    Two objectives are selected automatically in alphabetical order. For larger
    vectors, pass objectives=(x_name, y_name) to project the full Pareto set;
    dominance is not recomputed in that projection. Two-objective points are
    connected in x order as a visual guide, while projections use markers only.
    All objectives are minimized, retaining their saved signs and units. Runs
    must share objective/configuration (including failure policy) and dimension;
    different budgets are labeled. Empty archives with known axes are shown.
    """
    problem_name, names, runs = _pareto_archives(directories)
    if objectives is None:
        if len(names) != 2:
            raise ValueError('Select two objective names with objectives=(x, y) or --objectives X Y. '
                             f'Available: {", ".join(names)}')
        objectives = names
    if (not isinstance(objectives, (list, tuple)) or len(objectives) != 2
            or any(name not in names for name in objectives) or objectives[0] == objectives[1]):
        raise ValueError(f'Choose two distinct objective names from: {", ".join(names)}')
    x_index, y_index = (names.index(name) for name in objectives)
    if ax is None:
        ax = _figure((8, 5.5)).subplots()
    for index, run in enumerate(runs):
        points = sorted((point[x_index], point[y_index]) for point in run['points'])
        label = f'{run["algorithm"]}; seed {run["seed"]}; budget {run["budget"]}'
        label += (f'; {len(points)} {"design" if len(points) == 1 else "designs"}'
                  if points else '; no feasible Pareto points')
        ax.plot([point[0] for point in points], [point[1] for point in points],
                marker=('o', 's', '^', 'D')[(index // 10) % 4], color=f'C{index % 10}',
                linestyle='-' if len(names) == 2 else 'None', label=label, alpha=.85)
    title = 'Saved Pareto front' if len(names) == 2 else f'Saved Pareto front · projection from {len(names)} objectives'
    _style(ax, title + '\n' + textwrap.fill(problem_name, width=80),
           f'{objectives[0]} ↓', f'{objectives[1]} ↓')
    ax.title.set_fontsize(11)
    if not any(run['points'] for run in runs):
        ax.text(.5, .5, 'No feasible Pareto points in the saved archives',
                ha='center', va='center', transform=ax.transAxes)
    ax.legend(fontsize='small')
    _save(ax.figure, output)
    return ax


def plot_fixed_budget(directories, *, ax=None, output=None):
    """Final-incumbent distribution and feasibility counts; return an Axes.

    Completed runs that terminate early retain their final recommendation at the
    common budget. Runs without a feasible result appear in the counts, not as
    fabricated finite values in the boxplot. Incomplete runs are rejected.
    """
    return _plot(directories, _fixed_budget, ax=ax, output=output)


def _targets(values):
    values = list(values)
    if not values or len(set(values)) != len(values):
        raise ValueError('Provide at least one target, with no duplicates')
    return values


def plot_ert(directories, *, targets, ax=None, output=None):
    """Fixed-target ERT, marking never-reached targets as infinite; return an Axes."""
    return _plot(directories, _ert, targets=_targets(targets), ax=ax, output=output)


def plot_ecdf(directories, *, targets, ax=None, output=None):
    """Fraction of all run–target pairs reached by each attempt; return an Axes."""
    return _plot(directories, _ecdf, targets=_targets(targets), ax=ax, output=output)


def plot_benchmark(directories, *, targets=(), output=None):
    """Return a Figure with convergence/distribution and optional ERT/ECDF panels.

    Supply explicit objective targets to add the target-based panels. Output can
    be a PNG, SVG or PDF path. No files in the supplied run directories are edited.
    """
    targets = list(targets)
    if targets:
        targets = _targets(targets)
    report = _comparison(directories, targets)
    figure = _figure((12, 8 if targets else 4.5))
    axes = figure.subplots(2 if targets else 1, 2, squeeze=False).flat
    for draw, ax in zip((_convergence, _fixed_budget, _ert, _ecdf), axes):
        draw(report, ax)
    title = f'{report["problem_name"]} · dimension {report["dimension"]} · budget {report["budget"]}'
    figure.suptitle(textwrap.fill(title, width=105), fontsize=11)
    _save(figure, output)
    return figure


def _geometry_data(directories, field):
    from ..problems.level4_high_dimensional_topology.p003.config import TopologyConfig, MODEL_REVISION
    from ..problems.level4_high_dimensional_topology.p003.design import decode, grid

    if isinstance(directories, (str, Path)):
        directories = [directories]
    paths = []
    for directory in directories:
        path = Path(directory)
        if path.is_file():
            paths.append(path)
        elif (path / 'best-feasible.json').is_file():
            paths.append(path / 'best-feasible.json')
        else:
            paths.extend(run / 'best-feasible.json' for run in _run_directories(path))
    if len(paths) != 1:
        raise ValueError('Geometry plotting requires exactly one run or saved best-feasible.json')
    path = paths[0]
    if not path.is_file():
        raise ValueError(f'No saved best-feasible.json found in {path.parent}')
    record = json.loads(path.read_text())
    if (not isinstance(record, dict) or record.get('valid') is not True
            or record.get('feasible') is not True):
        raise ValueError(f'Geometry plotting requires a valid, feasible saved evaluation: {path}')
    result = record.get('result')
    if not isinstance(result, dict) or result.get('problem') != 'P003':
        raise ValueError('Geometry plotting currently supports P003 topology results only')
    if result.get('model_revision', MODEL_REVISION) != MODEL_REVISION:
        raise ValueError('Unsupported P003 model revision for geometry reconstruction')
    saved_config = result.get('config')
    required = {'design_nx', 'design_ny', 'mesh', 'length_m', 'inlet_width_m',
                'outlet_width_m', 'filter_radius_m', 'projection_beta'}
    if not isinstance(saved_config, dict) or not required.issubset(saved_config):
        raise ValueError('Saved P003 result is missing its geometry configuration')
    config = TopologyConfig(**saved_config)
    values = record.get('x')  # Evaluated coordinates, never proposal.proposed_x.
    if (not isinstance(values, list) or len(values) != config.dimension
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 <= v <= 1 for v in values)):
        raise ValueError(f'Saved P003 design must contain {config.dimension} finite values in [0, 1]')
    rows, columns = config.design_ny + 1, config.design_nx + 1
    if field == 'design':
        pixels = [values[j*columns:(j+1)*columns] for j in range(rows)]
    else:
        # The CFD grid is port-fitted and is not the nodal design grid. Rebuild
        # the actual piecewise-constant cell indicator, then sample the cell
        # containing each uniform output pixel's center in physical coordinates.
        indicator = decode(values, config)['indicator']
        x_edges, y_edges = grid(config)
        cell_columns = len(x_edges) - 1
        ix = [bisect_right(x_edges, (i+.5)*config.width_m/columns)-1 for i in range(columns)]
        iy = [bisect_right(y_edges, (j+.5)*config.length_m/rows)-1 for j in range(rows)]
        pixels = [[indicator[j*cell_columns+i] for i in ix] for j in iy]
    return pixels, config, record


def plot_geometry(directories, *, field='indicator', threshold=None, ax=None,
                  output=None, pixels_output=None):
    """Plot one saved P003 best feasible topology without CFD or case files.

    Accept a run, a parent containing exactly one run, or a standalone saved
    best-feasible.json (including its result.config). Return a Matplotlib Axes;
    ``ax.images[0].get_array()`` gives the displayed data with shape (65, 33)
    for the default design grid: y rows, x columns, x fastest, y=0 at the bottom.
    Custom design grid sizes follow the saved configuration.

    ``indicator`` reconstructs the solver's radial filter and projection, then
    samples its cell field at uniform pixel centers. It is a visualization of
    the physical half-domain, not the native CFD mesh or a new simulation.
    ``design`` shows the evaluated nodal coordinates without filtering/projection.
    Values use a fixed [0, 1] grayscale (black rib, white channel for indicator).
    Optional ``threshold`` binarizes the indicator with channel >= threshold;
    this display-only mask has not been evaluated by CFD.

    ``output`` saves an annotated PNG/SVG/PDF. ``pixels_output`` saves a PNG
    with exactly one 8-bit grayscale pixel per array element and no axes or margins
    (default: width 33, height 65). Both exports use the same orientation.
    """
    if field not in ('indicator', 'design'):
        raise ValueError("Geometry field must be 'indicator' or 'design'")
    if threshold is not None:
        if (field != 'indicator' or isinstance(threshold, bool)
                or not isinstance(threshold, (int, float)) or not math.isfinite(threshold)
                or not 0 <= threshold <= 1):
            raise ValueError('threshold requires the indicator field and a finite value in [0, 1]')
    if pixels_output is not None:
        pixels_output = Path(pixels_output)
        if pixels_output.suffix.lower() != '.png':
            raise ValueError('Pixel output must have a .png extension')
        if output is not None and pixels_output.resolve() == Path(output).resolve():
            raise ValueError('Annotated output and pixel output must use different paths')
    pixels, config, record = _geometry_data(directories, field)
    if threshold is not None:
        pixels = [[float(value >= threshold) for value in row] for row in pixels]
    if ax is None:
        ax = _figure((5, 7)).subplots()
    artist = ax.imshow(pixels, origin='lower', interpolation='nearest', cmap='gray',
                       vmin=0, vmax=1, aspect='equal',
                       extent=(0, config.width_m*1000, 0, config.length_m*1000))
    label = 'Channel indicator' if field == 'indicator' else 'Raw nodal design'
    if threshold is not None:
        label = f'Channel indicator ≥ {threshold:g} (display only)'
    evaluation = record.get('evaluation', '?')
    ax.set(title=f'P003 · best feasible · evaluation {evaluation}\n'
                 f'{label} · {len(pixels)} × {len(pixels[0])} pixels',
           xlabel='x [mm]', ylabel='y [mm]')
    ax.title.set_fontsize(11)
    colorbar = ax.figure.colorbar(artist, ax=ax, fraction=.05, pad=.04)
    if field == 'indicator':
        ticks = [0, 1] if threshold is not None else [0, .5, 1]
        labels = ['0 · rib', '1 · channel'] if threshold is not None else ['0 · rib', '0.5', '1 · channel']
        colorbar.set_ticks(ticks, labels=labels)
    else:
        colorbar.set_label('Normalized design variable')
    _save(ax.figure, output)
    if pixels_output is not None:
        from PIL import Image
        pixels_output.parent.mkdir(parents=True, exist_ok=True)
        # PNG rows start at the top. Quantize only once, without a colormap LUT.
        raster = (artist.get_array()[::-1]*255).round().astype('uint8')
        Image.fromarray(raster).save(pixels_output)
    return ax


def main(argv=None):
    parser = argparse.ArgumentParser(description='Plot saved FlowField runs; no CFD is executed')
    parser.add_argument('runs', nargs='+', type=Path,
                        help='Run directories or parents; geometry also accepts a saved best-feasible.json')
    parser.add_argument('--kind', choices=('all', 'convergence', 'mean-convergence', 'evaluations', 'objectives', 'pareto', 'geometry', 'fixed-budget', 'ert', 'ecdf'), default='all')
    parser.add_argument('--objectives', nargs=2, metavar=('X', 'Y'),
                        help='Raw objective names for the Pareto x and y axes')
    parser.add_argument('--targets', type=float, nargs='+', default=[])
    parser.add_argument('--field', choices=('indicator', 'design'),
                        help='Geometry field: reconstructed channel indicator (default) or raw nodal design')
    parser.add_argument('--threshold', type=float,
                        help='Geometry only: show a binary channel mask at this indicator threshold')
    parser.add_argument('--pixels-output', type=Path,
                        help='Geometry only: pixel-exact PNG without axes (default grid: 65 rows × 33 columns)')
    parser.add_argument('--output', type=Path, required=True, help='PNG, SVG or PDF file')
    args = parser.parse_args(argv)
    if args.objectives is not None and args.kind != 'pareto':
        parser.error('--objectives is only supported with --kind pareto')
    if args.kind != 'geometry' and any(value is not None for value in (args.field, args.threshold, args.pixels_output)):
        parser.error('--field, --threshold and --pixels-output are only supported with --kind geometry')
    try:
        if args.kind == 'all':
            plot_benchmark(args.runs, targets=args.targets, output=args.output)
        elif args.kind in ('ert', 'ecdf'):
            function = plot_ert if args.kind == 'ert' else plot_ecdf
            function(args.runs, targets=args.targets, output=args.output)
        elif args.kind == 'pareto':
            plot_pareto(args.runs, objectives=args.objectives, output=args.output)
        elif args.kind == 'geometry':
            plot_geometry(args.runs, field=args.field or 'indicator', threshold=args.threshold,
                          output=args.output, pixels_output=args.pixels_output)
        else:
            function = {'convergence': plot_convergence, 'evaluations': plot_evaluations,
                        'objectives': plot_objectives,
                        'mean-convergence': plot_mean_convergence,
                        'fixed-budget': plot_fixed_budget}[args.kind]
            function(args.runs, output=args.output)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(1, f'flowfield-plot: {exc}\n')
    print(args.output.resolve())
    if args.pixels_output is not None:
        print(args.pixels_output.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
