"""Plot saved optimization runs without importing an optimizer or running CFD.

Matplotlib is optional and loaded only when creating a figure. Functions accept
one run directory, a parent directory containing runs, or a list of either.
Only runs of the same objective/configuration, dimension and budget are compared.
"""

import argparse
from collections import Counter
from pathlib import Path
import textwrap

from .analysis import compare_runs


def _comparison(directories, targets=()):
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
    return compare_runs(runs, targets=targets)


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
        for number, run in enumerate(runs):
            values = [float('nan') if value is None else value for value in run['best_so_far']]
            ax.step(range(1, run['attempts'] + 1), values, where='post', color=f'C{index % 10}',
                    alpha=.8 if len(runs) == 1 else .55, marker='.' if run['attempts'] == 1 else None,
                    label=f'{algorithm} ({len(runs)} {"run" if len(runs) == 1 else "runs"})' if number == 0 else None)
    _style(ax, 'Convergence · individual runs', 'Evaluation attempts', 'Best feasible objective ↓')
    ax.set_xlim(.5, report['budget'] + .5)
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


def _plot(directories, draw, *, targets=(), ax=None, output=None):
    report = _comparison(directories, targets)
    if ax is None:
        ax = _figure((7, 4.5)).subplots()
    draw(report, ax)
    _save(ax.figure, output)
    return ax


def plot_convergence(directories, *, ax=None, output=None):
    """Best feasible objective per attempt, one line per seed; return an Axes.

    Failed/infeasible attempts consume x-axis budget but cannot improve the best.
    Lines stop at the last actual attempt; no observations are extrapolated.
    """
    return _plot(directories, _convergence, ax=ax, output=output)


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


def main(argv=None):
    parser = argparse.ArgumentParser(description='Plot saved FlowField runs; no CFD is executed')
    parser.add_argument('runs', nargs='+', type=Path, help='Run directories or parents containing runs of one problem')
    parser.add_argument('--kind', choices=('all', 'convergence', 'fixed-budget', 'ert', 'ecdf'), default='all')
    parser.add_argument('--targets', type=float, nargs='+', default=[])
    parser.add_argument('--output', type=Path, required=True, help='PNG, SVG or PDF file')
    args = parser.parse_args(argv)
    try:
        if args.kind == 'all':
            plot_benchmark(args.runs, targets=args.targets, output=args.output)
        elif args.kind in ('ert', 'ecdf'):
            function = plot_ert if args.kind == 'ert' else plot_ecdf
            function(args.runs, targets=args.targets, output=args.output)
        else:
            function = plot_convergence if args.kind == 'convergence' else plot_fixed_budget
            function(args.runs, output=args.output)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(1, f'flowfield-plot: {exc}\n')
    print(args.output.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
