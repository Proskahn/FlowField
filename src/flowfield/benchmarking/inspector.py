"""Read-only journal export and anytime analysis through IOHinspector.

Raw vectors are never replaced by indicator values. Invalid/infeasible attempts
remain in the evaluation clock but contribute no hypervolume. All objectives
use FlowField's minimization convention, including negative power.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from ..optimization.state import read_jsonl, save_json


def dependencies():
    try:
        import iohinspector as inspector
        import numpy as np
        import polars as pl
    except ImportError as exc:
        raise RuntimeError('Install analysis with: python -m pip install -e ".[inspector]"') from exc
    return inspector, np, pl


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _source_hashes(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (folder/name for name in ('experiment.json', 'evaluations.jsonl', 'events.jsonl', 'recoveries.jsonl'))
            if p.is_file()}


def _discover(inputs):
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    folders = set()
    for source in inputs:
        source = Path(source).resolve()
        if source.is_file() and source.name in ('evaluations.jsonl', 'experiment.json', 'benchmark.json'):
            source = source.parent
        if not source.is_dir():
            raise ValueError(f'Run directory does not exist: {source}')
        paths = ([source/'experiment.json'] if (source/'experiment.json').is_file()
                 else sorted(source.rglob('experiment.json')))
        for path in paths:
            if 'resume-sessions' not in path.relative_to(source).parts:
                folders.add(path.parent)
    if not folders:
        raise ValueError('No experiment.json files found in the selected inputs')
    return sorted(folders)


def _journal(folder, experiment):
    """Include failed/interrupted attempts, without altering recovery journals."""
    observations = read_jsonl(folder/'evaluations.jsonl')
    starts = [r for r in read_jsonl(folder/'events.jsonl') if r.get('event') == 'started']
    recoveries = read_jsonl(folder/'recoveries.jsonl')
    by_index = {}
    for row in observations:
        index = row.get('evaluation')
        if type(index) is not int or index < 1 or index in by_index:
            raise ValueError(f'Duplicate/invalid evaluation in {folder}')
        by_index[index] = row
    recovered = set()
    for row in recoveries:
        index = row.get('evaluation')
        if type(index) is not int or index < 1 or index in recovered:
            raise ValueError(f'Duplicate/invalid recovery in {folder}')
        if index in by_index and by_index[index].get('objective') is not None:
            raise ValueError(f'Completed evaluation also recovered in {folder}')
        by_index[index] = row
        recovered.add(index)
    if starts:
        if [r.get('evaluation') for r in starts] != list(range(1, len(starts)+1)):
            raise ValueError(f'Noncontiguous started evaluations in {folder}')
        if set(by_index) - set(range(1, len(starts)+1)):
            raise ValueError(f'Unstarted evaluation in {folder}')
        for start in starts:
            index = start['evaluation']
            if index in by_index:
                if any(by_index[index].get(k) != start.get(k) for k in ('x', 'phase', 'generation', 'proposal')):
                    raise ValueError(f'Evaluation {index} differs from its started event in {folder}')
            else:
                by_index[index] = dict(start, valid=False, feasible=False, objective=None,
                                      analysis_status='unfinished_attempt')
    if not by_index or sorted(by_index) != list(range(1, len(by_index)+1)):
        raise ValueError(f'Empty or noncontiguous evaluation log in {folder}')
    if len(by_index) > experiment['budget']:
        raise ValueError(f'Evaluations exceed the recorded budget in {folder}')
    return [by_index[i] for i in sorted(by_index)]


def _read_runs(inputs, regime):
    runs, identity, names, constraints, seeds = [], None, None, None, set()
    for folder in _discover(inputs):
        hashes = _source_hashes(folder)
        experiment = json.loads((folder/'experiment.json').read_text())
        if experiment.get('objective_direction', 'minimize') != 'minimize':
            raise ValueError('Analysis requires objectives in minimization convention')
        current = {k: experiment.get(k) for k in ('problem', 'problem_name', 'dimension', 'config', 'failure_penalty')}
        if identity is not None and current != identity:
            raise ValueError('Analyze the same problem/configuration, dimension and failure policy together; '
                             f'mismatch at {folder}')
        identity = current
        if regime == 'single' and (experiment.get('regime') == 'multi'
                                   or experiment.get('optimization_mode') == 'multi_objective'):
            raise ValueError('Multi-objective runs require --regime multi for benchmark analysis')
        key = (experiment['algorithm'], experiment['seed'])
        if key in seeds:
            raise ValueError(f'Duplicate algorithm/seed {key}; do not count a resumed/copied run twice')
        seeds.add(key)
        rows = _journal(folder, experiment)
        if _source_hashes(folder) != hashes:
            raise ValueError(f'Run changed while reading {folder}; retry analysis after the evaluation finishes')
        elapsed, complete_times = 0., True
        for row in rows:
            raw = row.get('result') or {}
            valid = (row.get('valid', False) and raw.get('status') not in ('failed', 'penalized')
                     and raw.get('metadata', {}).get('valid_cfd', True))
            objectives, cons = raw.get('objectives', {}), raw.get('constraints', {})
            if valid:
                if (not isinstance(objectives, dict) or not isinstance(cons, dict)
                        or any(not isinstance(k, str) or not k for k in (*objectives, *cons))
                        or any(not _finite(v) for v in (*objectives.values(), *cons.values()))
                        or not _finite(row.get('objective'))):
                    raise ValueError(f'Invalid objective/constraint values in {folder}, evaluation {row["evaluation"]}')
                found, found_constraints = sorted(objectives), sorted(cons)
                if regime == 'multi' and len(found) < 2:
                    raise ValueError(f'Multi analysis requires at least two named raw objectives: {folder}')
                if names is not None and (found != names or found_constraints != constraints):
                    raise ValueError('Objective or constraint names differ between observations')
                names, constraints = found, found_constraints
            row['_valid'] = bool(valid)
            row['_feasible'] = bool(valid and row.get('feasible', False) and raw.get('feasible', True)
                                    and all(v <= 1e-12 for v in cons.values()))
            duration = row.get('wall_time_s')
            if _finite(duration) and duration >= 0:
                elapsed += duration
            else:
                complete_times = False
            row['_cumulative_time'] = elapsed if complete_times else None
        runs.append(dict(folder=folder, experiment=experiment, rows=rows,
                         times_complete=complete_times, source_sha256=hashes))
    return runs, identity, names, constraints or []


def _settings(runs, identity, names, reference_point, settings):
    if reference_point is not None and settings is not None:
        raise ValueError('Use reference_point or saved settings, not both')
    if settings is not None:
        settings = json.loads(Path(settings).read_text()) if isinstance(settings, (str, Path)) else dict(settings)
        if settings.get('schema_version') != 1 or settings.get('identity') != identity:
            raise ValueError('Saved analysis settings do not match this problem/configuration/dimension')
        names = names or settings.get('objectives')
        if settings.get('objectives') != names:
            raise ValueError('Saved analysis settings have different objectives')
        if settings.get('direction') != 'minimize' or settings.get('feasibility_tolerance') != 1e-12:
            raise ValueError('Saved settings change the objective direction or feasibility policy')
    else:
        names = names or (sorted(reference_point) if isinstance(reference_point, dict) else None)
        if not names or len(names) < 2:
            raise ValueError('No objective schema; supply a named --reference-point for all-failed runs')
        settings = dict(schema_version=1, identity=identity, objectives=names,
                        direction='minimize', feasibility_tolerance=1e-12)
        if reference_point is not None:
            if not isinstance(reference_point, dict) or set(reference_point) != set(names):
                raise ValueError('Reference point must map every objective name to a finite minimized value')
            settings.update(normalization='none', offset={n: 0. for n in names}, scale={n: 1. for n in names},
                            reference_point=reference_point)
        else:
            vectors = [row['result']['objectives'] for run in runs for row in run['rows'] if row['_feasible']]
            if not vectors:
                raise ValueError('No feasible observations for automatic normalization; supply --reference-point')
            offset = {n: min(v[n] for v in vectors) for n in names}
            scale = {n: max(v[n] for v in vectors)-offset[n] for n in names}
            # A constant axis still needs a positive, recorded scale.
            scale = {n: s if s > 0 else max(abs(offset[n]), 1.) for n, s in scale.items()}
            settings.update(normalization='pooled_feasible_min_max', offset=offset, scale=scale,
                            reference_point={n: offset[n]+1.1*scale[n] for n in names},
                            constant_axis_policy='scale=max(abs(value), 1)',
                            note='Shared across selected runs. Reuse this file to compare later campaigns.')
    if not isinstance(names, list) or len(names) < 2 or len(set(names)) != len(names):
        raise ValueError('Analysis settings require at least two distinct objective names')
    for key in ('offset', 'scale', 'reference_point'):
        values = settings.get(key)
        if (not isinstance(values, dict) or set(values) != set(names)
                or any(not _finite(v) for v in values.values())
                or (key == 'scale' and any(v <= 0 for v in values.values()))):
            raise ValueError(f'Invalid analysis settings: {key}')
    return settings


def _export(runs, names, constraints, output):
    objective_columns = {name: f'objective_{i}' for i, name in enumerate(names)}
    constraint_columns = {name: f'constraint_{i}' for i, name in enumerate(constraints)}
    trace_columns = ['valid', 'feasible', 'evaluation_wall_time_s', 'cumulative_evaluation_wall_time_s',
                     *objective_columns.values(), *constraint_columns.values()]
    fields = ['run_key', 'algorithm', 'function', 'dimension', 'seed', 'budget', 'regime',
              'evaluations', 'scalar', *trace_columns]
    with (output/'evaluations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, run in enumerate(runs):
            exp = run['experiment']
            for row in run['rows']:
                raw = row.get('result') or {}
                entry = dict(run_key=str(index), algorithm=exp['algorithm'], function=exp['problem_name'],
                             dimension=exp['dimension'], seed=exp['seed'], budget=exp['budget'],
                             regime=exp.get('regime', 'multi' if exp.get('optimization_mode') == 'multi_objective' else 'single'),
                             evaluations=row['evaluation'], scalar=row.get('objective') if _finite(row.get('objective')) else 'nan',
                             valid=int(row['_valid']), feasible=int(row['_feasible']),
                             evaluation_wall_time_s=row.get('wall_time_s'),
                             cumulative_evaluation_wall_time_s=row['_cumulative_time'])
                for source, columns in (('objectives', objective_columns), ('constraints', constraint_columns)):
                    entry.update({column: raw.get(source, {}).get(name) if row['_valid'] else None
                                  for name, column in columns.items()})
                writer.writerow(entry)
    mapping = dict(version=1, evaluations='evaluations', value='scalar', run_by=['run_key'],
                   fields=dict(algorithm_name='algorithm', function_name='function', dimension='dimension',
                               maximization={'constant': False}, budget='budget'),
                   descriptors=['run_key', 'seed', 'regime'], trace_columns=trace_columns)
    save_json(output/'mapping.json', mapping)
    save_json(output/'columns.json', dict(objectives=objective_columns, constraints=constraint_columns,
                                         objective_direction='minimize', scalar_role='diagnostic in multi regime'))
    return mapping, objective_columns


def analyze_runs(inputs, *, output, regime='multi', reference_point=None, settings=None):
    """Export original vectors and compute indicators with no problem instantiation.

    Auto normalization is shared across all selected feasible observations. A
    saved settings file freezes it for future comparisons. With an explicit raw
    reference point, hypervolume is in raw objective units (no normalization).
    """
    if regime not in ('single', 'multi'):
        raise ValueError('regime must be single or multi')
    if regime == 'single' and (reference_point is not None or settings is not None):
        raise ValueError('Hypervolume reference/settings require --regime multi')
    runs, identity, names, constraints = _read_runs(inputs, regime)
    config = _settings(runs, identity, names, reference_point, settings) if regime == 'multi' else None
    names = config['objectives'] if config else names or []
    inspector, np, pl = dependencies()
    output = Path(output).resolve()
    if any(output == run['folder'] for run in runs):
        raise ValueError('Analysis output must be separate from source runs')
    output.mkdir(parents=True, exist_ok=False)
    mapping, columns = _export(runs, names, constraints, output)
    from iohinspector.convert import convert
    dataset_id = 'flowfield-' + hashlib.sha256((output/'evaluations.csv').read_bytes()).hexdigest()[:16]
    convert(output/'evaluations.csv', output/'iohinspector', mapping=mapping, dataset_id=dataset_id)
    manager = inspector.DataManager()
    manager.add_dataset(output/'iohinspector')
    # monotonic=True would discard scalar non-improvements that improve the Pareto front!
    data = manager.load(monotonic=False, include_meta_data=True)
    report = dict(schema_version=1, regime=regime, identity=identity, objectives=names,
                  indicator='hypervolume' if regime == 'multi' else 'best_feasible_objective',
                  indicator_direction='maximize' if regime == 'multi' else 'minimize',
                  versions={n: importlib.metadata.version(n) for n in ('iohinspector', 'moocore', 'polars')},
                  time_axis='sum of saved evaluation wall_time_s; excludes optimizer overhead; null if incomplete',
                  settings=config, runs=[])
    for index, run in enumerate(runs):
        group = data.filter(pl.col('run_key').cast(pl.String) == str(index)).sort('evaluations')
        if len(group) != len(run['rows']):
            raise RuntimeError('IOHinspector import did not preserve the evaluation count')
        if regime == 'multi':
            reference = np.array([(config['reference_point'][n]-config['offset'][n])/config['scale'][n] for n in names])
            prepared = []
            hv_columns = [f'hv_{i}' for i in range(len(names))]
            for i, name in enumerate(names):
                prepared.append(pl.when(pl.col('feasible') == 1)
                                .then((pl.col(columns[name]).cast(pl.Float64)-config['offset'][name])/config['scale'][name])
                                .otherwise(float(reference[i])).alias(hv_columns[i]))
            # An invalid attempt is represented only in this temporary scoring
            # frame by the reference point: zero volume, unchanged evaluation clock.
            scored = inspector.add_indicator(group.with_columns(prepared),
                        inspector.HyperVolume(reference), hv_columns,
                        evals=group['evaluations'].to_list()).sort('evaluations')
            values = scored['HyperVolume'].to_list()
        else:
            values, best = [], None
            for row in run['rows']:
                if row['_feasible']:
                    best = row['objective'] if best is None else min(best, row['objective'])
                values.append(best)
        exp = run['experiment']
        report['runs'].append(dict(directory=str(run['folder']), algorithm=exp['algorithm'],
            seed=exp['seed'], source_regime=exp.get('regime', 'multi' if exp.get('optimization_mode') == 'multi_objective' else 'single'),
            budget=exp['budget'], attempts=len(values), feasible_evaluations=sum(r['_feasible'] for r in run['rows']),
            failed_evaluations=sum(not r['_valid'] for r in run['rows']), final_value=values[-1],
            evaluations=list(range(1, len(values)+1)), values=values,
            cumulative_evaluation_wall_time_s=[r['_cumulative_time'] for r in run['rows']],
            source_sha256=run['source_sha256']))
    if config:
        save_json(output/'analysis-settings.json', config)
    save_json(output/'analysis.json', report)
    with (output/'indicators.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['algorithm', 'seed', 'evaluations', report['indicator'], 'cumulative_evaluation_wall_time_s'])
        for run in report['runs']:
            writer.writerows((run['algorithm'], run['seed'], e, v, t) for e, v, t in
                             zip(run['evaluations'], run['values'], run['cumulative_evaluation_wall_time_s']))
    _plot(report, output/'convergence.png')
    return report


def _plot(report, output):
    from .plotting import _figure, _save, _style
    figure = _figure((9, 5))
    ax = figure.subplots()
    algorithms = sorted({r['algorithm'] for r in report['runs']})
    for run in report['runs']:
        color = f'C{algorithms.index(run["algorithm"]) % 10}'
        ax.step(run['evaluations'], [float('nan') if v is None else v for v in run['values']],
                where='post', color=color, alpha=.75, label=f'{run["algorithm"]}; seed {run["seed"]}')
    ylabel = ('Feasible hypervolume ↑' if report['regime'] == 'multi' else 'Best feasible objective ↓')
    if report['regime'] == 'multi' and report['settings']['normalization'] != 'none':
        ylabel = 'Feasible hypervolume (normalized objectives) ↑'
    _style(ax, f'{report["identity"]["problem"]} · {report["identity"]["dimension"]} variables',
           'Evaluation attempts (including failures)', ylabel)
    ax.legend(fontsize='small')
    _save(figure, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Analyze saved FlowField journals with IOHinspector; never runs CFD')
    parser.add_argument('--input', nargs='+', required=True, type=Path, help='Run/benchmark directories or evaluations.jsonl files')
    parser.add_argument('--output', type=Path, required=True, help='New analysis directory')
    parser.add_argument('--regime', choices=('single', 'multi'), default='multi')
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument('--reference-point', type=json.loads, help='JSON map of objective names to RAW minimized values')
    reference.add_argument('--settings', type=Path, help='Reuse analysis-settings.json to freeze normalization and reference')
    args = parser.parse_args(argv)
    try:
        report = analyze_runs(args.input, output=args.output, regime=args.regime,
                              reference_point=args.reference_point, settings=args.settings)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(1, f'flowfield-analyze: {exc}\n')
    print(json.dumps(dict(output=str(args.output.resolve()), indicator=report['indicator'],
                          runs=[{k: r[k] for k in ('algorithm', 'seed', 'attempts', 'final_value')}
                                for r in report['runs']]), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
