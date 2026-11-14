"""Fixed-target and fixed-budget summaries from saved observations; no CFD."""

import json
import math
from pathlib import Path


def compare_runs(directories, *, targets=(), allow_unequal_budgets=False):
    """Compare runs of the SAME objective/configuration and dimension.

    Only successful feasible observations can reach a target. Failed attempts
    still consume budget. ERT = sum(first hit, or spent attempts for a miss) /
    successful runs. None represents infinite ERT when no run hits the target.
    Full evaluation traces remain available for IOHanalyzer ECDFs and plots.
    allow_unequal_budgets is for convergence traces only; it cannot be combined
    with fixed targets. Each run retains its own budget and final attempt.
    """
    targets = list(targets)
    if allow_unequal_budgets and targets:
        raise ValueError('Unequal-budget comparisons support convergence only, without targets')
    if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in targets):
        raise ValueError('Targets must be finite numbers')
    groups, identity = {}, None
    reference_folder = None
    for directory in directories:
        folder = Path(directory)
        experiment = json.loads((folder / 'experiment.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        current = (experiment['problem_name'], experiment['dimension'], experiment['budget'])
        compared = 2 if allow_unequal_budgets else 3
        if identity is not None and current[:compared] != identity[:compared]:
            differences = [f'{key}: {expected!r} != {actual!r}'
                           for key, expected, actual in zip(('problem_name', 'dimension', 'budget')[:compared],
                                                            identity[:compared], current[:compared])
                           if expected != actual]
            raise ValueError('Compare only the same objective/configuration, dimension, and budget. '
                             f'Mismatch between {reference_folder} and {folder}: '
                             + '; '.join(differences))
        if identity is None:
            reference_folder = folder
            identity = current
        exhausted_failures = summary['status'] == 'failed' and summary.get('stop_reason') == 'evaluation_budget'
        if summary['status'] not in ('completed', 'completed_with_failures') and not exhausted_failures:
            raise ValueError(f'Incomplete run: {folder}')
        # Include recovered attempts without counting their original partial rows twice.
        from ..optimization.state import load_history
        rows, pending = load_history(folder, experiment['budget'], experiment.get('failure_penalty'))
        if pending:
            raise ValueError(f'Run has unreconciled attempts: {folder}')
        if not rows or summary.get('attempts') != len(rows):
            raise ValueError(f'Run summary does not match its observation journal: {folder}')
        if exhausted_failures and len(rows) != experiment['budget']:
            raise ValueError(f'Failed run has not exhausted its budget: {folder}')
        feasible = [r for r in rows if r['valid'] and r['feasible']]
        best, trace = None, []
        for row in rows:
            if row['valid'] and row['feasible']:
                best = row['objective'] if best is None else min(best, row['objective'])
            trace.append(best)
        groups.setdefault(experiment['algorithm'], []).append(dict(
            seed=experiment['seed'], attempts=len(rows), budget=experiment['budget'],
            best_so_far=trace,
            best_feasible=min((r['objective'] for r in feasible), default=None),
            first_hits=[next((r['evaluation'] for r in feasible if r['objective'] <= t), None) for t in targets]))
    if not groups:
        raise ValueError('Provide at least one completed run')
    result = {}
    for algorithm, runs in groups.items():
        if len({r['seed'] for r in runs}) != len(runs):
            raise ValueError(f'Duplicate seeds for {algorithm}; do not count a resumed trace as a new run')
        target_rows = []
        for index, target in enumerate(targets):
            hits = [r['first_hits'][index] for r in runs]
            successes = sum(hit is not None for hit in hits)
            cost = sum(hit if hit is not None else r['attempts'] for hit, r in zip(hits, runs))
            target_rows.append(dict(target=target, successes=successes,
                success_rate=successes / len(runs), ert=cost / successes if successes else None,
                first_hits=hits))
        result[algorithm] = dict(runs=runs, targets=target_rows)
    budgets = sorted({run['budget'] for runs in groups.values() for run in runs})
    return dict(problem_name=identity[0], dimension=identity[1], budget=max(budgets),
                budgets=budgets, algorithms=result)
