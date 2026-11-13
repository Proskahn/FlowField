"""Fixed-target and fixed-budget summaries from saved observations; no CFD."""

import json
import math
from pathlib import Path


def compare_runs(directories, *, targets=()):
    """Compare runs of the SAME objective/configuration and dimension.

    Only successful feasible observations can reach a target. Failed attempts
    still consume budget. ERT = sum(first hit, or spent attempts for a miss) /
    successful runs. None represents infinite ERT when no run hits the target.
    Full evaluation traces remain available for IOHanalyzer ECDFs and plots.
    """
    targets = list(targets)
    if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in targets):
        raise ValueError('Targets must be finite numbers')
    groups, identity = {}, None
    for directory in directories:
        folder = Path(directory)
        experiment = json.loads((folder / 'experiment.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        current = (experiment['problem_name'], experiment['dimension'], experiment['budget'])
        if identity is not None and current != identity:
            raise ValueError('Compare only the same objective/configuration, dimension, and budget')
        identity = current
        if summary['status'] not in ('completed', 'completed_with_failures'):
            raise ValueError(f'Incomplete run: {folder}')
        # Include recovered attempts without counting their original partial rows twice.
        from ..optimization.state import load_history
        rows, pending = load_history(folder, experiment['budget'], experiment.get('failure_penalty'))
        if pending:
            raise ValueError(f'Run has unreconciled attempts: {folder}')
        feasible = [r for r in rows if r['valid'] and r['feasible']]
        groups.setdefault(experiment['algorithm'], []).append(dict(
            seed=experiment['seed'], attempts=len(rows),
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
    return dict(problem_name=identity[0], dimension=identity[1], budget=identity[2], algorithms=result)
