"""Run a selected problem/optimizer/seed matrix, retaining independent traces."""

import argparse
import json
from pathlib import Path

from ..evaluation import write_json
from ..optimization import optimize, list_optimizers
from ..optimization.registry import resolve_optimizer
from ..optimization.state import save_json
from ..registry import create_problem, list_problems, problem_config_type
from ..runtime import Runtime, DEFAULT_IMAGE


def run_benchmark(*, problems, optimizers, seeds, budget, output,
                  problem_options=None, optimizer_options=None, with_ioh=True):
    """One fresh problem and optimizer per run; no implicit full-suite execution.

    problem_options maps problem IDs to create_problem kwargs. optimizer_options
    maps optimizer IDs to optimize kwargs (e.g. population, sigma, bo_initial_points). Baselines and
    failures count against the same budget. Paths in benchmark.json are relative
    so the entire output directory can be moved for analysis.
    """
    problems, optimizers, seeds = list(problems), list(optimizers), list(seeds)
    for name, values in (('problems', problems), ('optimizers', optimizers), ('seeds', seeds)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f'{name} must be a nonempty sequence without duplicates')
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError('Seeds must be integers in [0, 2**32)')
    if type(budget) is not int or budget < 1:
        raise ValueError('budget must be a positive integer')
    if set(problems) - set(list_problems()) or set(optimizers) - set(list_optimizers()):
        raise ValueError('Select registered problem and optimizer IDs')
    problem_options = problem_options or {}
    optimizer_options = optimizer_options or {}
    for options in optimizer_options.values():
        if set(options) - {'population', 'sigma', 'failure_penalty', 'max_seconds', 'bo_initial_points', 'bo_candidates'}:
            raise ValueError('Optimizer options may only set population, sigma, failure_penalty, max_seconds, '
                             'bo_initial_points, bo_candidates')
    # Validate problem construction and algorithm compatibility before creating output.
    for identifier in problems:
        problem = create_problem(identifier, **problem_options.get(identifier, {}))
        for optimizer in optimizers:
            resolve_optimizer(problem, optimizer)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, status='running', problems=problems,
                    optimizers=optimizers, seeds=seeds, budget=budget, ioh_enabled=with_ioh,
                    runs=[], objective_direction='minimize')
    write_json(output / 'benchmark.json', manifest)
    try:
        for identifier in problems:
            for optimizer in optimizers:
                for seed in seeds:
                    relative = Path(identifier) / optimizer / f'seed-{seed}'
                    folder = output / relative
                    options = dict(problem_options.get(identifier, {}), output_root=folder / 'cases')
                    problem = create_problem(identifier, **options)
                    entry = dict(problem=identifier, optimizer=optimizer, seed=seed, directory=str(relative))
                    manifest['runs'].append(entry)
                    try:
                        summary = optimize(problem, optimizer=optimizer, budget=budget, seed=seed,
                            output=folder, with_ioh=with_ioh, **optimizer_options.get(optimizer, {}))
                        entry.update(status=summary['status'], attempts=summary['attempts'],
                                     best_objective=summary['best_objective'])
                    except (Exception, KeyboardInterrupt):
                        entry['status'] = 'failed'
                        raise
                    finally:
                        save_json(output / 'benchmark.json', manifest)
        manifest['status'] = ('completed' if all(r['status'] == 'completed' for r in manifest['runs'])
                              else 'completed_with_failures')
    except (Exception, KeyboardInterrupt):
        manifest['status'] = 'failed'
        raise
    finally:
        save_json(output / 'benchmark.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description='Benchmark selected optimizers over independent seeds')
    parser.add_argument('--problems', '--problem', nargs='+', choices=list_problems(), required=True)
    parser.add_argument('--optimizers', '--optimizer', nargs='+', choices=list_optimizers(), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--budget', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='JSON config for a single selected problem')
    parser.add_argument('--backend', choices=('auto', 'native', 'docker'), default='auto')
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--no-ioh', action='store_true', help='Keep JSON traces without IOH export')
    args = parser.parse_args(argv)
    try:
        if args.config and len(args.problems) != 1:
            raise ValueError('--config requires one problem; use problem_options in Python for multiple configs')
        options = {}
        for name in args.problems:
            options[name] = {}
            if problem_config_type(name) is not None:
                options[name]['runtime'] = Runtime(args.backend, args.image, args.timeout)
            if args.config:
                options[name]['config'] = args.config
        result = run_benchmark(problems=args.problems, optimizers=args.optimizers, seeds=args.seeds,
            budget=args.budget, output=args.output, problem_options=options, with_ioh=not args.no_ioh)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(1, f'flowfield-benchmark: {exc}\n')
    print(json.dumps(result, indent=2))
    return 0
