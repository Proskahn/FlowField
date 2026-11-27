"""Run a selected problem/optimizer/seed matrix, retaining independent traces."""

import argparse
import json
from pathlib import Path

from ..evaluation import write_json
from ..optimization import optimize, list_optimizers
from ..optimization.state import save_json
from ..problems.level4_high_dimensional_topology.p003.scalability import add_dimension_argument, config_from_dimension
from ..registry import create_problem, list_problems, problem_config_type
from ..runtime import Runtime, DEFAULT_IMAGE
from .planning import plan_benchmark


def run_benchmark(*, problems, optimizers, seeds, budget, output,
                  problem_options=None, optimizer_options=None, with_ioh=True):
    """One fresh problem and optimizer per run; no implicit full-suite execution.

    problem_options maps problem IDs to create_problem kwargs. optimizer_options
    maps optimizer IDs to optimize kwargs (e.g. population, sigma, bo_initial_points). Baselines and
    failures count against the same budget. Paths in benchmark.json are relative
    so the entire output directory can be moved for analysis.
    """
    problems, optimizers, seeds = list(problems), list(optimizers), list(seeds)
    problem_options = problem_options or {}
    optimizer_options = optimizer_options or {}
    plan = plan_benchmark(problems=problems, optimizers=optimizers, seeds=seeds, budget=budget,
                          problem_options=problem_options, optimizer_options=optimizer_options,
                          with_ioh=with_ioh)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(plan, status='running', runs=[])
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
    add_dimension_argument(parser)
    parser.add_argument('--backend', choices=('auto', 'native', 'docker'), default='auto')
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--no-ioh', action='store_true', help='Keep JSON traces without IOH export')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print the experiment plan without CFD or files')
    data = parser.add_mutually_exclusive_group()
    data.add_argument('--dataset', type=Path, help='Dataset for a single data-backed problem')
    data.add_argument('--model', type=Path, help='Trained surrogate for a single data-backed problem')
    args = parser.parse_args(argv)
    try:
        if args.dimension is not None and 'P003' not in args.problems:
            raise ValueError('--dimension is supported by P003 only; include --problem P003')
        if args.dimension is not None and args.config is not None:
            raise ValueError('Use --dimension or --config, not both')
        if (args.config or args.dataset or args.model) and len(args.problems) != 1:
            raise ValueError('--config/--dataset/--model require one problem; use problem_options in Python for multiple configs')
        options = {}
        for name in args.problems:
            options[name] = {}
            if problem_config_type(name) is not None:
                options[name]['runtime'] = Runtime(args.backend, args.image, args.timeout)
            if args.config:
                options[name]['config'] = args.config
            if name == 'P003' and args.dimension is not None:
                options[name]['config'] = config_from_dimension(args.dimension)
            if args.dataset:
                options[name]['dataset'] = args.dataset
            if args.model:
                options[name]['model'] = args.model
        arguments = dict(problems=args.problems, optimizers=args.optimizers, seeds=args.seeds,
                         budget=args.budget, problem_options=options, with_ioh=not args.no_ioh)
        result = (plan_benchmark(**arguments) if args.dry_run
                  else run_benchmark(**arguments, output=args.output))
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(1, f'flowfield-benchmark: {exc}\n')
    print(json.dumps(result, indent=2))
    return 0
