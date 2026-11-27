"""Run a selected problem/optimizer/seed matrix, retaining independent traces."""

import argparse
import json
from pathlib import Path

from ..evaluation import write_json
from ..optimization import optimize, list_optimizers
from ..optimization.state import save_json
from ..optimization.mobo import MOBO_METHODS
from ..problems.level4_high_dimensional_topology.p003.scalability import add_dimension_argument, config_from_dimension
from ..registry import create_problem, list_problems, problem_config_type
from ..runtime import Runtime, DEFAULT_IMAGE
from .planning import plan_benchmark


def run_benchmark(*, problems, optimizers, seeds, budget, output,
                  problem_options=None, optimizer_options=None, with_ioh=True, regime=None):
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
                          with_ioh=with_ioh, regime=regime)
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
                            output=folder, with_ioh=with_ioh, regime=regime,
                            **optimizer_options.get(optimizer, {}))
                        entry.update(status=summary['status'], attempts=summary['attempts'],
                                     best_objective=summary['best_objective'], regime=summary['regime'])
                        if summary.get('optimization_mode') == 'multi_objective':
                            entry.update(optimization_mode='multi_objective',pareto_size=summary['pareto_size'],
                                         objective_names=summary['objective_names'],
                                         pareto_file=str(relative/'pareto.json'),scalar_score_role='diagnostic only')
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
    parser.add_argument('--population', type=int, default=None,
                        help='NSGA-II/CMA-ES/lq-CMA-ES population size (default 4); baseline counts toward the budget')
    parser.add_argument('--failure-penalty', type=float,
                        help='Explicit positive scalar penalty to continue after CFD failures; failed vectors are excluded from Pareto/HV results')
    parser.add_argument('--regime', choices=('single', 'multi'),
                        help='Benchmark regime; omitted: infer from each optimizer. Random supports both.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='JSON config for a single selected problem')
    parser.add_argument('--bo-initial-points',type=int,default=5,help='BO/MOBO initialization attempts, including baseline')
    parser.add_argument('--bo-candidates',type=int,default=2048,help='BO/MOBO acquisition candidate count')
    parser.add_argument('--mobo-mc-samples',type=int,default=128,help='BoTorch acquisition samples (power of two >= 16)')
    parser.add_argument('--mobo-reference-point',type=json.loads,
                        help='qLogNEHVI acquisition reference: JSON mapping objective names to minimized values')
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
        population_methods = {'nsga2', 'cmaes', 'lqcmaes'}
        if args.population is not None and not population_methods.intersection(args.optimizers):
            raise ValueError('--population requires selecting nsga2, cmaes or lqcmaes')
        if args.mobo_reference_point is not None and 'qlognehvi' not in args.optimizers:
            raise ValueError('--mobo-reference-point requires selecting qlognehvi')
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
        optimizer_options = {}
        for method in args.optimizers:
            settings = {}
            if args.population is not None and method in population_methods:
                settings['population'] = args.population
            if args.failure_penalty is not None:
                settings['failure_penalty'] = args.failure_penalty
            if method in ('bo', *MOBO_METHODS):
                settings.update(bo_initial_points=args.bo_initial_points,bo_candidates=args.bo_candidates)
            if method in MOBO_METHODS:
                settings.update(mobo_mc_samples=args.mobo_mc_samples,
                    mobo_reference_point=args.mobo_reference_point if method == 'qlognehvi' else None)
            if settings:
                optimizer_options[method] = settings
        arguments = dict(problems=args.problems, optimizers=args.optimizers, seeds=args.seeds,
                         budget=args.budget, problem_options=options, optimizer_options=optimizer_options,
                         with_ioh=not args.no_ioh, regime=args.regime)
        result = (plan_benchmark(**arguments) if args.dry_run
                  else run_benchmark(**arguments, output=args.output))
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(1, f'flowfield-benchmark: {exc}\n')
    print(json.dumps(result, indent=2))
    return 0
