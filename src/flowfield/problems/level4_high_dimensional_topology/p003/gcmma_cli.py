"""CLI for explicit, full-grid P003 GCMMA continuation campaigns."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from ....runtime import Runtime
from .config import TopologyConfig
from .gcmma import ContinuationOptions, plan, run_continuation
from .scalability import add_dimension_argument, config_from_dimension, grid_summary


def add_arguments(parser):
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    add_dimension_argument(parser)
    parser.add_argument('--mesh', choices=('coarse', 'medium', 'fine'))
    parser.add_argument('--stages', type=int, nargs='+', help='Default: 1 2 3 above 4 Pa, otherwise 0 1 2 3')
    parser.add_argument('--iterations-per-stage', type=int, help='Accepted outer iterations per stage (default 60)')
    parser.add_argument('--fd-step', type=float, help='Raw nodal finite-difference step (default 0.0001)')
    parser.add_argument('--fd-scheme', choices=('forward', 'central'), help='Default forward: N CFD probes per gradient')
    parser.add_argument('--move', type=float, help='Maximum raw nodal change per outer iteration (default 0.01)')
    parser.add_argument('--max-inner', type=int, help='Maximum CFD trials per outer iteration (default 15)')
    parser.add_argument('--check-gradients', action=argparse.BooleanOptionalAction, default=None,
                        help='Check the first gradient of every stage against three directional differences at h and h/2')
    parser.add_argument('--gradient-rtol', type=float, help='Directional derivative relative tolerance (default 0.1)')
    parser.add_argument('--gradient-atol', type=float, help='Directional derivative absolute tolerance (default 1e-5)')
    parser.add_argument('--seed', type=int, help='Run label (default 42); uniform-start GCMMA is deterministic')
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--max-evaluations', type=int, help='Total CFD attempt limit, including calibration and probes')
    budget.add_argument('--extend-budget', type=int, help='With --resume, increase the total CFD attempt limit')
    parser.add_argument('--workers', type=int, default=1, help='Parallel independent serial CFD probes (default 1)')
    parser.add_argument('--resume', action='store_true', help='Replay this GCMMA campaign and reuse completed CFD requests')
    parser.add_argument('--retry-interrupted', action='store_true', help='Explicitly rerun unfinished CFD requests as new counted attempts')
    parser.add_argument('--retry-failed', action='store_true', help='After fixing a runtime failure, retry failed CFD requests as new counted attempts')
    parser.add_argument('--discard-probe-cases', action=argparse.BooleanOptionalAction, default=None,
                        help='Delete successful gradient/check CFD case directories after saving their results; retain baseline, trial and failed cases')
    parser.add_argument('--dry-run', action='store_true', help='Print the schedule and CFD cost without creating files or running CFD')


def execute(args):
    if args.dimension is not None and args.config is not None:
        raise ValueError('Use --dimension or --config, not both')
    saved = None
    if args.resume:
        path = args.output/'gcmma-run.json'
        if not path.is_file():
            raise ValueError('Missing gcmma-run.json; use a new directory for GCMMA, not --resume on TuRBO')
        saved = json.loads(path.read_text())
    values = json.loads(args.config.read_text()) if args.config else (saved['config'] if saved else {})
    config = (config_from_dimension(args.dimension,mesh=args.mesh)
              if args.dimension is not None and saved is None else TopologyConfig(**values))
    if saved and args.dimension is not None and args.dimension != config.dimension:
        raise ValueError('Cannot resume GCMMA with changed dimension')
    if args.mesh:
        config = replace(config, mesh=args.mesh)
    settings = dict(saved['options']) if saved else dict(stages=(1, 2, 3) if config.inlet_pressure_pa > 4 else (0, 1, 2, 3))
    for name in asdict(ContinuationOptions()):
        value = getattr(args, name, None)
        if value is not None:
            settings[name] = value
    settings['stages'] = tuple(settings['stages'])
    options = ContinuationOptions(**settings)
    estimate = plan(config, options)
    estimate.update(grid_summary(config))
    budget = args.max_evaluations
    if args.extend_budget is not None:
        if not args.resume:
            raise ValueError('--extend-budget requires --resume')
        if args.extend_budget < 1:
            raise ValueError('--extend-budget must be positive')
        budget = saved['max_evaluations']+args.extend_budget
    if budget is None:
        budget = saved['max_evaluations'] if saved else estimate['maximum_evaluations']
    seed = args.seed if args.seed is not None else (saved['seed'] if saved else 42)
    if args.workers < 1 or budget < 1:
        raise ValueError('--workers and the evaluation budget must be positive')
    if (args.retry_interrupted or args.retry_failed) and not args.resume:
        raise ValueError('--retry-interrupted and --retry-failed require --resume')
    discard = args.discard_probe_cases
    if discard is None:
        discard = saved['discard_probe_cases'] if saved else False
    if args.dry_run:
        if saved:
            for name, current in (('config', config.to_dict()), ('options', asdict(options)),
                                  ('seed', seed), ('discard_probe_cases', discard)):
                if json.loads(json.dumps(current)) != saved[name]:
                    raise ValueError(f'Cannot resume GCMMA with changed {name}')
            if budget < saved['max_evaluations']:
                raise ValueError('Cannot decrease the saved CFD evaluation budget')
        print(json.dumps(dict(estimate, max_evaluations=budget, workers=args.workers,
                              discard_probe_cases=discard, output=str(args.output.resolve()),
                              options=asdict(options)), indent=2))
        return 0

    def progress(attempt, phase, iteration, record):
        if phase != 'gradient' or attempt % 25 == 0:
            stage = record['result'].get('config', {}).get('stage', '?')
            print(f'CFD {attempt}/{budget} stage={stage} iteration={iteration} phase={phase} '
                  f'objective={record["objective"]:.9g}', flush=True)

    result = run_continuation(args.output, config=config, options=options,
        runtime=Runtime(args.backend, args.image, args.timeout), seed=seed,
        max_evaluations=budget, workers=args.workers, resume=args.resume,
        retry_interrupted=args.retry_interrupted, retry_failed=args.retry_failed,
        discard_probe_cases=discard, progress=progress)
    print(json.dumps({key: value for key, value in result.items() if key != 'final_design'}, indent=2))
    return 0 if result['status'] == 'completed_schedule' else 2
