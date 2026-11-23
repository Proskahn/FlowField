"""Suite entry point: discovery, one evaluation, or explicit tool dispatch."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .inspection import describe_problem
from .problem import CFDProblem
from .registry import create_problem, list_problems, problem_info


_DISPATCH = {
    'optimize': 'flowfield.optimization',
    'benchmark': 'flowfield.benchmarking.runner',
    'plot': 'flowfield.benchmarking.plotting',
    'check': 'flowfield.stability.cli',
    'p000': 'flowfield.problems.p000.cli',
}


def _problem_args(parser):
    parser.add_argument('--problem', required=True, choices=list_problems())
    _config_args(parser)


def _config_args(parser):
    parser.add_argument('--config', type=Path, help='Validated problem configuration JSON')
    data = parser.add_mutually_exclusive_group()
    data.add_argument('--dataset', type=Path, help='Explicit dataset for a data-backed problem')
    data.add_argument('--model', type=Path, help='Explicit trained surrogate for a data-backed problem')


def _design_args(parser):
    design = parser.add_mutually_exclusive_group()
    design.add_argument('--x', type=float, nargs='+', help='Design vector in the problem\'s declared bounds')
    design.add_argument('--design', type=Path, help='JSON array of design variables; useful for topology fields')


def _runtime_args(parser):
    parser.add_argument('--backend', choices=('auto', 'native', 'docker'))
    parser.add_argument('--image', help='Docker image; defaults to the problem runtime')
    parser.add_argument('--timeout', type=float, help='Seconds per solver command; defaults to the problem runtime')


def _options(args):
    return {key: getattr(args, key) for key in ('config', 'dataset', 'model')
            if getattr(args, key, None) is not None}


def _design(args, problem):
    if args.design is not None:
        values = json.loads(args.design.read_text(encoding='utf-8'))
        if not isinstance(values, list):
            raise ValueError('--design must contain a JSON array')
        return values
    return args.x if args.x is not None else problem.initial_design()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Delegate complete argument parsing to each concern's own CLI.
    if argv and argv[0] in _DISPATCH:
        from importlib import import_module
        return import_module(_DISPATCH[argv[0]]).main(argv[1:])
    if argv and argv[0] == 'search':
        from .optimization import main as optimize_main
        return optimize_main(['--optimizer', 'random', *argv[1:]])
    # Preserve existing physical-unit P000 invocations. Generic evaluation uses
    # an explicit problem ID; no ambiguity between normalized and millimetre inputs.
    if argv and (argv[0] in ('reference', 'smoke') or (
            argv[0] in ('evaluate', 'generate')
            and not any(word == '--problem' or word.startswith('--problem=') for word in argv))):
        from .problems.p000.cli import main as p000_main
        return p000_main(argv)

    parser = argparse.ArgumentParser(description='FlowField benchmark suite: discover, inspect and evaluate problems')
    sub = parser.add_subparsers(dest='command', required=True)
    listing = sub.add_parser('list', help='Discover benchmarks without loading simulation implementations')
    listing.add_argument('--details', action='store_true', help='Include descriptions, objectives and requirements')
    listing.add_argument('--level', type=int, choices=(1, 2, 3, 4))
    listing.add_argument('--variable-type', choices=('real', 'integer'))
    listing.add_argument('--kind', choices=('cfd', 'data', 'custom'))
    description = sub.add_parser('describe', help='Inspect configured bounds, baseline and identity without CFD')
    description.add_argument('problem', choices=list_problems())
    _config_args(description)
    for name, help_text in (
            ('decode', 'Decode a design without files or simulation'),
            ('evaluate', 'Evaluate one design; omit the vector to use the problem baseline'),
            ('generate', 'Generate a standalone CFD case without executing it')):
        cmd = sub.add_parser(name, help=help_text)
        _problem_args(cmd)
        _design_args(cmd)
        if name == 'evaluate':
            cmd.add_argument('--output', type=Path, required=True, help='Root for unique per-evaluation case folders')
            _runtime_args(cmd)
        if name == 'generate':
            cmd.add_argument('--case', type=Path, required=True, help='New case directory')
    for name in ('run', 'extract'):
        cmd = sub.add_parser(name, help='Execute a generated case once' if name == 'run' else 'Extract saved results without CFD')
        cmd.add_argument('case', type=Path)
        cmd.add_argument('--problem', choices=list_problems(), help='Defaults to the problem in case.json')
        if name == 'run':
            _runtime_args(cmd)
    for name in (*_DISPATCH, 'search'):
        sub.add_parser(name, help=f'Delegate to the {name} command (use {name} --help)')

    args = parser.parse_args(argv)
    try:
        if args.command == 'list':
            ids = list_problems(level=args.level, variable_type=args.variable_type, evaluation_kind=args.kind)
            result = {'problems': [problem_info(name).to_dict() for name in ids] if args.details else list(ids)}
        elif args.command == 'describe':
            result = describe_problem(args.problem, **_options(args))
        else:
            identifier = args.problem
            options = _options(args)
            if args.command in ('run', 'extract'):
                manifest = json.loads((args.case / 'case.json').read_text(encoding='utf-8'))
                identifier = identifier or manifest['problem']
                # Existing cases own their configuration. Do not silently rebuild
                # a default instance with a different design grid or calibration.
                options['config'] = manifest['config']
            if args.command == 'evaluate':
                options['output_root'] = args.output
            problem = create_problem(identifier, **options)
            if args.command in ('generate', 'run', 'extract') and not isinstance(problem, CFDProblem):
                raise ValueError(f'{identifier} does not provide standalone CFD cases')
            overrides = {key: getattr(args, arg) for arg, key in (
                ('backend', 'backend'), ('image', 'image'), ('timeout', 'timeout_s'))
                if getattr(args, arg, None) is not None}
            if overrides:
                if not isinstance(problem, CFDProblem):
                    raise ValueError(f'{identifier} does not use an OpenFOAM runtime')
                problem.runtime = replace(problem.runtime, **overrides)
            if args.command == 'decode':
                result = problem.decode(_design(args, problem))
            elif args.command == 'evaluate':
                design = _design(args, problem)
                problem.validate_design(design)
                problem.preflight()
                problem(design)
                result = problem.last_result
            elif args.command == 'generate':
                result = {'case_directory': str(problem.write_case(_design(args, problem), args.case))}
            elif args.command == 'run':
                problem.preflight()
                result = problem.run_case(args.case)
            else:
                result = problem.extract_results(args.case)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, KeyError) as exc:
        print(f'flowfield: {exc}', file=sys.stderr)
        return 1
    return 0


def smoke(*args, **kwargs):
    """Compatibility for the former top-level P000 smoke helper."""
    from .problems.p000.cli import smoke as run_smoke
    return run_smoke(*args, **kwargs)
