"""Shell-safe entry point for P002 generation, evaluation and smoke checks."""

import argparse
import json
from pathlib import Path

from ....evaluation import write_json
from ....ioh_adapter import analyzer_logger, make_ioh_problem
from ....runtime import DEFAULT_IMAGE, Runtime, SimulationError
from . import BASELINE_X, PAPER_OPTIMUM_X, P002Problem, PEMConfig
from .case import generate_case
MESHES = ('coarse','medium','fine')
from .evaluation import run_case


def smoke(output, config, runtime, with_ioh=False, compare=False):
    from contextlib import nullcontext
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    problem = P002Problem(output/'cases', config=config, runtime=runtime)
    designs = [('baseline', BASELINE_X)]
    if compare:
        designs.append(('paper_table5', PAPER_OPTIMUM_X))
    summary = dict(status='failed', problem='P002', ioh_enabled=with_ioh,
                   paper_validated=False, scalarization=config.scalarization,
                   config_sha256=config.fingerprint, evaluations=[])
    try:
        objective = make_ioh_problem(problem) if with_ioh else problem
        context = analyzer_logger(objective, output/'ioh', algorithm_name='P002-design-check',
            algorithm_info='Baseline and optional Table 5 design; no optimization') if with_ioh else nullcontext()
        with context:
            for label,x in designs:
                value = float(objective(x))
                result = problem.last_result
                record = dict(label=label, design=list(x), objective=value,
                              objectives=result['objectives'], metrics=result['metrics'],
                              case_directory=result['case_directory'], mesh_validation=result['mesh_validation'])
                summary['evaluations'].append(record)
                write_json(output/(label+'.json'),result)
        if with_ioh:
            if objective.state.evaluations != len(designs):
                raise SimulationError('Unexpected IOH evaluation count')
            files = list((output/'ioh').glob('IOHprofiler*.json'))+list((output/'ioh').rglob('*.dat'))
            if len(files) < 2:
                raise SimulationError('Missing IOH Analyzer artifacts')
            summary.update(ioh_evaluations=int(objective.state.evaluations),
                           ioh_files=[str(p.relative_to(output)) for p in files])
        if compare:
            first,last = [v['objectives'] for v in summary['evaluations']]
            summary['relative_change_percent'] = {k:100*(last[k]/first[k]-1) for k in first}
        summary['status'] = 'passed'
    except Exception as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(output/'smoke.json',summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='P002 provisional coupled PEM electrolyzer reconstruction (not paper-validated)')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('decode', 'generate', 'evaluate', 'smoke', 'compare'):
        p = sub.add_parser(command)
        p.add_argument('--config', type=Path)
        p.add_argument('--mesh', choices=MESHES)
        if command not in ('smoke','compare'):
            p.add_argument('--x', type=float, nargs=7, default=BASELINE_X,
                metavar=('W1','W2','W3','W4','W5','Ha','Hc'), help='Normalized coordinates in [0,1], Table 5 order')
        if command == 'generate':
            p.add_argument('--case', type=Path, required=True)
        if command in ('evaluate', 'smoke', 'compare'):
            p.add_argument('--output', type=Path, required=True)
            _runtime_arguments(p)
        if command in ('smoke','compare'):
            p.add_argument('--ioh', action='store_true')
    p = sub.add_parser('run')
    p.add_argument('case', type=Path)
    _runtime_arguments(p)
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            result = run_case(args.case, Runtime(args.backend, args.image, args.timeout))
        else:
            values = json.loads(args.config.read_text()) if args.config else {}
            if args.mesh:
                values['mesh'] = args.mesh
            config = PEMConfig(**values)
            problem = P002Problem(config=config)
            if args.command == 'decode':
                result = problem.decode(args.x)
            elif args.command == 'generate':
                result = {'case_directory':str(generate_case(args.case, problem.decode(args.x), config)),
                          'status':'generated_not_run', 'paper_validated':False}
            else:
                runtime = Runtime(args.backend, args.image, args.timeout)
                if args.command in ('smoke','compare'):
                    result = smoke(args.output, config, runtime, args.ioh, compare=args.command == 'compare')
                else:
                    result = P002Problem(args.output, config=config, runtime=runtime).evaluate_raw(args.x)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(1, f'flowfield-p002: {exc}\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def _runtime_arguments(parser):
    parser.add_argument('--backend', choices=('auto','native','docker'), default='auto')
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--timeout', type=float, default=1800, help='Seconds per meshing/solver command')


if __name__ == '__main__':
    raise SystemExit(main())
