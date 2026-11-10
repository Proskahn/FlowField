"""Shell-safe entry point for P001 generation, evaluation and smoke checks."""

import argparse
import json
from pathlib import Path

from ....evaluation import write_json
from ....ioh_adapter import analyzer_logger, make_ioh_problem
from ....runtime import DEFAULT_IMAGE, Runtime, SimulationError
from . import BASELINE_X, P001Problem, PPFCConfig
from .case import generate_case
from .config import MESHES
from .evaluation import run_case


def smoke(output, config, runtime, with_ioh=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    problem = P001Problem(output/'cases', config=config, runtime=runtime)
    summary = {'status':'failed', 'problem':'P001', 'design':list(BASELINE_X),
               'ioh_enabled':with_ioh, 'paper_validated':False,
               'replication_status':'provisional_geometry_reconstruction'}
    try:
        if with_ioh:
            objective = make_ioh_problem(problem)
            with analyzer_logger(objective, output/'ioh', algorithm_name='P001-smoke',
                    algorithm_info='One reconstructed-geometry baseline; no optimization'):
                value = float(objective(BASELINE_X))
            if objective.state.evaluations != 1:
                raise SimulationError('IOH evaluation count must be exactly one')
            info_files = list((output/'ioh').glob('IOHprofiler*.json'))
            dat_files = list((output/'ioh').rglob('*.dat'))
            if not info_files or not dat_files:
                raise SimulationError('Missing IOH Analyzer artifacts')
            summary['ioh_files'] = [str(p.relative_to(output)) for p in info_files+dat_files]
            summary['ioh_evaluations'] = int(objective.state.evaluations)
        else:
            value = problem(BASELINE_X)
        result = problem.last_result
        summary.update(status='passed', objective=value, metrics=result['metrics'],
                       case_directory=result['case_directory'], config_sha256=config.fingerprint)
    except Exception as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(output/'smoke.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='P001 provisional alkaline PPFC reconstruction (not paper-validated)')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('decode', 'generate', 'evaluate', 'smoke'):
        p = sub.add_parser(command)
        p.add_argument('--config', type=Path)
        p.add_argument('--mesh', choices=MESHES)
        if command != 'smoke':
            p.add_argument('--x', type=float, nargs=6, default=BASELINE_X,
                metavar=('X11','X12','X2','X3','H11','H12'), help='Normalized coordinates in [0,1], in Table 6 order')
        if command == 'generate':
            p.add_argument('--case', type=Path, required=True)
        if command in ('evaluate', 'smoke'):
            p.add_argument('--output', type=Path, required=True)
            _runtime_arguments(p)
        if command == 'smoke':
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
            config = PPFCConfig(**values)
            problem = P001Problem(config=config)
            if args.command == 'decode':
                result = problem.decode(args.x)
            elif args.command == 'generate':
                result = {'case_directory':str(generate_case(args.case, problem.decode(args.x), config)),
                          'status':'generated_not_run', 'paper_validated':False}
            else:
                runtime = Runtime(args.backend, args.image, args.timeout)
                if args.command == 'smoke':
                    result = smoke(args.output, config, runtime, args.ioh)
                else:
                    result = P001Problem(args.output, config=config, runtime=runtime).evaluate_raw(args.x)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(1, f'flowfield-p001: {exc}\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def _runtime_arguments(parser):
    parser.add_argument('--backend', choices=('auto','native','docker'), default='auto')
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--timeout', type=float, default=1800, help='Seconds per meshing/solver command')


if __name__ == '__main__':
    raise SystemExit(main())
