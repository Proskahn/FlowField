"""Tapered PEMFC cases, reference designs, and direct CFD comparisons."""
import argparse
from dataclasses import replace
import json
from ....config import load_config
from ....runtime import Runtime
from . import P011Problem, TaperedConfig
from .config import BASELINE, REFERENCE_WIDTHS, normalized


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('generate', 'run', 'extract', 'evaluate', 'reference', 'compare'))
    parser.add_argument('--case', help='New destination or existing case for run/extract')
    parser.add_argument('--config')
    parser.add_argument('--mesh', choices=('coarse', 'medium', 'fine'))
    parser.add_argument('--voltage', type=float)
    design = parser.add_mutually_exclusive_group()
    design.add_argument('--x', type=float, nargs=4, help='Four normalized endpoint widths')
    design.add_argument('--widths-mm', type=float, nargs=4, help='Cathode inlet/outlet then anode inlet/outlet')
    design.add_argument('--design', choices=tuple(REFERENCE_WIDTHS))
    parser.add_argument('--backend', choices=('auto', 'native', 'docker'), default='auto')
    parser.add_argument('--timeout', type=float, default=14400)
    args = parser.parse_args(argv)
    if args.action != 'reference' and args.case is None:
        parser.error('--case is required for CFD and case operations')
    overrides = (args.config, args.mesh, args.voltage, args.x, args.widths_mm, args.design)
    if args.action in ('run', 'extract', 'reference') and any(v is not None for v in overrides):
        parser.error('run/extract use immutable recorded inputs; reference uses published values only')
    if args.action == 'compare' and any(v is not None for v in (args.x, args.widths_mm, args.design)):
        parser.error('compare uses the declared CFC, GA and IGA designs')
    try:
        if args.action == 'reference':
            from .reference import reference_summary
            result = reference_summary()
        else:
            config = load_config(TaperedConfig, args.config) if args.config else TaperedConfig()
            if args.mesh is not None:
                config = replace(config, mesh=args.mesh)
            if args.voltage is not None:
                config = replace(config, cell_voltage=args.voltage)
            runtime = Runtime(args.backend, timeout_s=args.timeout)
            problem = P011Problem(config=config, runtime=runtime, output_root=args.case)
            x = args.x if args.x is not None else normalized(
                args.widths_mm if args.widths_mm is not None else REFERENCE_WIDTHS.get(args.design, BASELINE))
            if args.action == 'generate':
                result = str(problem.write_case(x, args.case))
            elif args.action == 'run':
                result = problem.run_case(args.case)
            elif args.action == 'extract':
                result = problem.extract_results(args.case)
            elif args.action == 'evaluate':
                problem.preflight()
                runtime.resolve(problem.case_definition.commands)
                result = problem.evaluate_raw(x)
            else:
                from .study import comparison_study
                result = comparison_study(args.case, config, runtime)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'flowfield-p011: {exc}\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
