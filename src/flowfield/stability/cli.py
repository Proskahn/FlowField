"""Run baseline checks or inspect existing cases without modifying them."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from ..registry import list_problems
from ..runtime import DEFAULT_IMAGE, Runtime
from .audit import inspect_paths
from .report import overall, write_reports
from .runner import make_plan, run_suite


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Must be positive and finite')
    return number


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('Must be nonnegative and finite')
    return number


def mappings(values, *, read_json=False):
    result = {}
    for item in values:
        name, sep, path = item.partition('=')
        if not sep or not name or not path or name in result:
            raise ValueError('Use unique PROBLEM=FILE arguments, e.g. P006=configs/p006.json')
        p = Path(path)
        if not p.is_file():
            raise ValueError(f'File not found: {p}')
        result[name] = json.loads(p.read_text()) if read_json else p
    return result


def exit_code(status):
    return 0 if status in ('passed', 'not_applicable') else 1 if status == 'failed' else 2


def summary(report, output=None):
    for record in report['cases']:
        audit = record.get('audit', record)
        label = f'{record.get("problem", audit.get("problem", "?"))}/{record.get("experiment", "inspect")}'
        step = audit.get('last_step')
        print(f'{label}: {audit["status"]}' + (f' at step {step:g}' if step is not None else '')
              + (f' ({audit["stage"]})' if audit.get('stage') else ''), flush=True)
        details = [c['name'] + ': ' + c['message'] for c in audit.get('checks', []) if c['status'] == 'failed']
        if audit.get('error') or audit.get('reason'):
            details.insert(0, str(audit.get('error', audit.get('reason'))))
        for message in details[:4]:
            print('  ' + message, flush=True)
    for comparison in report.get('comparisons', []):
        print(f'{comparison["problem"]}/{comparison["study"]} comparison: {comparison["status"]}', flush=True)
    print(f'Overall: {report["status"]}', flush=True)
    if output:
        print(f'Reports: {Path(output).resolve()}/report.html (also report.json and summary.csv)', flush=True)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Run fresh baseline cases and optional numerical sensitivity studies')
    run.add_argument('--problem', nargs='+', default=['all'], help='Registered identifiers or all; default: all')
    run.add_argument('--output', required=True, type=Path, help='New report/case directory, or identical suite with --resume')
    run.add_argument('--config', action='append', default=[], metavar='PROBLEM=FILE', help='JSON configuration for a selected problem')
    run.add_argument('--design', action='append', default=[], metavar='PROBLEM=FILE', help='JSON design-vector array; default: registered initial design')
    run.add_argument('--study', nargs='+', default=[], choices=('repeat', 'mesh', 'numerics', 'parallel', 'all'))
    run.add_argument('--processes', type=int, help='MPI ranks for P006 only; default: its configuration (2)')
    run.add_argument('--backend', choices=('auto', 'native', 'docker'), default='auto')
    run.add_argument('--image', default=DEFAULT_IMAGE)
    run.add_argument('--timeout', type=positive, default=14400, help='Seconds per runtime command; not a whole-suite deadline')
    run.add_argument('--progress-interval', type=positive, default=30, help='Seconds between progress messages')
    run.add_argument('--rtol', type=nonnegative, default=.02, help='Relative tolerance for physical-output comparisons (default 2%%)')
    run.add_argument('--atol', type=nonnegative, default=1e-10, help='Absolute tolerance in each reported observable\'s units')
    run.add_argument('--resume', action='store_true', help='Inspect started/finished jobs and execute only unstarted jobs; never restart an old solver case')
    run.add_argument('--dry-run', action='store_true', help='Print complete plan, without creating files or launching CFD')
    inspect = sub.add_parser('inspect', help='Read existing cases, including unfinished runs; never launch CFD')
    inspect.add_argument('--case', nargs='+', required=True, type=Path, help='Case directory or root to search recursively for case.json')
    inspect.add_argument('--output', type=Path, help='New directory for reports, outside inspected cases')
    inspect.add_argument('--watch', type=positive, metavar='SECONDS', help='Refresh until all runs end; Ctrl+C stops watching only')
    for command in (run, inspect):
        command.add_argument('--max-field-mib', type=positive, default=128, help='Per-field decompressed read limit; larger/binary fields are inconclusive')
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        max_bytes = max(1, int(args.max_field_mib * 1024 * 1024))
        if args.command == 'run':
            if 'all' in args.problem and args.problem != ['all']:
                raise ValueError('Use --problem all alone, or list problem identifiers')
            names = list(list_problems()) if args.problem == ['all'] else args.problem
            studies = ['repeat', 'mesh', 'numerics', 'parallel'] if 'all' in args.study else list(dict.fromkeys(args.study))
            plan = make_plan(names, mappings(args.config), studies, mappings(args.design, read_json=True), args.processes)
            if args.dry_run:
                print(json.dumps(plan, indent=2, allow_nan=False))
                return 0
            print(f'Planned {plan["case_count"]} CFD cases, sequentially. No optimization.', flush=True)
            report = run_suite(plan, args.output, runtime=Runtime(args.backend, args.image, args.timeout),
                resume=args.resume, progress_interval=args.progress_interval, rtol=args.rtol, atol=args.atol,
                max_field_bytes=max_bytes)
            summary(report, args.output)
            return exit_code(report['status'])
        # A report must not mutate an inspected case. Reject ancestors containing
        # case.json, including if an as-yet nonexistent subdirectory was chosen.
        if args.output:
            out = args.output.resolve()
            if any((p / 'case.json').is_file() for p in (out, *out.parents)):
                raise ValueError('Inspection reports must be outside solver case directories')
            out.mkdir(parents=True, exist_ok=False)
        previous = None
        while True:
            cases = inspect_paths(args.case, max_field_bytes=max_bytes)
            report = dict(schema_version=1, inspected_utc=datetime.now(timezone.utc).isoformat(),
                status=overall(cases), cases=[dict(problem=a.get('problem', '?'), experiment='inspect', audit=a) for a in cases],
                comparisons=[], coverage=dict(mode='read-only inspection', optimization_run=False,
                    process_liveness_verified=False, mesh_independence_established=False))
            if args.output:
                write_reports(args.output, report)
            signature = [(c['status'], c.get('last_step'), c.get('stage')) for c in cases]
            if previous != signature:
                summary(report, args.output)
                previous = signature
            if not args.watch or all(c.get('execution_state') in ('finished_record', 'ended_log', 'fatal_log')
                                     or c['status'] in ('incompatible', 'not_applicable') for c in cases):
                return exit_code(report['status'])
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print('\nInterrupted; reports and case logs are retained. Inspect mode never stops a solver.', flush=True)
        return 130
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        print(f'Checker error: {exc}', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
