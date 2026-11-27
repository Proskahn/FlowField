#!/usr/bin/env python3
"""Summarize P006 phase timings without double-counting MPI ranks.

Usage: .venv/bin/python scripts/profile_p006.py CASE [CASE ...] --first 11
Timings are inclusive: linear contains factor/matrix/precondition/matvec/
orthogonal, and assembly contains exchange. This report does not qualify CFD.
Run performance comparisons sequentially with the same CPU allocation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


NUMBER = r'[-+\d.eE]+'


def summarize(case, first=1, last=None):
    case = Path(case).resolve()
    manifest = json.loads((case/'case.json').read_text())
    processes = manifest['config']['processes']
    rows = {}
    clocks = {0: 0.0}
    iteration = None
    text = (case/'log.flowfieldP006Foam').read_text()
    for line in text.splitlines():
        match = re.match(r'^Iteration = (\d+)', line)
        if match:
            iteration = int(match[1])
        match = re.search(r'ClockTime = ('+NUMBER+r') s', line)
        if match and iteration is not None:
            clocks[iteration] = float(match[1])
        if 'PHASE_TIMING ' not in line:
            continue
        row = {k: float(v) for k, v in re.findall(r'(\w+)=('+NUMBER+r')', line)}
        step, rank = int(row['iteration']), int(row['rank'])
        if step < first or (last is not None and step > last):
            continue
        if rank in rows.setdefault(step, {}):
            raise ValueError(f'Duplicate timing: step {step}, rank {rank}')
        rows[step][rank] = row
    if not rows:
        raise ValueError('No PHASE_TIMING samples in the requested interval; '
                         'older PHASE_PROFILE timers omit the gas-mass correction')
    expected = set(range(processes))
    for step, ranks in rows.items():
        if set(ranks) != expected:
            raise ValueError(f'Incomplete timing at step {step}: ranks {sorted(ranks)}')
    # One critical-rank sample per outer step, never the sum of MPI rank times.
    selected = [max(rows[step].values(), key=lambda row: row['total_s'])
                for step in sorted(rows)]
    seconds = {key: sum(row[key] for row in selected)
               for key in selected[0] if key.endswith('_s')}
    per_rank = {str(rank): {key: sum(ranks[rank][key] for ranks in rows.values())
                           for key in seconds}
                for rank in range(processes)}
    counts = {key: int(sum(row[key] for row in selected))
              for key in selected[0] if key not in ('iteration', 'rank') and not key.endswith('_s')}
    outer = [clocks[step]-clocks[step-1] for step in rows
             if step in clocks and step-1 in clocks]
    status_path = case/'postProcessing/p006/status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    result_path = case/'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    return dict(case=str(case), cells=manifest['cells'], processes=processes,
                source_sha256=hashlib.sha256((case/'solver/FlowNetwork.H').read_bytes()).hexdigest(),
                selection=dict(first=min(rows), last=max(rows), samples=len(rows)),
                phase_seconds=seconds, per_rank_seconds=per_rank, counts=counts,
                phase_median_s=statistics.median(row['total_s'] for row in selected),
                outer_wall_s=sum(outer), outer_clock_samples=len(outer),
                completed=bool(re.search(r'^End\s*$', text, re.M)),
                solver_converged=status.get('converged'),
                execution_status=result.get('status', 'not_recorded'),
                qualification='Not assessed by this profiling report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path, nargs='+')
    parser.add_argument('--first', type=int, default=1)
    parser.add_argument('--last', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.first < 1 or (args.last is not None and args.last < args.first):
        parser.error('Expected 1 <= first <= last')
    report = dict(schema_version=1,
                  timing_note='Inclusive categories; one slowest-rank sample per step. '
                              'ClockTime is rounded by OpenFOAM. Compare matching stages '
                              'and CPU allocations; timing alone does not establish accuracy.',
                  cases=[summarize(case, args.first, args.last) for case in args.cases])
    rendered = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
