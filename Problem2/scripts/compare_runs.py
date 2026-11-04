#!/usr/bin/env python3
"""Compare converged serial/MPI solutions after reconstructing parallel fields.

Runs conservation checks independently, then compares complete cell fields and
reported integral quantities. Requires the same geometry, mesh and parameters.
"""
import argparse
import json
import math
from pathlib import Path
import re

from analyze import summarize
from verify_results import verify, values


def fields(case, iteration, count):
    directory = case / f'{iteration:g}'
    result = {name: values((directory / name).read_text(), 'internalField', count)
              for name in ('p', 'T', 'Cgas', 'phiS', 'phiE', 'iVol', 'heatSource')}
    text = (directory / 'U').read_text()
    match = re.search(r'internalField\s+uniform\s+\(([^)]+)\);', text)
    if match:
        velocity = [float(v) for v in match.group(1).split()] * count
    else:
        match = re.search(r'internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;', text, re.S)
        if not match or int(match.group(1)) != count:
            raise ValueError(f'Cannot read velocity in {directory}')
        velocity = [float(v) for v in re.findall(r'[-+\d.eE]+', match.group(2))]
    if len(velocity) != 3 * count:
        raise ValueError(f'Truncated velocity in {directory}')
    result['U'] = velocity
    if not all(math.isfinite(v) for field in result.values() for v in field):
        raise ValueError(f'Nonfinite saved field in {directory}')
    return result


def read_run(case):
    metadata = json.loads((case / 'case.json').read_text())
    summary = summarize(case)
    checks = verify(case)
    status = json.loads((case / 'postProcessing/pem/status.json').read_text())
    if not summary['converged'] or not checks['passed']:
        raise ValueError(f'{case} has not passed convergence and conservation checks')
    if checks['iteration'] != summary['last_iteration']['iteration']:
        raise ValueError(f'{case} has stale reconstructed fields')
    return metadata, summary, checks, status, fields(case, checks['iteration'], metadata['cells'])


def compare(reference, candidate, tolerance):
    rm, rs, rc, _, rf = read_run(reference)
    cm, cs, cc, status, cf = read_run(candidate)
    for key in ('geometry', 'mesh_level', 'cells', 'parameters'):
        if rm[key] != cm[key]:
            raise ValueError(f'Incompatible {key}: {reference} vs {candidate}')
    errors = {}
    for name, values_ref in rf.items():
        absolute = max(abs(a-b) for a, b in zip(values_ref, cf[name]))
        scale = max(max(abs(v) for v in values_ref), 1e-12)
        errors[name] = {'max_absolute_difference': absolute, 'relative_Linf': absolute/scale}
    quantities = ('voltage_V', 'dp_anode_Pa', 'dp_cathode_Pa', 'Tmax_K', 'heat_W',
                  'current_anode_A', 'current_cathode_A', 'O2_out_mol_s', 'H2_out_mol_s')
    integrals = {key: abs(cs['last_iteration'][key]-rs['last_iteration'][key]) /
                 max(abs(rs['last_iteration'][key]), 1e-12) for key in quantities}
    maximum = max([v['relative_Linf'] for v in errors.values()] + list(integrals.values()))
    return {'reference': str(reference), 'candidate': str(candidate),
            'processes': status.get('processes', 1), 'passed': maximum <= tolerance,
            'relative_tolerance': tolerance, 'maximum_relative_difference': maximum,
            'fields': errors, 'integrals': integrals,
            'reference_conservation_checks': rc['checks'], 'candidate_conservation_checks': cc['checks']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidates', type=Path, nargs='+')
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.rtol) or args.rtol <= 0:
        parser.error('--rtol must be finite and positive')
    reports = [compare(args.reference, case, args.rtol) for case in args.candidates]
    result = {'passed': all(r['passed'] for r in reports), 'comparisons': reports}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    for report in reports:
        print(f"{report['candidate']}: {report['processes']} processes, passed={report['passed']}, "
              f"maximum relative difference={report['maximum_relative_difference']:.3g}")
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
