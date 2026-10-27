#!/usr/bin/env python3
"""Summarize an actual solver run and check Faraday/outlet conservation."""
import argparse
import csv
import json
import math
from pathlib import Path

F = 96485.33212
R = 8.314462618


def summarize(case):
    files = list((case/'postProcessing/pem').glob('history_*.csv'))
    if not files:
        raise ValueError(f'No solver history in {case}')
    rows = {}
    for path in sorted(files, key=lambda p: float(p.stem.removeprefix('history_'))):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                if None in row.values():
                    continue
                numeric = {k: float(v) for k, v in row.items()}
                if not all(math.isfinite(v) for v in numeric.values()):
                    raise ValueError(f'Nonfinite value in {path}')
                rows[numeric['iteration']] = numeric
    if not rows:
        raise ValueError('No complete history rows yet')
    last = rows[max(rows)]
    metadata = json.loads((case/'case.json').read_text())
    status_path = case/'postProcessing/pem/status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'converged': False}
    p = metadata['parameters']['model']
    target = metadata['current_A']
    checks = {
        'anode_current_relative_error': abs(last['current_anode_A']/target-1),
        'cathode_current_relative_error': abs(last['current_cathode_A']/target-1),
        'oxygen_out_relative_error': abs(last['O2_out_mol_s']/(target/(4*F))-1),
        'hydrogen_out_relative_error': abs(last['H2_out_mol_s']/(target/(2*F))-1),
        'mass_relative_error': last['mass_error'],
        'bounded_cells': int(last['bounded_cells']),
    }
    passed = (status.get('converged', False) and
              status.get('iteration') == last['iteration'] and
              max(v for k,v in checks.items() if k != 'bounded_cells') < 0.001 and
              checks['bounded_cells'] == 0)
    result = {
        'case': str(case), 'converged': passed, 'last_iteration': last,
        'checks': checks, 'mesh_cells': metadata['cells'],
        'expected_O2_mol_s': target/(4*F), 'expected_H2_mol_s': target/(2*F),
        'expected_O2_mL_min': target/(4*F)*R*p['inletTemperature']/p['operatingPressure']*60e6,
        'expected_H2_mL_min': target/(2*F)*R*p['inletTemperature']/p['operatingPressure']*60e6,
        'electrical_power_W': target*last['voltage_V'],
        'pump_power_W': p['inletFlow_mL_min']*1e-6/60*(last['dp_anode_Pa']+last['dp_cathode_Pa']),
        'validation_status': 'Numerical reconstruction with estimated closures; not calibrated to the paper.'
    }
    (case/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path)
    args=parser.parse_args()
    result=summarize(args.case.resolve())
    last=result['last_iteration']
    print(f"Converged: {result['converged']} | iteration {last['iteration']:g} | {result['mesh_cells']:,} cells")
    print(f"Cell voltage: {last['voltage_V']:.6f} V")
    print(f"Pressure drop, anode/cathode: {last['dp_anode_Pa']:.6g} / {last['dp_cathode_Pa']:.6g} Pa")
    print(f"Maximum temperature: {last['Tmax_K']:.6f} K")
    print(f"O2/H2 production: {result['expected_O2_mL_min']:.6g} / {result['expected_H2_mL_min']:.6g} mL/min at inlet T,p")
    print('Checks:',json.dumps(result['checks']))
    print('Saved',args.case.resolve()/'summary.json')
    return 0 if result['converged'] else 2


if __name__=='__main__':
    raise SystemExit(main())
