#!/usr/bin/env python3
"""Prepare a short same-geometry P006 continuation, or summarize its field changes.

Preparation does not run CFD. Reporting does not qualify a case. The solver's
original field-change scales, physical balances and acceptance remain enforced.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

from flowfield.problems.level1_small_continuous.p006 import ManifoldConfig, P006Problem
from flowfield.problems.level1_small_continuous.p006.case import generate_case
from flowfield.stability.diagnostics import history

FIELDS=('T','sg','p','Ul','Ug','lambda')
STAGE=('momentum_fraction','thermal_active','applied_voltage_V')


def prepare(source, output, iterations=200):
    """Copy a completed guess into new inputs; retain every other configuration value."""
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output or source in output.parents:
        raise ValueError('Diagnostic output must be outside the source case')
    manifest=json.loads((source/'case.json').read_text())
    config=replace(ManifoldConfig(**manifest['config']),iterations=iterations)
    problem=P006Problem(config=config)
    geometry=problem.decode(manifest['design_geometry']['normalized_design'])
    # Existing loader checks completion, input hashes, reconstructed fields and
    # stage evidence before creating output. It never inherits qualification.
    case=generate_case(output,geometry,config,initial_state=source)
    return dict(case_directory=str(case),iterations=config.iterations,
                source_case=str(source),field_tolerance=config.field_tolerance,
                balance_tolerance=config.balance_tolerance,
                note='Prepared only. Run this new case, then report its history.')


def summarize(case, last=100):
    if type(last) is not int or last<1:raise ValueError('last must be a positive integer')
    case=Path(case).resolve();manifest=json.loads((case/'case.json').read_text())
    if manifest.get('problem')!='P006':raise ValueError('Expected a P006 case')
    config=ManifoldConfig(**manifest['config'])
    if config.fingerprint!=manifest['config_sha256']:raise ValueError('Configuration fingerprint differs')
    result_path=case/'result.json'
    result=json.loads(result_path.read_text()) if result_path.exists() else {}
    parsed=history(case/'postProcessing/p006/history.csv',unfinished=not result_path.exists(),keep=last)
    if parsed['errors']:raise ValueError('; '.join(parsed['errors']))
    rows=parsed['recent']
    if not rows:raise ValueError('No complete diagnostic history samples')
    required={'field_change',*STAGE,*(f'field_change_{name}' for name in FIELDS)}
    if not required.issubset(rows[-1]):
        raise ValueError('Per-field diagnostics are missing; prepare and run a fresh case with the updated solver')
    # Do not mix heat-off or partial-momentum samples with the final stage.
    start=len(rows)-1
    while start>0 and all(rows[start-1][key]==rows[-1][key] for key in STAGE):start-=1
    rows=rows[start:]
    dominant={name:0 for name in FIELDS}
    streak=longest=0
    for row in rows:
        values={name:row[f'field_change_{name}'] for name in FIELDS}
        if min(values.values())<0 or not math.isclose(max(values.values()),row['field_change'],rel_tol=1e-9,abs_tol=1e-15):
            raise ValueError('Per-field changes disagree with the acceptance field_change')
        dominant[max(values,key=values.get)]+=1
        streak=streak+1 if row['field_change']<config.field_tolerance else 0
        longest=max(longest,streak)
    def stats(key):
        samples=[r[key] for r in rows]
        return dict(last=samples[-1],minimum=min(samples),maximum=max(samples),
                    samples_at_or_above_limit=sum(v>=config.field_tolerance for v in samples))
    checks=('charge_residual','energy_balance','liquid_balance','gas_balance',
            'local_liquid_balance','local_gas_balance','liquid_momentum_residual',
            'gas_momentum_residual','bounded_cells')
    return dict(case_directory=str(case),execution_status=result.get('status','not_recorded'),
        assessment='Diagnostic only; not a convergence certificate.',
        total_history_samples=parsed['samples'],window=dict(first=rows[0]['iteration'],
            last=rows[-1]['iteration'],samples=len(rows),stage={k:rows[-1][k] for k in STAGE}),
        field_tolerance=config.field_tolerance,field_change=dict(stats('field_change'),
            longest_streak_below_limit=longest),
        components={name:dict(stats(f'field_change_{name}'),dominant_samples=dominant[name]) for name in FIELDS},
        final_checks={key:rows[-1][key] for key in checks},
        note='Changes are normalized block-solve updates after Anderson input; dominance identifies a blocker, not its cause.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_subparsers(dest='action',required=True)
    create=actions.add_parser('prepare',help='Generate new diagnostic inputs from a completed case')
    create.add_argument('source',type=Path);create.add_argument('output',type=Path)
    create.add_argument('--iterations',type=int,default=200)
    report=actions.add_parser('report',help='Summarize the final stage in the recent history')
    report.add_argument('case',type=Path);report.add_argument('--last',type=int,default=100)
    args=parser.parse_args(argv)
    try:
        result=(prepare(args.source,args.output,args.iterations) if args.action=='prepare'
                else summarize(args.case,args.last))
    except (OSError,ValueError,KeyError,TypeError) as exc:parser.error(str(exc))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
