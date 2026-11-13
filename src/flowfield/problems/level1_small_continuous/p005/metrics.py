"""Strict transient verification and right-censored time objective."""
import csv
import math
import re
from pathlib import Path
from ....runtime import SimulationError


def extract_metrics(case,config):
    case=Path(case);log=(case/'log.interFoam').read_text()
    if not re.search(r'^End\s*$',log,re.M) or re.search(r'FOAM FATAL|signal SIGFPE|Segmentation fault',log,re.I):
        raise SimulationError('interFoam did not finish normally')
    with (case/'postProcessing/detachment.csv').open() as stream:
        rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    if not rows or any(not math.isfinite(v) for row in rows for v in row.values()):
        raise SimulationError('Missing/nonfinite detachment diagnostics')
    if any(b['time_s']<=a['time_s'] for a,b in zip(rows,rows[1:])):
        raise SimulationError('Nonmonotonic diagnostic time')
    final=rows[-1]
    for row in rows:
        if (row['min_alpha_water'] < -config.alpha_tolerance or row['max_alpha_water'] > 1+config.alpha_tolerance
            or row['max_gas_balance_relative']>config.gas_balance_tolerance
            or row['max_continuity_relative']>config.continuity_tolerance):
            raise SimulationError('Phase bounds or volume conservation failed')
    detached=final['confirmed']==1
    if final['injected_volume_m3']<=0 or final['gas_volume_m3']<=0:
        raise SimulationError('No oxygen injected or retained')
    expected=config.oxygen_velocity_m_s*config.gas_inlet_side_m**2*final['time_s']
    if abs(final['injected_volume_m3']/expected-1)>.001:
        raise SimulationError('Oxygen inlet flow differs from specified velocity and area')
    if detached:
        t=final['event_time_s'];lo=final['event_bracket_start_s']
        if not 0<=lo<t<=final['time_s'] or final['time_s']-t<config.persistence_s*(1-1e-8):
            raise SimulationError('Invalid detachment event timing')
        objective=t
    else:
        if abs(final['time_s']-config.end_time_s)>max(1e-10,config.end_time_s*1e-7):
            raise SimulationError('Simulation ended before event or observation horizon')
        t=None;lo=None;objective=2*config.end_time_s
    return dict(status='ok',objective=objective,objectives={'censored_detachment_score_s':objective},
        constraints={'detachment_observed':-1. if detached else 1.},
        metrics=dict(detached=detached,detachment_time_s=t,detachment_bracket_s=[lo,t] if detached else None,
            right_censored=not detached,detachment_time_lower_bound_s=config.end_time_s if not detached else None,
            final_time_s=final['time_s'],diagnostic_samples=len(rows),**{k:final[k] for k in (
                'gas_volume_m3','injected_volume_m3','max_gas_balance_relative','max_continuity_relative',
                'min_alpha_water','max_alpha_water')}))


def verify_mesh_log(log, expected):
    if 'Mesh OK.' not in log or 'Failed ' in log:
        raise SimulationError('P005 mesh check failed')
    if 'Mesh has 3 geometric' not in log or 'Number of regions: 1 (OK).' not in log:
        raise SimulationError('P005 requires one connected three-dimensional fluid region')
    volume=re.search(r'Total volume = ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)',log)
    cells=re.search(r'\bcells:\s+(\d+)',log)
    inlet=re.search(r'^\s*gasInlet\s+(\d+)\s+',log,re.M)
    if not volume or not cells or not inlet:
        raise SimulationError('Missing mesh counts/volume/patch diagnostics')
    volume=float(volume.group(1))
    if (not math.isfinite(volume) or abs(volume/expected['expected_volume_m3']-1)>1e-8
        or int(cells.group(1))!=expected['cells'] or int(inlet.group(1))!=expected['gas_inlet_faces']):
        raise SimulationError('Generated mesh differs from physical geometry or expected cell/patch counts')
    return dict(all_checks_passed=True,volume_m3=volume,cells=int(cells.group(1)),
                gas_inlet_faces=int(inlet.group(1)))
