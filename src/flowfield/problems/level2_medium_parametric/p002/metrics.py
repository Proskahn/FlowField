"""Reject unconverged/stale output; preserve the paper's two physical objectives."""

import csv
import json
import math
from pathlib import Path
import re

from .config import scalar_objective
from .verification import verify, values
from .geometry import geometry as build_geometry

HISTORY_COLUMNS = ('iteration,voltage_V,current_anode_A,current_cathode_A,charge_residual,'
    'dp_anode_Pa,dp_cathode_Pa,Tmax_K,gas_volume_max,O2_source_mol_s,H2_source_mol_s,'
    'O2_out_mol_s,H2_out_mol_s,mass_error,gas_error,field_change,heat_W,bounded_cells').split(',')


def read_history(case):
    files = list((case/'postProcessing/pem').glob('history_*.csv'))
    if len(files) != 1 or files[0].stem != 'history_0':
        raise ValueError('Expected exactly one fresh history starting from zero')
    with files[0].open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != HISTORY_COLUMNS:
            raise ValueError('Unexpected P002 history schema')
        rows = []
        for record in reader:
            if set(record) != set(HISTORY_COLUMNS):
                raise ValueError('Malformed P002 history')
            row = {k:float(v) for k,v in record.items()}
            if not all(math.isfinite(v) for v in row.values()):
                raise ValueError('Nonfinite P002 history')
            if row['iteration'] != len(rows)+1:
                raise ValueError('History is not a consecutive fresh run')
            rows.append(row)
    if len(rows) < 5:
        raise ValueError('Need five consecutive converged iterations')
    return rows


def field_pressure_drops(case, iteration, manifest):
    """Independently area-average saved p in Pa at the physical inlet/outlet ports."""
    m = build_geometry(manifest['parameters']['geometry'], 'parallel', manifest['mesh_level'])
    poly = case/'constant/polyMesh'
    owner_text = (poly/'owner').read_text().split('}',1)[1]
    owner = [int(v) for v in re.search(r'\d+\s*\((.*?)\)',owner_text,re.S).group(1).split()]
    boundary = (poly/'boundary').read_text()
    ptext = (case/str(int(iteration))/'p').read_text()
    if not re.search(r'dimensions\s*\[1 -1 -2 0 0 0 0\]', ptext):
        raise ValueError('P002 pressure must be dimensional Pa')
    internal = values(ptext,'internalField',len(m['cells']))
    pressure = {}
    for patch in ('anodeInlet','anodeOutlet','cathodeInlet','cathodeOutlet'):
        block = re.search(r'\b'+patch+r'\s*\{(.*?)\}',boundary,re.S).group(1)
        count = int(re.search(r'nFaces\s+(\d+)',block).group(1))
        start = int(re.search(r'startFace\s+(\d+)',block).group(1))
        pblock = re.search(r'\b'+patch+r'\s*\{(.*?)\}',ptext,re.S).group(1)
        if re.search(r'type\s+zeroGradient\s*;',pblock):
            pv = [internal[c] for c in owner[start:start+count]]
        elif re.search(r'type\s+fixedValue\s*;',pblock):
            pv = values(pblock,'value',count)
        else:
            raise ValueError('Unexpected pressure boundary condition')
        areas = []
        for cell in owner[start:start+count]:
            i,j,k = m['cells'][cell]
            areas.append((m['y'][j+1]-m['y'][j])*(m['z'][k+1]-m['z'][k])*1e-6)
        pressure[patch] = sum(a*v for a,v in zip(areas,pv))/sum(areas)
    return {side:pressure[side+'Inlet']-pressure[side+'Outlet'] for side in ('anode','cathode')}


def extract_metrics(case, config):
    case = Path(case)
    manifest = json.loads((case/'case.json').read_text())
    status = json.loads((case/'postProcessing/pem/status.json').read_text())
    rows = read_history(case)
    last = rows[-1]
    log = (case/'log.flowfieldP002Foam').read_text()
    if ('CONVERGED: coupled fields and charge/mass/species balances passed.' not in log
            or not re.search(r'^End\s*$',log,re.M) or status.get('converged') is not True
            or status.get('iteration') != last['iteration'] or status.get('cells') != manifest['cells']
            or status.get('processes') != 1):
        raise ValueError('Solver did not finish a verified serial run')
    for row in rows[-5:]:
        if not (0 <= row['charge_residual'] < 1e-6
                and 0 <= row['mass_error'] < config.model['balanceTolerance']
                and 0 <= row['gas_error'] < config.model['balanceTolerance']
                and 0 <= row['field_change'] < config.model['outerTolerance']
                and row['bounded_cells'] == 0
                and row['iteration'] >= config.model['minimumIterations']):
            raise ValueError('Last five iterations do not meet convergence and balance limits')
    if last['Tmax_K'] <= 0 or not 0 <= last['gas_volume_max'] < .995:
        raise ValueError('Unphysical final state or active gas-fraction cap')
    verification = verify(case)
    if not verification['passed'] or verification['iteration'] != last['iteration']:
        raise ValueError('Independent saved-field conservation checks failed or fields are stale')
    dp = field_pressure_drops(case, last['iteration'], manifest)
    for side in ('anode','cathode'):
        if abs(dp[side]-last[f'dp_{side}_Pa']) > max(abs(dp[side])*1e-7,1e-8):
            raise ValueError('Saved-field pressure differs from runtime history')
    voltage = verification['voltage_V']
    if abs(voltage-last['voltage_V']) > 1e-7:
        raise ValueError('Saved-field voltage differs from runtime history')
    pressure_sum = sum(dp.values())
    objective = scalar_objective(voltage, pressure_sum, config)
    metrics = dict(last, voltage_V=voltage, pressure_sum_Pa=pressure_sum,
                   dp_anode_Pa=dp['anode'], dp_cathode_Pa=dp['cathode'], cells=manifest['cells'],
                   electrical_power_W=manifest['current_A']*voltage,
                   pumping_power_W=config.model['inletFlow_mL_min']*1e-6/60*pressure_sum,
                   field_verification=verification)
    return dict(objective=objective, objectives=dict(voltage_V=voltage, pressure_sum_Pa=pressure_sum),
                constraints={}, metrics=metrics)
