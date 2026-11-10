"""Accept only aligned, converged, conservative CFD output."""

import json
import math
from pathlib import Path
import re

from ....metrics import read_surface_history
from ....runtime import SimulationError


def extract_metrics(case, config):
    case = Path(case)
    mesh_log = (case/'log.checkMesh').read_text()
    log = (case/'log.simpleFoam').read_text()
    if 'Mesh OK.' not in mesh_log or 'Failed ' in mesh_log:
        raise SimulationError('P001 mesh failed validation')
    if not re.search(r'SIMPLE solution converged in \d+ iterations', log) or not re.search(r'^End\s*$', log, re.M):
        raise SimulationError('P001 requires SIMPLE residual convergence and normal solver completion')
    times = re.findall(r'^Time = ([0-9.eE+\-]+)\s*$', log, re.M)
    if not times:
        raise SimulationError('Missing solver times')
    final_time = float(times[-1])
    columns = {'p_inlet':'areaAverage(p)', 'p_outlet':'areaAverage(p)',
        'q_inlet':'sum(phi)', 'q_outlet':'sum(phi)', 'uniformity_midplane':'uniformity(speed)',
        'speed_midplane':'areaAverage(speed)',
        **{f'uniformity_section{i}':'uniformity(speed)' for i in range(1, 5)}}
    histories = {}
    for name, column in columns.items():
        files = list((case/'postProcessing'/name).glob('*/surfaceFieldValue.dat'))
        if len(files) != 1:
            raise SimulationError(f'Expected one fresh {name} history, got {len(files)}')
        histories[name] = read_surface_history(files[0], column)
    if any(max(h) != final_time for h in histories.values()):
        raise SimulationError('Stale or unaligned P001 surface histories')
    values = {name:h[final_time] for name,h in histories.items()}
    qi, qo = values['q_inlet'], values['q_outlet']
    flow = config.flow_rate_m3_s
    balance = abs(qi+qo)/flow
    error = max(abs(qi/flow+1), abs(qo/flow-1))
    if qi >= 0 or qo <= 0 or max(balance, error) > config.mass_balance_tolerance:
        raise SimulationError(f'Flow conservation failed: balance={balance}, flow error={error}')
    ui = values['uniformity_midplane']
    if values['speed_midplane'] <= 0 or any(not 0 <= v <= 1 for n,v in values.items() if n.startswith('uniformity')):
        raise SimulationError('Undefined speed uniformity or values outside [0,1]')
    tail = list(histories['uniformity_midplane'].values())[-config.stability_window:]
    drift = max(tail)-min(tail)
    if len(tail) != config.stability_window or drift > config.stability_tolerance:
        raise SimulationError(f'Uniformity is not stable over the final {config.stability_window} samples: range={drift}')
    dp = config.density_kg_m3*(values['p_inlet']-values['p_outlet'])
    if not math.isfinite(dp) or dp <= 0:
        raise SimulationError('Pressure drop must be finite and positive')
    return {'uniformity':ui, 'mean_speed_m_s':values['speed_midplane'], 'pressure_drop_pa':dp,
        'section_uniformity':{str(y):values[f'uniformity_section{i}'] for i,y in enumerate((-135,-45,45,135),1)},
        'inlet_flux_m3_s':qi, 'outlet_flux_m3_s':qo, 'mass_balance_relative_error':balance,
        'flow_rate_relative_error':error, 'uniformity_final_window_range':drift,
        'iterations':int(final_time), 'cell_count':int(re.search(r'^\s*cells:\s*(\d+)', mesh_log, re.M).group(1)),
        'gmsh_cells':json.loads((case/'mesh-summary.json').read_text())['cells'],
        'mesh_ok':True, 'converged':True, 'paper_validated':False}
