"""Reject unqualified/stale CFD; measure F using the units of Figure 31."""
import json
import math
from pathlib import Path
import re
from ....openfoam import read_surface_history
from ....runtime import SimulationError


def extract_metrics(case, config):
    case = Path(case)
    manifest = json.loads((case/'case.json').read_text())
    mesh_log = (case/'log.checkMesh').read_text()
    log = (case/'log.simpleFoam').read_text()
    if 'Mesh OK.' not in mesh_log or re.search(r'Failed\s+\d+\s+mesh checks', mesh_log):
        raise SimulationError('P009 mesh is not qualified')
    if ('FOAM FATAL' in log or not re.search(r'SIMPLE solution converged in \d+ iterations', log)
            or not re.search(r'^End\s*$', log, re.M)):
        raise SimulationError('P009 requires SIMPLE convergence and normal completion; exhausted budgets are rejected')
    times = re.findall(r'^Time = ([0-9.eE+\-]+)\s*$', log, re.M)
    if not times:
        raise SimulationError('Missing solver iteration records')
    final = float(times[-1])
    columns = dict(p_inlet='areaAverage(p)', p_outlet='areaAverage(p)',
        q_inlet='sum(phi)', q_outlet='sum(phi)', q_walls='sumMag(phi)',
        plane_speed='areaAverage(speed)', plane_speed2='areaAverage(speed2)')
    histories = {}
    for name, column in columns.items():
        files = list((case/'postProcessing'/name).glob('*/surfaceFieldValue.dat'))
        if len(files) != 1:
            raise SimulationError(f'Expected one fresh {name} history')
        histories[name] = read_surface_history(files[0], column)
    keys = list(histories['p_inlet'])
    if any(list(h) != keys for h in histories.values()) or not keys or keys[-1] != final:
        raise SimulationError('Stale or unaligned P009 histories')
    if len(keys) < config.stability_window:
        raise SimulationError('Insufficient stationary history')
    flow = manifest['geometry']['flow_m3_s']
    balance, imposed, wall = [], [], []
    dp, speed, cv = [], [], []
    for t in keys[-config.stability_window:]:
        v = {name:h[t] for name,h in histories.items()}
        qi, qo = v['q_inlet'], v['q_outlet']
        balance.append(abs(qi+qo)/flow)
        imposed.append(max(abs(qi/flow+1), abs(qo/flow-1)))
        wall.append(abs(v['q_walls'])/flow)
        p = config.density_kg_m3*(v['p_inlet']-v['p_outlet'])
        mean, square = v['plane_speed'], v['plane_speed2']
        if qi >= 0 or qo <= 0 or p <= 0 or mean <= 0 or square < mean*mean*(1-1e-5):
            raise SimulationError('Invalid pressure, flow direction or speed moments')
        dp.append(p); speed.append(mean)
        cv.append(math.sqrt(max(0., square-mean*mean))/mean)
    if max(balance+imposed+wall) > config.balance_tolerance:
        raise SimulationError('P009 flow conservation failed over the final window')
    drift = max((max(v)-min(v))/max(abs(v[-1]), 1e-15) for v in (dp, speed, cv))
    if drift > config.steady_tolerance:
        raise SimulationError(f'P009 objective/flow statistics have not settled: relative range {drift:g}')
    cells = int(re.search(r'^\s*cells:\s*(\d+)', mesh_log, re.M)[1])
    from ....stability.fields import final_fields
    fields = final_fields(case, 'P009', config, final, cells)
    if fields['status'] != 'passed':
        raise SimulationError('P009 saved fields failed inspection: '+str(fields))
    from .verification import verify_saved
    verification = verify_saved(case,config,final,cells,dict(dp=dp[-1],speed=speed[-1],
        speed2=histories['plane_speed2'][final],
        **{key:histories[key][final] for key in ('q_inlet','q_outlet','q_walls')}))
    g = manifest['geometry']
    hydraulic_diameter = 2*g['inlet_width_mm']*g['inlet_height_mm']/(g['inlet_width_mm']+g['inlet_height_mm'])*.001
    reynolds = config.density_kg_m3*g['inlet_speed_m_s']*hydraulic_diameter/config.dynamic_viscosity_pa_s
    return dict(pressure_drop_pa=dp[-1], mean_speed_m_s=speed[-1], flow_cv=cv[-1],
        performance_m_s_per_bar=speed[-1]/(dp[-1]/1e5), pumping_power_W=dp[-1]*flow,
        mass_balance_relative_error=max(balance), imposed_flow_relative_error=max(imposed),
        sealed_wall_relative_flux=max(wall), statistics_final_window_relative_range=drift,
        inlet_reynolds=reynolds, paper_reynolds_under_1000=reynolds < 1000,
        iterations=int(final), cells=cells, mesh_ok=True, converged=True, paper_validated=False,
        saved_field_verification=verification,
        measurement='area-weighted speed moments on nickel-only plane, 1 mm above electrode wall',
        velocity_convention='single continuous mixture/superficial velocity; Eq.3 porosity factors retained')
