"""Fresh-case auxiliary CFD and saved-field verification, separate from PEMFC current."""
import json,math,re
from pathlib import Path
from ....runtime import Runtime,SimulationError
from ....evaluation import CaseDefinition, CaseEvaluator
from ....mesh import PolyMesh
from ....openfoam import read_surface_history
from .case import INPUTS,generate_case
from .config import HydraulicConfig,HYDRAULIC_REVISION
from .geometry import decode


def _values(text,key,count):
    match=re.search(r'\b'+key+r'\s+uniform\s+([^;]+);',text)
    if match:values=[float(match.group(1))]*count
    else:
        match=re.search(r'\b'+key+r'\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
        if not match or int(match.group(1))!=count:raise SimulationError('Malformed saved scalar field')
        values=[float(v) for v in match.group(2).split()]
    if len(values)!=count or not all(math.isfinite(v) for v in values):raise SimulationError('Invalid saved scalar values')
    return values


def extract_metrics(case,config):
    case=Path(case);manifest=json.loads((case/'case.json').read_text());mesh=manifest['mesh']
    text=(case/'log.checkMesh').read_text()
    if 'Mesh OK.' not in text or 'Failed ' in text:raise SimulationError('Mesh validation failed')
    volume=re.search(r'Total volume\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)',text)
    if not volume or abs(float(volume.group(1))/98e-9-1)>1e-8:raise SimulationError('Mesh volume differs from physical channel volume')
    text=(case/'log.simpleFoam').read_text()
    if not re.search(r'SIMPLE solution converged in \d+ iterations',text) or not re.search(r'^End\s*$',text,re.M):
        raise SimulationError('Solver did not converge normally')
    times=re.findall(r'^Time = ([\d.eE+\-]+)\s*$',text,re.M);iteration=float(times[-1])
    histories={}
    for name,col in [('p_inlet','areaAverage(p)'),('p_outlet','areaAverage(p)'),('q_inlet','sum(phi)'),('q_outlet','sum(phi)')]:
        files=list((case/'postProcessing'/name).glob('*/surfaceFieldValue.dat'))
        if len(files)!=1 or files[0].parent.name!='0':raise SimulationError('Expected a fresh surface history')
        histories[name]=read_surface_history(files[0],col)
        if max(histories[name])!=iteration or len(histories[name])<20:raise SimulationError('Stale or short convergence history')
    keys=list(histories['p_inlet'])[-20:]
    dp=[config.density_kg_m3*(histories['p_inlet'][k]-histories['p_outlet'][k]) for k in keys]
    if min(dp)<=0:raise SimulationError('Pressure drop must be positive')
    steady=max(abs(v/dp[-1]-1) for v in dp)
    qi=histories['q_inlet'][iteration];qo=histories['q_outlet'][iteration]
    balance=abs(qi+qo)/config.flow_rate_m3_s
    flow_error=max(abs(-qi/config.flow_rate_m3_s-1),abs(qo/config.flow_rate_m3_s-1))
    if max(balance,flow_error)>config.balance_tolerance or steady>config.steady_tolerance:
        raise SimulationError('Flow balance or final pressure stability failed')
    folders=[p for p in case.iterdir() if p.is_dir() and re.fullmatch(r'\d+(?:\.\d+)?',p.name)]
    latest=max(folders,key=lambda p:float(p.name))
    if float(latest.name)!=iteration:raise SimulationError('Saved fields do not match the final iteration')
    pressure=(latest/'p').read_text();flux=(latest/'phi').read_text()
    if not re.search(r'dimensions\s*\[0 2 -2 0 0 0 0\]',pressure) or not re.search(r'dimensions\s*\[0 3 -1 0 0 0 0\]',flux):
        raise SimulationError('Wrong pressure/flux units')
    internal_p=_values(pressure,'internalField',mesh['cells']);saved_p={};saved_q={}
    owner_text=(case/'constant/polyMesh/owner').read_text()
    owner_match=re.search(r'\}\s*(\d+)\s*\((.*?)\)',owner_text,re.S)
    if not owner_match:raise SimulationError('Cannot read face owners')
    owners=[int(v) for v in owner_match.group(2).split()]
    if len(owners)!=mesh['faces'] or any(v<0 or v>=mesh['cells'] for v in owners):
        raise SimulationError('Invalid face owners')
    boundary=(case/'constant/polyMesh/boundary').read_text()
    for name in ('inlet','outlet'):
        count=mesh['patch_faces'][name]
        block=re.search(r'\b'+name+r'\s*\{(.*?)\}',pressure,re.S).group(1)
        if re.search(r'type\s+zeroGradient\s*;',block):
            # OpenFOAM omits values for zeroGradient; the patch equals its face-owner cells.
            patch=re.search(r'\b'+name+r'\s*\{(.*?)\}',boundary,re.S).group(1)
            first=int(re.search(r'startFace\s+(\d+)',patch).group(1))
            pv=[internal_p[v] for v in owners[first:first+count]]
        else:pv=_values(block,'value',count)
        saved_p[name]=sum(pv)/count
        block=re.search(r'\b'+name+r'\s*\{(.*?)\}',flux,re.S).group(1)
        saved_q[name]=sum(_values(block,'value',count))
    saved_dp=config.density_kg_m3*(saved_p['inlet']-saved_p['outlet'])
    wall_block=re.search(r'\bwalls\s*\{(.*?)\}',flux,re.S).group(1)
    wall_leak=sum(abs(v) for v in _values(wall_block,'value',mesh['patch_faces']['walls']))/config.flow_rate_m3_s
    if wall_leak>config.balance_tolerance:raise SimulationError('Nonzero sealed-wall flux')
    error=max(abs(saved_dp/dp[-1]-1),abs(saved_q['inlet']/qi-1),abs(saved_q['outlet']/qo-1))
    if error>1e-8:raise SimulationError('Saved fields differ from history')
    return dict(pressure_drop_pa=saved_dp,pumping_power_w=saved_dp*config.flow_rate_m3_s,
        inlet_flow_m3_s=-qi,outlet_flow_m3_s=qo,mass_balance_relative_error=balance,
        imposed_flow_relative_error=flow_error,pressure_last20_relative_change=steady,
        saved_fields_relative_error=error,sealed_wall_relative_flux=wall_leak,mesh_volume_m3=float(volume.group(1)),
        reynolds_number=config.reynolds_number,cells=mesh['cells'],iterations=int(iteration),
        current_density_A_m2=None,electrochemistry_solved=False)


def _validate(case, manifest, config):
    if (manifest['geometry']!=decode(manifest['geometry']['path'])
        or json.loads((case/'geometry.json').read_text())!=manifest['geometry']):
        raise ValueError('Geometry manifest mismatch')
    if manifest['mesh']['cells']!=98*config.cells_per_width**3:raise ValueError('Mesh/config mismatch')


def _execute(case, config, manifest, runtime, info, result):
    PolyMesh.from_case(case)
    result.update(library_id=manifest['geometry']['library_id'], evaluation_kind='auxiliary_channel_only_CFD', paper_current_density_reproduced=False)
    log=runtime.run(case,'checkMesh',('-allGeometry','-allTopology'),info=info).read_text()
    if 'Mesh OK.' not in log or 'Failed ' in log:raise SimulationError('Full mesh check failed')
    runtime.run(case,'simpleFoam',info=info)


def extract_results(case, config):
    metrics = extract_metrics(case, config)
    return dict(objective=metrics['pressure_drop_pa'], objectives={'pressure_drop_pa': metrics['pressure_drop_pa']},
                constraints={}, objective_unit='Pa', metrics=metrics)


CASE = CaseDefinition(
    name='P004-hydraulic', config_type=HydraulicConfig, inputs=frozenset(INPUTS),
    commands=('checkMesh', 'simpleFoam'), validate=_validate, execute=_execute,
    generate=generate_case, extract=extract_results, revision=HYDRAULIC_REVISION,
    execution_outputs=('log.checkMesh', 'log.simpleFoam'))


def run_case(case, runtime=None):
    return CaseEvaluator(CASE, runtime).run_case(case)


def evaluate(path, output_root, config, runtime):
    return CaseEvaluator(CASE, runtime).evaluate(path, config, output_root=output_root)
