"""Problem-specific diagnostics and numerical experiments; no physics changes."""
from dataclasses import replace

PROFILES = {
    'P000': ('log.simpleFoam', None),
    'P001': ('log.simpleFoam', None),
    'P002': ('log.flowfieldP002Foam', 'postProcessing/pem/history_0.csv'),
    'P003': ('log.flowfieldP003Foam', 'postProcessing/p003/history_0.csv'),
    'P004-hydraulic': ('log.simpleFoam', None),
    'P005': ('log.interFoam', 'postProcessing/detachment.csv'),
    'P006': ('log.flowfieldP006Foam', 'postProcessing/p006/history.csv'),
    'P008': ('log.flowfieldP008Foam', 'postProcessing/p008/history.csv'),
    'P007': ('log.flowfieldP007Foam', 'postProcessing/p007/history.csv'),
}


def limits(name, c):
    """Final physical acceptance limits, not linear-solver residual tolerances."""
    if name == 'P008':
        return dict(field_change=c.field_tolerance,charge_residual=1e-7,**dict.fromkeys(('liquid_balance','gas_balance','momentum_residual','energy_balance'),c.balance_tolerance))
    if name == 'P007':
        return dict(field_change=c.outer_tolerance, **dict.fromkeys(
            ('volume_error','local_continuity','power_error','momentum_residual','energy_balance'), c.balance_tolerance))
    if name == 'P006':
        return dict(field_change=c.field_tolerance, charge_residual=1e-6,
            **dict.fromkeys(('liquid_balance','gas_balance','local_liquid_balance','local_gas_balance',
                'liquid_momentum_residual','gas_momentum_residual','energy_balance'), c.balance_tolerance))
    if name == 'P002':
        return dict(field_change=c.model['outerTolerance'], charge_residual=1e-6,
                    mass_error=c.model['balanceTolerance'], gas_error=c.model['balanceTolerance'])
    if name == 'P003':
        return dict(field_change=c.outer_tolerance, **dict.fromkeys(
            ('volume_error','oxygen_error','water_error'), c.balance_tolerance))
    if name == 'P005':
        return dict(max_gas_balance_relative=c.gas_balance_tolerance,
                    max_continuity_relative=c.continuity_tolerance)
    return {}


def refined(c):
    if hasattr(c, 'mesh'):
        levels=('coarse','medium','fine');start=levels.index(c.mesh)
        return [(f'mesh-{v}', replace(c,mesh=v)) for v in levels[start+1:]]
    return [(f'mesh-x{factor}', replace(c,cells=tuple(factor*n for n in c.cells))) for factor in (2,4)]


def stricter(name, c):
    if name == 'P005':
        return replace(c,max_delta_t_s=c.max_delta_t_s/2,max_courant=c.max_courant/2,
                       monitor_interval_s=c.monitor_interval_s/2)
    if name == 'P002':
        return replace(c,model_overrides=dict(c.model_overrides,
            outerTolerance=c.model['outerTolerance']/10,balanceTolerance=c.model['balanceTolerance']/10))
    keys = {'P000':('residual_tolerance','mass_balance_tolerance'),
            'P001':('residual_tolerance','mass_balance_tolerance','stability_tolerance'),
            'P003':('outer_tolerance','balance_tolerance'),
            'P004-hydraulic':('residual_tolerance','balance_tolerance','steady_tolerance'),
            'P006':('field_tolerance','balance_tolerance'),
            'P008':('field_tolerance','balance_tolerance'),
            'P007':('outer_tolerance','balance_tolerance')}[name]
    return replace(c,**{k:getattr(c,k)/10 for k in keys})


def experiments(name, config, studies):
    jobs=[('baseline','baseline',config)];unavailable=[]
    for study in studies:
        if study == 'repeat': jobs.append(('repeat','repeat',config))
        elif study == 'numerics': jobs.append(('numerics','numerics',stricter(name,config)))
        elif study == 'mesh':
            variants=refined(config)
            jobs.extend((label,'mesh',c) for label,c in variants)
            if not variants:unavailable.append(dict(study=study,reason='Already at the finest configured mesh.'))
        elif study == 'parallel':
            if name == 'P006':
                n=1 if config.processes!=1 else 2
                jobs.append((f'processes-{n}','parallel',replace(config,processes=n)))
            else:unavailable.append(dict(study=study,reason='This benchmark implementation is serial only.'))
    return jobs,unavailable


def observables(name, extracted):
    """Compare dimensional physical outputs, never penalty scores."""
    m=extracted['metrics']
    if name=='P008':
        from ..problems.level1_small_continuous.p008.ewm import INDICATORS
        return {k:m[k] for k,_ in INDICATORS}
    keys={'P000':('pressure_drop_pa',), 'P001':('uniformity','pressure_drop_pa'),
          'P002':('voltage_V','pressure_sum_Pa'),
          'P003':('power_domain_W','current_std_A_m2','inlet_flow_half_cell_m3_s'),
          'P004-hydraulic':('pressure_drop_pa',),
          'P006':('temperature_std_K','current_anode_A','dp_anode_Pa','dp_cathode_Pa','flow_cv'),
          'P007':('heat_transfer','dissipation','inlet_flow','inlet_pressure')}
    if name == 'P005': return {'detachment_time_s':m['detachment_time_s']} if m['detached'] else {}
    return {k:m[k] for k in keys[name]}
