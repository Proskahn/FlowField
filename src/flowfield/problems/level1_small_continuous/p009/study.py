"""Run Table 5 CFD, fit a fixed RSM, search it with GA, verify its candidate."""
import json
from pathlib import Path
from ....evaluation import write_json
from ....stability.report import atomic
from .config import PARAMETERS
from .reference import TABLE5, PRODUCT_BASELINE, REPORTED_OPTIMUM
from .rsm import fit, optimize_ga


def paper_study(output, config, runtime, *, seed=42):
    from . import P009Problem
    if config.diameter_mm != 160:
        raise ValueError('Table 5 is the 160 mm optimization study')
    probe = P009Problem(config=config, runtime=runtime)
    probe.preflight()
    runtime.resolve(probe.case_definition.commands)  # Before creating outputs.
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    records, cache = [], {}
    def evaluate(label, physical):
        problem = P009Problem(config=config, runtime=runtime, output_root=root/label)
        result = problem.evaluate_physical(*physical)
        if result['status'] != 'ok':
            raise ValueError('An unqualified case cannot enter an RSM fit')
        row = dict(label=label, physical_design=list(physical), case_directory=result['case_directory'], metrics=result['metrics'])
        records.append(row)
        atomic(root/'observations.json', json.dumps(records, indent=2, allow_nan=False)+'\n')
        return row
    baseline = evaluate('product-baseline', PRODUCT_BASELINE)
    table = []
    for i, (*physical, reference) in enumerate(TABLE5,1):
        key = tuple(physical)
        if key not in cache:
            cache[key] = evaluate(f'table5-{i:02d}', physical)
        table.append(dict(case_number=i, physical_design=physical,
            case_directory=cache[key]['case_directory'], repeated_design=any(t['physical_design']==physical for t in table),
            reference_F=reference, cfd_F=cache[key]['metrics']['performance_m_s_per_bar']))
    model = fit([t['physical_design'] for t in table], [t['cfd_F'] for t in table])
    model.update(qualified_cfd_panel=True, config_sha256=config.fingerprint,
        provenance=[dict(physical_design=r['physical_design'],case_directory=r['case_directory']) for r in records[1:]])
    write_json(root/'response-surface.json', model)
    ga = optimize_ga(model, seed=seed)
    candidate = evaluate('ga-confirmation', [ga['physical_design'][k] for k in PARAMETERS])
    measured = candidate['metrics']['performance_m_s_per_bar']
    proposed = cache[REPORTED_OPTIMUM]
    result = dict(status='completed', paper_validated=False, baseline=baseline, table5=table,
        model=model, ga=ga, confirmation=candidate, reported_optimum=proposed,
        ga_prediction_relative_error=abs(measured-ga['predicted_performance'])/abs(measured),
        cfd_improvement_percent=100*(measured/baseline['metrics']['performance_m_s_per_bar']-1),
        optimization_config=str(root/'optimization-config.json'),
        note='CFD and published scores remain separate; repeated Table 5 rows reuse deterministic CFD and retain regression multiplicity.')
    write_json(root/'optimization-config.json',config.to_dict())
    write_json(root/'study.json',result)
    return result
