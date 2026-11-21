"""Qualified CFD observations, BBD regression and fresh optimum confirmation."""
from dataclasses import replace
import json
from pathlib import Path
from ....evaluation import write_json
from ....stability.report import atomic
from .config import BASELINE,PARAMETERS,FLOW_FIELDS,normalized
from .reference import ROWS,COMPARISONS
from .rsm import fit,maximize,interval

def recorder(output,config,runtime):
    from . import P010Problem
    probe=P010Problem(config=config,runtime=runtime);probe.preflight();runtime.resolve(probe.case_definition.commands)
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);records=[]
    def evaluate(label,physical,settings=config):
        p=P010Problem(config=settings,runtime=runtime,output_root=root/label)
        result=p.evaluate_raw(normalized(physical))
        if result['status']!='ok':raise ValueError('Unqualified CFD cannot enter a response surface')
        row=dict(label=label,physical_design=list(physical),config=settings.to_dict(),case_directory=result['case_directory'],metrics=result['metrics'])
        records.append(row);atomic(root/'observations.json',json.dumps(records,indent=2,allow_nan=False)+'\n');return row
    return root,evaluate

def paper_study(output,config,runtime):
    if config.cell_voltage!=.4 or config.flow_field!='ESCFF-3':raise ValueError('Table 6 requires ESCFF-3 at 0.4 V')
    root,evaluate=recorder(output,config,runtime)
    baseline=evaluate('baseline',BASELINE)  # Fail here before any DOE/optimization.
    cache={};table=[]
    for i,row in enumerate(ROWS,1):
        physical=row[:4];repeated=physical in cache
        if not repeated:cache[physical]=evaluate(f'table6-{i:02d}',physical)
        observed=cache[physical]
        table.append(dict(case_number=i,physical_design=list(physical),repeated_design=repeated,
            cfd_net_power_W=observed['metrics']['net_power_W'],reference_net_power_W=row[4],case_directory=observed['case_directory']))
        atomic(root/'table6.json',json.dumps(table,indent=2,allow_nan=False)+'\n')
    model=fit([normalized(r['physical_design']) for r in table],[r['cfd_net_power_W'] for r in table],source='Qualified independent CFD, Table 6 design panel')
    model.update(config_sha256=config.fingerprint,qualified_cfd_panel=True,
                 replicate_note='Repeated center designs reuse deterministic CFD; row multiplicities retained, not independent replicates.')
    write_json(root/'response-surface.json',model)
    optimum=maximize(model);validation=evaluate('optimum-confirmation',[optimum['physical_design'][k] for k in PARAMETERS])
    measured=validation['metrics']['net_power_W'];predicted=optimum['predicted_net_power_W']
    result=dict(status='completed',paper_validated=False,baseline=baseline,table6=table,model=model,
        optimum=optimum,conditional_intervals=interval(model,optimum['normalized_design']),confirmation=validation,
        prediction_relative_error=abs(measured-predicted)/abs(measured),
        measured_improvement_percent=100*(measured/baseline['metrics']['net_power_W']-1))
    write_json(root/'study.json',result);return result

def comparison_study(output,config,runtime,*,polarization=False):
    root,evaluate=recorder(output,config,runtime)
    baseline=evaluate('baseline',BASELINE)
    rows=[]
    if polarization:
        for voltage in (.9,.8,.7,.6,.5,.4,.3):
            rows.append(evaluate(f'voltage-{voltage:.1f}',BASELINE,replace(config,cell_voltage=voltage)))
    else:
        for name in FLOW_FIELDS:
            row=baseline if name==config.flow_field else evaluate(name,BASELINE,replace(config,flow_field=name))
            rows.append(dict(flow_field=name,computed=row,published_at_04V=COMPARISONS[name]))
    result=dict(status='completed',paper_validated=False,baseline=baseline,results=rows,
        note='Reconstructed CAD and declared missing closure estimates; published values are comparisons only.')
    write_json(root/'comparison.json',result);return result
