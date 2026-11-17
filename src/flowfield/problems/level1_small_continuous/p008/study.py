"""Paper's sequential width/depth panels. Every panel uses qualified CFD only."""
import json
from pathlib import Path
from dataclasses import replace
from ....evaluation import write_json
from ....runtime import SimulationError
from .ewm import fit
from .config import MODEL_REVISION

def paper_study(output,config,runtime):
    from . import P008Problem
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    if config.ewm_calibration:raise ValueError('Paper panels fit their own EWM weights; do not provide an existing calibration')
    records=[];panels={}
    def evaluate(label,x):
        problem=P008Problem(output_root=root/label,config=config,runtime=runtime)
        result=problem.evaluate_raw(x)
        if result['status']!='ok':raise SimulationError('Unqualified paper panel member')
        record=dict(label=label,design=x,geometry=problem.decode(x),case_directory=result['case_directory'],metrics=result['metrics'])
        records.append(record)
        from ....stability.report import atomic
        atomic(root/'observations.json',json.dumps(records,indent=2,allow_nan=False)+'\n');return record
    # Qualify the actual Table1 baseline first, before either comparison panel.
    baseline=evaluate('baseline',[1/6,2/3])
    widths=[baseline,evaluate('width-1.2mm',[.5,2/3]),evaluate('width-1.0mm',[5/6,2/3])]
    calibration=fit([r['metrics'] for r in widths]);panels['width']=dict(records=widths,calibration=calibration)
    best=max(range(3),key=lambda i:calibration['scores'][i]);width_x=widths[best]['design'][0]
    depth=[evaluate('depth-1.0mm',[width_x,1.]),widths[best],evaluate('depth-0.6mm',[width_x,1/3])]
    depth_cal=fit([r['metrics'] for r in depth]);panels['depth']=dict(records=depth,calibration=depth_cal)
    # A separate, frozen calibration spanning both completed stages is used by
    # generic optimizers. It does not alter either paper panel's published method.
    frozen=fit([r['metrics'] for r in records]);frozen.update(qualified_cfd_panel=True,model_revision=MODEL_REVISION,config_sha256=config.fingerprint);frozen['provenance']=[dict(case_directory=r['case_directory'],design=r['design']) for r in records]
    write_json(root/'calibration.json',frozen)
    write_json(root/'optimization-config.json',replace(config,ewm_calibration=frozen).to_dict())
    result=dict(status='completed',paper_validated=False,panels=panels,calibration=str(root/'calibration.json'),
        best_depth=depth[max(range(3),key=lambda i:depth_cal['scores'][i])]['geometry'])
    write_json(root/'study.json',result);return result
