"""Xie ESCFF coupled PEMFC, qualification, and four-factor RSM study."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from ....config import load_config
from ....evaluation import write_json
from ....runtime import Runtime
from . import PEMFCConfig,P010Problem
from .config import BASELINE,FLOW_FIELDS,normalized

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('generate','run','extract','evaluate','reference-audit','paper-study','compare-fields','polarization'))
    p.add_argument('--case',required=True,help='New output directory, existing case for run/extract, JSON path for reference-audit')
    p.add_argument('--config');p.add_argument('--mesh',choices=('coarse','medium','fine'))
    p.add_argument('--flow-field',choices=FLOW_FIELDS)
    for name in ('temperature-k','humidity-percent','cathode-stoichiometry','gdl-porosity'):p.add_argument('--'+name,type=float)
    p.add_argument('--voltage',type=float)
    p.add_argument('--backend',choices=('auto','native','docker'),default='auto')
    p.add_argument('--timeout',type=float,default=14400)
    args=p.parse_args(argv)
    design=(args.temperature_k,args.humidity_percent,args.cathode_stoichiometry,args.gdl_porosity)
    overrides=(args.config,args.mesh,args.flow_field,args.voltage,*design)
    if args.action in ('run','extract','reference-audit') and any(v is not None for v in overrides):
        p.error('run/extract use immutable recorded inputs; reference-audit uses published data only')
    if args.action in ('paper-study','compare-fields','polarization') and any(v is not None for v in design):
        p.error('Studies use their declared baseline/paper design panel')
    if args.action=='paper-study' and (args.voltage is not None and args.voltage!=.4 or args.flow_field not in (None,'ESCFF-3')):
        p.error('The paper optimization uses ESCFF-3 at 0.4 V')
    try:
        if args.action=='reference-audit':
            from .rsm import reference_audit
            dest=Path(args.case)
            if dest.exists():raise ValueError('Output already exists')
            result=reference_audit();dest.parent.mkdir(parents=True,exist_ok=True);write_json(dest,result)
        else:
            config=load_config(PEMFCConfig,args.config) if args.config else PEMFCConfig()
            for key,value in (('mesh',args.mesh),('flow_field',args.flow_field),('cell_voltage',args.voltage)):
                if value is not None:config=replace(config,**{key:value})
            runtime=Runtime(args.backend,timeout_s=args.timeout)
            problem=P010Problem(config=config,runtime=runtime,output_root=args.case)
            x=normalized([d if v is None else v for v,d in zip(design,BASELINE)])
            if args.action=='generate':result=str(problem.write_case(x,args.case))
            elif args.action=='run':result=problem.run_case(args.case)
            elif args.action=='extract':result=problem.extract_results(args.case)
            elif args.action=='evaluate':
                problem.preflight();runtime.resolve(problem.case_definition.commands);result=problem.evaluate_raw(x)
            else:
                from .study import paper_study,comparison_study
                result=paper_study(args.case,config,runtime) if args.action=='paper-study' else comparison_study(args.case,config,runtime,polarization=args.action=='polarization')
    except (OSError,ValueError,RuntimeError) as exc:p.exit(1,f'flowfield-p010: {exc}\n')
    print(json.dumps(result,indent=2,allow_nan=False));return 0

if __name__=='__main__':raise SystemExit(main())
