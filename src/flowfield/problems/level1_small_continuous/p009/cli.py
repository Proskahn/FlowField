"""Chen 2025 hydrodynamic reconstruction and reference-data audit."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from ....config import load_config
from ....runtime import Runtime
from . import ChenConfig, P009Problem
from .config import physical_geometry


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('generate','run','extract','evaluate','paper-study','reference-audit'))
    p.add_argument('--case', required=True, help='New output root, existing case for run/extract, or audit JSON path')
    p.add_argument('--config'); p.add_argument('--mesh', choices=('coarse','medium','fine'))
    p.add_argument('--diameter-mm',type=float,choices=(100.,160.))
    p.add_argument('--angle-deg',type=float); p.add_argument('--flow-lpm',type=float); p.add_argument('--thickness-mm',type=float)
    p.add_argument('--backend',choices=('auto','native','docker'),default='auto')
    p.add_argument('--timeout',type=float,default=14400); p.add_argument('--seed',type=int,default=42)
    args = p.parse_args(argv)
    design_options = (args.angle_deg,args.flow_lpm,args.thickness_mm)
    config_options = (args.config,args.mesh,args.diameter_mm)
    if args.action in ('run','extract','reference-audit') and any(v is not None for v in (*design_options,*config_options)):
        p.error('run/extract use immutable recorded inputs; reference-audit uses only published data')
    if args.action=='paper-study' and any(v is not None for v in design_options):
        p.error('paper-study uses the published Table 5 designs')
    try:
        if args.action=='reference-audit':
            from .rsm import reference_audit
            from ....evaluation import write_json
            result = reference_audit(args.seed)
            dest = Path(args.case); dest.parent.mkdir(parents=True,exist_ok=True)
            write_json(dest,result)
        else:
            config = load_config(ChenConfig,args.config) if args.config else ChenConfig()
            if args.mesh: config=replace(config,mesh=args.mesh)
            if args.diameter_mm: config=replace(config,diameter_mm=args.diameter_mm)
            runtime=Runtime(backend=args.backend,timeout_s=args.timeout)
            problem=P009Problem(config=config,runtime=runtime,output_root=args.case)
            values=[v if v is not None else d for v,d in zip(design_options,(92.5,.75,1.92))]
            if args.action in ('run','extract') and not (Path(args.case)/'case.json').is_file():
                raise ValueError('Not a complete generated case: case.json is missing')
            if args.action=='generate':
                from .case import generate_case
                result=str(generate_case(args.case,physical_geometry(*values,config),config))
            elif args.action=='run':
                runtime.resolve(problem.case_definition.commands)
                result=problem.run_case(args.case)
            elif args.action=='extract': result=problem.extract_results(args.case)
            elif args.action=='paper-study':
                from .study import paper_study
                result=paper_study(args.case,config,runtime,seed=args.seed)
            else:
                problem.preflight(); runtime.resolve(problem.case_definition.commands)
                result=problem.evaluate_physical(*values)
    except (OSError,ValueError,RuntimeError) as exc:
        p.exit(1,f'flowfield-p009: {exc}\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
