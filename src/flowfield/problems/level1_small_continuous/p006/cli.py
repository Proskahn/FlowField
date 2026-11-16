"""P006 case generation, execution and independent extraction."""
import argparse
from dataclasses import replace
import json
from ....config import load_config
from ....runtime import Runtime
from . import P006Problem,ManifoldConfig


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('generate','run','extract','evaluate'))
    p.add_argument('--case',required=True,help='New case, existing case, or evaluation output root')
    p.add_argument('--config')
    p.add_argument('--initial-state',help='Finished same-geometry P006 case to use as an initial guess (generate only)')
    p.add_argument('--widths-mm',type=float,nargs=3,metavar=('WIN','WMID','WOUT'))
    p.add_argument('--manifold-mm',type=float)
    p.add_argument('--mesh',choices=('coarse','medium','fine'))
    p.add_argument('--processes',type=int)
    p.add_argument('--backend',choices=('auto','native','docker'),default='auto')
    p.add_argument('--timeout',type=float,default=3600)
    args=p.parse_args(argv)
    if args.initial_state and args.action!='generate':
        p.error('--initial-state is supported by generate only; run/extract retain the recorded inputs')
    if args.action in ('run','extract') and any(v is not None for v in
            (args.config,args.widths_mm,args.manifold_mm,args.mesh,args.processes)):
        p.error('run/extract use the immutable case configuration; set geometry, mesh and processes when generating a new case')
    config=load_config(ManifoldConfig,args.config) if args.config else ManifoldConfig()
    overrides={k:v for k,v in dict(manifold_width_mm=args.manifold_mm,mesh=args.mesh,processes=args.processes).items() if v is not None}
    config=replace(config,**overrides)
    problem=P006Problem(config=config,output_root=args.case,runtime=Runtime(backend=args.backend,timeout_s=args.timeout))
    x=[(v-.7)/.8 for v in (args.widths_mm or [1.,1.,1.])]
    if args.action=='generate':
        from .case import generate_case
        result=str(generate_case(args.case,problem.decode(x),config,initial_state=args.initial_state))
    elif args.action=='run':result=problem.run_case(args.case)
    elif args.action=='extract':result=problem.extract_results(args.case)
    else:result=problem.evaluate_raw(x)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
