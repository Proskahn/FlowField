"""Generate, run and extract a standalone P007 case; optimization uses the shared CLI."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from ....config import load_config
from ....runtime import DEFAULT_IMAGE,Runtime
from . import P007Problem,TopologyConfig
from .evaluation import run_case


def main(argv=None):
    parser=argparse.ArgumentParser(description='P007 dimensionless 2D flow/heat topology model')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('decode','generate','evaluate','run','extract'):
        p=sub.add_parser(command)
        if command not in ('run','extract'):
            p.add_argument('--config',type=Path);p.add_argument('--scheme',type=int,choices=range(1,6))
            p.add_argument('--mesh',choices=('coarse','medium','fine'));p.add_argument('--design',type=Path)
        if command in ('generate','run','extract'):p.add_argument('--case',type=Path,required=True)
        if command=='evaluate':p.add_argument('--output',type=Path,required=True)
        if command in ('run','evaluate'):
            p.add_argument('--backend',choices=('auto','native','docker'),default='auto')
            p.add_argument('--image',default=DEFAULT_IMAGE);p.add_argument('--timeout',type=float,default=1800)
    args=parser.parse_args(argv)
    try:
        if args.command=='run':result=run_case(args.case,Runtime(args.backend,args.image,args.timeout))
        elif args.command=='extract':result=P007Problem().extract_results(args.case)
        else:
            config=load_config(TopologyConfig,args.config)
            changes={k:getattr(args,k) for k in ('scheme','mesh') if getattr(args,k) is not None}
            config=replace(config,**changes)
            problem=P007Problem(config=config,output_root=getattr(args,'output','runs'))
            x=json.loads(args.design.read_text()) if args.design else problem.initial_design()
            if args.command=='decode':result=problem.decode(x)
            elif args.command=='generate':result=dict(status='generated_not_run',case_directory=str(problem.write_case(x,args.case)))
            else:
                problem.runtime=Runtime(args.backend,args.image,args.timeout);problem.preflight()
                result=problem.evaluate_raw(x)
    except (OSError,ValueError,TypeError,RuntimeError,ImportError) as exc:parser.exit(1,f'flowfield-p007: {exc}\n')
    print(json.dumps(result,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
