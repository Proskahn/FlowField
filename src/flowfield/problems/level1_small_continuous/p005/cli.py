"""Generate, evaluate and smoke-test the physical VOF benchmark from a shell."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
from ....evaluation import write_json
from ....runtime import Runtime,DEFAULT_IMAGE,SimulationError
from ....benchmarking.ioh import analyzer_logger,make_ioh_problem
from . import P005Problem,DualLayerConfig,BASELINE_X
from .case import generate_case
from .evaluation import run_case


def smoke(output,config,runtime,with_ioh=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    # A short solver/adapter test, explicitly not a detachment-time validation.
    c=replace(config,end_time_s=5e-5,monitor_interval_s=1e-5)
    problem=P005Problem(output/'cases',config=c,runtime=runtime)
    summary=dict(problem='P005',status='failed',paper_validated=False,
        scope='short_transient_execution_and_IOH_only',observation_horizon_s=c.end_time_s)
    try:
        objective=make_ioh_problem(problem) if with_ioh else problem
        context=analyzer_logger(objective,output/'ioh',algorithm_name='P005-VOF-smoke') if with_ioh else nullcontext()
        with context:
            value=float(objective(problem.initial_design()))
            summary.update(objective=value,metrics=problem.last_result['metrics'],case_directory=problem.last_result['case_directory'])
        if with_ioh:
            if objective.state.evaluations!=1 or not list((output/'ioh').rglob('*.dat')):
                raise SimulationError('Missing IOH trace')
            summary['ioh_evaluations']=int(objective.state.evaluations)
        summary['status']='passed'
    except Exception as exc:summary['error']=str(exc);raise
    finally:write_json(output/'smoke.json',summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description='P005 physical-only dual-layer VOF reconstruction (provisional)')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('decode','generate','evaluate','smoke'):
        p=sub.add_parser(name);p.add_argument('--config',type=Path)
        p.add_argument('--mesh',choices=('coarse','medium','fine'))
        p.add_argument('--oxygen-velocity',type=float,help='Fixed operating condition in m/s; paper uses 1,3,5,10')
        if name!='smoke':p.add_argument('--x',nargs=2,type=float,default=BASELINE_X)
        if name=='generate':p.add_argument('--case',type=Path,required=True)
        if name in ('evaluate','smoke'):
            p.add_argument('--output',type=Path,required=True);runtime_arguments(p)
        if name=='smoke':p.add_argument('--ioh',action='store_true')
    p=sub.add_parser('run');p.add_argument('case',type=Path);runtime_arguments(p)
    args=parser.parse_args(argv)
    try:
        if args.command=='run':result=run_case(args.case,Runtime(args.backend,args.image,args.timeout))
        else:
            values=json.loads(args.config.read_text()) if args.config else {}
            if args.mesh:values['mesh']=args.mesh
            if args.oxygen_velocity is not None:values['oxygen_velocity_m_s']=args.oxygen_velocity
            config=DualLayerConfig(**values);problem=P005Problem(config=config)
            if args.command=='decode':result=problem.decode(args.x)
            elif args.command=='generate':result=dict(status='generated_not_run',case_directory=str(generate_case(args.case,problem.decode(args.x),config)))
            else:
                runtime=Runtime(args.backend,args.image,args.timeout)
                if args.command=='smoke':result=smoke(args.output,config,runtime,args.ioh)
                else:result=P005Problem(args.output,config=config,runtime=runtime).evaluate_raw(args.x)
    except (OSError,ValueError,TypeError,RuntimeError) as exc:parser.exit(1,f'flowfield-p005: {exc}\n')
    print(json.dumps(result,indent=2,allow_nan=False));return 0


def runtime_arguments(parser):
    parser.add_argument('--backend',choices=('auto','native','docker'),default='auto')
    parser.add_argument('--image',default=DEFAULT_IMAGE)
    parser.add_argument('--timeout',type=float,default=1800,help='Wall seconds per meshing, compilation or solver command')


if __name__=='__main__':raise SystemExit(main())
