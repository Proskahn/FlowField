"""Shell-safe P003 commands; design vectors are stored in JSON files."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path

from ....evaluation import write_json
from ....benchmarking.ioh import analyzer_logger, make_ioh_problem
from ....runtime import DEFAULT_IMAGE, Runtime, SimulationError
from . import P003Problem, TopologyConfig
from .case import generate_case
from .evaluation import run_case


def smoke(output, config, runtime, with_ioh=False, compare=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    problem=P003Problem(output/'cases',config=config,runtime=runtime)
    summary=dict(status='failed',problem='P003',paper_validated=False,ioh_enabled=with_ioh,
                 config_sha256=config.fingerprint,evaluations=[])
    seeds=['uniform','parallel'] if compare else ['uniform']
    try:
        objective=make_ioh_problem(problem) if with_ioh else problem
        context=analyzer_logger(objective,output/'ioh',algorithm_name='P003-seed-check',
            algorithm_info='Forward CFD verification; no topology optimization') if with_ioh else nullcontext()
        with context:
            for label in seeds:
                x=problem.preset(label);value=float(objective(x));result=problem.last_result
                write_json(output/(label+'.json'),result)
                write_json(output/(label+'-design.json'),x)
                summary['evaluations'].append(dict(label=label,objective=value,
                    objectives=result['objectives'],constraints=result['constraints'],metrics=result['metrics'],
                    case_directory=result['case_directory']))
        if with_ioh:
            files=list((output/'ioh').glob('IOHprofiler*.json'))+list((output/'ioh').rglob('*.dat'))
            if objective.state.evaluations!=len(seeds) or len(files)<2:
                raise SimulationError('Missing or inconsistent IOH evaluations/logs')
            summary.update(ioh_evaluations=int(objective.state.evaluations),
                           ioh_files=[str(p.relative_to(output)) for p in files])
        summary['status']='passed'
    except Exception as exc:
        summary['error']=f'{type(exc).__name__}: {exc}';raise
    finally:write_json(output/'smoke.json',summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description='P003 provisional depth-averaged PEMFC topology benchmark')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('decode','generate','evaluate','smoke','compare','calibrate'):
        p=sub.add_parser(command);p.add_argument('--config',type=Path)
        p.add_argument('--mesh',choices=('coarse','medium','fine'))
        if command in ('decode','generate','evaluate'):
            group=p.add_mutually_exclusive_group()
            group.add_argument('--design',type=Path,help='JSON list of normalized nodal values')
            group.add_argument('--preset',choices=('uniform','parallel','all_rib','all_channel'),default='uniform')
        if command=='generate':p.add_argument('--case',type=Path,required=True)
        if command in ('evaluate','smoke','compare','calibrate'):
            p.add_argument('--output',type=Path,required=True);_runtime_arguments(p)
        if command in ('smoke','compare'):p.add_argument('--ioh',action='store_true')
    p=sub.add_parser('run');p.add_argument('case',type=Path);_runtime_arguments(p)
    from .gcmma_cli import add_arguments, execute
    p=sub.add_parser('gcmma',help='Full-grid finite-difference GCMMA with gray-material continuation')
    add_arguments(p);_runtime_arguments(p)
    from . import scalability
    p=sub.add_parser('scalable-config',help='Create a configurable design grid on a fixed CFD mesh')
    scalability.add_arguments(p)
    args=parser.parse_args(argv)
    try:
        if args.command=='gcmma':return execute(args)
        if args.command=='scalable-config':return scalability.execute(args)
        if args.command=='run':result=run_case(args.case,Runtime(args.backend,args.image,args.timeout))
        else:
            values=json.loads(args.config.read_text()) if args.config else {}
            if args.mesh:values['mesh']=args.mesh
            config=TopologyConfig(**values);problem=P003Problem(config=config)
            if args.command in ('decode','generate','evaluate'):
                x=json.loads(args.design.read_text()) if args.design else problem.preset(args.preset)
            if args.command=='decode':result=problem.decode(x)
            elif args.command=='generate':
                result=dict(case_directory=str(generate_case(args.case,problem.decode(x),config)),status='generated_not_run')
            else:
                runtime=Runtime(args.backend,args.image,args.timeout)
                if args.command in ('smoke','compare'):
                    result=smoke(args.output,config,runtime,args.ioh,args.command=='compare')
                elif args.command=='calibrate':
                    # An explicit extra CFD call, outside any optimizer's evaluation budget.
                    args.output.mkdir(parents=True,exist_ok=False)
                    raw=P003Problem(args.output/'cases',config=config,runtime=runtime).evaluate_raw(problem.initial_design())
                    calibrated=replace(config,power_scale_w=raw['metrics']['power_domain_W'],
                        current_std_scale_a_m2=raw['metrics']['current_std_A_m2'])
                    write_json(args.output/'normalized-config.json',calibrated.to_dict())
                    result=dict(status='calibrated',baseline_case=raw['case_directory'],
                        config_file=str((args.output/'normalized-config.json').resolve()),
                        config_sha256=calibrated.fingerprint,calibration_cfd_calls=1)
                    write_json(args.output/'calibration.json',result)
                else:result=P003Problem(args.output,config=config,runtime=runtime).evaluate_raw(x)
    except (OSError,ValueError,TypeError,RuntimeError,ImportError) as exc:parser.exit(1,f'flowfield-p003: {exc}\n')
    print(json.dumps(result,indent=2,allow_nan=False));return 0


def _runtime_arguments(parser):
    parser.add_argument('--backend',choices=('auto','native','docker'),default='auto')
    parser.add_argument('--image',default=DEFAULT_IMAGE)
    parser.add_argument('--timeout',type=float,default=1800,help='Seconds per build/solver command')


if __name__=='__main__':raise SystemExit(main())
