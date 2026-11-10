"""P004 library/data/ML commands and separately named hydraulic verification."""
import argparse
from contextlib import nullcontext
import json,random
from pathlib import Path
from ....evaluation import write_json
from ....runtime import Runtime,DEFAULT_IMAGE
from ....ioh_adapter import make_ioh_problem,analyzer_logger
from ....experiments import random_search
from . import P004Problem,P004HydraulicProblem,HydraulicConfig
from .library import catalog,enumerate_paths,path_id,serpentine,COUNT
from .case import generate_case
from .hydraulics import run_case


def presets():return json.loads((Path(__file__).parent/'data/figure4-paths.json').read_text())['paths']


def hydraulic_smoke(output,config,runtime,with_ioh=False,compare=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    p=P004HydraulicProblem(output/'cases',config=config,runtime=runtime)
    labels=['serpentine','figure4_spiral'] if compare else ['serpentine']
    result=dict(status='failed',problem='P004-hydraulic',electrochemistry_solved=False,
                paper_current_density_reproduced=False,ioh_enabled=with_ioh,evaluations=[])
    try:
        f=make_ioh_problem(p) if with_ioh else p
        context=analyzer_logger(f,output/'ioh',algorithm_name='P004-hydraulic-geometry-check') if with_ioh else nullcontext()
        with context:
            for label in labels:
                value=float(f(presets()[label]));raw=p.last_result;write_json(output/(label+'.json'),raw)
                result['evaluations'].append(dict(label=label,objective=value,metrics=raw['metrics'],case_directory=raw['case_directory'],library_id=raw['library_id']))
        if with_ioh:
            if f.state.evaluations!=len(labels):raise RuntimeError('Unexpected IOH evaluation count')
            files=list((output/'ioh').glob('IOHprofiler*.json'))+list((output/'ioh').rglob('*.dat'))
            if len(files)<2:raise RuntimeError('IOH logs are missing')
            result.update(ioh_evaluations=int(f.state.evaluations),ioh_files=[str(v.relative_to(output)) for v in files])
        result['status']='passed'
    except Exception as exc:result['error']=str(exc);raise
    finally:write_json(output/'smoke.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description='P004 discrete PEMFC design library; fuel-cell labels required for current objective')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('library');p.add_argument('--verify',action='store_true')
    p=sub.add_parser('sample');p.add_argument('--count',type=int,default=480);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,required=True)
    for name in ('decode','generate','evaluate','hydraulic-evaluate'):
        p=sub.add_parser(name);g=p.add_mutually_exclusive_group()
        g.add_argument('--id',type=int);g.add_argument('--design',type=Path)
        g.add_argument('--preset',choices=tuple(presets()),default='serpentine')
        if name=='generate':p.add_argument('--case',type=Path,required=True);p.add_argument('--config',type=Path)
        if name=='hydraulic-evaluate':p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True);_runtime(p)
        if name=='evaluate':_current_data(p)
    p=sub.add_parser('hydraulic-smoke');p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path)
    p.add_argument('--compare',action='store_true');p.add_argument('--ioh',action='store_true');_runtime(p)
    p=sub.add_parser('run');p.add_argument('case',type=Path);_runtime(p)
    p=sub.add_parser('train');p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--training-config',type=Path)
    p=sub.add_parser('rank');p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('search');_current_data(p);p.add_argument('--output',type=Path,required=True);p.add_argument('--budget',type=int,default=20);p.add_argument('--seed',type=int,default=42);p.add_argument('--ioh',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='library':
            paths=catalog()
            if args.verify and tuple(enumerate_paths())!=paths:raise ValueError('Bundled library differs from fresh DFS enumeration')
            result=dict(count=len(paths),grid=[7,7],fresh_DFS_verified=args.verify,figure4_ids={k:path_id(v) for k,v in presets().items()},
                        paper_current_density_available=False)
        elif args.command=='sample':
            if not 1<=args.count<=COUNT:raise ValueError('Invalid sample count')
            args.output.mkdir(parents=True,exist_ok=False);ids=random.Random(args.seed).sample(range(COUNT),args.count)
            result=dict(schema_version=1,status='unlabelled_designs',seed=args.seed,quantity='mean_current_density_A_m2',voltage_v=.475,
                source={'kind':'external_cfd','description':'','model_id':''},samples=[dict(library_id=i,path=list(catalog()[i]),current_density_A_m2=None) for i in ids])
            write_json(args.output/'labels-to-fill.json',result)
            result=dict(status='unlabelled_designs',count=len(ids),file=str((args.output/'labels-to-fill.json').resolve()),note='No author labels or CFD results generated')
        elif args.command=='train':
            from .surrogate import train,TrainingConfig
            config=TrainingConfig(**json.loads(args.training_config.read_text())) if args.training_config else TrainingConfig()
            result=train(args.dataset,args.output,config)
        elif args.command=='rank':
            from .surrogate import rank
            result=rank(args.model,args.output)
        elif args.command=='search':
            p=P004Problem(dataset=args.dataset,model=args.model)
            result=random_search(p,budget=args.budget,seed=args.seed,output=args.output,with_ioh=args.ioh)
        elif args.command=='run':result=run_case(args.case,Runtime(args.backend,args.image,args.timeout))
        elif args.command=='hydraulic-smoke':
            config=HydraulicConfig(**json.loads(args.config.read_text())) if args.config else HydraulicConfig()
            result=hydraulic_smoke(args.output,config,Runtime(args.backend,args.image,args.timeout),args.ioh,args.compare)
        else:
            p=P004Problem();path=json.loads(args.design.read_text()) if args.design else p.design_from_id(args.id) if args.id is not None else presets()[args.preset]
            if args.command=='decode':result=p.decode(path)
            elif args.command=='evaluate':result=P004Problem(dataset=args.dataset,model=args.model).evaluate_raw(path)
            else:
                config=HydraulicConfig(**json.loads(args.config.read_text())) if args.config else HydraulicConfig()
                if args.command=='generate':result=dict(case_directory=str(generate_case(args.case,path,config)),evaluation_kind='auxiliary_channel_only_CFD',status='generated_not_run')
                else:result=P004HydraulicProblem(args.output,config=config,runtime=Runtime(args.backend,args.image,args.timeout)).evaluate_raw(path)
    except (OSError,ValueError,TypeError,RuntimeError) as exc:parser.exit(1,f'flowfield-p004: {exc}\n')
    print(json.dumps(result,indent=2,allow_nan=False));return 0


def _runtime(p):
    p.add_argument('--backend',choices=('auto','native','docker'),default='auto');p.add_argument('--image',default=DEFAULT_IMAGE)
    p.add_argument('--timeout',type=float,default=1800)

def _current_data(p):
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--dataset',type=Path);group.add_argument('--model',type=Path)

if __name__=='__main__':raise SystemExit(main())
