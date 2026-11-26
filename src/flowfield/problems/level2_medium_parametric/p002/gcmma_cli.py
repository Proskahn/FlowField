"""P002 discrete-adjoint and GCMMA command-line interfaces."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path


def add_arguments(parser):
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--mesh',choices=('coarse','medium','fine'))
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument('--initial',choices=('paper','baseline'),help='Default paper; the 1 mm baseline is a mesh transition')
    initial.add_argument('--x',type=float,nargs=7,help='Seven normalized design coordinates')
    parser.add_argument('--iterations',type=int,help='Accepted outer iterations (default 30)')
    parser.add_argument('--move',type=float,help='Normalized move limit (default 0.01)')
    parser.add_argument('--max-inner',type=int,help='Trials per outer iteration (default 15)')
    parser.add_argument('--max-evaluations',type=int,help='Total CFD attempts, including checks and failures (default 100)')
    parser.add_argument('--check-gradients',action=argparse.BooleanOptionalAction,default=None,
                        help='Check both objectives using eight independent CFD probes before optimization (default on)')
    parser.add_argument('--check-step',type=float,help='CFD directional check step (default 0.001)')
    parser.add_argument('--check-rtol',type=float,help='Directional derivative relative tolerance (default 0.03)')
    parser.add_argument('--max-unknowns',type=int,help='Explicit sparse adjoint resource guard (default 150000; coarse mesh)')
    parser.add_argument('--resume',action='store_true',help='Replay saved successful/failed requests with the same sources and settings')
    parser.add_argument('--dry-run',action='store_true',help='Validate the design and print costs without running CFD or writing files')


def execute(args):
    from ....runtime import Runtime
    from . import BASELINE_X,PAPER_OPTIMUM_X,PEMConfig
    from .adjoint import AdjointOptions
    from .adjoint_residual import Residual
    from .gcmma import Options,run
    saved = json.loads((args.output/'gcmma-run.json').read_text()) if args.resume else None
    contract = saved['contract'] if saved else {}
    config = PEMConfig(**(json.loads(args.config.read_text()) if args.config else contract.get('config',{})))
    if args.mesh:
        config = replace(config,mesh=args.mesh)
    initial = (args.x if args.x is not None else BASELINE_X if args.initial=='baseline' else
               PAPER_OPTIMUM_X if args.initial=='paper' else contract.get('initial',PAPER_OPTIMUM_X))
    settings = dict(contract.get('options',{}))
    for name in asdict(Options()):
        value = getattr(args,name,None)
        if value is not None:
            settings[name] = value
    options = Options(**settings)
    settings = dict(contract.get('adjoint_options',{}))
    if args.max_unknowns is not None:
        settings['max_unknowns'] = args.max_unknowns
    adjoint_options = AdjointOptions(**settings)
    budget = args.max_evaluations if args.max_evaluations is not None else saved['max_evaluations'] if saved else 100
    if budget < 1:
        raise ValueError('max_evaluations must be positive')
    if args.dry_run:
        model = Residual(initial,config,geometry_step=adjoint_options.geometry_step)
        if model.size > adjoint_options.max_unknowns:
            raise ValueError(f'Adjoint unknowns {model.size} exceed max_unknowns={adjoint_options.max_unknowns}')
        print(json.dumps(dict(problem='P002',algorithm='GCMMA',gradient_method='coupled_discrete_adjoint',
            config=config.to_dict(),initial=list(initial),options=asdict(options),unknowns=model.size,
            adjoint_options=asdict(adjoint_options),max_evaluations=budget,
            nominal_cfd_attempts=1+8*options.check_gradients+options.iterations,
            gradient_cfd_probes=0,validation_cfd_probes=8*options.check_gradients,
            transpose_solves_per_gradient=2,initial_policy='deterministic',
            scalarization=config.scalarization,resource_note='Sparse factorization cost is separate from CFD counts'),indent=2))
        return 0
    def progress(record):
        print(f'CFD {record["evaluation"]}/{budget}: {record["phase"]} '
              f'{record["status"]} objective={record["objective"]}',flush=True)
    result = run(args.output,config=config,initial=initial,options=options,adjoint_options=adjoint_options,
        runtime=Runtime(args.backend,args.image,args.timeout),max_evaluations=budget,resume=args.resume,progress=progress)
    print(json.dumps(result,indent=2))
    return 0 if result['status'] in ('completed_schedule','step_tolerance_reached') else 2
