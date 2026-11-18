"""Generate, solve, inspect and compare the coupled 3D P008 AEM model."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from ....config import load_config
from ....runtime import Runtime
from . import AEMConfig,P008Problem

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('generate','run','extract','evaluate','paper-study','calibrate'))
    p.add_argument('--case',required=True,help='New case/output root, or existing case for run/extract')
    p.add_argument('--config');p.add_argument('--mesh',choices=('coarse','medium','fine'))
    p.add_argument('--width-mm',type=float,choices=(1.,1.2,1.5),default=None);p.add_argument('--depth-mm',type=float)
    p.add_argument('--backend',choices=('auto','native','docker'),default='auto');p.add_argument('--timeout',type=float,default=14400)
    p.add_argument('--cases',nargs='+',help='Qualified case directories to fit a frozen EWM calibration')
    args=p.parse_args(argv)
    case=Path(args.case)
    if args.action=='generate' and case.exists():
        p.error(f'Case directory already exists: {case}. Use a new --case directory; existing files are preserved.')
    if args.action in ('run','extract') and not (case/'case.json').is_file():
        p.error(f'Missing {case / "case.json"}. This is not a complete generated P008 case. '
                'Generate a new case successfully before run/extract; preserve the existing directory.')
    if args.action in ('run','extract','calibrate') and any(v is not None for v in (args.config,args.mesh,args.width_mm,args.depth_mm)):
        p.error('run/extract/calibrate use recorded immutable case configurations')
    if args.action=='paper-study' and (args.width_mm is not None or args.depth_mm is not None):p.error('Paper study uses the reported width/depth panels')
    if args.action=='calibrate':
        if not args.cases:p.error('calibrate requires --cases')
        from .ewm import fit
        from ....registry import create_problem
        from ....evaluation import write_json
        metrics=[];provenance=[];config_fingerprints=set()
        for case in args.cases:
            manifest=json.loads((Path(case)/'case.json').read_text())
            if manifest['problem']!='P008':raise ValueError('All calibration cases must be P008')
            problem=create_problem('P008',config=manifest['config']);result=problem.extract_results(case)
            metrics.append(result['metrics']);provenance.append(dict(case_directory=str(Path(case).resolve()),design=manifest['design_geometry']['normalized_design']))
            config_fingerprints.add(replace(problem.config,ewm_calibration=None).fingerprint)
        if len(config_fingerprints)!=1:raise ValueError('Calibration members must share physical/numerical configuration')
        result=fit(metrics);result['provenance']=provenance;result['config_sha256']=config_fingerprints.pop()
        from .config import MODEL_REVISION
        result.update(qualified_cfd_panel=True,model_revision=MODEL_REVISION)
        output=Path(args.case);output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
        write_json(output.with_name(output.stem+'-config.json'),replace(problem.config,ewm_calibration=result).to_dict())
    else:
        config=load_config(AEMConfig,args.config) if args.config else AEMConfig()
        if args.mesh:config=replace(config,mesh=args.mesh)
        runtime=Runtime(backend=args.backend,timeout_s=args.timeout)
        problem=P008Problem(config=config,output_root=args.case,runtime=runtime)
        x=[{1.5:1/6,1.2:.5,1.:5/6}[args.width_mm or 1.5],((args.depth_mm if args.depth_mm is not None else .8)-.4)/.6]
        if args.action=='generate':result=str(problem.write_case(x,args.case))
        elif args.action=='run':result=problem.run_case(args.case)
        elif args.action=='extract':result=problem.extract_results(args.case)
        elif args.action=='paper-study':
            from .study import paper_study
            result=paper_study(args.case,config,runtime)
        else:result=problem.evaluate_raw(x)
    print(json.dumps(result,indent=2,allow_nan=False));return 0
if __name__=='__main__':raise SystemExit(main())
