#!/usr/bin/env python3
"""Verify P006's phase projection against analytic two-phase Darcy flow.

Requires native OpenCFD v2312 or the installed Docker image. Checks uniform
and 1e4 mobility contrast, in serial or MPI, optionally with a solid-only rank.
This is an operator test, not qualification of a full P006 design.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from flowfield.runtime import Runtime
from flowfield.problems.level1_small_continuous.p006.runtime import run_phase


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('auto','native','docker'),default='auto')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--processes',type=int,choices=(1,2),default=2)
    parser.add_argument('--solid-half',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=root/'src/flowfield/problems/level1_small_continuous/p006/solver'
    case=args.output.resolve();case.mkdir(parents=True,exist_ok=False)
    (case/'system').mkdir();(case/'constant').mkdir();(case/'0').mkdir();(case/'solver/Make').mkdir(parents=True)
    def dictionary(path,body):
        (case/path).write_text('FoamFile { version 2.0; format ascii; class dictionary; object '+Path(path).name+'; }\n'+body+'\n')
    dictionary('system/blockMeshDict','''scale 1;
vertices ((0 0 0)(1 0 0)(1 1 0)(0 1 0)(0 0 1)(1 0 1)(1 1 1)(0 1 1));
blocks (hex (0 1 2 3 4 5 6 7) (8 8 8) simpleGrading (1 1 1));
edges ();
boundary (
testInlet { type patch; faces ((0 4 7 3)); }
testOutlet { type patch; faces ((1 2 6 5)); }
walls { type wall; faces ((0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); });''')
    dictionary('system/controlDict','application p006ProjectionCheck; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;')
    dictionary('system/fvSchemes','ddtSchemes { default steadyState; } gradSchemes { default Gauss linear; } divSchemes { default Gauss linear; } laplacianSchemes { default Gauss linear orthogonal; } interpolationSchemes { default linear; } snGradSchemes { default orthogonal; }')
    dictionary('system/fvSolution','solvers {}')
    split=f'1 {args.processes} 1' if args.solid_half else f'{args.processes} 1 1'
    dictionary('system/decomposeParDict',f'numberOfSubdomains {args.processes}; method simple; simpleCoeffs {{ n ({split}); delta 0.001; }}')
    dictionary('constant/modelProperties',f'''solid_half {str(args.solid_half).lower()};
momentum_fraction 0; pressure_relaxation 1;
inlet_flow_ml_min 10.29; inlet_liquid_saturation .7;
contact_angle_deg 70; surface_tension .0627;
porosity_ptl .4; porosity_cl .3; permeability_ptl 1e-12; permeability_cl 1e-13;
permeability_channel 1e-6; liquid_viscosity .000355; hydrogen_viscosity 1e-5; oxygen_viscosity 2.3e-5;''')
    shutil.copy2(root/'tests/openfoam/p006_projection.C',case/'solver/flowfieldP006Foam.C')
    for name in ('Tools.H','ChargeNetwork.H','FlowNetwork.H','Make/files','Make/options'):
        shutil.copy2(source/name,case/'solver'/name)
    runtime=Runtime(backend=args.backend,timeout_s=600)
    info=runtime.resolve({'blockMesh','checkMesh'})
    report=dict(kind='analytic_phase_projection',qualified_cfd=False,runtime=info,
        processes=args.processes,solid_half=args.solid_half,
        input_sha256={str(p.relative_to(case)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(case.rglob('*')) if p.is_file()})
    try:
        runtime.run(case,'blockMesh',info=info)
        log=runtime.run(case,'checkMesh',info=info)
        if 'Mesh OK.' not in log.read_text():raise RuntimeError('Analytic mesh failed checkMesh')
        run_phase(case,'build',runtime,info)
        log=run_phase(case,'solve',runtime,info,processes=args.processes)
        report['checks']=[line for line in log.read_text().splitlines() if line.startswith('PROJECTION_')]
        report['passed']='PROJECTION_CHECK PASSED' in report['checks']
        if not report['passed']:raise RuntimeError('Missing projection check success marker')
    except Exception as exc:
        report.update(passed=False,error=str(exc));raise
    finally:
        (case/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
