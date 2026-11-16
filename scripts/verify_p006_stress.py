#!/usr/bin/env python3
"""Compile and check P006's stress operator against three analytic fields.

Requires the P006 build dependencies and native v2312 or the installed Docker
image. Creates a fresh, small test case. Does not run or qualify P006 CFD.
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
    parser.add_argument('--processes',type=int,choices=(1,2),default=1)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=root/'src/flowfield/problems/level1_small_continuous/p006/solver'
    case=args.output.resolve();case.mkdir(parents=True,exist_ok=False)
    (case/'system').mkdir();(case/'solver/Make').mkdir(parents=True)
    def dictionary(name,body):
        (case/'system'/name).write_text(
            'FoamFile { version 2.0; format ascii; class dictionary; object '+name+'; }\n'+body+'\n')
    dictionary('blockMeshDict','''scale 1;
vertices ((0 0 0)(1 0 0)(1 1 0)(0 1 0)(0 0 1)(1 0 1)(1 1 1)(0 1 1));
blocks (hex (0 1 2 3 4 5 6 7) (8 8 8) simpleGrading (1 1 1));
edges ();
boundary (walls { type wall; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); });''')
    dictionary('controlDict','application p006StressCheck; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;')
    dictionary('fvSchemes','ddtSchemes { default steadyState; } gradSchemes { default Gauss linear; } divSchemes { default Gauss linear; } laplacianSchemes { default Gauss linear orthogonal; } interpolationSchemes { default linear; } snGradSchemes { default orthogonal; }')
    dictionary('fvSolution','solvers { qCheck { solver PBiCGStab; preconditioner DILU; tolerance 1e-10; relTol 0; maxIter 4000; } }')
    dictionary('decomposeParDict',f'numberOfSubdomains {args.processes}; method simple; simpleCoeffs {{ n ({args.processes} 1 1); delta 0.001; }}')
    shutil.copy2(root/'tests/openfoam/p006_stress.C',case/'solver/flowfieldP006Foam.C')
    shutil.copy2(source/'Tools.H',case/'solver/Tools.H')
    for name in ('files','options'):shutil.copy2(source/'Make'/name,case/'solver/Make'/name)
    runtime=Runtime(backend=args.backend,timeout_s=600)
    info=runtime.resolve({'blockMesh','checkMesh'})
    report=dict(kind='manufactured_stress_operator',qualified_cfd=False,runtime=info,
        processes=args.processes,input_sha256={str(p.relative_to(case)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(case.rglob('*')) if p.is_file()})
    try:
        runtime.run(case,'blockMesh',info=info)
        log=runtime.run(case,'checkMesh',info=info)
        if 'Mesh OK.' not in log.read_text():raise RuntimeError('Manufactured mesh failed checkMesh')
        run_phase(case,'build',runtime,info)
        log=run_phase(case,'solve',runtime,info,processes=args.processes)
        report['checks']=[line for line in log.read_text().splitlines()
                          if line.startswith(('STRESS_CHECK','ROW_SCALE_CHECK'))]
        report['passed']='STRESS_CHECK PASSED' in report['checks']
        if not report['passed']:raise RuntimeError('Missing stress check success marker')
    except Exception as exc:
        report.update(passed=False,error=str(exc));raise
    finally:
        (case/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
