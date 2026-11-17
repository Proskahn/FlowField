"""Fresh immutable cases for the nondimensional 2D model (paper Eqs. 1–16)."""
import hashlib
from pathlib import Path
import shutil
from ....evaluation import write_json
from .config import MODEL_REVISION, DOI
from .mesh import mesh_description, mesh, field, write

SOLVER_INPUTS=('flowfieldP007Foam.C','Make/files','Make/options')
INPUTS=({'0/'+n for n in ('U','p','T','gamma','brinkman')}
    | {'constant/polyMesh/'+n for n in ('points','faces','owner','neighbour','boundary','cellZones','faceZones')}
    | {'system/'+n for n in ('controlDict','fvSchemes','fvSolution')}
    | {'constant/topologyProperties','design.json'} | {'solver/'+n for n in SOLVER_INPUTS})


def generate_case(case,geometry,config):
    from . import P007Problem
    if geometry != P007Problem(config=config).decode(geometry['normalized_design']):
        raise ValueError('P007 geometry must come from decode')
    case=Path(case).resolve();case.mkdir(parents=True,exist_ok=False)
    m=mesh_description(config);mesh(case,m)
    common={'frontBack':'type empty;'}
    field(case,'U','0 1 -1 0 0 0 0','(0 0 0)',dict(common,
        inlet='type pressureInletVelocity; value uniform (0 0 0);',
        outlet='type zeroGradient;',walls='type fixedValue; value uniform (0 0 0);'),True)
    field(case,'p','0 2 -2 0 0 0 0',0,dict(common,
        inlet='type fixedValue; value uniform 100;',outlet='type fixedValue; value uniform 0;'))
    field(case,'T','0 0 0 0 0 0 0',.5,dict(common,inlet='type fixedValue; value uniform 0;'))
    field(case,'gamma','0 0 0 0 0 0 0',geometry['indicator'],common)
    field(case,'brinkman','0 0 -1 0 0 0 0',geometry['brinkman'],common)
    props=dict(Re=config.reynolds,Pr=config.prandtl,heatCoefficient=config.heat_coefficient,
        pumpingPower=config.pumping_power,depth=.1,outerTolerance=config.outer_tolerance,
        balanceTolerance=config.balance_tolerance,powerRelaxation=config.power_relaxation,
        diagnosticInterval=config.diagnostic_interval)
    write(case/'constant/topologyProperties',''.join(f'{k} {v};\n' for k,v in props.items()))
    write(case/'system/controlDict',f'''application flowfieldP007Foam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {config.iterations};
deltaT 1;
writeControl timeStep;
writeInterval {config.write_interval};
purgeWrite 2;
writeFormat ascii;
writePrecision 14;
runTimeModifiable false;
''')
    write(case/'system/fvSchemes','''ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) Gauss upwind; div(phi,T) Gauss upwind; }
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
fluxRequired { default no; p; T; }
''')
    write(case/'system/fvSolution',f'''solvers
{{
 p {{ solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; maxIter 6000; }}
 U {{ solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; maxIter 3000; }}
 T {{ solver PBiCGStab; preconditioner DILU; tolerance 1e-13; relTol 0; maxIter 4000; }}
}}
relaxationFactors {{ fields {{ p {config.pressure_relaxation}; }} equations {{ U {config.velocity_relaxation}; }} }}
''')
    write_json(case/'design.json',geometry)
    for name in SOLVER_INPUTS:
        dest=case/'solver'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(__file__).parent/'solver'/name,dest)
    write_json(case/'case.json',dict(problem='P007',model_revision=MODEL_REVISION,
        config=config.to_dict(),config_sha256=config.fingerprint,dimension=config.dimension,
        cells=len(m['cells']),mesh_shape=[len(m['y'])-1,len(m['x'])-1],paper_doi=DOI,
        paper_validated=False,scope='2D dimensionless flow and heat topology; no electrochemistry',
        volume_fraction=geometry['volume_fraction'],
        estimated_inputs=['darcy','heat_coefficient','filter_radius'],
        input_sha256={n:hashlib.sha256((case/n).read_bytes()).hexdigest() for n in sorted(INPUTS)}))
    (case/'P007.foam').touch()
    return case

