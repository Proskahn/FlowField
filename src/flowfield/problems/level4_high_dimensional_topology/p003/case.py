"""Fresh, hash-pinned cases for the half-cell topology evaluator."""
import hashlib
from pathlib import Path
import shutil
from ....evaluation import write_json
from .config import MODEL_REVISION
from .mesh import mesh_description, mesh, field, write

SOLVER_INPUTS=('flowfieldP003Foam.C','Make/files','Make/options')
INPUTS=({'0/'+n for n in ('U','p','oxygen','water','design','porosity','diffusionFactor','brinkman')}
        | {'constant/polyMesh/'+n for n in ('points','faces','owner','neighbour','boundary','cellZones','faceZones')}
        | {'system/'+n for n in ('controlDict','fvSchemes','fvSolution')}
        | {'constant/topologyProperties','design.json'} | {'solver/'+n for n in SOLVER_INPUTS})


def generate_case(case,geometry,config):
    from . import P003Problem
    if geometry != P003Problem(config=config).decode(geometry['normalized_design']):
        raise ValueError('P003 geometry must come from decode')
    case=Path(case).resolve();case.mkdir(parents=True,exist_ok=False)
    m=mesh_description(config);mesh(case,m)
    common={'frontBack':'type empty;','symmetry':'type symmetryPlane;'}
    vel=dict(common,inlet='type pressureInletOutletVelocity; value uniform (0 0 0);',
             outlet='type zeroGradient;',walls='type fixedValue; value uniform (0 0 0);')
    press=dict(common,inlet=f'type fixedValue; value uniform {config.inlet_pressure_pa};',
               outlet='type fixedValue; value uniform 0;')
    field(case,'U','0 1 -1 0 0 0 0','(0 0 0)',vel,True)
    field(case,'p','1 -1 -2 0 0 0 0',0,press)
    for name,value in [('oxygen',config.oxygen_inlet),('water',config.water_inlet)]:
        field(case,name,'0 0 0 0 0 0 0',value,dict(common,inlet=f'type fixedValue; value uniform {value};'))
    for name,dim,key in [('design','0 0 0 0 0 0 0','indicator'),('porosity','0 0 0 0 0 0 0','porosity'),
                         ('diffusionFactor','0 0 0 0 0 0 0','diffusivity_factor'),
                         ('brinkman','1 -3 -1 0 0 0 0','brinkman_pa_s_m2')]:
        field(case,name,dim,geometry[key],common)
    props=dict(depth=config.depth_m,rho=config.gas_density_kg_m3,mu=config.viscosity_pa_s,
        temperature=config.temperature_k,voltage=config.cell_voltage_v,openCircuit=config.open_circuit_voltage_v,
        ohmic=config.ohmic_resistance_ohm_m2,exchangeCurrent=config.exchange_current_a_m2,
        referenceOxygen=config.oxygen_reference_mol_m3,alpha=config.cathodic_alpha,
        outerTolerance=config.outer_tolerance,balanceTolerance=config.balance_tolerance)
    write(case/'constant/topologyProperties',''.join(f'{k} {v};\n' for k,v in props.items()))
    write(case/'system/controlDict',f'''application flowfieldP003Foam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {config.iterations};
deltaT 1;
writeControl timeStep;
writeInterval {config.write_interval};
purgeWrite 2;
writeFormat ascii;
writePrecision 12;
runTimeModifiable false;
''')
    write(case/'system/fvSchemes','''ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
 default none;
 div(phi,U) Gauss upwind;
 div(phi,oxygen) Gauss upwind;
 div(phi,water) Gauss upwind;
 div((mu*T(grad(U)))) Gauss linear;
}
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
fluxRequired { default no; p; oxygen; water; }
''')
    write(case/'system/fvSolution','''solvers
{
 p { solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0.0001; maxIter 6000; }
 U { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0.0001; maxIter 3000; }
 oxygen { solver PBiCGStab; preconditioner DILU; tolerance 1e-13; relTol 0.00001; maxIter 4000; }
 water { solver PBiCGStab; preconditioner DILU; tolerance 1e-13; relTol 0.00001; maxIter 4000; }
}
relaxationFactors { fields { p 0.4; } equations { U 0.7; } }
''')
    write_json(case/'design.json',geometry)
    for name in SOLVER_INPUTS:
        dest=case/'solver'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(__file__).parent/'solver'/name,dest)
    manifest=dict(problem='P003',model_revision=MODEL_REVISION,config=config.to_dict(),config_sha256=config.fingerprint,
        dimension=config.dimension,cells=len(m['cells']),mesh_shape=[len(m['y'])-1,len(m['x'])-1],
        area_m2=config.area_m2,volume_fraction=geometry['volume_fraction'],paper_validated=False,
        input_sha256={n:hashlib.sha256((case/n).read_bytes()).hexdigest() for n in sorted(INPUTS)})
    write_json(case/'case.json',manifest);(case/'P003.foam').touch()
    return case
