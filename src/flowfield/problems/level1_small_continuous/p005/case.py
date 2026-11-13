"""Generate a fresh, fully specified stock interFoam v2312 case."""
import hashlib
import json
from pathlib import Path
import shutil
from ....openfoam import foam_file
from .config import DualLayerConfig,MODEL_REVISION,DOI
from .geometry import block_mesh

MONITOR_INPUTS=('Detector.H','detachmentMonitor.C','Make/files','Make/options')
INPUTS={'0/U','0/alpha.water','0/p_rgh','constant/transportProperties','constant/turbulenceProperties',
        'constant/g','system/blockMeshDict','system/controlDict','system/fvSchemes','system/fvSolution','geometry.json'} | {
        'monitor/'+name for name in MONITOR_INPUTS}


def generate_case(path,geometry,config=None):
    from . import P005Problem
    c=config or DualLayerConfig()
    if geometry!=P005Problem(config=c).decode(geometry['normalized_design']):
        raise ValueError('Geometry does not match P005 configuration')
    mesh,description=block_mesh(geometry,c)
    case=Path(path).resolve();case.mkdir(parents=True,exist_ok=False)
    f=lambda name,body,cls='dictionary':foam_file(name,body,cls).replace('FlowField P000','FlowField P005')
    files={
      'system/blockMeshDict':f('blockMeshDict',mesh),
      '0/U':f('U',f'''dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{{
 waterInlet {{ type fixedValue; value uniform (0 0 {c.water_velocity_m_s:.16g}); }}
 gasInlet {{ type fixedValue; value uniform (0 {c.oxygen_velocity_m_s:.16g} 0); }}
 outlet {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}
 electrode {{ type noSlip; }}
 walls {{ type noSlip; }}
}}
''','volVectorField'),
      '0/alpha.water':f('alpha.water',f'''dimensions [0 0 0 0 0 0 0];
internalField uniform 1;
boundaryField
{{
 waterInlet {{ type fixedValue; value uniform 1; }}
 gasInlet {{ type fixedValue; value uniform 0; }}
 outlet {{ type inletOutlet; inletValue uniform 1; value uniform 1; }}
 electrode {{ type constantAlphaContactAngle; theta0 {c.electrode_water_contact_angle_deg}; limit gradient; value uniform 1; }}
 walls {{ type constantAlphaContactAngle; theta0 {c.wall_water_contact_angle_deg}; limit gradient; value uniform 1; }}
}}
''','volScalarField'),
      '0/p_rgh':f('p_rgh','''dimensions [1 -1 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
 waterInlet { type fixedFluxPressure; value uniform 0; }
 gasInlet { type fixedFluxPressure; value uniform 0; }
 outlet { type prghPressure; p uniform 0; value uniform 0; }
 electrode { type fixedFluxPressure; value uniform 0; }
 walls { type fixedFluxPressure; value uniform 0; }
}
''','volScalarField'),
      'constant/transportProperties':f('transportProperties',f'''phases (water oxygen);
water {{ transportModel Newtonian; nu [0 2 -1 0 0 0 0] {c.water_viscosity_pa_s/c.water_density_kg_m3:.16g}; rho [1 -3 0 0 0 0 0] {c.water_density_kg_m3}; }}
oxygen {{ transportModel Newtonian; nu [0 2 -1 0 0 0 0] {c.oxygen_viscosity_pa_s/c.oxygen_density_kg_m3:.16g}; rho [1 -3 0 0 0 0 0] {c.oxygen_density_kg_m3}; }}
sigma [1 0 -2 0 0 0 0] {c.surface_tension_n_m};
'''),
      'constant/turbulenceProperties':f('turbulenceProperties','simulationType laminar;\n'),
      'constant/g':f('g',f'dimensions [0 1 -2 0 0 0 0];\nvalue (0 {-c.gravity_m_s2} 0);\n','uniformDimensionedVectorField'),
      'system/controlDict':f('controlDict',f'''application interFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {c.end_time_s:.16g};
deltaT {min(1e-7,description['maximum_delta_t_s']):.16g};
adjustTimeStep yes;
maxCo {c.max_courant};
maxAlphaCo {c.max_courant};
maxDeltaT {description['maximum_delta_t_s']:.16g};
writeControl adjustableRunTime;
writeInterval {min(c.write_interval_s,c.end_time_s):.16g};
purgeWrite 0;
writeFormat ascii;
writePrecision 12;
writeCompression off;
timeFormat general;
timePrecision 12;
runTimeModifiable false;
functions
{{
 bubbleDetachment
 {{
  type flowfieldDetachment;
  libs ("$FOAM_CASE/lib/libflowfieldP005.so");
  executeControl timeStep;
  executeInterval 1;
  writeControl none;
  waterDensity {c.water_density_kg_m3};
  oxygenDensity {c.oxygen_density_kg_m3};
  gasThreshold {c.gas_threshold};
  minimumBubbleVolume {c.min_bubble_volume_m3:.16g};
  detachedFraction {c.detached_fraction};
  persistence {c.persistence_s:.16g};
  sampleInterval {c.monitor_interval_s:.16g};
  gasTolerance {c.gas_balance_tolerance};
  continuityTolerance {c.continuity_tolerance};
  alphaTolerance {c.alpha_tolerance};
 }}
}}
'''),
      'system/fvSchemes':f('fvSchemes','''ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes
{
 default none;
 div(rhoPhi,U) Gauss linearUpwind grad(U);
 div(phi,alpha) Gauss vanLeer;
 div(phirb,alpha) Gauss linear;
 div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
'''),
      'system/fvSolution':f('fvSolution','''solvers
{
 "alpha.water.*"
 {
  nAlphaCorr 2; nAlphaSubCycles 1; cAlpha 1;
  MULESCorr yes; nLimiterIter 5;
  solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0;
 }
 "pcorr.*" { solver PCG; preconditioner DIC; tolerance 1e-10; relTol 0; }
 p_rgh { solver PCG; preconditioner DIC; tolerance 1e-9; relTol 0.01; }
 p_rghFinal { $p_rgh; relTol 0; }
 U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0; }
}
PIMPLE { momentumPredictor no; nOuterCorrectors 2; nCorrectors 3; nNonOrthogonalCorrectors 0; }
relaxationFactors { equations { ".*" 1; } }
'''),
      'geometry.json':json.dumps(geometry,indent=2)+'\n'}
    for name,body in files.items():
        path=case/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(body)
    for name in MONITOR_INPUTS:
        dest=case/'monitor'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(__file__).parent/'monitor'/name,dest)
    inputs={name:hashlib.sha256((case/name).read_bytes()).hexdigest() for name in sorted(INPUTS)}
    manifest=dict(problem='P005',model_revision=MODEL_REVISION,paper_doi=DOI,paper_validated=False,
        config=c.to_dict(),config_sha256=c.fingerprint,geometry=geometry,mesh=description,input_sha256=inputs)
    (case/'case.json').write_text(json.dumps(manifest,indent=2)+'\n');(case/'P005.foam').touch()
    return case
