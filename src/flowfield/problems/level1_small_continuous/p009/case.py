"""Write immutable inputs for the single-phase Brinkman reconstruction."""
import hashlib
import json
from pathlib import Path
from ....openfoam import foam_file, surface_function
from .config import DOI, MODEL_REVISION


def foam(name, body, cls='dictionary'):
    return foam_file(name, body, cls).replace('FlowField P000', 'FlowField P009')


def plane(name, field, z):
    return f'''
    {name}
    {{
        type surfaceFieldValue; libs ("libfieldFunctionObjects.so");
        writeControl timeStep; writeInterval 1; log false; writeFields false;
        regionType sampledSurface; name {name}; operation areaAverage; fields ({field});
        sampledSurfaceDict
        {{
            type cuttingPlane; planeType pointAndNormal; zone nickel;
            pointAndNormalDict {{ point (0 0 {z}); normal (0 0 1); }}
            interpolate false;
        }}
    }}
'''


def generate_case(path, geometry, config):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    c, g = config, geometry
    files = {}
    files['0/U'] = foam('U', f'''
dimensions [0 1 -1 0 0 0 0]; internalField uniform (0 0 0);
boundaryField
{{
    inlet {{ type fixedValue; value uniform (0 {-g['inlet_speed_m_s']:.16g} 0); }}
    outlet {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}
    walls {{ type noSlip; }}
}}
''', 'volVectorField')
    files['0/p'] = foam('p', '''
dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    walls { type zeroGradient; }
}
''', 'volScalarField')
    files['constant/transportProperties'] = foam('transportProperties',
        f'transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {c.dynamic_viscosity_pa_s/c.density_kg_m3:.16g};')
    files['constant/turbulenceProperties'] = foam('turbulenceProperties', 'simulationType laminar;')
    d = c.porosity**2/c.permeability_m2
    f = c.porosity**3*c.inertial_coefficient_m_inv
    files['constant/fvOptions'] = foam('fvOptions', f'''
nickelResistance
{{
    type explicitPorositySource; active true;
    explicitPorositySourceCoeffs
    {{
        selectionMode cellZone; cellZone nickel; type DarcyForchheimer;
        DarcyForchheimerCoeffs
        {{
            d ({d:.16g} {d:.16g} {d:.16g});
            f ({f:.16g} {f:.16g} {f:.16g});
            coordinateSystem
            {{
                type cartesian; origin (0 0 0);
                coordinateRotation {{ type axesRotation; e1 (1 0 0); e2 (0 1 0); }}
            }}
        }}
    }}
}}
''')
    functions = '''
    speedMagnitude { type mag; libs ("libfieldFunctionObjects.so"); field U; result speed;
        executeControl timeStep; executeInterval 1; writeControl none; }
    speedSquared { type magSqr; libs ("libfieldFunctionObjects.so"); field U; result speed2;
        executeControl timeStep; executeInterval 1; writeControl none; }
'''
    functions += ''.join(surface_function(*args) for args in (
        ('p_inlet', 'inlet', 'areaAverage', 'p'), ('p_outlet', 'outlet', 'areaAverage', 'p'),
        ('q_inlet', 'inlet', 'sum', 'phi'), ('q_outlet', 'outlet', 'sum', 'phi'),
        ('q_walls', 'walls', 'sumMag', 'phi')))
    functions += plane('plane_speed', 'speed', c.measurement_height_mm*.001)
    functions += plane('plane_speed2', 'speed2', c.measurement_height_mm*.001)
    files['system/controlDict'] = foam('controlDict', f'''
application simpleFoam; startFrom startTime; startTime 0; stopAt endTime;
endTime {c.max_iterations}; deltaT 1; writeControl timeStep; writeInterval {c.write_interval};
purgeWrite 2; writeFormat ascii; writePrecision 14; writeCompression off;
timeFormat general; timePrecision 12; runTimeModifiable false;
functions {{{functions}}}
''')
    files['system/fvSchemes'] = foam('fvSchemes', '''
ddtSchemes { default steadyState; }
gradSchemes { default cellLimited Gauss linear 1; }
divSchemes { default none; div(phi,U) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear limited 0.5; }
interpolationSchemes { default linear; }
snGradSchemes { default limited 0.5; }
fluxRequired { default no; p; }
''')
    files['system/fvSolution'] = foam('fvSolution', f'''
solvers
{{
    p {{ solver GAMG; tolerance 1e-10; relTol 0.01; smoother GaussSeidel; }}
    U {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0.01; }}
}}
SIMPLE {{ nNonOrthogonalCorrectors 2; consistent yes;
    residualControl {{ p {c.residual_tolerance}; U {c.residual_tolerance}; }} }}
relaxationFactors {{ fields {{ p 0.3; }} equations {{ U 0.5; }} }}
''')
    # Thin (0.1 mm) sub-inlet layers are anisotropic but well formed. Apply
    # explicit determinant, skewness and nonorthogonality limits together.
    files['system/meshQualityDict'] = foam('meshQualityDict', '''
maxNonOrtho 65;
maxBoundarySkewness 1;
maxInternalSkewness 1;
maxConcave 80;
minVol 1e-20;
minTetQuality 1e-12;
minArea -1;
minTwist .02;
minTriangleTwist -1;
minFaceWeight .1;
minVolRatio .1;
minDeterminant 1e-5;
''')
    files['mesher.py'] = Path(__file__).with_name('geometry.py').read_text()
    for name, value in files.items():
        dest = path/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(value)
    metadata = dict(schema_version=1, problem='P009', model_revision=MODEL_REVISION, paper_doi=DOI,
        replication_status='provisional_hydrodynamic_reconstruction', paper_validated=False,
        config=c.to_dict(), config_sha256=c.fingerprint, geometry=g,
        objective='negative area-weighted mean speed at 1 mm / pressure drop in bar',
        input_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()})
    (path/'case.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    (path/'P009.foam').touch()
    return path


INPUTS = frozenset(('0/U', '0/p', 'constant/transportProperties', 'constant/turbulenceProperties',
    'constant/fvOptions', 'system/controlDict', 'system/fvSchemes', 'system/fvSolution',
    'system/meshQualityDict', 'mesher.py'))
