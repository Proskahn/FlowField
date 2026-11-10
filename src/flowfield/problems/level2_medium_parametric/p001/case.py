"""Generate P001 inputs without running or overwriting any case."""

import hashlib
import json
from pathlib import Path

from ....case import foam_file as _foam_file, _surface
from .config import DOI, MODEL_REVISION
from .geometry import layout_svg, unit_layout


def foam(name, body, cls='dictionary'):
    return _foam_file(name, body, cls).replace('FlowField P000;', 'FlowField P001 reconstruction;')


def _plane(name, point, normal, operation):
    return f'''
    {name}
    {{
        type surfaceFieldValue; libs ("libfieldFunctionObjects.so");
        writeControl timeStep; writeInterval 1; log false; writeFields false;
        regionType sampledSurface; name {name}; operation {operation}; fields (speed);
        sampledSurfaceDict
        {{
            type cuttingPlane; planeType pointAndNormal;
            pointAndNormalDict {{ point {point}; normal {normal}; }}
            interpolate false;
        }}
    }}
'''


def generate_case(path, geometry, config):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    for directory in ('0', 'constant', 'system'):
        (path/directory).mkdir()
    k = 1.5*(config.inlet_velocity_m_s*config.turbulence_intensity)**2
    epsilon = 0.0845**0.75*k**1.5/config.turbulence_length_m
    files = {}
    files['0/U'] = foam('U', f'''
dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0.01 0);
boundaryField
{{
    inlet {{ type fixedValue; value uniform (0 {config.inlet_velocity_m_s} 0); }}
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
    for name, value, dimensions, wall in (
            ('k', k, '[0 2 -2 0 0 0 0]', 'kqRWallFunction'),
            ('epsilon', epsilon, '[0 2 -3 0 0 0 0]', 'epsilonWallFunction')):
        files['0/'+name] = foam(name, f'''
dimensions {dimensions}; internalField uniform {value:.16g};
boundaryField
{{
    inlet {{ type fixedValue; value uniform {value:.16g}; }}
    outlet {{ type inletOutlet; inletValue uniform {value:.16g}; value uniform {value:.16g}; }}
    walls {{ type {wall}; value uniform {value:.16g}; }}
}}
''', 'volScalarField')
    files['0/nut'] = foam('nut', '''
dimensions [0 2 -1 0 0 0 0]; internalField uniform 0;
boundaryField
{
    inlet { type calculated; value uniform 0; }
    outlet { type calculated; value uniform 0; }
    walls { type nutkWallFunction; value uniform 0; }
}
''', 'volScalarField')
    files['constant/transportProperties'] = foam('transportProperties',
        f'transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {config.kinematic_viscosity_m2_s:.16g};')
    files['constant/turbulenceProperties'] = foam('turbulenceProperties', '''
simulationType RAS;
RAS { RASModel RNGkEpsilon; turbulence on; printCoeffs on; }
''')
    functions = '''
    speedMagnitude
    {
        type mag; libs ("libfieldFunctionObjects.so"); field U; result speed;
        executeControl timeStep; executeInterval 1; writeControl none;
    }
'''
    functions += ''.join(_surface(*args) for args in (
        ('p_inlet', 'inlet', 'areaAverage', 'p'), ('p_outlet', 'outlet', 'areaAverage', 'p'),
        ('q_inlet', 'inlet', 'sum', 'phi'), ('q_outlet', 'outlet', 'sum', 'phi')))
    functions += _plane('uniformity_midplane', '(0 0 0.00375)', '(0 0 1)', 'uniformity')
    functions += _plane('speed_midplane', '(0 0 0.00375)', '(0 0 1)', 'areaAverage')
    for index, y in enumerate((-.135, -.045, .045, .135)):
        functions += _plane(f'uniformity_section{index+1}', f'(0 {y} 0)', '(0 1 0)', 'uniformity')
    files['system/controlDict'] = foam('controlDict', f'''
application simpleFoam; startFrom startTime; startTime 0; stopAt endTime;
endTime {config.max_iterations}; deltaT 1; writeControl timeStep; writeInterval 100;
purgeWrite 2; writeFormat ascii; writePrecision 12; writeCompression off;
timeFormat general; timePrecision 10; runTimeModifiable false;
functions {{{functions}}}
''')
    files['system/fvSchemes'] = foam('fvSchemes', '''
ddtSchemes { default steadyState; }
gradSchemes { default cellLimited Gauss linear 1; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss upwind;
    div(phi,k) bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited 0.5; }
interpolationSchemes { default linear; }
snGradSchemes { default limited 0.5; }
wallDist { method meshWave; }
''')
    files['system/fvSolution'] = foam('fvSolution', f'''
solvers
{{
    p {{ solver GAMG; tolerance 1e-9; relTol 0.01; smoother GaussSeidel; }}
    "(U|k|epsilon)" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.05; }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors 2; consistent yes;
    residualControl {{ p {config.residual_tolerance}; U {config.residual_tolerance};
                      k {config.residual_tolerance}; epsilon {config.residual_tolerance}; }}
}}
relaxationFactors {{ fields {{ p 0.3; }} equations {{ U 0.5; k 0.5; epsilon 0.5; }} }}
''')
    files['layout.svg'] = layout_svg(geometry)
    files['units.json'] = json.dumps(unit_layout(geometry), indent=2)+'\n'
    for name, content in files.items():
        (path/name).write_text(content, encoding='utf-8')
    metadata = {'schema_version': 1, 'problem': 'P001', 'model_revision': MODEL_REVISION,
        'replication_status': 'provisional_geometry_reconstruction', 'paper_doi': DOI,
        'config': config.to_dict(), 'config_sha256': config.fingerprint, 'geometry': geometry,
        'objective': 'negative area-weighted speed uniformity on z=3.75 mm plane, including ports',
        'input_sha256': {name:hashlib.sha256(content.encode()).hexdigest() for name,content in files.items()}}
    (path/'case.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    (path/'P001.foam').touch()
    return path
