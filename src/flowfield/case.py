"""Generate a complete, inspectable OpenCFD OpenFOAM v2312 case."""

import hashlib
import json
from pathlib import Path

from .model import ChannelConfig


def foam_file(name, body, cls="dictionary"):
    return f"""/* FlowField P000; OpenCFD OpenFOAM v2312; SI units. */
FoamFile
{{
    version 2.0;
    format ascii;
    class {cls};
    object {name};
}}
{body.strip()}
"""


def _surface(name, patch, operation, field):
    return f"""
    {name}
    {{
        type surfaceFieldValue;
        libs ("libfieldFunctionObjects.so");
        writeControl timeStep;
        writeInterval 1;
        log false;
        writeFields false;
        regionType patch;
        name {patch};
        operation {operation};
        fields ({field});
    }}
"""


def generate_case(path, config: ChannelConfig):
    """Refuse even an empty existing destination; never overwrite a case."""
    path = Path(path).absolute()
    path.mkdir(parents=True, exist_ok=False)
    for name in ("0", "constant", "system"):
        (path / name).mkdir()
    L, w, h = config.length_m, config.width_m, config.height_m
    nx, ny, nz = config.cells
    files = {}
    files["system/blockMeshDict"] = foam_file("blockMeshDict", f"""
convertToMeters 1;
vertices
(
    (0 0 0) ({L:.16g} 0 0) ({L:.16g} {w:.16g} 0) (0 {w:.16g} 0)
    (0 0 {h:.16g}) ({L:.16g} 0 {h:.16g}) ({L:.16g} {w:.16g} {h:.16g}) (0 {w:.16g} {h:.16g})
);
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1));
edges ();
boundary
(
    inlet {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    walls {{ type wall; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
""")
    files["0/U"] = foam_file("U", f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({config.mean_velocity_m_s:.16g} 0 0);
boundaryField
{{
    inlet {{ type fixedValue; value uniform ({config.mean_velocity_m_s:.16g} 0 0); }}
    outlet {{ type zeroGradient; }}
    walls {{ type noSlip; }}
}}
""", "volVectorField")
    files["0/p"] = foam_file("p", """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    walls { type zeroGradient; }
}
""", "volScalarField")
    files["constant/transportProperties"] = foam_file("transportProperties", f"""
transportModel Newtonian;
nu [0 2 -1 0 0 0 0] {config.kinematic_viscosity_m2_s:.16g};
""")
    files["constant/turbulenceProperties"] = foam_file("turbulenceProperties", "simulationType laminar;")
    functions = "".join(_surface(*args) for args in (
        ("p_inlet", "inlet", "areaAverage", "p"),
        ("p_outlet", "outlet", "areaAverage", "p"),
        ("q_inlet", "inlet", "sum", "phi"),
        ("q_outlet", "outlet", "sum", "phi")))
    files["system/controlDict"] = foam_file("controlDict", f"""
application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {config.max_iterations};
deltaT 1;
writeControl timeStep;
writeInterval 100;
purgeWrite 2;
writeFormat ascii;
writePrecision 12;
writeCompression off;
timeFormat general;
timePrecision 10;
runTimeModifiable false;
functions
{{{functions}}}
""")
    files["system/fvSchemes"] = foam_file("fvSchemes", """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
wallDist { method meshWave; }
""")
    files["system/fvSolution"] = foam_file("fvSolution", f"""
solvers
{{
    p {{ solver GAMG; tolerance 1e-10; relTol 0.01; smoother GaussSeidel; }}
    U {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0.01; }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    consistent yes;
    residualControl {{ p {config.residual_tolerance:.16g}; U {config.residual_tolerance:.16g}; }}
}}
relaxationFactors
{{
    fields {{ p 0.3; }}
    equations {{ U 0.7; }}
}}
""")
    for name, contents in files.items():
        (path / name).write_text(contents, encoding="utf-8")
    (path / "case.json").write_text(json.dumps({"schema_version": 1, "problem": "P000",
        "openfoam_version": "v2312", "config": config.to_dict(),
        "config_sha256": config.fingerprint,
        "input_sha256": {name: hashlib.sha256(contents.encode()).hexdigest()
                         for name, contents in files.items()}}, indent=2) + "\n", encoding="utf-8")
    (path / "P000.foam").touch()
    return path
