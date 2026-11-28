"""Immutable full-cell cases; generation is separate from qualification."""
import hashlib
import json
import math
from pathlib import Path
import shutil
from ....evaluation import write_json
from .config import DOI,MODEL_REVISION,decode
from .geometry import build_geometry,mesh,field,write,FLUID,PATCHES

SOLVER_INPUTS=('flowfieldP011Foam.C','Common.H','Charge.H','ChargeNewton.H','Gas.H','WaterHeat.H','Make/files','Make/options')
INITIAL=('material','p','T','s','lambda','yr','yw','phiS','phiE','U')
INPUTS=({'0/'+v for v in INITIAL}|{'solver/'+v for v in SOLVER_INPUTS}
 |{'constant/polyMesh/'+v for v in ('points','faces','owner','neighbour','boundary','cellZones','faceZones')}
 |{'system/'+v for v in ('controlDict','fvSchemes','fvSolution','meshQualityDict')}
 |{'constant/modelProperties','design.json'})

def saturation_pressure(t):
    c=t-273.15
    return 101325*10**(-2.1794+.02953*c-9.1837e-5*c*c+1.4454e-7*c**3)

def equilibrium_water(a):
    return .043+17.81*a-39.85*a*a+36*a**3 if a<=1 else 14.003+1.4*(a-1)

def generate_case(case,g,config):
    if g!=decode(g['normalized_design'],config):raise ValueError('Design must come from P011.decode')
    case=Path(case).resolve();case.mkdir(parents=True,exist_ok=False)
    m=build_geometry(g,config);mesh(case,m);mat=m['ids'];t=g['temperature_K']
    w=g['humidity_percent']*.01*saturation_pressure(t)/config.outlet_pressure
    r=[((1-w)*(1. if a>=5 else .21)) if a in FLUID else 0 for a in mat]
    generic={p:'type zeroGradient;' for p in PATCHES}
    vel={p:'type fixedValue; value uniform (0 0 0);' for p in PATCHES}
    pressure={p:'type calculated; value uniform 0;' for p in PATCHES}
    for p in PATCHES:
        if 'Outlet' in p:vel[p]='type zeroGradient;'
    field(case,'U','0 1 -1 0 0 0 0','(0 0 0)',vel,True)
    field(case,'p','1 -1 -2 0 0 0 0',0,pressure)
    field(case,'T','0 0 0 1 0 0 0',t,{p:f'type fixedValue; value uniform {t};' if 'Outlet' not in p else generic[p] for p in PATCHES})
    field(case,'material','0 0 0 0 0 0 0',mat,generic)
    field(case,'s','0 0 0 0 0 0 0',[0 for a in mat],generic)
    field(case,'yr','0 0 0 0 0 0 0',r,generic)
    field(case,'yw','0 0 0 0 0 0 0',[w if a in FLUID else 0 for a in mat],generic)
    field(case,'lambda','0 0 0 0 0 0 0',[equilibrium_water(g['humidity_percent']*.01) if a in (3,4,5) else 0 for a in mat],generic)
    field(case,'phiS','1 2 -3 0 0 -1 0',[config.open_circuit_voltage if a in (0,2,3) else 0 for a in mat],
        dict(generic,cathodeTerminal=f'type fixedValue; value uniform {config.cell_voltage};',anodeTerminal='type fixedValue; value uniform 0;'))
    field(case,'phiE','1 2 -3 0 0 -1 0',[-.001 if a in (3,4,5) else 0 for a in mat],generic)
    properties={k:v for k,v in config.to_dict().items() if k not in ('mesh',)}
    properties.update({k:g[k] for k in ('temperature_K','humidity_percent','cathode_stoichiometry','gdl_porosity')});properties['reactionArea']=g['active_area_m2']
    write(case/'constant/modelProperties',''.join(f'{k} {v};\n' for k,v in properties.items()))
    write(case/'system/controlDict',f'''application flowfieldP011Foam;
startFrom startTime; startTime 0; stopAt endTime; endTime {config.iterations}; deltaT 1;
writeControl timeStep; writeInterval {config.write_interval}; purgeWrite 2;
writeFormat ascii; writePrecision 14; writeCompression off;
timeFormat general; timePrecision 10; runTimeModifiable false;
''')
    write(case/'system/fvSchemes','ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\nfluxRequired { default no; }\n')
    write(case/'system/fvSolution','solvers {}\n')
    write(case/'system/meshQualityDict','maxNonOrtho 10; maxBoundarySkewness 1; maxInternalSkewness 1; maxConcave 80; minVol 1e-20; minTetQuality 1e-12; minArea 1e-20; minTwist 0.02; minDeterminant 0.0001; minFaceWeight 0.1; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;\n')
    for name in SOLVER_INPUTS:
        dest=case/'solver'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(__file__).parent/'solver'/name,dest)
    write_json(case/'design.json',g)
    manifest=dict(problem='P011',model_revision=MODEL_REVISION,paper_doi=DOI,paper_validated=False,
        design_geometry=g,config=config.to_dict(),config_sha256=config.fingerprint,cells=len(mat),
        mesh_description='Orthogonal 3D staircase of linearly tapered channels; endpoint heights are reciprocal widths. Geometric and mesh convergence remain separate verification requirements.',
        geometry_source_sha256=hashlib.sha256((Path(__file__).parent/'geometry.py').read_bytes()).hexdigest(),
        inlet_areas_m2=m['inlet_areas'],geometry_diagnostics=m['approximation'],input_sha256={name:hashlib.sha256((case/name).read_bytes()).hexdigest() for name in sorted(INPUTS)})
    write_json(case/'case.json',manifest);(case/'P011.foam').touch();return case
