"""Auxiliary channel-only hydraulic case; this is not the paper's PEMFC model."""
import hashlib,json
from pathlib import Path
from ....case import foam_file as _template,_surface
from .geometry import decode,footprint_cells,svg
from .config import HydraulicConfig,HYDRAULIC_REVISION

INPUTS={'0/U','0/p','constant/transportProperties','constant/turbulenceProperties',
        'system/controlDict','system/fvSchemes','system/fvSolution','geometry.json','geometry.svg'} | {
        'constant/polyMesh/'+name for name in ('points','faces','owner','neighbour','boundary','cellZones','faceZones')}

def foam_file(name,body,cls='dictionary'):return _template(name,body,cls).replace('FlowField P000','FlowField P004 auxiliary hydraulics')


def _mesh(case,path,config):
    q=config.cells_per_width;h=.001/q
    footprint=footprint_cells(path,q)
    cells=[(i,j,k) for k in range(q) for i,j in footprint];ids={v:i for i,v in enumerate(cells)}
    points=[];point_ids={};internal=[];boundary={p:[] for p in ('inlet','outlet','walls')}
    directions=[((-1,0,0),((0,0,0),(0,0,1),(0,1,1),(0,1,0))),
                ((1,0,0),((1,0,0),(1,1,0),(1,1,1),(1,0,1))),
                ((0,-1,0),((0,0,0),(1,0,0),(1,0,1),(0,0,1))),
                ((0,1,0),((0,1,0),(0,1,1),(1,1,1),(1,1,0))),
                ((0,0,-1),((0,0,0),(0,1,0),(1,1,0),(1,0,0))),
                ((0,0,1),((0,0,1),(1,0,1),(1,1,1),(0,1,1)))]
    for cell,(i,j,k) in enumerate(cells):
        for side,((di,dj,dk),corners) in enumerate(directions):
            neighbor=ids.get((i+di,j+dj,k+dk),-1)
            if 0<=neighbor<cell:continue
            face=[]
            for a,b,c in corners:
                point=(i+a,j+b,k+c)
                if point not in point_ids:point_ids[point]=len(points);points.append(point)
                face.append(point_ids[point])
            if neighbor>=0:internal.append((face,cell,neighbor))
            else:
                patch='inlet' if side==2 and j==0 else 'outlet' if side==3 and j==14*q-1 else 'walls'
                boundary[patch].append((face,cell,-1))
    poly=case/'constant/polyMesh';poly.mkdir(parents=True)
    def write(name,body,cls): (poly/name).write_text(foam_file(name,body,cls))
    write('points',str(len(points))+'\n(\n'+''.join(f'({i*h:.12g} {j*h:.12g} {k*h:.12g})\n' for i,j,k in points)+')\n','vectorField')
    faces=internal+[v for group in boundary.values() for v in group]
    write('faces',str(len(faces))+'\n(\n'+''.join('4('+' '.join(map(str,f))+')\n' for f,_,_ in faces)+')\n','faceList')
    write('owner',str(len(faces))+'\n(\n'+''.join(f'{o}\n' for _,o,_ in faces)+')\n','labelList')
    write('neighbour',str(len(internal))+'\n(\n'+''.join(f'{n}\n' for _,_,n in internal)+')\n','labelList')
    offset=len(internal);body='3\n(\n'
    for patch,group in boundary.items():
        typ='wall' if patch=='walls' else 'patch'
        body+=f'{patch}\n{{ type {typ}; nFaces {len(group)}; startFace {offset}; }}\n';offset+=len(group)
    write('boundary',body+')\n','polyBoundaryMesh')
    for name in ('cellZones','faceZones'):write(name,'0\n()\n','regIOobject')
    return dict(cells=len(cells),faces=len(faces),patch_faces={k:len(v) for k,v in boundary.items()},cell_size_m=h,
                expected_volume_m3=len(cells)*h**3,expected_port_area_m2=q*q*h*h)


def generate_case(path,design,config=None):
    config=config or HydraulicConfig();geometry=decode(design);case=Path(path).resolve();case.mkdir(parents=True,exist_ok=False)
    mesh=_mesh(case,geometry['path'],config)
    files={'0/U':foam_file('U',f'''dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{{
 inlet {{ type fixedValue; value uniform (0 {config.mean_velocity_m_s:.16g} 0); }}
 outlet {{ type zeroGradient; }}
 walls {{ type noSlip; }}
}}
''','volVectorField'),
    '0/p':foam_file('p','''dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
 inlet { type zeroGradient; }
 outlet { type fixedValue; value uniform 0; }
 walls { type zeroGradient; }
}
''','volScalarField'),
    'constant/transportProperties':foam_file('transportProperties',f'transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {config.viscosity_pa_s/config.density_kg_m3:.16g};'),
    'constant/turbulenceProperties':foam_file('turbulenceProperties','simulationType laminar;'),
    'system/controlDict':foam_file('controlDict',f'''application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {config.max_iterations};
deltaT 1;
writeControl timeStep;
writeInterval 1;
purgeWrite 2;
writeFormat ascii;
writePrecision 12;
writeCompression off;
timeFormat general;
runTimeModifiable false;
functions
{{{''.join(_surface(*args) for args in [('p_inlet','inlet','areaAverage','p'),('p_outlet','outlet','areaAverage','p'),('q_inlet','inlet','sum','phi'),('q_outlet','outlet','sum','phi')])}}}
'''),
    'system/fvSchemes':foam_file('fvSchemes','''ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
wallDist { method meshWave; }
fluxRequired { default no; p; }
'''),
    'system/fvSolution':foam_file('fvSolution',f'''solvers
{{
 p {{ solver GAMG; tolerance 1e-12; relTol 0.001; smoother GaussSeidel; }}
 U {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0.001; }}
}}
SIMPLE {{ nNonOrthogonalCorrectors 0; consistent yes; residualControl {{ p {config.residual_tolerance:.16g}; U {config.residual_tolerance:.16g}; }} }}
relaxationFactors {{ fields {{ p 0.3; }} equations {{ U 0.7; }} }}
'''),
    'geometry.json':json.dumps(geometry,indent=2)+'\n','geometry.svg':svg(geometry)}
    for name,text in files.items():
        dest=case/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
    inputs={p.relative_to(case).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in case.rglob('*') if p.is_file()}
    if set(inputs)!=INPUTS:raise ValueError('Unexpected generated case input set')
    manifest=dict(problem='P004-hydraulic',model_revision=HYDRAULIC_REVISION,evaluation_kind='auxiliary_channel_only_CFD',
        paper_current_density_reproduced=False,config=config.to_dict(),config_sha256=config.fingerprint,
        geometry=geometry,mesh=mesh,input_sha256=inputs)
    (case/'case.json').write_text(json.dumps(manifest,indent=2)+'\n');(case/'P004-hydraulic.foam').touch()
    return case
