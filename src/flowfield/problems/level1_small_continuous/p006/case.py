"""Fresh P006 case generation, with immutable inputs and bundled solver."""
import hashlib
from pathlib import Path
import shutil
from ....evaluation import write_json
from .config import MODEL_REVISION, SCREENING_SPACING_MM, geometry as decode_geometry
from .geometry import build_geometry, mesh, case_files

SOLVER_INPUTS=('flowfieldP006Foam.C','ChargeNetwork.H','Tools.H','FlowNetwork.H','Anderson.H','Make/files','Make/options')
INPUTS=({'0/'+n for n in ('Ul','Ug','p','T','sg','lambda','material','phiS','phiE','phiL','phiG')}
    | {'constant/polyMesh/'+n for n in ('points','faces','owner','neighbour','boundary','cellZones','faceZones')}
    | {'system/'+n for n in ('controlDict','fvSchemes','fvSolution','decomposeParDict','meshQualityDict')}
    | {'constant/modelProperties'} | {'solver/'+n for n in SOLVER_INPUTS})


def generate_case(case, geometry, config, *, initial_state=None):
    from . import P006Problem
    if geometry != P006Problem(config=config).decode(geometry['normalized_design']):
        raise ValueError('Geometry must come from P006.decode')
    if min(geometry['ribs_mm']) <= 0 or min(geometry['widths_mm']) <= 0:
        raise ValueError('Channel geometry overlaps or has a nonpositive width')
    case=Path(case).resolve()
    m=build_geometry(geometry,config)
    initial=None
    if initial_state is not None:
        from .warm_start import load
        initial=load(initial_state,geometry,config,INPUTS,m['ids'])
    case.mkdir(parents=True,exist_ok=False)
    mesh(case,m)
    report=case_files(case,m,config)
    if initial is not None:
        from .warm_start import write as write_initial
        report['initial_state']=write_initial(case,initial)
    if config.mesh in SCREENING_SPACING_MM and not (
            initial is not None and initial[1]['coupling']['restored']):
        # The zero-momentum Darcy projection has zero pseudo-storage and can
        # stall at saturation bounds on reduced grids. Start at the solver's
        # existing first nonzero continuation stage so its adaptive storage is
        # available. Final qualification still requires fraction=1 and heat on.
        with (case/'constant/modelProperties').open('a') as stream:
            stream.write('\ninitial_momentum_fraction 1e-6;\n')
        report['numerical_initialization']=dict(momentum_fraction=1e-6)
    for name in SOLVER_INPUTS:
        target=case/'solver'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(__file__).parent/'solver'/name,target)
    report.update(problem='P006',model_revision=MODEL_REVISION,design_geometry=geometry,
        config=config.to_dict(),config_sha256=config.fingerprint,fidelity=config.fidelity,paper_validated=False,
        input_sha256={n:hashlib.sha256((case/n).read_bytes()).hexdigest() for n in sorted(INPUTS)})
    write_json(case/'case.json',report)
    return case
