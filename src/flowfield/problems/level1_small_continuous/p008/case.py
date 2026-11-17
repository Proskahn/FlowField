"""Fresh 3D cases, pinned model sources and EWM calibration."""
import hashlib
from pathlib import Path
import json
import shutil
from ....evaluation import write_json
from .config import MODEL_REVISION,DOI,geometry as decode_geometry
from .geometry import build_geometry,mesh,case_files
SOLVER_INPUTS=('flowfieldP008Foam.C','Transport.H','ChargeNetwork.H','Make/files','Make/options')
INPUTS=({'0/'+n for n in ('U','Ul','Ug','p','T','sg','material','phiS','phiE')}
 |{'constant/polyMesh/'+n for n in ('points','faces','owner','neighbour','boundary','cellZones','faceZones')}
 |{'system/'+n for n in ('controlDict','fvSchemes','fvSolution','meshQualityDict')}
 |{'constant/modelProperties','design.json','calibration.json'}|{'solver/'+n for n in SOLVER_INPUTS})

def generate_case(case,geometry,config):
    from . import P008Problem
    if geometry!=P008Problem(config=config).decode(geometry['normalized_design']):raise ValueError('Geometry must come from P008.decode')
    calibration=config.ewm_calibration
    if calibration:
        from .ewm import score
        score(dict(zip(calibration['names'],calibration['minimum'])),calibration)
    case=Path(case).resolve();case.mkdir(parents=True,exist_ok=False)
    m=build_geometry(geometry,config);mesh(case,m);report=case_files(case,m,config)
    write_json(case/'design.json',geometry);write_json(case/'calibration.json',calibration)
    for name in SOLVER_INPUTS:
        dest=case/'solver'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(__file__).parent/'solver'/name,dest)
    report.update(problem='P008',model_revision=MODEL_REVISION,paper_doi=DOI,paper_validated=False,
        design_geometry=geometry,config=config.to_dict(),config_sha256=config.fingerprint,
        scope='3D AEM: homogeneous channel mixture, separate porous Darcy phases, electrochemistry and heat',
        input_sha256={n:hashlib.sha256((case/n).read_bytes()).hexdigest() for n in sorted(INPUTS)})
    write_json(case/'case.json',report);return case
