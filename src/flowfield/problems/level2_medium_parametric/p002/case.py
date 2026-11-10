"""Generate a new full-cell mesh, case dictionaries and pinned solver sources."""

import hashlib
from pathlib import Path
import shutil

from ....evaluation import write_json
from .config import MODEL_REVISION, parameters_for
from .geometry import geometry as build_geometry, mesh, case_files

SOLVER_INPUTS = ('pemElectrolyzerFoam.C','ChargeNetwork.H','Make/files','Make/options')
INPUTS = ({'0/'+name for name in ('U','p','T','Cgas','material','phiS','phiE')}
          | {'constant/polyMesh/'+name for name in ('points','faces','owner','neighbour','boundary','cellZones','faceZones')}
          | {'system/'+name for name in ('controlDict','fvSchemes','fvSolution','decomposeParDict')}
          | {'constant/pemProperties'} | {'solver/'+name for name in SOLVER_INPUTS})


def generate_case(case, geometry, config):
    from . import P002Problem
    if geometry != P002Problem(config=config).decode(geometry['normalized_design']):
        raise ValueError('Geometry must come from P002.decode')
    case = Path(case).resolve()
    params = parameters_for(geometry, config)
    m = build_geometry(params['geometry'], 'parallel', config.mesh)
    case.mkdir(parents=True, exist_ok=False)
    mesh(case, m)
    report = case_files(case, m, params, config.iterations)
    for name in SOLVER_INPUTS:
        target = case/'solver'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).parent/'solver'/name, target)
    report.update(problem='P002', model_revision=MODEL_REVISION, design_geometry=geometry,
                  config=config.to_dict(), config_sha256=config.fingerprint,
                  paper_validated=False, input_sha256={
                      name:hashlib.sha256((case/name).read_bytes()).hexdigest() for name in sorted(INPUTS)})
    write_json(case/'case.json', report)
    return case
