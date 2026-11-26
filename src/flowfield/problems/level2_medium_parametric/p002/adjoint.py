"""Checked discrete adjoints of the coupled P002 fixed point.

Solve R_y.T lambda = J_y.T and return J_x - lambda.T R_x. State
derivatives are sparse forward automatic differentiation. Geometry derivatives
are coordinate-only differences on a verified unchanged mesh topology.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .adjoint_residual import Residual
from .config import PEMConfig, parameters_for
from .metrics import extract_metrics

REVISION = 'p002-coupled-discrete-adjoint-1'
# A changed primal discretization requires an explicit residual audit and new
# CFD directional checks before these compatibility hashes may be updated.
AUDITED_SOURCES = {
    'solver/pemElectrolyzerFoam.C': '89d1644de3cc8cc85f17eed22182f1a229fcbc81e2c9277832a096d162bc5957',
    'solver/ChargeNetwork.H': '2e8f53d5e623ed8932358bb12c3f6f57c6f70166a1bf19e0479f669085650cfb',
    'geometry.py': '64dece63f0a20c983530502166fe763047b13d2996d0fe8cb78f3a152fd3f888',
}


@dataclass(frozen=True)
class AdjointOptions:
    geometry_step: float = 1e-5
    primal_tolerance: float = 5e-6
    linear_tolerance: float = 1e-8
    max_unknowns: int = 150000

    def __post_init__(self):
        for name in ('geometry_step','primal_tolerance','linear_tolerance'):
            value = getattr(self,name)
            if isinstance(value,bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.geometry_step > .001:
            raise ValueError('geometry_step must be <= 0.001')
        if type(self.max_unknowns) is not int or self.max_unknowns < 1:
            raise ValueError('max_unknowns must be a positive integer')


def weights(config):
    if config.scalarization == 'voltage_v':
        return np.array([1.,0.])
    if config.scalarization == 'pressure_sum_pa':
        return np.array([0.,1.])
    return np.array([config.model['currentDensity']*121e-6,
                     config.model['inletFlow_mL_min']*1e-6/60])


def validate_case(case):
    from . import P002Problem
    from .case import INPUTS
    case = Path(case).resolve()
    manifest = json.loads((case/'case.json').read_text())
    config = PEMConfig(**manifest['config'])
    x = manifest['design_geometry']['normalized_design']
    decoded = P002Problem(config=config).decode(x)
    if (manifest['config_sha256'] != config.fingerprint
            or manifest['design_geometry'] != decoded
            or manifest['parameters'] != parameters_for(decoded,config)):
        raise ValueError('Adjoint case provenance/configuration mismatch')
    source = Path(__file__).parent
    for name,digest in AUDITED_SOURCES.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'P002 adjoint has not been audited for the current {name}')
        if name.startswith('solver/') and hashlib.sha256((case/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'P002 case uses an unaudited {name}')
    if set(manifest['input_sha256']) != INPUTS:
        raise ValueError('Incomplete adjoint primal input manifest')
    for name,digest in manifest['input_sha256'].items():
        if hashlib.sha256((case/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'P002 primal input changed after generation: {name}')
    # Verify generated discretization and boundary dictionaries, not only their
    # self-declared hashes. This is cheap and performs no CFD or compilation.
    import tempfile
    from .geometry import case_files
    from .adjoint_residual import design_mesh
    mesh = design_mesh(x,config)
    mesh['inlet_areas'] = manifest['inlet_areas_m2']
    with tempfile.TemporaryDirectory(prefix='p002-adjoint-inputs-') as temporary:
        expected = Path(temporary)
        case_files(expected,mesh,parameters_for(decoded,config),config.iterations)
        for name in INPUTS:
            if name.startswith(('0/','system/')) or name=='constant/pemProperties':
                if (expected/name).read_bytes() != (case/name).read_bytes():
                    raise ValueError(f'P002 adjoint requires the generated discretization: {name}')
    metrics = extract_metrics(case,config,write_report=False)
    return config,x,metrics


def solve_adjoint(jacobian, objective_jacobian, state_scales, *, tolerance=1e-8):
    """One sparse factorization, two transpose solves; no nonlinear probes."""
    n = len(state_scales)
    A = jacobian[:,:n]
    row_scale = np.asarray(abs(A).multiply(state_scales).sum(axis=1)).ravel()
    if np.any(row_scale <= 0) or not np.all(np.isfinite(row_scale)):
        raise ValueError('Singular/nonfinite P002 residual scaling')
    row_inverse = sparse.diags(1/row_scale)
    equilibrated = (row_inverse@A@sparse.diags(state_scales)).tocsc()
    rhs = objective_jacobian[:,:n].T*state_scales[:,None]
    start = time.monotonic()
    if n > 1000:
        try:
            import pymetis
        except ImportError as exc:
            raise RuntimeError('P002 adjoint requires PyMETIS; install with pip install -e ".[gcmma]"') from exc
        graph = (abs(equilibrated)+abs(equilibrated.T)).tocsr()
        graph.setdiag(0); graph.eliminate_zeros()
        permutation,_ = pymetis.nested_dissection(
            adjacency=pymetis.CSRAdjacency(graph.indptr,graph.indices),options=pymetis.Options(seed=0))
        permutation = np.asarray(permutation)
        ordered = equilibrated[permutation,:][:,permutation].tocsc()
        # The continuity row operation in Residual exposes a pressure diagonal.
        # Nested dissection and static pivoting avoid excessive fill. Explicit
        # iterative refinement and the *unpermuted* residual below validate the
        # resulting solve; a failed/unstable factorization is never accepted.
        lu = splu(ordered,permc_spec='NATURAL',diag_pivot_thresh=0.,options={'Equil':False})
        ordered_rhs = rhs[permutation]
        solution = lu.solve(ordered_rhs,trans='T')
        for _ in range(3):
            solution += lu.solve(ordered_rhs-ordered.T@solution,trans='T')
        adjoints = np.empty_like(solution)
        adjoints[permutation] = solution
        ordering = 'metis_nested_dissection_static_pivoting_with_refinement'
    else:
        lu = splu(equilibrated)
        adjoints = lu.solve(rhs,trans='T')
        ordering = 'superlu_default'
    errors = np.linalg.norm(equilibrated.T@adjoints-rhs,axis=0)/np.maximum(np.linalg.norm(rhs,axis=0),1e-30)
    if not np.all(np.isfinite(adjoints)) or np.any(errors > tolerance):
        raise RuntimeError(f'P002 transpose solve failed its residual test: {errors.tolist()}')
    gradients = objective_jacobian[:,n:]-adjoints.T@(row_inverse@jacobian[:,n:])
    if not np.all(np.isfinite(gradients)):
        raise RuntimeError('Nonfinite P002 adjoint gradient')
    return gradients, dict(linear_relative_residual=errors.tolist(),factor_nonzeros=int(lu.nnz),ordering=ordering,
                          factor_and_solve_seconds=time.monotonic()-start),row_scale


def gradient(case, *, options=None):
    options = options or AdjointOptions()
    start = time.monotonic()
    config,x,metrics = validate_case(case)
    model = Residual(x,config,geometry_step=options.geometry_step)
    if model.size > options.max_unknowns:
        raise ValueError(f'P002 adjoint has {model.size} unknowns; exceeds max_unknowns={options.max_unknowns}. '
                         'Increase the explicit resource limit only with enough sparse-factorization memory.')
    state = model.read_state(case)
    residual,objective = model.evaluate(state)
    row_scale = np.asarray(abs(residual.j[:,:model.size]).multiply(model.scales).sum(axis=1)).ravel()
    if np.any(row_scale <= 0):
        raise RuntimeError('P002 residual has a zero Jacobian row')
    normalized = np.abs(residual.v)/row_scale
    blocks = {name:float(np.max(normalized[part])) for name,part in model.slices.items()}
    if not np.all(np.isfinite(normalized)) or max(blocks.values()) > options.primal_tolerance:
        raise RuntimeError(f'P002 fixed-point residual too large for adjoint: {blocks}; tighten outerTolerance')
    expected = np.array([metrics['objectives'][k] for k in ('voltage_V','pressure_sum_Pa')])
    if not np.allclose(objective.v,expected,rtol=1e-10,atol=1e-10):
        raise RuntimeError('P002 adjoint objective differs from the verified primal')
    gradients,diagnostics,_ = solve_adjoint(residual.j,objective.j.toarray(),model.scales,
                                          tolerance=options.linear_tolerance)
    w = weights(config)
    return dict(status='ok',problem='P002',revision=REVISION,case_directory=str(Path(case).resolve()),
        config=config.to_dict(),config_sha256=config.fingerprint,x=list(x),
        gradient_method='coupled_discrete_adjoint',state_derivatives='sparse_forward_automatic_differentiation',
        geometry_derivatives='fixed_connectivity_coordinate_differences',geometry_step=options.geometry_step,
        objective=float(w@objective.v),objectives=dict(zip(('voltage_V','pressure_sum_Pa'),objective.v.tolist())),
        scalarization=config.scalarization,weights=w.tolist(),gradient=(w@gradients).tolist(),
        objective_gradients=dict(zip(('voltage_V','pressure_sum_Pa'),gradients.tolist())),
        gradient_coordinates='normalized seven-variable design',unknowns=model.size,
        jacobian_nonzeros=int(residual.j.nnz),primal_scaled_residual=blocks,
        nonlinear_cfd_probes=0,transpose_solves=2,wall_time_s=time.monotonic()-start,**diagnostics)
