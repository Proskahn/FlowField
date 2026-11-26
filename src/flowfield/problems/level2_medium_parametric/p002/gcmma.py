"""Bounded, replayable P002 GCMMA using checked coupled discrete adjoints."""

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np

from ....optimization.gcmma import GCMMA, GCMMAOptions
from ....optimization.state import ResumeError, append_json, run_lock, save_json
from ....runtime import SimulationError
from . import PAPER_OPTIMUM_X, PEMConfig, P002Problem
from .adjoint import AdjointOptions, REVISION, gradient
from .adjoint_residual import NondifferentiableMesh, design_mesh, mesh_derivatives


@dataclass(frozen=True)
class Options:
    iterations: int = 30
    move: float = .01
    max_inner: int = 15
    check_gradients: bool = True
    check_step: float = .001
    check_rtol: float = .03
    voltage_atol: float = 1e-6
    pressure_atol: float = .05
    step_tolerance: float = 1e-5

    def __post_init__(self):
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if type(self.check_gradients) is not bool:
            raise ValueError('check_gradients must be boolean')
        for name in ('check_step','check_rtol','voltage_atol','pressure_atol','step_tolerance'):
            v = getattr(self,name)
            if isinstance(v,bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.check_step > .01:
            raise ValueError('check_step must be <= 0.01')
        GCMMAOptions(move=self.move,max_inner=self.max_inner)


class BudgetReached(RuntimeError):
    pass


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def source_hash():
    root = Path(__file__).parent
    files = sorted(root.glob('*.py'))+sorted((root/'solver').rglob('*'))
    files.append(Path(__file__).parents[3]/'optimization/gcmma.py')
    return _digest([(str(p.relative_to(Path(__file__).parents[3])),hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in files if p.is_file()])


def check_gradient(x, adjoint, evaluate, config, options):
    """Two independent directions, h and h/2, checking BOTH raw objectives.

    These eight primal probes validate the adjoint; they never supply GCMMA's
    gradient. Central probes must remain in bounds and in the same mesh region.
    """
    x = np.asarray(x)
    base = design_mesh(x,config)
    signature = lambda m: ([len(m[a]) for a in 'xyz'],m['ids'])
    directions = [np.ones(7),np.array([1.,-1.,1.,-1.,1.,-1.,1.])]
    jacobian = np.array([adjoint['objective_gradients'][k] for k in ('voltage_V','pressure_sum_Pa')])
    checks = []
    for i,direction in enumerate(directions):
        # Interior tangent directions at a design bound; exclude fixed axes.
        direction[(x < options.check_step) | (x > 1-options.check_step)] = 0
        if not np.any(direction):
            raise ValueError('No interior coordinates for a centered CFD gradient check')
        direction /= np.linalg.norm(direction)
        estimates = []
        for j,h in enumerate((options.check_step,options.check_step/2)):
            values = []
            for sign in (-1,1):
                candidate = x+sign*h*direction
                if signature(design_mesh(candidate,config)) != signature(base):
                    raise NondifferentiableMesh('CFD gradient-check probes cross a mesh transition; reduce check_step')
                record = evaluate(f'check-{i}-{j}-{sign}',candidate.tolist(),'gradient-check')
                values.append([record['result']['objectives'][k] for k in ('voltage_V','pressure_sum_Pa')])
            estimates.append((np.asarray(values[1])-values[0])/(2*h))
        prediction = jacobian@direction
        tolerance = np.array([options.voltage_atol,options.pressure_atol])+options.check_rtol*np.maximum(np.abs(prediction),np.abs(estimates[1]))
        passed = bool(np.all(np.abs(prediction-estimates[1]) <= tolerance)
                      and np.all(np.abs(estimates[0]-estimates[1]) <= tolerance))
        checks.append(dict(direction=direction.tolist(),predicted=prediction.tolist(),
            centered_h=estimates[0].tolist(),centered_half_h=estimates[1].tolist(),
            tolerance=tolerance.tolist(),passed=passed))
    return dict(passed=all(c['passed'] for c in checks),checks=checks,
                objectives=['voltage_V','pressure_sum_Pa'],cfd_probes=8)


class Store:
    def __init__(self, output, config, runtime, limit, evaluate=None, progress=None):
        self.output,self.config,self.runtime,self.limit = output,config,runtime,limit
        self.evaluate,self.progress = evaluate,progress
        self.attempts = len(list((output/'attempts').glob('*.json')))
        self.best = json.loads((output/'best.json').read_text()) if (output/'best.json').exists() else None

    def get(self, key, x, phase):
        request = dict(x=list(x),config_sha256=self.config.fingerprint,phase=phase,key=key)
        digest = _digest(request)
        path = self.output/'requests'/(key+'.json')
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['request_sha256'] != digest:
                raise ResumeError(f'Replayed P002 request changed: {key}')
            if saved['status'] == 'started':
                raise ResumeError(f'Interrupted CFD request {key}; its spent attempt is retained. '
                                  'Inspect the case and start a new run from the last accepted design.')
            return saved
        if self.attempts >= self.limit:
            raise BudgetReached('P002 CFD attempt budget reached')
        self.attempts += 1
        record = dict(request,request_sha256=digest,evaluation=self.attempts,status='started')
        ledger = self.output/'attempts'/f'{self.attempts:06d}.json'
        save_json(ledger,record); save_json(path,record)
        start = time.monotonic()
        try:
            result = (self.evaluate(x,self.config,self.output/'cases') if self.evaluate else
                      P002Problem(self.output/'cases',config=self.config,runtime=self.runtime).evaluate_raw(x))
            if result.get('status') != 'ok' or not math.isfinite(result['objective']):
                raise SimulationError('P002 GCMMA requires valid, unpenalized CFD')
            record.update(status='ok',valid=True,feasible=True,objective=result['objective'],result=result)
        except (SimulationError,ValueError) as exc:
            record.update(status='failed',valid=False,feasible=False,objective=None,error=str(exc),
                          case_directory=getattr(exc,'case_directory',None))
        record['wall_time_s'] = time.monotonic()-start
        save_json(ledger,record); save_json(path,record)
        with (self.output/'evaluations.jsonl').open('a') as stream:
            append_json(stream,record)
        if record['valid'] and (self.best is None or record['objective'] < self.best['objective']):
            self.best = record; save_json(self.output/'best.json',record)
        if self.progress:
            self.progress(record)
        return record


def run(output, *, config=None, initial=PAPER_OPTIMUM_X, options=None, adjoint_options=None,
        runtime=None, max_evaluations=100, resume=False, progress=None,
        evaluate=None, differentiate=None):
    """Deterministically replay cached CFD and adjoints after budget exhaustion.

    Failed trials are counted and rejected by shrinking the move box. Failed
    baseline/check/adjoint calculations stop the run. No penalty is differentiated.
    ``evaluate``/``differentiate`` are testing hooks; CLI always uses real CFD.
    """
    config = config or PEMConfig()
    options,adjoint_options = options or Options(),adjoint_options or AdjointOptions()
    initial = P002Problem(config=config).decode(initial)['normalized_design']
    mesh_derivatives(initial,config,adjoint_options.geometry_step)
    if type(max_evaluations) is not int or max_evaluations < 1:
        raise ValueError('max_evaluations must be a positive integer')
    output = Path(output).resolve()
    contract = dict(revision=REVISION,problem='P002',algorithm='GCMMA',gradient_method='coupled_discrete_adjoint',
        config=config.to_dict(),config_sha256=config.fingerprint,initial=initial,options=asdict(options),
        adjoint_options=asdict(adjoint_options),source_sha256=source_hash(),
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','mmapy','pymetis')})
    if resume:
        old = json.loads((output/'gcmma-run.json').read_text())
        if old['contract'] != contract or max_evaluations < old['max_evaluations']:
            raise ResumeError('P002 resume requires identical sources/config/options and a nondecreasing budget')
    else:
        output.mkdir(parents=True,exist_ok=False)
    with run_lock(output/'.lock'):
        for name in ('requests','attempts','adjoints'):
            (output/name).mkdir(exist_ok=True)
        save_json(output/'gcmma-run.json',dict(contract=contract,max_evaluations=max_evaluations))
        store = Store(output,config,runtime,max_evaluations,evaluate,progress)
        accepted, adjoint_calls = 0,0
        summary = dict(status='running',accepted_iterations=0)
        try:
            current = store.get('initial',initial,'initial')
            if not current['valid']:
                raise RuntimeError(f'P002 initial CFD failed: {current["error"]}')
            optimizer = GCMMA(initial,options=GCMMAOptions(move=options.move,max_inner=options.max_inner))
            for iteration in range(options.iterations):
                x = optimizer.x.ravel()
                path = output/'adjoints'/f'{iteration:04d}.json'
                if path.exists():
                    adjoint = json.loads(path.read_text())
                    if adjoint['x'] != x.tolist() or adjoint['case_directory'] != current['result']['case_directory']:
                        raise ResumeError('Cached P002 adjoint belongs to a different primal')
                else:
                    adjoint = (differentiate(current['result']) if differentiate else
                               gradient(current['result']['case_directory'],options=adjoint_options))
                    adjoint_calls += 1
                    save_json(path,adjoint)
                if iteration == 0 and options.check_gradients:
                    def checked(key, candidate, phase):
                        record = store.get(key,candidate,phase)
                        if not record['valid']:
                            raise RuntimeError(f'Gradient-check CFD failed: {key}')
                        return record
                    check = check_gradient(x,adjoint,checked,config,options)
                    save_json(output/'gradient-check.json',check)
                    if not check['passed']:
                        raise RuntimeError('P002 adjoint failed independent CFD directional checks')
                # P002 has only box bounds. The generic MMA implementation needs
                # m>=1: this inactive constant inequality adds no physical constraint.
                trial = optimizer.propose(current['objective'],adjoint['gradient'],[-1.],[[0.]*7])
                while True:
                    key = f'iteration-{iteration:04d}-inner-{optimizer.inner:02d}'
                    try:
                        mesh_derivatives(trial,config,adjoint_options.geometry_step)
                    except NondifferentiableMesh as exc:
                        save_json(output/(key+'-mesh-rejection.json'),dict(x=trial.tolist(),reason=str(exc)))
                        trial = optimizer.reject_trial(); continue
                    record = store.get(key,trial.tolist(),'trial')
                    if not record['valid']:
                        trial = optimizer.reject_trial(); continue
                    if optimizer.observe(record['objective'],[-1.]):
                        step = float(np.max(np.abs(optimizer.x.ravel()-x)))
                        current = record; accepted += 1
                        save_json(output/'current.json',dict(record,accepted_iteration=accepted))
                        break
                    trial = optimizer.trial.ravel()
                if step < options.step_tolerance:
                    summary['status'] = 'step_tolerance_reached'; break
            else:
                summary['status'] = 'completed_schedule'
        except BudgetReached:
            summary['status'] = 'budget_exhausted'
        except BaseException as exc:
            summary.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                           error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            summary.update(problem='P002',algorithm='GCMMA',gradient_method='coupled_discrete_adjoint',
                config_sha256=config.fingerprint,accepted_iterations=accepted,cfd_attempts=store.attempts,
                new_adjoint_evaluations=adjoint_calls,best_objective=store.best['objective'] if store.best else None,
                best_design=store.best['x'] if store.best else None,
                gradient_checked=bool(options.check_gradients and (output/'gradient-check.json').exists()
                    and json.loads((output/'gradient-check.json').read_text())['passed']))
            save_json(output/'summary.json',summary)
        return summary
