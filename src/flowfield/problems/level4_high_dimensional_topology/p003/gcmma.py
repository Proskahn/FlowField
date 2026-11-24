"""Full-grid finite-difference GCMMA with explicit material continuation.

Each CFD request has a durable result and identity, so a restart can replay
accepted steps and reuse even a partially completed gradient. Stages have
separate configurations and best-feasible records. No CFD adjoint is claimed.
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import threading
import time

from ....optimization.gcmma import GCMMA, GCMMAOptions, dependencies
from ....optimization.state import ResumeError, run_lock, save_json
from .config import CONTINUATION, TopologyConfig
from .design import decode, preset, volume_gradient

REVISION = 'p003-gcmma-fd-continuation-1'


@dataclass(frozen=True)
class ContinuationOptions:
    stages: tuple = (1, 2, 3)
    iterations_per_stage: int = 60
    fd_step: float = 1e-4
    fd_scheme: str = 'forward'
    move: float = .01
    max_inner: int = 15
    volume_margin: float = 1e-8
    check_gradients: bool = False
    gradient_rtol: float = .1
    gradient_atol: float = 1e-5

    def __post_init__(self):
        if (not self.stages or any(type(s) is not int or s not in range(4) for s in self.stages)
                or list(self.stages) != sorted(set(self.stages))):
            raise ValueError('stages must be distinct, increasing integers in 0..3')
        if type(self.iterations_per_stage) is not int or self.iterations_per_stage < 1:
            raise ValueError('iterations_per_stage must be a positive integer')
        if self.fd_scheme not in ('forward', 'central'):
            raise ValueError('fd_scheme must be forward or central')
        for name in ('fd_step', 'volume_margin', 'gradient_rtol', 'gradient_atol'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.fd_step > .1 or self.volume_margin > 1e-4:
            raise ValueError('Require fd_step <= 0.1 and volume_margin <= 1e-4')
        GCMMAOptions(move=self.move, max_inner=self.max_inner)


class EvaluationBudgetReached(RuntimeError):
    pass


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _versions():
    return {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'mmapy')}


def _source_hash():
    directory = Path(__file__).parent
    paths = sorted(directory.glob('*.py')) + sorted((directory/'solver').rglob('*'))
    paths.append(Path(__file__).parents[3]/'optimization/gcmma.py')
    return _digest([(str(p.relative_to(Path(__file__).parents[3])),
                     hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths if p.is_file()])


def plan(config, options):
    probes = config.dimension*(2 if options.fd_scheme == 'central' else 1)
    steps = len(options.stages)*options.iterations_per_stage
    checks = 12*len(options.stages) if options.check_gradients else 0
    return dict(problem='P003', algorithm='GCMMA', gradient_method='finite_difference',
                cfd_adjoint_available=False, design_variables=config.dimension,
                design_shape=[config.design_ny+1, config.design_nx+1],
                probes_per_gradient=probes, outer_iterations=steps,
                baseline_and_calibration_evaluations=1+len(options.stages),
                maximum_evaluations=1+len(options.stages)+steps*(probes+options.max_inner)+checks,
                nominal_evaluations=1+len(options.stages)+steps*(probes+1)+checks,
                stages=[dict(stage=s, p=CONTINUATION[s][0], q=CONTINUATION[s][1],
                             iterations=options.iterations_per_stage) for s in options.stages],
                normalization='initial calibration power and current standard deviation, fixed across stages',
                constraint_handling='explicit volume inequality; no repair of derivative probes or GCMMA trials',
                seed_policy='deterministic uniform initialization; seeds do not randomize the design')


class EvaluationStore:
    """Bounded parallel CFD and atomic per-request results, without a RAM history."""

    def __init__(self, output, evaluate, max_evaluations, *, workers=1,
                 retry_interrupted=False, retry_failed=False,
                 discard_probe_cases=False, progress=None):
        self.root, self.evaluate = Path(output), evaluate
        self.limit, self.workers = max_evaluations, workers
        self.retry_interrupted, self.progress = retry_interrupted, progress
        self.retry_failed = retry_failed
        self.discard_probe_cases = discard_probe_cases
        self.lock = threading.Lock()
        counter = self.root/'attempt-counter.json'
        self.attempts = json.loads(counter.read_text())['attempts'] if counter.exists() else 0
        self.best = {}

    def _track(self, folder, record):
        if not record['valid'] or not record['feasible']:
            return
        with self.lock:
            if folder not in self.best:
                path = folder/'best-feasible.json'
                self.best[folder] = json.loads(path.read_text()) if path.exists() else None
            incumbent = self.best[folder]
            if incumbent is None or record['objective'] < incumbent['objective']:
                self.best[folder] = record
                save_json(folder/'best-feasible.json', record)

    def _discard_probe_case(self, folder, record):
        if (not self.discard_probe_cases or not record['valid']
                or record['phase'] not in ('gradient', 'gradient-check')):
            return
        location = record['result'].get('case_directory')
        if location is None:
            return
        case = Path(location)
        # Only remove this runner's newly generated, direct-child P003 cases.
        if (case.is_symlink() or case.resolve().parent != (folder/'cases').resolve()
                or not case.name.startswith('P003-')):
            raise RuntimeError(f'Refusing to discard a case outside this run: {case}')
        if case.exists():
            shutil.rmtree(case)

    def get(self, folder, key, x, config, *, phase, iteration):
        folder = Path(folder)
        request_hash = _digest(dict(x=list(x), config=config.to_dict(), phase=phase, iteration=iteration))
        result_path = folder/'evaluations'/f'{key}.json'
        started_path = folder/'started'/f'{key}.json'
        failed = None
        if result_path.exists():
            record = json.loads(result_path.read_text())
            if record.get('request_sha256') != request_hash:
                raise ResumeError(f'Cannot reproduce saved CFD request: {result_path}')
            if not record.get('valid'):
                if not self.retry_failed:
                    raise ResumeError(f'Saved CFD failed: {result_path}; fix the runtime cause, then '
                                      'use --resume --retry-failed for a new counted attempt')
                failed = record
            else:
                self._track(folder, record)
                self._discard_probe_case(folder, record)
                return record
        with self.lock:
            if started_path.exists():
                started = json.loads(started_path.read_text())
                if started.get('request_sha256') != request_hash:
                    raise ResumeError(f'Interrupted CFD request changed: {started_path}')
                if failed is None and not self.retry_interrupted:
                    raise ResumeError('A CFD attempt was interrupted before saving its result. '
                                      'Use --resume --retry-interrupted to authorize a new counted attempt: '
                                      f'{started_path}')
            if self.attempts >= self.limit:
                raise EvaluationBudgetReached('CFD evaluation budget reached')
            if failed is not None:
                archive = self.root/'failed-attempts'
                archive.mkdir(exist_ok=True)
                save_json(archive/f'{failed["evaluation"]:08d}.json', failed)
            self.attempts += 1
            attempt = self.attempts
            save_json(self.root/'attempt-counter.json', dict(attempts=self.attempts))
            started_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            started = dict(evaluation=attempt, request_sha256=request_hash,
                           phase=phase, generation=iteration, key=key)
            # Retain a ledger for failed/interrupted/retried attempts as well.
            ledger = self.root/'attempts'/f'{attempt:08d}.json'
            ledger.parent.mkdir(exist_ok=True)
            save_json(ledger, dict(started, folder=str(folder.relative_to(self.root))))
            save_json(started_path, started)
        start = time.monotonic()
        try:
            result = self.evaluate(list(x), config, folder/'cases')
            objective = result.get('objective')
            unpenalized = result.get('unpenalized_objective')
            if (result.get('status') != 'ok' or result.get('metadata', {}).get('valid_cfd') is False
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in (objective, unpenalized))):
                raise RuntimeError('GCMMA requires successful CFD with a finite unpenalized objective')
            record = dict(started, x=list(x), objective=objective, valid=True,
                          feasible=result.get('feasible') is True, result=result,
                          case_retention=('discard_successful_probe' if self.discard_probe_cases
                                          and phase in ('gradient', 'gradient-check') else 'retain'),
                          wall_time_s=time.monotonic()-start)
        except Exception as exc:
            record = dict(started, x=list(x), objective=None, valid=False, feasible=False,
                          error=f'{type(exc).__name__}: {exc}', wall_time_s=time.monotonic()-start)
            save_json(result_path, record)
            raise
        save_json(result_path, record)
        self._track(folder, record)
        self._discard_probe_case(folder, record)
        if self.progress is not None:
            with self.lock:
                self.progress(attempt, phase, iteration, record)
        return record

    def batch_values(self, folder, requests, config, *, phase, iteration):
        """Keep only scalar probe results in RAM; stop scheduling on failure."""
        def one(request):
            key, x = request
            return self.get(folder, key, x, config, phase=phase,
                            iteration=iteration)['result']['unpenalized_objective']
        results = []
        iterator = iter(requests)
        if self.workers == 1:
            return [one(request) for request in iterator]
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            pending = {}
            def submit_one():
                try:
                    request = next(iterator)
                except StopIteration:
                    return False
                index = len(results)
                results.append(None)
                pending[executor.submit(one, request)] = index
                return True
            for _ in range(self.workers):
                if not submit_one():
                    break
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    results[pending.pop(future)] = future.result()
                for _ in done:
                    submit_one()
        return results


def finite_difference_gradient(x, value, evaluate_batch, *, step=1e-4, scheme='forward'):
    """Unrepaired unit-box probes; central uses second-order one-sided bounds."""
    np, _ = dependencies()
    x = np.asarray(x, dtype=float)
    if (x.ndim != 1 or not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x > 1)
            or not math.isfinite(step) or not 0 < step <= .1 or scheme not in ('forward', 'central')):
        raise ValueError('Invalid finite-difference design or settings')
    rules = []
    for coordinate, v in enumerate(x):
        if scheme == 'forward':
            h = step if v+step <= 1 else -step
            rules.append(((h,), (-1/h, 1/h)))
        elif v-step >= 0 and v+step <= 1:
            rules.append(((-step, step), (0., -1/(2*step), 1/(2*step))))
        else:
            h = step if v+2*step <= 1 else -step
            rules.append(((h, 2*h), (-3/(2*h), 2/h, -1/(2*h))))
    def requests():
        for coordinate, (offsets, _) in enumerate(rules):
            for probe, offset in enumerate(offsets):
                candidate = x.copy()
                candidate[coordinate] += offset
                yield f'fd-{coordinate:05d}-{probe}', candidate.tolist()
    values = iter(evaluate_batch(requests()))
    gradient = []
    for offsets, weights in rules:
        gradient.append(weights[0]*value+sum(w*next(values) for w in weights[1:]))
    gradient = np.asarray(gradient)
    if not np.all(np.isfinite(gradient)):
        raise RuntimeError('Nonfinite finite-difference gradient')
    return gradient


def _check_gradient(x, gradient, evaluate_batch, options):
    np, _ = dependencies()
    x = np.asarray(x)
    directions = []
    rng = np.random.default_rng(0)
    for _ in range(3):
        direction = rng.normal(size=len(x))
        direction[(x <= 2*options.fd_step) | (x >= 1-2*options.fd_step)] = 0
        if not np.any(direction):
            raise RuntimeError('No interior coordinates for a centered gradient check')
        directions.append(direction/np.max(np.abs(direction)))
    def requests():
        for i, direction in enumerate(directions):
            for j, scale in enumerate((1., -1., .5, -.5)):
                yield f'check-{i}-{j}', (x+scale*options.fd_step*direction).tolist()
    values = evaluate_batch(requests())
    checks = []
    for i, direction in enumerate(directions):
        a, b, c, d = values[i*4:i*4+4]
        coarse, fine = (a-b)/(2*options.fd_step), (c-d)/options.fd_step
        prediction = float(gradient@direction)
        tolerance = options.gradient_atol+options.gradient_rtol*max(abs(fine), abs(prediction))
        checks.append(dict(direction=i, predicted=prediction, centered_h=coarse,
                           centered_half_h=fine, tolerance=tolerance,
                           passed=abs(prediction-fine) <= tolerance and abs(coarse-fine) <= tolerance))
    return dict(passed=all(check['passed'] for check in checks), checks=checks,
                method='directional finite differences at h and h/2; no adjoint')


def run_continuation(output, *, config=None, options=None, runtime=None, seed=42,
                     max_evaluations=390000, workers=1, resume=False,
                     retry_interrupted=False, retry_failed=False,
                     discard_probe_cases=False, evaluate=None, progress=None):
    """Run or replay the schedule; every calibration, probe and trial is counted.

    ``evaluate`` is a testing/embedding hook with signature (x, config, case_root).
    The CLI always uses real P003 CFD. A failed solve aborts; no failure penalties
    are differentiated. Resuming never changes the schedule or model settings.
    """
    dependencies()
    config, options = config or TopologyConfig(), options or ContinuationOptions()
    if any(type(value) is not int or value < 1 for value in (max_evaluations, workers)):
        raise ValueError('max_evaluations and workers must be positive integers')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if options.volume_margin >= config.volume_limit:
        raise ValueError('volume_margin must be smaller than volume_limit')
    if (retry_interrupted or retry_failed) and not resume:
        raise ValueError('retry_interrupted and retry_failed require resume')
    output = Path(output).resolve()
    manifest = dict(revision=REVISION, config=config.to_dict(), options=asdict(options),
                    seed=seed, versions=_versions(), source_sha256=_source_hash(),
                    max_evaluations=max_evaluations, discard_probe_cases=discard_probe_cases,
                    plan=plan(config, options))
    manifest = json.loads(json.dumps(manifest))
    if output.exists() and not resume:
        raise FileExistsError(f'GCMMA requires a new output directory: {output}')
    if resume and not (output/'gcmma-run.json').is_file():
        raise ResumeError('Missing gcmma-run.json; a TuRBO run cannot be resumed as GCMMA')
    if evaluate is None:
        from . import P003Problem
        P003Problem(config=config, runtime=runtime).preflight()
        def evaluate(x, c, case_root):
            return P003Problem(case_root, config=c, runtime=runtime).evaluate_raw(x)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output/'.gcmma.lock'):
        if resume:
            saved = json.loads((output/'gcmma-run.json').read_text())
            for key in ('revision', 'config', 'options', 'seed', 'versions', 'source_sha256', 'discard_probe_cases'):
                if saved.get(key) != manifest[key]:
                    raise ResumeError(f'Cannot resume GCMMA with changed {key}')
            if max_evaluations < saved['max_evaluations']:
                raise ResumeError('Cannot decrease the saved CFD evaluation budget')
            if max_evaluations != saved['max_evaluations']:
                saved.setdefault('budget_extensions', []).append(dict(
                    previous=saved['max_evaluations'], current=max_evaluations))
                saved['max_evaluations'] = max_evaluations
                save_json(output/'gcmma-run.json', saved)
        else:
            save_json(output/'gcmma-run.json', manifest)
        store = EvaluationStore(output, evaluate, max_evaluations, workers=workers,
                                retry_interrupted=retry_interrupted, retry_failed=retry_failed,
                                discard_probe_cases=discard_probe_cases, progress=progress)
        summary = dict(status='running', algorithm='GCMMA', gradient_method='finite_difference',
                       cfd_adjoint_available=False, paper_validated=False, completed_stages=[])
        try:
            first_config = replace(config, stage=options.stages[0])
            initial = preset('uniform', first_config)
            calibration = store.get(output/'calibration', 'baseline', initial, first_config,
                                    phase='calibration', iteration=0)
            metrics = calibration['result']['metrics']
            power_scale, std_scale = metrics['power_domain_W'], metrics['current_std_A_m2']
            if not math.isfinite(power_scale) or power_scale <= 0 or not math.isfinite(std_scale) or std_scale < 0:
                raise RuntimeError('Invalid normalization metrics from calibration')
            if std_scale == 0 and config.uniformity_weight:
                raise RuntimeError('Cannot normalize a nonzero uniformity weight by zero initial standard deviation')
            normalized = replace(config, power_scale_w=power_scale, current_std_scale_a_m2=std_scale or 1.)
            save_json(output/'normalized-config.json', normalized.to_dict())
            for stage in options.stages:
                stage_config = replace(normalized, stage=stage)
                folder = output/f'stage-{stage}'
                folder.mkdir(exist_ok=True)
                # These manifests allow geometry discovery; objective histories
                # use accepted-iterations.json, not black-box comparison journals.
                save_json(folder/'experiment.json', dict(
                    algorithm='GCMMA-FD', problem='P003', seed=seed, dimension=config.dimension,
                    config=stage_config.to_dict(), config_sha256=stage_config.fingerprint,
                    gradient_method=options.fd_scheme, iterations=options.iterations_per_stage))
                base = store.get(folder, 'baseline', initial, stage_config, phase='baseline', iteration=0)
                optimizer = GCMMA(initial, options=GCMMAOptions(
                    move=options.move, max_inner=options.max_inner, constraint_tolerance=1e-10))
                accepted = []
                for iteration in range(1, options.iterations_per_stage+1):
                    prefix = f'iteration-{iteration:04d}'
                    x = optimizer.x.ravel().tolist()
                    f = base['result']['unpenalized_objective']
                    def batch(requests, phase='gradient'):
                        return store.batch_values(folder, ((f'{prefix}-{k}', v) for k, v in requests),
                                                  stage_config, phase=phase, iteration=iteration)
                    df = finite_difference_gradient(x, f, batch, step=options.fd_step, scheme=options.fd_scheme)
                    if options.check_gradients and iteration == 1:
                        check = _check_gradient(x, df, lambda requests: batch(requests, 'gradient-check'), options)
                        save_json(folder/'gradient-check.json', check)
                        if not check['passed']:
                            raise RuntimeError(f'Gradient verification failed in {folder}; inspect gradient-check.json '
                                               'and CFD convergence before optimizing')
                    g = decode(x, stage_config)['volume_fraction']-stage_config.volume_limit+options.volume_margin
                    dg = volume_gradient(x, stage_config)
                    optimizer.propose(f, df, [g], [dg])
                    while True:
                        trial_x = optimizer.trial.ravel().tolist()
                        trial = store.get(folder, f'{prefix}-trial-{optimizer.inner:02d}', trial_x, stage_config,
                                          phase='trial', iteration=iteration)
                        trial_g = decode(trial_x, stage_config)['volume_fraction']-stage_config.volume_limit+options.volume_margin
                        if optimizer.observe(trial['result']['unpenalized_objective'], [trial_g]):
                            break
                    base = trial
                    accepted.append(dict(iteration=iteration, evaluation=trial['evaluation'],
                                         objective=trial['result']['unpenalized_objective'],
                                         volume_fraction=trial_g+stage_config.volume_limit-options.volume_margin,
                                         inner_corrections=optimizer.inner))
                    save_json(folder/'accepted-iterations.json', accepted)
                    save_json(folder/'current.json', trial)
                initial = optimizer.x.ravel().tolist()
                stage_summary = dict(status='completed_schedule', stage=stage,
                                     iterations=len(accepted), p=CONTINUATION[stage][0], q=CONTINUATION[stage][1],
                                     current=accepted[-1], best_feasible_file=str(folder/'best-feasible.json'))
                save_json(folder/'summary.json', stage_summary)
                summary['completed_stages'].append(stage_summary)
                save_json(output/'summary.json', dict(summary, attempts=store.attempts))
            summary.update(status='completed_schedule', stop_reason='stage_iteration_limits',
                           final_stage=options.stages[-1], final_design=initial)
        except EvaluationBudgetReached:
            summary.update(status='budget_exhausted', stop_reason='cfd_evaluation_budget')
        except BaseException as exc:
            summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                           error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            summary['attempts'] = store.attempts
            save_json(output/'summary.json', summary)
        return summary
