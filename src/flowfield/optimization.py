"""Strict-budget serial CMA-ES experiments and feasible discrete baselines."""

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import shutil
import sys
import time

from .base import FlowFieldProblem
from .ioh_adapter import analyzer_logger, make_ioh_problem
from .model import ChannelConfig, MESHES
from .registry import get_problem
from .runtime import DEFAULT_IMAGE, Runtime, SimulationError

PROBLEMS = ('P000', 'P001', 'P002', 'P003', 'P004', 'P004-hydraulic')


def _save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _versions():
    result = {'python': platform.python_version()}
    for name in ('cma', 'numpy', 'ioh', 'gmsh'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _validate_options(budget, seed, population, sigma, failure_penalty, max_seconds):
    if type(budget) is not int or budget < 1:
        raise ValueError('budget must be a positive integer')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if type(population) is not int or population < 4:
        raise ValueError('population must be an integer of at least 4')
    for label, value in (('sigma', sigma), ('failure_penalty', failure_penalty), ('max_seconds', max_seconds)):
        if value is None and label != 'sigma':
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{label} must be positive and finite')
    if sigma > .5:
        raise ValueError('sigma is measured in normalized coordinates and must be <= 0.5')


def _method(problem, method):
    if method not in ('auto', 'cmaes', 'random'):
        raise ValueError('Unknown optimization method')
    discrete = getattr(problem, 'variable_type', 'real') == 'integer'
    if method == 'cmaes' and discrete:
        raise ValueError('CMA-ES does not preserve P004 Hamiltonian paths; use auto or random')
    if discrete and not hasattr(problem, 'sample_design'):
        raise ValueError('Discrete search requires a feasible sample_design method')
    return ('random' if discrete else 'cmaes') if method == 'auto' else method


def _initial(problem):
    if problem.name == 'P004' and getattr(problem, 'dataset', None) is not None:
        identifier = 23005 if 23005 in problem.dataset.labels else min(problem.dataset.labels)
        return list(problem.design_from_id(identifier))
    if hasattr(problem, 'initial_design'):
        return list(problem.initial_design())
    if problem.name == 'P000':
        return [1/3, 1/3]
    if problem.name == 'P001':
        from .problems.level2_medium_parametric.p001 import BASELINE_X
        return list(BASELINE_X)
    if problem.name == 'P002':
        from .problems.level2_medium_parametric.p002 import BASELINE_X
        return list(BASELINE_X)
    return [(lo + hi)/2 for lo, hi in zip(problem.lower_bounds, problem.upper_bounds)]


class _FailurePolicy(FlowFieldProblem):
    """Make explicitly penalized CFD failures visible to the same IOH call."""

    def __init__(self, problem, penalty):
        super().__init__(problem.name, problem.dimension, problem.lower_bounds, problem.upper_bounds)
        self.problem, self.penalty = problem, penalty
        self.variable_type = getattr(problem, 'variable_type', 'real')

    @property
    def ioh_name(self):
        suffix = 'raise' if self.penalty is None else format(self.penalty, '.12g')
        return f'{self.problem.ioh_name}_optimizer_failure_{suffix}'

    def decode(self, x):
        return self.problem.decode(x)

    def evaluate_raw(self, x):
        try:
            self.problem(x)
        except SimulationError as exc:
            if self.penalty is None:
                raise
            return dict(status='penalized', objective=self.penalty, objectives={}, constraints={},
                        error=str(exc), metadata=dict(valid_cfd=False,
                        failure_penalty=self.penalty, case_directory=exc.case_directory))
        return self.problem.last_result


def optimize(problem, *, budget=20, seed=42, output, method='auto', population=4,
             sigma=.1, with_ioh=True, failure_penalty=None, max_seconds=None,
             initial=None, progress=None):
    """One baseline plus at most budget-1 proposals; failures consume the budget.

    No restart, final-mean evaluation, validation solve or automatic retry is made.
    A final incomplete population is evaluated but not passed to CMA-ES tell().
    max_seconds is a soft limit checked before starting each evaluation.
    """
    if not isinstance(problem, FlowFieldProblem):
        raise TypeError('optimize requires a FlowFieldProblem')
    _validate_options(budget, seed, population, sigma, failure_penalty, max_seconds)
    method = _method(problem, method)
    x0 = _initial(problem) if initial is None else list(initial)
    problem.decode(x0)
    if problem.name == 'P004' and getattr(problem, 'dataset', None) is None and getattr(problem, 'model', None) is None:
        raise ValueError('P004 current evaluation requires --p004-dataset or --p004-model')
    cma, np = None, None
    if method == 'cmaes':
        try:
            import cma
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('Install CMA-ES with: python -m pip install -e ".[optim]"') from exc
    if with_ioh:
        from .ioh_adapter import _ioh
        _ioh()
    wrapped = _FailurePolicy(problem, failure_penalty)
    objective = make_ioh_problem(wrapped) if with_ioh else wrapped
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    diagonal = method == 'cmaes' and problem.dimension > 100
    algorithm = ('Sep-CMA-ES' if diagonal else 'CMA-ES') if method == 'cmaes' else (
        'FeasiblePathRandomSearch' if wrapped.variable_type == 'integer' else 'RandomSearch')
    config = getattr(problem, 'config', None)
    _save(output/'experiment.json', dict(schema_version=1, algorithm=algorithm, problem=problem.name,
        problem_name=wrapped.ioh_name, dimension=problem.dimension, budget=budget, seed=seed,
        initial_design=x0, population=population if method == 'cmaes' else None,
        sigma_normalized=sigma if method == 'cmaes' else None, diagonal_covariance=diagonal,
        lower_bounds=problem.lower_bounds, upper_bounds=problem.upper_bounds,
        objective_direction='minimize', failure_penalty=failure_penalty, ioh_enabled=with_ioh,
        max_seconds_soft=max_seconds, versions=_versions(),
        config=config.to_dict() if config is not None else None,
        budget_includes_baseline_and_failed_attempts=True, automatic_retries=False))
    summary = dict(status='running', problem=problem.name, algorithm=algorithm, budget=budget,
        seed=seed, attempts=0, successful_evaluations=0, failed_evaluations=0,
        baseline_objective=None, best_objective=None, best_design=None, best_case_directory=None,
        best_feasible_objective=None, best_feasible_design=None, best_feasible_case_directory=None,
        complete_generations=0, partial_generation_evaluations=0, stop_reason=None)
    started = time.monotonic()

    def snapshot():
        summary['wall_time_s'] = time.monotonic() - started
        _save(output/'summary.json', summary)

    def can_start():
        if summary['attempts'] >= budget:
            summary['stop_reason'] = 'evaluation_budget'
            return False
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            summary['stop_reason'] = 'soft_wall_time_limit'
            return False
        return True

    snapshot()
    rng = random.Random(seed)
    seen = set()
    context = analyzer_logger(objective, output/'ioh', algorithm_name=algorithm,
        algorithm_info=f'seed={seed}; budget={budget}; baseline included; failure_penalty={failure_penalty}') if with_ioh else nullcontext()
    try:
        with context, (output/'evaluations.jsonl').open('x') as records, (output/'events.jsonl').open('x') as events:
            def evaluate(x, phase, generation):
                # Start is persisted before entering CFD, even if the process is interrupted.
                if not can_start():
                    return None
                summary['attempts'] += 1
                index = summary['attempts']
                x = [int(v) if wrapped.variable_type == 'integer' else float(v) for v in x]
                seen.add(tuple(x))
                events.write(json.dumps(dict(event='started', evaluation=index, x=x,
                    phase=phase, generation=generation), allow_nan=False) + '\n')
                events.flush()
                snapshot()
                if progress:
                    progress(problem.name, index, budget, 'started')
                began = time.monotonic()
                try:
                    value = float(objective(x))
                    raw = wrapped.last_result
                    valid = raw.get('status') not in ('failed', 'penalized') and raw.get('metadata', {}).get('valid_cfd', True)
                    feasible = valid and raw.get('feasible', True) and all(
                        float(v) <= 1e-12 for v in raw.get('constraints', {}).values())
                    record = dict(evaluation=index, x=x, objective=value, valid=bool(valid),
                        feasible=bool(feasible), phase=phase, generation=generation, result=raw)
                except (Exception, KeyboardInterrupt) as exc:
                    summary['failed_evaluations'] += 1
                    records.write(json.dumps(dict(evaluation=index, x=x, objective=None, valid=False,
                        feasible=False, phase=phase, generation=generation,
                        error=f'{type(exc).__name__}: {exc}', case_directory=getattr(exc,'case_directory',None)), allow_nan=False)+'\n')
                    records.flush()
                    raise
                record['wall_time_s'] = time.monotonic() - began
                records.write(json.dumps(record, allow_nan=False)+'\n')
                records.flush()
                summary['successful_evaluations' if valid else 'failed_evaluations'] += 1
                if phase == 'baseline' and valid:
                    summary['baseline_objective'] = value
                case = raw.get('case_directory', raw.get('metadata',{}).get('case_directory'))
                if valid and (summary['best_objective'] is None or value < summary['best_objective']):
                    summary.update(best_objective=value, best_design=x, best_case_directory=case)
                    _save(output/'best.json', record)
                if feasible and (summary['best_feasible_objective'] is None or value < summary['best_feasible_objective']):
                    summary.update(best_feasible_objective=value, best_feasible_design=x, best_feasible_case_directory=case)
                    _save(output/'best-feasible.json', record)
                events.write(json.dumps(dict(event='finished', evaluation=index, valid=bool(valid)))+'\n')
                events.flush()
                snapshot()
                if progress:
                    progress(problem.name, index, budget, 'ok' if valid else 'failed/penalized')
                return value

            evaluate(x0, 'baseline', 0)
            if method == 'cmaes' and can_start():
                lower, span = np.asarray(problem.lower_bounds), np.asarray(problem.upper_bounds)-problem.lower_bounds
                normalized = (np.asarray(x0)-lower)/span
                es = cma.CMAEvolutionStrategy(normalized.tolist(), sigma, dict(
                    bounds=[0,1], popsize=population, CMA_diagonal=diagonal, CMA_mirrors=0,
                    randn=np.random.RandomState(seed).randn, seed=float('nan'),
                    verbose=-9, verb_log=0, verb_disp=0))
                while can_start():
                    stopped = es.stop()
                    if stopped:
                        summary['stop_reason'] = 'cma_termination'
                        summary['cma_stop'] = {str(k):str(v) for k,v in stopped.items()}
                        break
                    proposals = es.ask()
                    values = []
                    for candidate in proposals:
                        if not can_start():
                            break
                        x = (lower + span*np.asarray(candidate)).tolist()
                        value = evaluate(x, 'candidate', int(es.countiter)+1)
                        if value is None:
                            break
                        values.append(value)
                    if len(values) == population:
                        es.tell(proposals, values)
                        summary['complete_generations'] += 1
                    else:
                        summary['partial_generation_evaluations'] = len(values)
                        break
                    snapshot()
            elif method == 'random':
                while can_start():
                    for _ in range(1000):
                        x = list(problem.sample_design(rng)) if hasattr(problem,'sample_design') else [
                            rng.uniform(lo,hi) for lo,hi in zip(problem.lower_bounds, problem.upper_bounds)]
                        if tuple(x) not in seen:
                            break
                    else:
                        summary['stop_reason'] = 'no_new_design_found'
                        break
                    evaluate(x, 'candidate', None)
            if summary['stop_reason'] is None:
                can_start()
            summary['status'] = ('completed' if summary['failed_evaluations'] == 0 else
                                 'completed_with_failures' if summary['successful_evaluations'] else 'failed')
    except KeyboardInterrupt:
        summary.update(status='interrupted', stop_reason='user_interrupt')
        raise
    except Exception as exc:
        summary.update(status='failed', stop_reason='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if summary['baseline_objective'] is not None and summary['best_objective'] is not None:
            summary['objective_improvement'] = summary['baseline_objective'] - summary['best_objective']
        snapshot()
    return summary


def _make_problem(identifier, args):
    common = dict(output_root=args.output/identifier/'cases', runtime=Runtime(args.backend,args.image,args.timeout))
    if identifier == 'P000':
        return get_problem(identifier, config=ChannelConfig(cells=MESHES[args.mesh]), **common)
    if identifier == 'P001':
        from .problems.level2_medium_parametric.p001 import PPFCConfig
        return get_problem(identifier, config=PPFCConfig(mesh=args.mesh), **common)
    if identifier == 'P002':
        from .problems.level2_medium_parametric.p002 import PEMConfig
        return get_problem(identifier, config=PEMConfig(mesh=args.mesh), **common)
    if identifier == 'P003':
        from .problems.level4_high_dimensional_topology.p003 import TopologyConfig
        return get_problem(identifier, config=TopologyConfig(mesh=args.mesh), **common)
    if identifier == 'P004-hydraulic':
        from .problems.level3_discrete_combinatorial.p004 import HydraulicConfig
        return get_problem(identifier, config=HydraulicConfig(mesh=args.mesh), **common)
    return get_problem('P004', output_root=args.output/identifier/'cases',
                       dataset=args.p004_dataset, model=args.p004_model)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Bounded FlowField optimization: CMA-ES and feasible discrete search')
    parser.add_argument('--problems', nargs='+', choices=PROBLEMS+('all',), default=['all'])
    parser.add_argument('--budget', type=int, default=20, help='Maximum attempts PER problem, including baseline/failures')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--population', type=int, default=4)
    parser.add_argument('--sigma', type=float, default=.1, help='Initial step size in normalized coordinates')
    parser.add_argument('--p003-sigma', type=float, default=.005, help='P003 step size near its sharply projected feasible baseline')
    parser.add_argument('--method', choices=('auto','cmaes','random'), default='auto')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mesh', choices=('coarse','medium','fine'), default='coarse')
    parser.add_argument('--backend', choices=('auto','native','docker'), default='auto')
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--timeout', type=float, default=1800, help='Seconds per mesh/build/solver command')
    parser.add_argument('--ioh', action='store_true')
    parser.add_argument('--failure-penalty', type=float, help='Explicit finite penalty for SimulationError; otherwise stop that problem')
    parser.add_argument('--max-seconds-per-problem', type=float, help='Soft cap; finish an in-flight evaluation before stopping')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--p004-dataset', type=Path)
    group.add_argument('--p004-model', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Print design/algorithm/budget plan without CFD or creating output')
    args = parser.parse_args(argv)
    try:
        _validate_options(args.budget,args.seed,args.population,args.sigma,args.failure_penalty,args.max_seconds_per_problem)
        _validate_options(args.budget,args.seed,args.population,args.p003_sigma,args.failure_penalty,args.max_seconds_per_problem)
        if 'all' in args.problems and len(args.problems) != 1:
            raise ValueError('Use all alone or list individual problem IDs')
        identifiers = list(PROBLEMS) if args.problems == ['all'] else args.problems
        if len(set(identifiers)) != len(identifiers):
            raise ValueError('Duplicate problem IDs would obscure the per-problem budget')
        args.output = args.output.resolve()
        if args.output.exists():
            raise FileExistsError(f'Output must be new: {args.output}')
        problems, entries = [], []
        for identifier in identifiers:
            if identifier == 'P004' and args.p004_dataset is None and args.p004_model is None:
                entries.append(dict(problem=identifier,status='skipped',reason='Current labels/model unavailable; no coupled fuel-cell solver'))
                continue
            problem = _make_problem(identifier,args)
            method = _method(problem,args.method)
            problems.append(problem)
            entries.append(dict(problem=identifier,status='ready',dimension=problem.dimension,
                algorithm=('Sep-CMA-ES' if problem.dimension>100 else 'CMA-ES') if method=='cmaes' else (
                    'FeasiblePathRandomSearch' if getattr(problem,'variable_type','real')=='integer' else 'RandomSearch'),
                maximum_attempts=args.budget, evaluation_kind='current_lookup_or_surrogate' if identifier=='P004' else 'CFD',
                sigma_normalized=(args.p003_sigma if identifier=='P003' else args.sigma) if method=='cmaes' else None,
                initial_design_length=len(_initial(problem))))
        plan = dict(budget_per_problem=args.budget,maximum_attempts=len(problems)*args.budget,
                    maximum_cfd_attempts=sum(p.name!='P004' for p in problems)*args.budget,
                    mesh=args.mesh, seed=args.seed, problems=entries)
        if args.dry_run:
            print(json.dumps(plan,indent=2))
            return 0
        if not problems:
            raise ValueError('No runnable problems selected; P004 requires a current dataset/model')
        # Detect missing Python dependencies before any CFD budget is spent.
        if any(_method(p,args.method)=='cmaes' for p in problems):
            import cma
        if any(p.name=='P001' for p in problems):
            import gmsh
            if gmsh.__version__ != '4.13.1':
                raise RuntimeError('P001 requires gmsh==4.13.1')
        if args.ioh:
            from .ioh_adapter import _ioh
            _ioh()
        if any(p.name != 'P004' for p in problems):
            runtime = Runtime(args.backend,args.image,args.timeout)
            commands = {'checkMesh'}
            if any(p.name in ('P000','P001','P004-hydraulic') for p in problems):
                commands.add('simpleFoam')
            if any(p.name == 'P000' for p in problems):
                commands.add('blockMesh')
            if any(p.name == 'P001' for p in problems):
                commands.add('gmshToFoam')
            resolved = runtime.resolve(commands)
            if resolved['backend']=='native' and any(p.name in ('P002','P003') for p in problems) and not shutil.which('wmake'):
                raise RuntimeError('Native P002/P003 require OpenFOAM wmake')
            plan['runtime_preflight'] = resolved
        args.output.mkdir(parents=True,exist_ok=False)
        _save(args.output/'plan.json',plan)
        suite = dict(status='running', budget_per_problem=args.budget, results=[],
                     skipped=[e for e in entries if e['status']=='skipped'])
        def progress(name,index,budget,state):
            print(f'{name}: {index}/{budget} {state}',file=sys.stderr,flush=True)
        _save(args.output/'suite-summary.json',suite)
        for problem in problems:
            try:
                result = optimize(problem, budget=args.budget, seed=args.seed, output=args.output/problem.name,
                    method=args.method,population=args.population,
                    sigma=args.p003_sigma if problem.name=='P003' else args.sigma,with_ioh=args.ioh,
                    failure_penalty=args.failure_penalty,max_seconds=args.max_seconds_per_problem,progress=progress)
            except Exception as exc:
                path = args.output/problem.name/'summary.json'
                result = json.loads(path.read_text()) if path.exists() else dict(
                    problem=problem.name,status='failed',attempts=0,error=f'{type(exc).__name__}: {exc}')
            except KeyboardInterrupt:
                suite['status']='interrupted'
                path = args.output/problem.name/'summary.json'
                if path.exists():
                    suite['results'].append(json.loads(path.read_text()))
                _save(args.output/'suite-summary.json',suite)
                return 130
            suite['results'].append(result)
            _save(args.output/'suite-summary.json',suite)
        suite['status'] = 'completed' if all(r['status']=='completed' for r in suite['results']) else 'completed_with_failures'
        suite['total_attempts'] = sum(r['attempts'] for r in suite['results'])
        _save(args.output/'suite-summary.json',suite)
        print(json.dumps(suite,indent=2))
        return 0 if suite['status']=='completed' else 1
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        parser.exit(1,f'flowfield-optimize: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
