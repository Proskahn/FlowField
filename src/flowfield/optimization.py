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
from .optimization_state import (ResumeError, append_json, check_legacy_pid, load_history,
    next_session, pristine_initialization, restore_options, run_lock, save_json, validate_experiment)

PROBLEMS = ('P000', 'P001', 'P002', 'P003', 'P004', 'P004-hydraulic')


_save = save_json


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
        self.replay_result = None
        self.variable_type = getattr(problem, 'variable_type', 'real')

    @property
    def ioh_name(self):
        suffix = 'raise' if self.penalty is None else format(self.penalty, '.12g')
        return f'{self.problem.ioh_name}_optimizer_failure_{suffix}'

    def decode(self, x):
        return self.problem.decode(x)

    def evaluate_raw(self, x):
        if self.replay_result is not None:
            result, self.replay_result = self.replay_result, None
            return result
        try:
            self.problem(x)
        except SimulationError as exc:
            if self.penalty is None:
                raise
            return dict(status='penalized', objective=self.penalty, objectives={}, constraints={},
                        error=str(exc), metadata=dict(valid_cfd=False,
                        failure_penalty=self.penalty, case_directory=exc.case_directory))
        return self.problem.last_result


def _proposals(problem, x0, method, population, sigma, seed, state):
    """Yield designs and accept scalar feedback; deterministic across restarts."""
    seen = {tuple(x0)}
    yield list(x0), 'baseline', 0
    if method == 'cmaes':
        import cma
        import numpy as np
        lower = np.asarray(problem.lower_bounds)
        span = np.asarray(problem.upper_bounds)-problem.lower_bounds
        normalized = (np.asarray(x0)-lower)/span
        es = cma.CMAEvolutionStrategy(normalized.tolist(), sigma, dict(
            bounds=[0,1], popsize=population, CMA_diagonal=problem.dimension>100,
            CMA_mirrors=0, randn=np.random.RandomState(seed).randn, seed=float('nan'),
            verbose=-9, verb_log=0, verb_disp=0))
        while True:
            stopped = es.stop()
            if stopped:
                state['stop_reason'] = 'cma_termination'
                state['cma_stop'] = {str(k):str(v) for k,v in stopped.items()}
                return
            proposals = es.ask()
            values = []
            for candidate in proposals:
                x = (lower + span*np.asarray(candidate)).tolist()
                values.append((yield x, 'candidate', int(es.countiter)+1))
                state['partial_generation_evaluations'] = len(values)
            es.tell(proposals, values)
            state['complete_generations'] += 1
            state['partial_generation_evaluations'] = 0
    else:
        rng = random.Random(seed)
        while True:
            for _ in range(1000):
                x = list(problem.sample_design(rng)) if hasattr(problem,'sample_design') else [
                    rng.uniform(lo,hi) for lo,hi in zip(problem.lower_bounds,problem.upper_bounds)]
                if tuple(x) not in seen:
                    break
            else:
                state['stop_reason'] = 'no_new_design_found'
                return
            seen.add(tuple(x))
            yield x, 'candidate', None


def _verify_history(problem, x0, method, population, sigma, seed, history):
    state = dict(complete_generations=0, partial_generation_evaluations=0)
    proposals = _proposals(problem, x0, method, population, sigma, seed, state)
    try:
        request = next(proposals)
        for index, row in enumerate(history):
            x, phase, generation = request
            if row['x'] != x or row['phase'] != phase or row['generation'] != generation:
                raise ResumeError(f'Cannot reproduce saved proposal {index+1}; '
                                  'check original code, dependencies and seed. No new CFD was started.')
            if index+1 < len(history):
                request = proposals.send(row['objective'])
    except StopIteration as exc:
        raise ResumeError('Saved history extends past optimizer termination') from exc
    finally:
        proposals.close()


def optimize(problem, *, output, resume=False, **options):
    """Run or resume a fixed-budget experiment under an exclusive process lock."""
    if not isinstance(problem, FlowFieldProblem):
        raise TypeError('optimize requires a FlowFieldProblem')
    _validate_options(options.get('budget',20),options.get('seed',42),options.get('population',4),
                      options.get('sigma',.1),options.get('failure_penalty'),options.get('max_seconds'))
    _method(problem,options.get('method','auto'))
    output = Path(output).resolve()
    if resume:
        if not output.is_dir() or (not (output/'experiment.json').is_file() and not pristine_initialization(output)):
            raise ResumeError(f'Missing experiment.json in {output}; a console log alone cannot resume a run')
    else:
        output.mkdir(parents=True,exist_ok=False)
    with run_lock(output/'.optimizer.lock'):
        return _optimize(problem,output=output,resume=resume and (output/'experiment.json').exists(),**options)


def _optimize(problem, *, budget=20, seed=42, output, method='auto', population=4,
              sigma=.1, with_ioh=True, failure_penalty=None, max_seconds=None,
              initial=None, progress=None, resume=False):
    method = _method(problem,method)
    x0 = _initial(problem) if initial is None else list(initial)
    problem.decode(x0)
    if problem.name == 'P004' and getattr(problem,'dataset',None) is None and getattr(problem,'model',None) is None:
        raise ValueError('P004 current evaluation requires --p004-dataset or --p004-model')
    if method == 'cmaes':
        try:
            import cma
            import numpy
        except ImportError as exc:
            raise RuntimeError('Install CMA-ES with: python -m pip install -e ".[optim]"') from exc
    if with_ioh:
        from .ioh_adapter import _ioh
        _ioh()
    wrapped = _FailurePolicy(problem,failure_penalty)
    diagonal = method == 'cmaes' and problem.dimension > 100
    algorithm = ('Sep-CMA-ES' if diagonal else 'CMA-ES') if method == 'cmaes' else (
        'FeasiblePathRandomSearch' if wrapped.variable_type == 'integer' else 'RandomSearch')
    config = getattr(problem,'config',None)
    experiment = dict(schema_version=2, algorithm=algorithm, problem=problem.name,
        problem_name=wrapped.ioh_name, dimension=problem.dimension, budget=budget, seed=seed,
        initial_design=x0, population=population if method == 'cmaes' else None,
        sigma_normalized=sigma if method == 'cmaes' else None, diagonal_covariance=diagonal,
        lower_bounds=problem.lower_bounds, upper_bounds=problem.upper_bounds,
        objective_direction='minimize', failure_penalty=failure_penalty, ioh_enabled=with_ioh,
        max_seconds_soft=max_seconds, versions=_versions(),
        config=config.to_dict() if config is not None else None,
        budget_includes_baseline_and_failed_attempts=True, automatic_retries=False,
        recovery='deterministic_journal_replay', soft_time_limit_scope='per_invocation')
    history, pending, previous = [], [], {}
    ioh_output = output/'ioh'
    session = None
    if resume:
        saved = json.loads((output/'experiment.json').read_text())
        validate_experiment(saved,experiment)
        history, pending = load_history(output,budget,failure_penalty)
        _verify_history(problem,x0,method,population,sigma,seed,history)
        if (output/'summary.json').exists():
            previous = json.loads((output/'summary.json').read_text())
        session = next_session(output)
        for name in ('summary.json','best.json','best-feasible.json','checkpoint.json'):
            if (output/name).exists():
                shutil.copyfile(output/name,session/name)
        _save(session/'session.json',dict(replayed_evaluations=len(history),
            recovered_evaluations=[r['evaluation'] for r in pending],versions=_versions(),
            strategy='replay_without_CFD', io_trace='full_history_plus_new_evaluations'))
        if pending:
            with (output/'recoveries.jsonl').open('a') as stream:
                for row in pending:
                    append_json(stream,row)
        ioh_output = session/'ioh'
    else:
        _save(output/'experiment.json',experiment)
    objective = make_ioh_problem(wrapped) if with_ioh else wrapped
    summary = dict(status='running',problem=problem.name,algorithm=algorithm,budget=budget,
        seed=seed,attempts=0,successful_evaluations=0,failed_evaluations=0,
        baseline_objective=None,best_objective=None,best_design=None,best_case_directory=None,
        best_feasible_objective=None,best_feasible_design=None,best_feasible_case_directory=None,
        complete_generations=0,partial_generation_evaluations=0,stop_reason=None,
        resumed=resume,replayed_evaluations=len(history),new_attempts=0,
        recovered_interruptions=sum(bool(r.get('recovered')) for r in history),
        ioh_directory=str(ioh_output) if with_ioh else None)
    started = time.monotonic()
    pending_index = None

    def snapshot():
        summary['session_wall_time_s'] = time.monotonic()-started
        summary['wall_time_s'] = previous.get('wall_time_s',0.)+summary['session_wall_time_s']
        _save(output/'summary.json',summary)
        _save(output/'checkpoint.json',dict(schema_version=1,recovery='deterministic_journal_replay',
            attempts=summary['attempts'],pending_evaluation=pending_index,
            original_budget=budget,experiment='experiment.json',events='events.jsonl',
            evaluations='evaluations.jsonl',recoveries='recoveries.jsonl'))

    def remember(record):
        value, x = record['objective'], record['x']
        valid, feasible, raw = record['valid'],record['feasible'],record['result']
        summary['successful_evaluations' if valid else 'failed_evaluations'] += 1
        if record['phase'] == 'baseline' and valid:
            summary['baseline_objective'] = value
        case = raw.get('case_directory',raw.get('metadata',{}).get('case_directory'))
        if valid and (summary['best_objective'] is None or value < summary['best_objective']):
            summary.update(best_objective=value,best_design=x,best_case_directory=case)
            _save(output/'best.json',record)
        if feasible and (summary['best_feasible_objective'] is None or value < summary['best_feasible_objective']):
            summary.update(best_feasible_objective=value,best_feasible_design=x,best_feasible_case_directory=case)
            _save(output/'best-feasible.json',record)

    proposals = _proposals(problem,x0,method,population,sigma,seed,summary)
    context = analyzer_logger(objective,ioh_output,algorithm_name=algorithm,
        algorithm_info=f'seed={seed}; budget={budget}; resumed={resume}; replay is not new CFD') if with_ioh else nullcontext()
    try:
        with context, (output/'evaluations.jsonl').open('a') as records, (output/'events.jsonl').open('a') as events:
            request = next(proposals)
            while summary['attempts'] < budget:
                replaying = summary['attempts'] < len(history)
                if not replaying and max_seconds is not None and time.monotonic()-started >= max_seconds:
                    summary['stop_reason'] = 'soft_wall_time_limit'
                    break
                x, phase, generation = request
                x = [int(v) if wrapped.variable_type == 'integer' else float(v) for v in x]
                summary['attempts'] += 1
                index = summary['attempts']
                if replaying:
                    record = history[index-1]
                    wrapped.replay_result = record['result']
                    objective(x)  # IOH gets cached values, the CFD problem is not called.
                else:
                    summary['new_attempts'] += 1
                    pending_index = index
                    append_json(events,dict(event='started',evaluation=index,x=x,phase=phase,generation=generation))
                    snapshot()
                    if progress:
                        progress(problem.name,index,budget,'started')
                    began = time.monotonic()
                    try:
                        value = float(objective(x))
                        raw = wrapped.last_result
                        valid = raw.get('status') not in ('failed','penalized') and raw.get('metadata',{}).get('valid_cfd',True)
                        feasible = valid and raw.get('feasible',True) and all(float(v)<=1e-12 for v in raw.get('constraints',{}).values())
                        record = dict(evaluation=index,x=x,objective=value,valid=bool(valid),feasible=bool(feasible),
                            phase=phase,generation=generation,result=raw,wall_time_s=time.monotonic()-began)
                    except (Exception,KeyboardInterrupt) as exc:
                        summary['failed_evaluations'] += 1
                        append_json(records,dict(evaluation=index,x=x,objective=None,valid=False,
                            feasible=False,phase=phase,generation=generation,
                            error=f'{type(exc).__name__}: {exc}',case_directory=getattr(exc,'case_directory',None)))
                        raise
                    append_json(records,record)
                    pending_index = None
                    append_json(events,dict(event='finished',evaluation=index,valid=record['valid']))
                remember(record)
                if progress:
                    progress(problem.name,index,budget,'replayed' if replaying else 'ok' if record['valid'] else 'failed/penalized')
                try:
                    request = proposals.send(record['objective'])
                except StopIteration:
                    break
                snapshot()
            if summary['attempts'] < len(history):
                raise ResumeError('Optimizer terminated before all saved history was consumed')
            if summary['attempts'] >= budget:
                summary['stop_reason'] = 'evaluation_budget'
            summary['status'] = ('completed' if summary['failed_evaluations']==0 else
                'completed_with_failures' if summary['successful_evaluations'] else 'failed')
    except KeyboardInterrupt:
        summary.update(status='interrupted',stop_reason='user_interrupt')
        raise
    except Exception as exc:
        summary.update(status='failed',stop_reason='error',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        proposals.close()
        if summary['baseline_objective'] is not None and summary['best_objective'] is not None:
            summary['objective_improvement'] = summary['baseline_objective']-summary['best_objective']
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
    parser.add_argument('--resume',action='store_true',help='Continue an existing run from its JSON journals; keep the original total budget')
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    try:
        if args.resume:
            args = restore_options(args,{word.split('=')[0] for word in argv if word.startswith('--')})
            if not args.dry_run:
                check_legacy_pid(args.output.resolve())
        return _run_suite(args)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        parser.exit(1,f'flowfield-optimize: {exc}\n')


def _run_suite(args):
    _validate_options(args.budget,args.seed,args.population,args.sigma,args.failure_penalty,args.max_seconds_per_problem)
    _validate_options(args.budget,args.seed,args.population,args.p003_sigma,args.failure_penalty,args.max_seconds_per_problem)
    if 'all' in args.problems and len(args.problems) != 1:
        raise ValueError('Use all alone or list individual problem IDs')
    identifiers = list(PROBLEMS) if args.problems == ['all'] else args.problems
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate problem IDs would obscure the per-problem budget')
    args.output = args.output.resolve()
    if args.output.exists() and not args.resume:
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
    finished = {}
    if args.resume:
        remaining = 0
        for problem in problems:
            folder = args.output/problem.name
            if (folder/'experiment.json').exists():
                history, pending = load_history(folder,args.budget,args.failure_penalty)
                spent = len(history)
                saved = json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {}
                if (saved.get('attempts') == spent and not pending and
                    saved.get('stop_reason') in ('evaluation_budget','cma_termination','no_new_design_found') and
                    saved.get('status') in ('completed','completed_with_failures','failed')):
                    finished[problem.name] = saved
            elif folder.exists() and not pristine_initialization(folder):
                raise ResumeError(f'Incomplete experiment initialization: {folder}; preserve it for inspection')
            else:
                spent = 0
            entry = next(e for e in entries if e['problem']==problem.name)
            entry.update(saved_attempts=spent,remaining_attempts=0 if problem.name in finished else args.budget-spent)
            remaining += entry['remaining_attempts']
        plan.update(resume=True,remaining_attempts=remaining)
    if args.dry_run:
        print(json.dumps(plan,indent=2))
        return 0
    if not problems:
        raise ValueError('No runnable problems selected; P004 requires a current dataset/model')
    active = [p for p in problems if p.name not in finished]
    # Detect missing Python dependencies before any CFD budget is spent.
    if any(_method(p,args.method)=='cmaes' for p in active):
        import cma
    if any(p.name=='P001' for p in active):
        import gmsh
        if gmsh.__version__ != '4.13.1':
            raise RuntimeError('P001 requires gmsh==4.13.1')
    if args.ioh:
        from .ioh_adapter import _ioh
        _ioh()
    if any(p.name != 'P004' for p in active):
        runtime = Runtime(args.backend,args.image,args.timeout)
        commands = {'checkMesh'}
        if any(p.name in ('P000','P001','P004-hydraulic') for p in active):
            commands.add('simpleFoam')
        if any(p.name == 'P000' for p in active):
            commands.add('blockMesh')
        if any(p.name == 'P001' for p in active):
            commands.add('gmshToFoam')
        resolved = runtime.resolve(commands)
        if resolved['backend']=='native' and any(p.name in ('P002','P003') for p in active) and not shutil.which('wmake'):
            raise RuntimeError('Native P002/P003 require OpenFOAM wmake')
        plan['runtime_preflight'] = resolved
    if not args.resume:
        args.output.mkdir(parents=True,exist_ok=False)
    with run_lock(args.output/'.suite.lock'):
        if args.resume:
            session = next_session(args.output)
            if (args.output/'suite-summary.json').exists():
                shutil.copyfile(args.output/'suite-summary.json',session/'suite-summary.json')
            _save(session/'plan.json',plan)
        else:
            _save(args.output/'plan.json',plan)
            options = {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()
                       if k not in ('output','resume','dry_run')}
            _save(args.output/'run-options.json',options)
        return _execute_suite(args,problems,entries,finished)


def _execute_suite(args,problems,entries,finished):
    suite = dict(status='running', budget_per_problem=args.budget, results=[],
                 skipped=[e for e in entries if e['status']=='skipped'])
    def progress(name,index,budget,state):
        print(f'{name}: {index}/{budget} {state}',file=sys.stderr,flush=True)
    _save(args.output/'suite-summary.json',suite)
    for problem in problems:
        if problem.name in finished:
            suite['results'].append(finished[problem.name])
            _save(args.output/'suite-summary.json',suite)
            print(f'{problem.name}: already finished; no evaluations repeated',file=sys.stderr,flush=True)
            continue
        try:
            result = optimize(problem, budget=args.budget, seed=args.seed, output=args.output/problem.name,
                method=args.method,population=args.population,
                sigma=args.p003_sigma if problem.name=='P003' else args.sigma,with_ioh=args.ioh,
                failure_penalty=args.failure_penalty,max_seconds=args.max_seconds_per_problem,progress=progress,
                resume=args.resume and (args.output/problem.name).exists())
        except ResumeError:
            raise
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


if __name__ == '__main__':
    raise SystemExit(main())
