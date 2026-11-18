"""Strict-budget optimization, raw observations, Pareto archives and resume."""

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time

from ..problem import AbstractFlowFieldProblem
from ..benchmarking.ioh import analyzer_logger, make_ioh_problem
from ..problems.p000.config import MESHES
from ..problems.level1_small_continuous.p006.config import DEFAULT_TIMEOUT_S as P006_TIMEOUT_S
from ..registry import create_problem, list_problems, problem_config_type
from ..config import load_config
from ..problem import CFDProblem
from .registry import create_sampler, list_optimizers, resolve_optimizer, optimizer_label, optimizer_definition
from ..runtime import DEFAULT_IMAGE, Runtime, SimulationError
from .state import (ResumeError, append_json, check_legacy_pid, load_history,
    next_session, pristine_initialization, restore_options, run_lock, save_json, validate_experiment)
from .nsga2 import ParetoArchive
from .bo import bo_settings
from .turbo import turbo_settings
from .lqcmaes import lq_settings

PROBLEMS = list_problems()


_save = save_json


def _versions():
    result = {'python': platform.python_version()}
    for name in ('cma', 'numpy', 'scipy', 'scikit-learn', 'ioh', 'gmsh'):
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
    return resolve_optimizer(problem, method)


def _initial(problem):
    return list(problem.initial_design())


class _FailurePolicy(AbstractFlowFieldProblem):
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


def _proposals(problem, x0, method, population, sigma, seed, state, *,
               budget=None, bo_initial_points=5, bo_candidates=2048,
               turbo_initial_points=5, turbo_candidates=512):
    """Bridge serial, budgeted evaluations to batch ask/tell algorithms."""
    extra = dict(bo_initial_points=bo_initial_points, bo_candidates=bo_candidates) if method == 'bo' else {}
    if method == 'turbo':
        extra = dict(turbo_initial_points=turbo_initial_points, turbo_candidates=turbo_candidates)
    sampler = create_sampler(method, problem, seed=seed, initial=x0,
                             population=population, sigma=sigma, **extra)
    baseline = yield list(x0), 'baseline', 0, None
    if getattr(problem,'requires_qualified_baseline',False) and (
            baseline.get('status')!='ok' or not baseline.get('metadata',{}).get('valid_cfd',True)
            or not baseline.get('feasible',True)
            or any(float(v)>1e-12 for v in baseline.get('constraints',{}).values())):
        detail = baseline.get('error') or 'The baseline must be valid CFD and satisfy its constraints.'
        case = baseline.get('case_directory') or baseline.get('metadata', {}).get('case_directory')
        raise SimulationError(f'{problem.name}: baseline did not qualify; optimization was not started. {detail}',
                              case_directory=case)
    sampler.initialize(list(x0), baseline)
    state.update(getattr(sampler, 'diagnostics', {}))
    spent = 1
    while budget is None or spent < budget:
        batch = sampler.ask()
        state.update(getattr(sampler, 'diagnostics', {}))
        if not batch:
            state['stop_reason'] = sampler.stop_reason or 'optimizer_termination'
            state.update(getattr(sampler, 'diagnostics', {}))
            return
        results = []
        for x in batch:
            problem.validate_design(x)
            results.append((yield list(x), 'candidate', sampler.generation,
                            getattr(sampler, 'proposal_metadata', None)))
            spent += 1
            if sampler.generation is not None and not sampler.manages_generations:
                state['partial_generation_evaluations'] = len(results)
            if budget is not None and spent >= budget and len(results) < len(batch):
                return
        sampler.tell(batch, results)
        state.update(getattr(sampler, 'diagnostics', {}))
        if sampler.generation is not None and not sampler.manages_generations:
            state['complete_generations'] += 1
            state['partial_generation_evaluations'] = 0


def _verify_history(problem, x0, method, population, sigma, seed, history, **options):
    state = dict(complete_generations=0, partial_generation_evaluations=0)
    proposals = _proposals(problem, x0, method, population, sigma, seed, state, **options)
    try:
        request = next(proposals)
        for index, row in enumerate(history):
            x, phase, generation, proposal = request
            if (row['x'] != x or row['phase'] != phase or row['generation'] != generation
                    or row.get('proposal') != proposal):
                raise ResumeError(f'Cannot reproduce saved proposal {index+1}; '
                                  'check original code, dependencies and seed. No new CFD was started.')
            if index+1 < len(history):
                request = proposals.send(row['result'])
    except StopIteration as exc:
        raise ResumeError('Saved history extends past optimizer termination') from exc
    finally:
        proposals.close()


def optimize(problem, *, output, resume=False, **options):
    """Run or resume a fixed-budget experiment under an exclusive process lock."""
    if not isinstance(problem, AbstractFlowFieldProblem):
        raise TypeError('optimize requires a AbstractFlowFieldProblem')
    if 'optimizer' in options:
        if 'method' in options:
            raise ValueError('Specify optimizer or method, not both')
        options['method'] = options.pop('optimizer')
    _validate_options(options.get('budget',20),options.get('seed',42),options.get('population',4),
                      options.get('sigma',.1),options.get('failure_penalty'),options.get('max_seconds'))
    _method(problem,options.get('method','auto'))
    bo_settings(options.get('bo_initial_points',5), options.get('bo_candidates',2048))
    turbo_settings(problem.dimension, options.get('turbo_initial_points',5), options.get('turbo_candidates',512))
    output = Path(output).resolve()
    if resume:
        if not output.is_dir() or (not (output/'experiment.json').is_file() and not pristine_initialization(output)):
            raise ResumeError(f'Missing experiment.json in {output}; a console log alone cannot resume a run')
    elif output.exists():
        raise FileExistsError(f'Output must be new: {output}')
    if hasattr(problem,'validate_optimization'):problem.validate_optimization()
    problem.preflight()
    if not resume:
        output.mkdir(parents=True,exist_ok=False)
    with run_lock(output/'.optimizer.lock'):
        return _optimize(problem,output=output,resume=resume and (output/'experiment.json').exists(),**options)


def _optimize(problem, *, budget=20, seed=42, output, method='auto', population=4,
              sigma=.1, with_ioh=True, failure_penalty=None, max_seconds=None,
              initial=None, progress=None, resume=False, bo_initial_points=5, bo_candidates=2048,
              turbo_initial_points=5, turbo_candidates=512):
    method = _method(problem,method)
    x0 = _initial(problem) if initial is None else list(initial)
    problem.decode(x0)
    if problem.name == 'P004' and getattr(problem,'dataset',None) is None and getattr(problem,'model',None) is None:
        raise ValueError('P004 current evaluation requires --p004-dataset or --p004-model')
    if method in ('cmaes', 'lqcmaes'):
        try:
            import cma
            import numpy
        except ImportError as exc:
            raise RuntimeError('Install CMA-ES with: python -m pip install -e ".[optim]"') from exc
    if method in ('bo', 'turbo'):
        from .bo import dependencies
        dependencies()
    if method == 'cobyla':
        from .cobyla import dependencies
        dependencies()
    if with_ioh:
        from ..benchmarking.ioh import _ioh
        _ioh()
    wrapped = _FailurePolicy(problem,failure_penalty)
    archive = ParetoArchive() if optimizer_definition(method).multi_objective else None
    diagonal = method in ('cmaes', 'lqcmaes') and problem.dimension > 100
    algorithm = optimizer_label(method, problem)
    config = getattr(problem,'config',None)
    experiment = dict(schema_version=3, algorithm=algorithm, optimizer=method,
        optimizer_revision=optimizer_definition(method).revision, problem=problem.name,
        problem_name=wrapped.ioh_name, dimension=problem.dimension, budget=budget, seed=seed,
        initial_design=x0, population=population if method not in ('random', 'bo', 'turbo', 'cobyla') else None,
        sigma_normalized=sigma if method not in ('random', 'nsga2', 'bo', 'turbo', 'cobyla') else None, diagonal_covariance=diagonal,
        lower_bounds=problem.lower_bounds, upper_bounds=problem.upper_bounds,
        objective_direction='minimize', failure_penalty=failure_penalty, ioh_enabled=with_ioh,
        max_seconds_soft=max_seconds, versions=_versions(),
        config=config.to_dict() if config is not None else None,
        budget_includes_baseline_and_failed_attempts=True, automatic_retries=False,
        recovery='deterministic_journal_replay', soft_time_limit_scope='per_invocation')
    if archive is not None:
        experiment.update(optimization_mode='multi_objective',
                          objective_source='named raw objectives, all minimized; scalar fallback if empty',
                          ioh_trace='scalar diagnostic only; Pareto results in pareto.json')
    if method == 'nsga2':
        experiment['optimizer_settings'] = dict(initial_population='baseline plus uniform samples',
            crossover='bounded SBX', crossover_probability=.9, crossover_eta=15,
            mutation='bounded polynomial', mutation_probability=1/problem.dimension, mutation_eta=20,
            selection='rank and crowding binary tournament',
            constraint_handling='feasibility first, then sum of positive violations; invalid CFD last')
    if method == 'bo':
        experiment['optimizer_settings'] = bo_settings(bo_initial_points, bo_candidates)
    if method == 'lqcmaes':
        experiment['optimizer_settings'] = lq_settings()
    if method == 'turbo':
        experiment['optimizer_settings'] = turbo_settings(problem.dimension, turbo_initial_points, turbo_candidates)
    if method == 'cobyla':
        experiment['optimizer_settings'] = dict(rhobeg_normalized=.1, trust_region_tolerance=1e-5,
            constraint_tolerance=1e-8, relative_objective_tolerance=1e-5,
            constraint_handling='native geometric inequalities and bounds',
            objective_source='unpenalized_objective when provided, otherwise objective',
            proposal_recovery='SciPy replay; previously evaluated CFD values are reused')
    if method in ('bo', 'turbo'):
        experiment['optimizer_settings']['candidate_repair'] = getattr(problem, 'candidate_repair', None)
    history, pending, previous = [], [], {}
    ioh_output = output/'ioh'
    session = None
    if resume:
        saved = json.loads((output/'experiment.json').read_text())
        validate_experiment(saved,experiment)
        history, pending = load_history(output,budget,failure_penalty)
        _verify_history(problem,x0,method,population,sigma,seed,history,
                        budget=budget, bo_initial_points=bo_initial_points, bo_candidates=bo_candidates,
                        turbo_initial_points=turbo_initial_points, turbo_candidates=turbo_candidates)
        if (output/'summary.json').exists():
            previous = json.loads((output/'summary.json').read_text())
        session = next_session(output)
        for name in ('summary.json','best.json','best-feasible.json','checkpoint.json','pareto.json'):
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
        if archive is not None:
            summary.update(pareto_size=len(archive.entries), pareto_file='pareto.json',
                           objective_names=list(archive.space.names or ()),
                           optimization_mode='multi_objective')
            _save(output/'pareto.json', archive.to_dict())
        summary['session_wall_time_s'] = time.monotonic()-started
        summary['wall_time_s'] = previous.get('wall_time_s',0.)+summary['session_wall_time_s']
        _save(output/'summary.json',summary)
        _save(output/'checkpoint.json',dict(schema_version=1,recovery='deterministic_journal_replay',
            attempts=summary['attempts'],pending_evaluation=pending_index,
            original_budget=budget,experiment='experiment.json',events='events.jsonl',
            evaluations='evaluations.jsonl',recoveries='recoveries.jsonl'))

    def remember(record):
        if archive is not None:
            archive.add(record)
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

    proposals = _proposals(problem,x0,method,population,sigma,seed,summary,
                           budget=budget, bo_initial_points=bo_initial_points, bo_candidates=bo_candidates,
                           turbo_initial_points=turbo_initial_points, turbo_candidates=turbo_candidates)
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
                x, phase, generation, proposal = request
                proposal_fields = dict(proposal=proposal) if proposal is not None else {}
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
                    append_json(events,dict(event='started',evaluation=index,x=x,phase=phase,generation=generation,
                                           **proposal_fields))
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
                            phase=phase,generation=generation,result=raw,wall_time_s=time.monotonic()-began,
                            **proposal_fields)
                    except (Exception,KeyboardInterrupt) as exc:
                        summary['failed_evaluations'] += 1
                        append_json(records,dict(evaluation=index,x=x,objective=None,valid=False,
                            feasible=False,phase=phase,generation=generation,
                            error=f'{type(exc).__name__}: {exc}',case_directory=getattr(exc,'case_directory',None),
                            **proposal_fields))
                        raise
                    append_json(records,record)
                    pending_index = None
                    append_json(events,dict(event='finished',evaluation=index,valid=record['valid']))
                remember(record)
                if progress:
                    progress(problem.name,index,budget,'replayed' if replaying else 'ok' if record['valid'] else 'failed/penalized')
                try:
                    request = proposals.send(record['result'])
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


def _command_timeout(identifier, args):
    if args.timeout is not None:
        return args.timeout
    # A resumed run keeps the effective limits recorded in its plan.
    saved = getattr(args, 'command_timeouts', {})
    return saved.get(identifier, P006_TIMEOUT_S if identifier == 'P006' else 1800)


def _make_problem(identifier, args):
    common = dict(output_root=args.output / identifier / 'cases')
    config_type = problem_config_type(identifier)
    if config_type is not None:
        source = args.config
        if identifier == 'P005' and args.p005_config:
            if source is not None:
                raise ValueError('Use --config or --p005-config, not both')
            source = args.p005_config
        overrides = {}
        if args.mesh is not None:
            defaults = config_type()
            overrides['mesh' if hasattr(defaults, 'mesh') else 'cells'] = (
                args.mesh if hasattr(defaults, 'mesh') else MESHES[args.mesh])
        if identifier == 'P005' and args.p005_oxygen_velocity is not None:
            overrides['oxygen_velocity_m_s'] = args.p005_oxygen_velocity
        common.update(config=load_config(config_type, source, **overrides),
                      runtime=Runtime(args.backend, args.image, _command_timeout(identifier, args)))
    elif args.config is not None:
        raise ValueError(f'{identifier} has no registered configuration type')
    if identifier == 'P004':
        common.update(dataset=args.p004_dataset, model=args.p004_model)
    return create_problem(identifier, **common)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run selected FlowField problems with a registered optimizer')
    parser.add_argument('--problem', '--problems', dest='problems', nargs='+', choices=list_problems()+('all',), default=['P000'])
    parser.add_argument('--budget', type=int, default=20, help='Maximum attempts PER problem, including baseline/failures')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--population', type=int, default=4, help='Population size; NSGA-II includes the baseline in its initial population')
    parser.add_argument('--sigma', type=float, default=.1, help='CMA-ES initial step size in normalized coordinates')
    parser.add_argument('--bo-initial-points', type=int, default=5,
                        help='BO initialization attempts including the baseline (default 5)')
    parser.add_argument('--bo-candidates', type=int, default=2048,
                        help='BO random acquisition candidates per step (default 2048)')
    parser.add_argument('--turbo-initial-points', type=int, default=5,
                        help='TuRBO initialization attempts per region, including the first baseline (default 5)')
    parser.add_argument('--turbo-candidates', type=int, default=512,
                        help='TuRBO candidates in each joint Thompson sample (default 512)')
    parser.add_argument('--p005-config',type=Path,help='P005 physical/numerical configuration JSON; retain it for resume')
    parser.add_argument('--p005-oxygen-velocity',type=float,help='P005 fixed oxygen inlet speed in m/s (default 5)')
    parser.add_argument('--p003-sigma', type=float, default=.005, help='P003 step size near its sharply projected feasible baseline')
    parser.add_argument('--optimizer', '--method', dest='method', choices=('auto',)+list_optimizers(), default='auto')
    parser.add_argument('--config', type=Path, help='Problem configuration JSON; --mesh overrides its mesh setting')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mesh', choices=('coarse','medium','fine'), default=None)
    parser.add_argument('--backend', choices=('auto','native','docker'), default='auto')
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--timeout', type=float, default=None,
                        help='Seconds per mesh/build/solver command (default: P006 14400; other problems 1800)')
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
    supplied = {word.split('=')[0] for word in argv if word.startswith('--')}
    if '--problem' in supplied: supplied.add('--problems')
    if '--optimizer' in supplied: supplied.add('--method')
    try:
        if args.resume:
            args = restore_options(args, supplied)
            if not args.dry_run:
                check_legacy_pid(args.output.resolve())
        return _run_suite(args)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        parser.exit(1,f'flowfield-optimize: {exc}\n')


def _run_suite(args):
    _validate_options(args.budget,args.seed,args.population,args.sigma,args.failure_penalty,args.max_seconds_per_problem)
    _validate_options(args.budget,args.seed,args.population,args.p003_sigma,args.failure_penalty,args.max_seconds_per_problem)
    bo_settings(args.bo_initial_points,args.bo_candidates)
    turbo_settings(1,args.turbo_initial_points,args.turbo_candidates)
    if 'all' in args.problems and len(args.problems) != 1:
        raise ValueError('Use all alone or list individual problem IDs')
    identifiers = list(list_problems()) if args.problems == ['all'] else args.problems
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
        if hasattr(problem, 'validate_optimization'):
            try:
                problem.validate_optimization()
            except ValueError as exc:
                if args.problems != ['all']:
                    raise
                entries.append(dict(problem=identifier, status='skipped', reason=str(exc)))
                continue
        method = _method(problem,args.method)
        problems.append(problem)
        entries.append(dict(problem=identifier,status='ready',dimension=problem.dimension,
            algorithm=optimizer_label(method, problem),
            maximum_attempts=args.budget, evaluation_kind='current_lookup_or_surrogate' if identifier=='P004' else 'CFD',
            sigma_normalized=(args.p003_sigma if identifier=='P003' else args.sigma) if method in ('cmaes','lqcmaes') else None,
            initial_design_length=len(_initial(problem))))
        if isinstance(problem, CFDProblem):
            entries[-1]['command_timeout_s'] = problem.runtime.timeout_s
        if method in ('bo', 'turbo'):
            entries[-1]['candidate_repair'] = getattr(problem, 'candidate_repair', None)
    plan = dict(budget_per_problem=args.budget,maximum_attempts=len(problems)*args.budget,
                maximum_cfd_attempts=sum(p.name!='P004' for p in problems)*args.budget,
                mesh=args.mesh or 'coarse', seed=args.seed, problems=entries)
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
                    saved.get('stop_reason') in ('evaluation_budget','cma_termination','no_new_design_found','optimizer_termination') and
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
        raise ValueError('No runnable problems selected; review the required datasets and calibration inputs')
    active = [p for p in problems if p.name not in finished]
    # Construct adapters before any CFD budget is spent.
    for problem in active:
        if hasattr(problem,'validate_optimization'):problem.validate_optimization()
        problem.preflight()
        extra = (dict(bo_initial_points=args.bo_initial_points, bo_candidates=args.bo_candidates)
                 if _method(problem,args.method) == 'bo' else {})
        if _method(problem,args.method) == 'turbo':
            extra = dict(turbo_initial_points=args.turbo_initial_points, turbo_candidates=args.turbo_candidates)
        create_sampler(args.method, problem, seed=args.seed, initial=_initial(problem),
                       population=args.population, sigma=args.p003_sigma if problem.name=='P003' else args.sigma, **extra)
    if any(_method(p,args.method) in ('cmaes','lqcmaes') for p in active):
        import cma
    if any(p.name=='P001' for p in active):
        import gmsh
        if gmsh.__version__ != '4.13.1':
            raise RuntimeError('P001 requires gmsh==4.13.1')
    if args.ioh:
        from ..benchmarking.ioh import _ioh
        _ioh()
    cfd = [p for p in active if isinstance(p, CFDProblem)]
    if cfd:
        runtime = Runtime(args.backend, args.image, max(p.runtime.timeout_s for p in cfd))
        commands = {command for p in cfd for command in p.case_definition.commands}
        plan['runtime_preflight'] = runtime.resolve(commands)
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
        if isinstance(problem, CFDProblem):
            message = f'{problem.name}: command timeout = {problem.runtime.timeout_s:g} s'
            if getattr(problem, 'requires_qualified_baseline', False):
                message += '; baseline must qualify before optimization candidates'
            print(message, file=sys.stderr, flush=True)
        try:
            result = optimize(problem, budget=args.budget, seed=args.seed, output=args.output/problem.name,
                method=args.method,population=args.population,
                sigma=args.p003_sigma if problem.name=='P003' else args.sigma,with_ioh=args.ioh,
                failure_penalty=args.failure_penalty,max_seconds=args.max_seconds_per_problem,progress=progress,
                bo_initial_points=args.bo_initial_points,bo_candidates=args.bo_candidates,
                turbo_initial_points=args.turbo_initial_points,turbo_candidates=args.turbo_candidates,
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
