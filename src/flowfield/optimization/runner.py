"""Strict-budget optimization, raw observations, Pareto archives and resume."""

from contextlib import ExitStack, nullcontext
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import time

from ..problem import AbstractFlowFieldProblem
from ..benchmarking.ioh import analyzer_logger, make_ioh_problem
from .registry import create_sampler, resolve_optimizer, resolve_regime, optimizer_label, optimizer_revision
from ..runtime import SimulationError
from .state import (ResumeError, append_json, load_history,
    next_session, pristine_initialization, run_lock, save_json, validate_experiment,
    extend_budget_metadata, validate_budget_extension)
from .nsga2 import ParetoArchive
from .bo import bo_settings
from .turbo import turbo_settings
from .lqcmaes import lq_settings
from .random import random_settings
from .repair import evaluation_repair_settings
from .mobo import MOBO_METHODS, mobo_settings
from .replay import verify

_save = save_json


def _versions():
    result = {'python': platform.python_version()}
    for name in ('cma', 'numpy', 'scipy', 'scikit-learn', 'ioh', 'gmsh',
                 'torch', 'botorch', 'gpytorch', 'linear-operator'):
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

    def __init__(self, problem, penalty, *, require_multi=False):
        super().__init__(problem.name, problem.dimension, problem.lower_bounds, problem.upper_bounds)
        self.problem, self.penalty = problem, penalty
        self.replay_result = None
        self.variable_type = getattr(problem, 'variable_type', 'real')
        self.require_multi = require_multi
        self.objective_names = None

    @property
    def ioh_name(self):
        suffix = 'raise' if self.penalty is None else format(self.penalty, '.12g')
        return f'{self.problem.ioh_name}_optimizer_failure_{suffix}'

    def decode(self, x):
        return self.problem.decode(x)

    def evaluate_raw(self, x):
        if self.replay_result is not None:
            result, self.replay_result = self.replay_result, None
        else:
            try:
                self.problem(x)
            except SimulationError as exc:
                if self.penalty is None:
                    raise
                return dict(status='penalized', objective=self.penalty, objectives={}, constraints={},
                            error=str(exc), metadata=dict(valid_cfd=False,
                            failure_penalty=self.penalty, case_directory=exc.case_directory))
            result = self.problem.last_result
        if (self.require_multi and result.get('status') not in ('failed', 'penalized')
                and result.get('metadata', {}).get('valid_cfd', True)):
            objectives = result.get('objectives')
            if (not isinstance(objectives, dict) or len(objectives) < 2
                    or any(not isinstance(k, str) or not k for k in objectives)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in objectives.values())):
                raise ValueError('The multi regime requires at least two finite named raw objectives')
            names = tuple(sorted(objectives))
            if self.objective_names is not None and names != self.objective_names:
                raise ValueError('Multi-objective names changed during the run')
            self.objective_names = names
        return result


def _proposals(problem, x0, method, population, sigma, seed, state, *,
               budget=None, bo_initial_points=5, bo_candidates=2048,
               turbo_initial_points=5, turbo_candidates=512,
               mobo_mc_samples=128, mobo_reference_point=None):
    """Bridge serial, budgeted evaluations to batch ask/tell algorithms."""
    extra = dict(bo_initial_points=bo_initial_points, bo_candidates=bo_candidates) if method == 'bo' else {}
    if method == 'turbo':
        extra = dict(turbo_initial_points=turbo_initial_points, turbo_candidates=turbo_candidates)
    if method in MOBO_METHODS:
        extra = dict(bo_initial_points=bo_initial_points, bo_candidates=bo_candidates,
                     mobo_mc_samples=mobo_mc_samples, mobo_reference_point=mobo_reference_point)
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
        metadata = sampler.batch_proposal_metadata
        if metadata is not None and len(metadata) != len(batch):
            raise ValueError('Proposal metadata must match the complete batch')
        for index, x in enumerate(batch):
            problem.validate_design(x)
            results.append((yield list(x), 'candidate', sampler.generation,
                            metadata[index] if metadata is not None else getattr(sampler, 'proposal_metadata', None)))
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


def _verify_history(problem, x0, method, population, sigma, seed, history, *,
                    saved=None, current=None, progress=None, **options):
    state = dict(complete_generations=0, partial_generation_evaluations=0)
    proposals = _proposals(problem, x0, method, population, sigma, seed, state, **options)
    return verify(proposals,state,history,problem.name,saved=saved,current=current,progress=progress)


def optimize(problem, *, output, resume=False, extend_budget=None, _verified_replay=None, **options):
    """Resume journals without new CFD for replay; extend_budget is a new total."""
    if not isinstance(problem, AbstractFlowFieldProblem):
        raise TypeError('optimize requires a AbstractFlowFieldProblem')
    if 'optimizer' in options:
        if 'method' in options:
            raise ValueError('Specify optimizer or method, not both')
        options['method'] = options.pop('optimizer')
    if extend_budget is not None:
        if not resume:
            raise ValueError('extend_budget requires resume=True')
        validate_budget_extension(1, extend_budget)
        options['budget'] = extend_budget
    _validate_options(options.get('budget',20),options.get('seed',42),options.get('population',4),
                      options.get('sigma',.1),options.get('failure_penalty'),options.get('max_seconds'))
    method = _method(problem,options.get('method','auto'))
    if resume and 'regime' not in options and (Path(output)/'experiment.json').is_file():
        saved = json.loads((Path(output)/'experiment.json').read_text())
        if saved.get('strict_multi_objective'):
            options['regime'] = saved['regime']
    resolve_regime(method, options.get('regime'))
    bo_settings(options.get('bo_initial_points',5), options.get('bo_candidates',2048))
    turbo_settings(problem.dimension, options.get('turbo_initial_points',5), options.get('turbo_candidates',512))
    if method in MOBO_METHODS:
        mobo_settings(method,options.get('bo_initial_points',5),options.get('bo_candidates',2048),
                      options.get('mobo_mc_samples',128),options.get('mobo_reference_point'))
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
    with run_lock(output/'.optimizer.lock'), ExitStack() as resources:
        return _optimize(problem,output=output,resume=resume and (output/'experiment.json').exists(),
                         allow_budget_extension=extend_budget is not None,
                         _verified_replay=_verified_replay,_resources=resources,**options)


def _experiment(problem, x0, method, *, budget, seed, population, sigma, with_ioh,
                failure_penalty, max_seconds, bo_initial_points, bo_candidates,
                turbo_initial_points, turbo_candidates, mobo_mc_samples=128, mobo_reference_point=None,
                regime=None):
    """Build the same manifest for read-only validation and actual execution."""
    diagonal = method in ('cmaes', 'lqcmaes') and problem.dimension > 100
    algorithm = optimizer_label(method, problem)
    config = getattr(problem,'config',None)
    experiment = dict(schema_version=3, algorithm=algorithm, optimizer=method,
        regime=resolve_regime(method, regime), strict_multi_objective=regime == 'multi',
        optimizer_revision=optimizer_revision(method, problem), problem=problem.name,
        problem_name=_FailurePolicy(problem, failure_penalty).ioh_name, dimension=problem.dimension, budget=budget, seed=seed,
        initial_design=x0, population=population if method not in ('random', 'bo', 'turbo', 'cobyla', *MOBO_METHODS) else None,
        sigma_normalized=sigma if method not in ('random', 'nsga2', 'bo', 'turbo', 'cobyla', *MOBO_METHODS) else None, diagonal_covariance=diagonal,
        lower_bounds=problem.lower_bounds, upper_bounds=problem.upper_bounds,
        objective_direction='minimize', failure_penalty=failure_penalty, ioh_enabled=with_ioh,
        max_seconds_soft=max_seconds, versions=_versions(),
        config=config.to_dict() if config is not None else None,
        budget_includes_baseline_and_failed_attempts=True, automatic_retries=False,
        recovery='deterministic_journal_replay', soft_time_limit_scope='per_invocation')
    if experiment['regime'] == 'multi':
        experiment.update(optimization_mode='multi_objective',
                          objective_source='named raw objectives, all minimized; scalar fallback if empty',
                          ioh_trace='scalar diagnostic only; Pareto results in pareto.json')
        if regime == 'multi':
            experiment['objective_source'] = 'at least two named raw objectives, all minimized; no scalar fallback'
    if method == 'random':
        experiment['optimizer_settings'] = random_settings(problem)
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
    if method in MOBO_METHODS:
        experiment['optimizer_settings'] = mobo_settings(method,bo_initial_points,bo_candidates,
                                                         mobo_mc_samples,mobo_reference_point)
        experiment['objective_source'] = 'at least two named raw objectives, all minimized; no scalar fallback'
    if method == 'cobyla':
        experiment['optimizer_settings'] = dict(rhobeg_normalized=.1, trust_region_tolerance=1e-5,
            constraint_tolerance=1e-8, relative_objective_tolerance=1e-5,
            constraint_handling='native geometric inequalities and bounds',
            objective_source='unpenalized_objective when provided, otherwise objective',
            proposal_recovery='SciPy replay; previously evaluated CFD values are reused')
    if method in ('bo', 'turbo', *MOBO_METHODS):
        experiment['optimizer_settings']['candidate_repair'] = getattr(problem, 'candidate_repair', None)
    repair_settings = evaluation_repair_settings(problem, method)
    if repair_settings:
        experiment.setdefault('optimizer_settings', {}).update(repair_settings)
        if method == 'cobyla':
            experiment['optimizer_settings']['constraint_handling'] = 'problem-owned repair before CFD; native bounds'
    return experiment


def _optimize(problem, *, budget=20, seed=42, output, method='auto', population=4,
              sigma=.1, with_ioh=True, failure_penalty=None, max_seconds=None,
              initial=None, progress=None, resume=False, bo_initial_points=5, bo_candidates=2048,
              turbo_initial_points=5, turbo_candidates=512, allow_budget_extension=False,
              _verified_replay=None, _resources=None, replay_progress=None,
              mobo_mc_samples=128, mobo_reference_point=None, regime=None):
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
    if method in MOBO_METHODS:
        from .mobo import dependencies
        dependencies()
    if method == 'cobyla':
        from .cobyla import dependencies
        dependencies()
    if with_ioh:
        from ..benchmarking.ioh import _ioh
        _ioh()
    wrapped = _FailurePolicy(problem,failure_penalty, require_multi=regime == 'multi')
    archive = ParetoArchive() if resolve_regime(method, regime) == 'multi' else None
    experiment = _experiment(problem, x0, method, budget=budget, seed=seed, population=population,
        sigma=sigma, with_ioh=with_ioh, failure_penalty=failure_penalty, max_seconds=max_seconds,
        bo_initial_points=bo_initial_points, bo_candidates=bo_candidates,
        turbo_initial_points=turbo_initial_points, turbo_candidates=turbo_candidates,
        mobo_mc_samples=mobo_mc_samples, mobo_reference_point=mobo_reference_point, regime=regime)
    algorithm = experiment['algorithm']
    history, pending, previous = [], [], {}
    ioh_output = output/'ioh'
    session = None
    if resume:
        saved = json.loads((output/'experiment.json').read_text())
        validate_experiment(saved,experiment,allow_budget_extension=allow_budget_extension)
        history, pending = load_history(output,saved['budget'],failure_penalty)
        if _verified_replay is None:
            _verified_replay = _verify_history(problem,x0,method,population,sigma,seed,history,
                saved=saved,current=experiment,progress=replay_progress,
                budget=budget, bo_initial_points=bo_initial_points, bo_candidates=bo_candidates,
                turbo_initial_points=turbo_initial_points, turbo_candidates=turbo_candidates,
                mobo_mc_samples=mobo_mc_samples, mobo_reference_point=mobo_reference_point)
        else:
            _verified_replay.validate(saved,experiment,history)
        _resources.callback(_verified_replay.close)
        if (output/'summary.json').exists():
            previous = json.loads((output/'summary.json').read_text())
        session = next_session(output)
        # A backup must not look like another run to recursive plot discovery.
        shutil.copyfile(output/'experiment.json',session/'previous-experiment.json')
        for name in ('summary.json','best.json','best-feasible.json','checkpoint.json','pareto.json'):
            if (output/name).exists():
                shutil.copyfile(output/name,session/name)
        _save(session/'session.json',dict(replayed_evaluations=len(history),
            recovered_evaluations=[r['evaluation'] for r in pending],versions=_versions(),
            previous_budget=saved['budget'], budget=budget,
            strategy='replay_without_CFD', io_trace='full_history_plus_new_evaluations'))
        experiment = extend_budget_metadata(saved, budget)
        if budget != saved['budget']:
            _save(output/'experiment.json',experiment)
        if pending:
            with (output/'recoveries.jsonl').open('a') as stream:
                for row in pending:
                    append_json(stream,row)
        ioh_output = session/'ioh'
    else:
        _save(output/'experiment.json',experiment)
    objective = make_ioh_problem(wrapped) if with_ioh else wrapped
    summary = dict(status='running',problem=problem.name,algorithm=algorithm,budget=budget,
        regime=resolve_regime(method, regime),
        seed=seed,attempts=0,successful_evaluations=0,failed_evaluations=0,
        baseline_objective=None,best_objective=None,best_design=None,best_case_directory=None,
        best_feasible_objective=None,best_feasible_design=None,best_feasible_case_directory=None,
        complete_generations=0,partial_generation_evaluations=0,stop_reason=None,
        resumed=resume,replayed_evaluations=len(history),new_attempts=0,
        recovered_interruptions=sum(bool(r.get('recovered')) for r in history),
        ioh_directory=str(ioh_output) if with_ioh else None)
    if experiment.get('budget_extensions'):
        summary.update(original_budget=experiment['original_budget'],
                       budget_extensions=experiment['budget_extensions'])
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
            original_budget=experiment.get('original_budget',budget),budget=budget,
            experiment='experiment.json',events='events.jsonl',
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

    if _verified_replay is not None:
        proposals, proposal_state = _verified_replay.proposals, _verified_replay.state
    else:
        proposals = _proposals(problem,x0,method,population,sigma,seed,summary,
                               budget=budget, bo_initial_points=bo_initial_points, bo_candidates=bo_candidates,
                               turbo_initial_points=turbo_initial_points, turbo_candidates=turbo_candidates,
                               mobo_mc_samples=mobo_mc_samples, mobo_reference_point=mobo_reference_point)
        proposal_state = summary
    context = analyzer_logger(objective,ioh_output,algorithm_name=algorithm,
        algorithm_info=f'seed={seed}; budget={budget}; resumed={resume}; replay is not new CFD') if with_ioh else nullcontext()
    try:
        with context, (output/'evaluations.jsonl').open('a') as records, (output/'events.jsonl').open('a') as events:
            request = None if history else next(proposals)
            while summary['attempts'] < budget:
                replaying = summary['attempts'] < len(history)
                if not replaying and max_seconds is not None and time.monotonic()-started >= max_seconds:
                    summary['stop_reason'] = 'soft_wall_time_limit'
                    break
                if replaying:
                    cached = history[summary['attempts']]
                    x, phase, generation, proposal = (cached['x'],cached['phase'],cached['generation'],cached.get('proposal'))
                else:
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
                if replaying and index < len(history):
                    # IOH/archive reconstruction uses already verified records;
                    # it must not ask the optimizer to generate them a second time.
                    continue
                summary.update(proposal_state)
                if progress and index < budget:
                    progress(problem.name,index+1,budget,'proposing')
                try:
                    request = proposals.send(record['result'])
                except StopIteration:
                    summary.update(proposal_state)
                    break
                summary.update(proposal_state)
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


def main(argv=None):
    """Compatibility entry point; CLI policy lives in flowfield.optimization.cli."""
    from .cli import main as cli_main
    return cli_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
