"""CLI construction and multi-problem orchestration, separate from run execution."""

import argparse
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import shutil
import sys

from ..config import load_config
from ..problem import CFDProblem
from ..problems.p000.config import MESHES
from ..problems.level1_small_continuous.p006.config import DEFAULT_TIMEOUT_S as P006_TIMEOUT_S, MESH_LEVELS as P006_MESH_LEVELS
from ..problems.level4_high_dimensional_topology.p003.scalability import (
    add_dimension_argument, config_from_dimension, grid_summary)
from ..registry import create_problem, list_problems, problem_config_type
from ..runtime import DEFAULT_IMAGE, Runtime
from .bo import bo_settings
from .turbo import turbo_settings
from .random import random_settings
from .repair import evaluation_repair_settings
from .mobo import MOBO_METHODS, mobo_settings
from .registry import create_sampler, list_optimizers, optimizer_label, optimizer_revision, resolve_regime
from .runner import optimize, _experiment, _initial, _method, _validate_options, _verify_history
from .state import (ResumeError, check_legacy_pid, load_history, next_session,
                    pristine_initialization, restore_options, run_lock, save_json,
                    validate_experiment, extend_budget_metadata)

_save = save_json


def _command_timeout(identifier, args):
    if args.timeout is not None:
        return args.timeout
    # A resumed run keeps the effective limits recorded in its plan.
    saved = getattr(args, 'command_timeouts', {})
    return saved.get(identifier, P006_TIMEOUT_S if identifier in ('P006','P010') else 1800)


def _make_problem(identifier, args):
    common = dict(output_root=args.output / identifier / 'cases')
    config_type = problem_config_type(identifier)
    if config_type is not None:
        source = args.config
        if identifier == 'P005' and args.p005_config:
            if source is not None:
                raise ValueError('Use --config or --p005-config, not both')
            source = args.p005_config
        if identifier == 'P003' and args.dimension is not None:
            source = config_from_dimension(args.dimension,source=source,mesh=args.mesh)
        overrides = {}
        if args.mesh is not None:
            if args.mesh.startswith('screening') and identifier != 'P006':
                raise ValueError('screening mesh overrides are supported by P006 only')
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
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument('--budget', type=int, default=20, help='Maximum attempts PER problem, including baseline/failures')
    budgets.add_argument('--extend-budget', type=int, metavar='TOTAL',
                        help='With --resume, increase the total budget while reusing all saved evaluations')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--regime', choices=('single', 'multi'),
                        help='Objective regime; omitted: infer from optimizer. Random supports both.')
    parser.add_argument('--population', type=int, default=4, help='Population size; NSGA-II includes the baseline in its initial population')
    parser.add_argument('--sigma', type=float, default=.1, help='CMA-ES initial step size in normalized coordinates')
    parser.add_argument('--bo-initial-points', type=int, default=5,
                        help='BO initialization attempts including the baseline (default 5)')
    parser.add_argument('--bo-candidates', type=int, default=2048,
                        help='BO random acquisition candidates per step (default 2048)')
    parser.add_argument('--mobo-mc-samples', type=int, default=128,
                        help='BoTorch acquisition samples; a power of two >= 16 (default 128)')
    parser.add_argument('--mobo-reference-point', type=json.loads,
                        help='qLogNEHVI fixed acquisition reference point: JSON mapping objective names to minimized values')
    parser.add_argument('--turbo-initial-points', type=int, default=5,
                        help='TuRBO initialization attempts per region, including the first baseline (default 5)')
    parser.add_argument('--turbo-candidates', type=int, default=512,
                        help='TuRBO candidates in each joint Thompson sample (default 512)')
    parser.add_argument('--p005-config',type=Path,help='P005 physical/numerical configuration JSON; retain it for resume')
    parser.add_argument('--p005-oxygen-velocity',type=float,help='P005 fixed oxygen inlet speed in m/s (default 5)')
    parser.add_argument('--p003-sigma', type=float, default=.005, help='P003 step size near its sharply projected feasible baseline')
    parser.add_argument('--optimizer', '--method', dest='method', choices=('auto',)+list_optimizers(), default='auto')
    parser.add_argument('--config', type=Path, help='Problem configuration JSON; --mesh overrides its mesh setting')
    add_dimension_argument(parser)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mesh', choices=P006_MESH_LEVELS, default=None,
                        help='Mesh override; screening and screening-coarse are P006-only')
    parser.add_argument('--backend', choices=('auto','native','docker'), default='auto')
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--timeout', type=float, default=None,
                        help='Seconds per mesh/build/solver command (default: P006/P010 14400; other problems 1800)')
    parser.add_argument('--ioh', action='store_true')
    parser.add_argument('--failure-penalty', type=float, help='Explicit finite penalty for SimulationError; otherwise stop that problem')
    parser.add_argument('--max-seconds-per-problem', type=float, help='Soft cap; finish an in-flight evaluation before stopping')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--p004-dataset', type=Path)
    group.add_argument('--p004-model', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Print design/algorithm/budget plan without CFD or creating output')
    parser.add_argument('--resume',action='store_true',help='Continue an existing run from its JSON journals; use --extend-budget to increase its total')
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    supplied = {word.split('=')[0] for word in argv if word.startswith('--')}
    if '--problem' in supplied: supplied.add('--problems')
    if '--optimizer' in supplied: supplied.add('--method')
    try:
        if args.extend_budget is not None and not args.resume:
            raise ValueError('--extend-budget requires --resume')
        if args.resume:
            if args.dry_run:
                return _run_suite(restore_options(args, supplied))
            if not args.output.is_dir():
                raise ResumeError(f'Missing run directory: {args.output}')
            check_legacy_pid(args.output.resolve())
            # Read settings and commit any increase while holding the same lock
            # as the original CLI run. Another process cannot change the budget.
            with run_lock(args.output/'.suite.lock'):
                args = restore_options(args, supplied)
                args._suite_locked = True
                return _run_suite(args)
        return _run_suite(args)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        parser.exit(1,f'flowfield-optimize: {exc}\n')


def _suite_history(problem, args, folder):
    extending = getattr(args, '_allow_budget_extension', False)
    # Reject an active API run before changing the suite budget. Dry runs never
    # create or modify lock files; the execution path checks locks again later.
    with (run_lock(folder/'.optimizer.lock') if extending and not args.dry_run else nullcontext()):
        saved_experiment = json.loads((folder/'experiment.json').read_text())
        current = _experiment(problem, _initial(problem), _method(problem,args.method),
            budget=args.budget, seed=args.seed, population=args.population,
            sigma=args.p003_sigma if problem.name=='P003' else args.sigma,
            with_ioh=args.ioh, failure_penalty=args.failure_penalty,
            max_seconds=args.max_seconds_per_problem,
            bo_initial_points=args.bo_initial_points, bo_candidates=args.bo_candidates,
            turbo_initial_points=args.turbo_initial_points, turbo_candidates=args.turbo_candidates,
            mobo_mc_samples=args.mobo_mc_samples, mobo_reference_point=args.mobo_reference_point,
            regime=args.regime)
        validate_experiment(saved_experiment,current,allow_budget_extension=extending)
        history, pending = load_history(folder,saved_experiment['budget'],args.failure_penalty)
        args._replay_inputs[problem.name] = (saved_experiment, current, history)
        return saved_experiment, history, pending


def _run_suite(args):
    with ExitStack() as resources:
        args._replay_resources = resources
        args._replay_inputs, args._verified_replays = {}, {}
        return _run_suite_impl(args)


def _replay_progress(name, index, total, stage):
    if stage == 'verification_started':
        message = f'rebuilding optimizer from {total} saved evaluations; no CFD is rerun'
    elif stage == 'verification_finished':
        message = 'history verified; retaining optimizer state for continuation'
    else:
        message = f'{stage} saved proposal {index}/{total}'
    print(f'{name}: {message}',file=sys.stderr,flush=True)


def _prepare_suite_replays(args, active):
    for problem in active:
        if problem.name not in args._replay_inputs:
            continue
        folder = args.output/problem.name
        saved, current, history = args._replay_inputs[problem.name]
        with (nullcontext() if args.dry_run else run_lock(folder/'.optimizer.lock')):
            if (json.loads((folder/'experiment.json').read_text()) != saved
                    or load_history(folder,saved['budget'],args.failure_penalty)[0] != history):
                raise ResumeError('Run changed during resume preparation; resume again. No new CFD was started.')
            replay = _verify_history(problem,_initial(problem),_method(problem,args.method),
                args.population,args.p003_sigma if problem.name=='P003' else args.sigma,args.seed,history,
                saved=saved,current=current,progress=_replay_progress,
                budget=args.budget, bo_initial_points=args.bo_initial_points, bo_candidates=args.bo_candidates,
                turbo_initial_points=args.turbo_initial_points, turbo_candidates=args.turbo_candidates,
                mobo_mc_samples=args.mobo_mc_samples, mobo_reference_point=args.mobo_reference_point)
            args._replay_resources.callback(replay.close)
            args._verified_replays[problem.name] = replay


def _run_suite_impl(args):
    _validate_options(args.budget,args.seed,args.population,args.sigma,args.failure_penalty,args.max_seconds_per_problem)
    _validate_options(args.budget,args.seed,args.population,args.p003_sigma,args.failure_penalty,args.max_seconds_per_problem)
    bo_settings(args.bo_initial_points,args.bo_candidates)
    turbo_settings(1,args.turbo_initial_points,args.turbo_candidates)
    if 'all' in args.problems and len(args.problems) != 1:
        raise ValueError('Use all alone or list individual problem IDs')
    identifiers = list(list_problems()) if args.problems == ['all'] else args.problems
    if args.dimension is not None and 'P003' not in identifiers:
        raise ValueError('--dimension is supported by P003 only; include --problem P003')
    if args.dimension is not None and args.config is not None:
        raise ValueError('Use --dimension or --config, not both')
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
        selected_regime = resolve_regime(method, args.regime)
        if args.mobo_reference_point is not None and method != 'qlognehvi':
            raise ValueError('--mobo-reference-point is only used by qlognehvi')
        problems.append(problem)
        entries.append(dict(problem=identifier,status='ready',dimension=problem.dimension,
            regime=selected_regime,
            algorithm=optimizer_label(method, problem),
            maximum_attempts=args.budget, evaluation_kind='current_lookup_or_surrogate' if identifier=='P004' else 'CFD',
            sigma_normalized=(args.p003_sigma if identifier=='P003' else args.sigma) if method in ('cmaes','lqcmaes') else None,
            initial_design_length=len(_initial(problem))))
        if isinstance(problem, CFDProblem):
            entries[-1]['command_timeout_s'] = problem.runtime.timeout_s
        if identifier == 'P003':
            entries[-1].update(grid_summary(problem.config))
        if identifier == 'P006':
            entries[-1].update(fidelity=problem.config.fidelity,config_sha256=problem.config.fingerprint)
        if method == 'random':
            entries[-1]['optimizer_settings'] = random_settings(problem)
        if method in MOBO_METHODS:
            entries[-1].update(optimization_mode='multi_objective',
                optimizer_settings=mobo_settings(method,args.bo_initial_points,args.bo_candidates,
                                                   args.mobo_mc_samples,args.mobo_reference_point),
                optimizer_revision=optimizer_revision(method,problem))
            entries[-1]['optimizer_settings']['candidate_repair'] = getattr(problem,'candidate_repair',None)
        repair_settings = evaluation_repair_settings(problem, method)
        if repair_settings:
            entries[-1]['optimizer_settings'] = repair_settings
            entries[-1]['optimizer_revision'] = optimizer_revision(method, problem)
        if method in ('random', 'bo', 'turbo', *MOBO_METHODS) or repair_settings:
            entries[-1]['candidate_repair'] = getattr(problem, 'candidate_repair', None)
    mesh_label = args.mesh or (problems[0].config.mesh if len(problems)==1 and problems[0].name in ('P003','P006') else 'coarse')
    plan = dict(budget_per_problem=args.budget,maximum_attempts=len(problems)*args.budget,
                maximum_cfd_attempts=sum(p.name!='P004' for p in problems)*args.budget,
                mesh=mesh_label, seed=args.seed, problems=entries)
    finished = {}
    if args.resume:
        remaining = 0
        for problem in problems:
            folder = args.output/problem.name
            terminal = False
            if (folder/'experiment.json').exists():
                saved_experiment, history, pending = _suite_history(problem,args,folder)
                spent = len(history)
                saved = json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {}
                reason = saved.get('stop_reason')
                if (saved.get('attempts') == spent and not pending and
                    reason in ('evaluation_budget','cma_termination','no_new_design_found','optimizer_termination') and
                    saved.get('status') in ('completed','completed_with_failures','failed')):
                    terminal = reason != 'evaluation_budget' or spent >= args.budget
                    if terminal and saved.get('budget') == args.budget and saved_experiment['budget'] == args.budget:
                        finished[problem.name] = saved
            elif folder.exists() and not pristine_initialization(folder):
                raise ResumeError(f'Incomplete experiment initialization: {folder}; preserve it for inspection')
            else:
                spent = 0
            entry = next(e for e in entries if e['problem']==problem.name)
            entry.update(saved_attempts=spent,remaining_attempts=0 if terminal else args.budget-spent)
            remaining += entry['remaining_attempts']
        plan.update(resume=True,remaining_attempts=remaining)
        if getattr(args,'_allow_budget_extension',False):
            plan.update(previous_budget_per_problem=args._saved_budget,
                        original_budget_per_problem=args._budget_metadata.get('original_budget',args._saved_budget))
    active = [p for p in problems if p.name not in finished]
    if args.dry_run:
        _prepare_suite_replays(args,active)
        print(json.dumps(plan,indent=2))
        return 0
    if not problems:
        raise ValueError('No runnable problems selected; review the required datasets and calibration inputs')
    # Construct adapters before any CFD budget is spent.
    for problem in active:
        if hasattr(problem,'validate_optimization'):problem.validate_optimization()
        problem.preflight()
        extra = (dict(bo_initial_points=args.bo_initial_points, bo_candidates=args.bo_candidates)
                 if _method(problem,args.method) == 'bo' else {})
        if _method(problem,args.method) == 'turbo':
            extra = dict(turbo_initial_points=args.turbo_initial_points, turbo_candidates=args.turbo_candidates)
        if _method(problem,args.method) in MOBO_METHODS:
            extra = dict(bo_initial_points=args.bo_initial_points,bo_candidates=args.bo_candidates,
                         mobo_mc_samples=args.mobo_mc_samples,mobo_reference_point=args.mobo_reference_point)
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
    # Fail fast on unavailable runtimes/dependencies, then verify once. The
    # still-live samplers are passed to execution rather than reconstructed.
    _prepare_suite_replays(args,active)
    if not args.resume:
        args.output.mkdir(parents=True,exist_ok=False)
    with (nullcontext() if getattr(args,'_suite_locked',False) else run_lock(args.output/'.suite.lock')):
        if args.resume:
            session = next_session(args.output)
            for name in ('suite-summary.json','plan.json','run-options.json'):
                if (args.output/name).exists():
                    shutil.copyfile(args.output/name,session/name)
            _save(session/'resume-plan.json',plan)
            if getattr(args,'_allow_budget_extension',False):
                metadata = extend_budget_metadata(args._budget_metadata,args.budget)
                options = _saved_options(args) | metadata
                # Commit authorization first. If a later write fails, ordinary
                # --resume reads this target and finishes updating the manifests.
                _save(args.output/'run-options.json',options)
                _save(args.output/'plan.json',plan)
        else:
            _save(args.output/'plan.json',plan)
            _save(args.output/'run-options.json',_saved_options(args))
        return _execute_suite(args,problems,entries,finished)


def _saved_options(args):
    return {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()
            if not k.startswith('_') and k not in ('output','resume','dry_run','extend_budget','command_timeouts')}


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
                mobo_mc_samples=args.mobo_mc_samples,mobo_reference_point=args.mobo_reference_point,
                regime=args.regime,
                resume=args.resume and (args.output/problem.name).exists(),
                extend_budget=args.budget if getattr(args,'_allow_budget_extension',False)
                              and (args.output/problem.name).exists() else None,
                _verified_replay=args._verified_replays.get(problem.name),
                replay_progress=_replay_progress)
        except ResumeError:
            raise
        except Exception as exc:
            path = args.output/problem.name/'summary.json'
            result = json.loads(path.read_text()) if path.exists() else dict(
                problem=problem.name,status='failed',attempts=0,error=f'{type(exc).__name__}: {exc}')
            # An extension can fail before replacing the previously completed
            # summary. Do not report that old success as this invocation's result.
            result.update(status='failed',stop_reason='error',error=f'{type(exc).__name__}: {exc}')
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
