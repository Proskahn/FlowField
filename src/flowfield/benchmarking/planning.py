"""Validate and describe an experiment matrix without executing evaluations."""

from ..problem import CFDProblem
from ..registry import create_problem, list_problems, problem_info
from ..optimization.mobo import MOBO_METHODS, mobo_settings


_OPTIMIZER_OPTIONS = {
    'population', 'sigma', 'failure_penalty', 'max_seconds',
    'bo_initial_points', 'bo_candidates', 'turbo_initial_points', 'turbo_candidates',
    'mobo_mc_samples', 'mobo_reference_point',
}


def plan_benchmark(*, problems, optimizers, seeds, budget, problem_options=None,
                   optimizer_options=None, with_ioh=True, regime=None):
    """Return a JSON-ready plan; no files, preflight, adapters or CFD calls.

    All combinations must be compatible. Problem-owned setup requirements such
    as frozen objective calibration must hold; unavailable problems are never
    silently omitted. Runtime/package availability is checked during execution.
    """
    from ..optimization.registry import list_optimizers, resolve_optimizer, resolve_regime
    from ..optimization.runner import _validate_options
    from ..optimization.bo import bo_settings
    from ..optimization.turbo import turbo_settings

    selections = {}
    for name, values in (('problems', problems), ('optimizers', optimizers), ('seeds', seeds)):
        if isinstance(values, (str, bytes)):
            raise ValueError(f'{name} must be a nonempty sequence without duplicates')
        values = list(values)
        if not values or len(set(values)) != len(values):
            raise ValueError(f'{name} must be a nonempty sequence without duplicates')
        selections[name] = values
    problems, optimizers, seeds = (selections[name] for name in ('problems', 'optimizers', 'seeds'))
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError('Seeds must be integers in [0, 2**32)')
    if type(budget) is not int or budget < 1:
        raise ValueError('budget must be a positive integer')
    if set(problems) - set(list_problems()) or set(optimizers) - set(list_optimizers()):
        raise ValueError('Select registered problem and optimizer IDs')
    regimes = {method: resolve_regime(method, regime) for method in optimizers}
    problem_options = problem_options or {}
    optimizer_options = optimizer_options or {}
    if set(problem_options) - set(problems):
        raise ValueError('problem_options contains unselected problem IDs')
    if set(optimizer_options) - set(optimizers):
        raise ValueError('optimizer_options contains unselected optimizer IDs')
    for optimizer, options in optimizer_options.items():
        if set(options) - _OPTIMIZER_OPTIONS:
            raise ValueError('Optimizer options may only set ' + ', '.join(sorted(_OPTIMIZER_OPTIONS)))
        _validate_options(budget, seeds[0], options.get('population', 4), options.get('sigma', .1),
                          options.get('failure_penalty'), options.get('max_seconds'))
        bo_settings(options.get('bo_initial_points', 5), options.get('bo_candidates', 2048))
        turbo_settings(1, options.get('turbo_initial_points', 5), options.get('turbo_candidates', 512))
        if optimizer in MOBO_METHODS:
            mobo_settings(optimizer,options.get('bo_initial_points',5),options.get('bo_candidates',2048),
                          options.get('mobo_mc_samples',128),options.get('mobo_reference_point'))
    instances = []
    for identifier in problems:
        problem = create_problem(identifier, **problem_options.get(identifier, {}))
        if hasattr(problem, 'validate_optimization'):
            problem.validate_optimization()
        for optimizer in optimizers:
            resolve_optimizer(problem, optimizer)
        instance = dict(problem=identifier, instance=problem.ioh_name,
                        dimension=problem.dimension,
                        variable_type=getattr(problem, 'variable_type', 'real'),
                        evaluation_kind=problem_info(identifier).evaluation_kind,
                        lower_bounds=problem.lower_bounds, upper_bounds=problem.upper_bounds,
                        initial_design=problem.initial_design())
        config = getattr(problem, 'config', None)
        if config is not None:
            instance.update(config=config.to_dict(), config_sha256=config.fingerprint)
        if identifier == 'P003':
            from ..problems.level4_high_dimensional_topology.p003.scalability import grid_summary
            instance.update(grid_summary(config))
        if isinstance(problem, CFDProblem):
            instance['runtime'] = dict(backend=problem.runtime.backend, image=problem.runtime.image,
                                       timeout_s=problem.runtime.timeout_s)
        instances.append(instance)
    count = len(problems) * len(optimizers) * len(seeds)
    return dict(schema_version=3, problems=problems, optimizers=optimizers, seeds=seeds,
                regime=regime or (next(iter(regimes.values())) if len(set(regimes.values())) == 1 else 'mixed'),
                optimizer_regimes=regimes,
                budget=budget, ioh_enabled=with_ioh, instances=instances,
                optimizer_options=optimizer_options, run_count=count, maximum_attempts=count * budget,
                objective_direction='minimize', constraint_convention='g(x) <= 0',
                budget_policy='Baseline and failed attempts count; saved-result analysis does not evaluate',
                runtime_checked=False)
