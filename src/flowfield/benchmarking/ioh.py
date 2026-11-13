"""Optional real IOHexperimenter wrapper and native IOHanalyzer-compatible logger."""

from contextlib import contextmanager
from pathlib import Path

from ..problem import AbstractFlowFieldProblem


def _ioh():
    try:
        import ioh
    except ImportError as exc:
        raise RuntimeError('Install the optional adapter with: python -m pip install -e ".[ioh]"') from exc
    return ioh


def make_ioh_problem(evaluator: AbstractFlowFieldProblem):
    """Adapt any AbstractFlowFieldProblem using its dimension, bounds and scalar call.

    Instance one, minimization, and no claimed known optimum. IOH's numeric ID
    is registry-assigned; the problem's ioh_name distinguishes configurations.
    """
    if not isinstance(evaluator, AbstractFlowFieldProblem):
        raise TypeError("make_ioh_problem requires a AbstractFlowFieldProblem")
    ioh = _ioh()
    kind = getattr(evaluator, 'variable_type', 'real')
    if kind not in ('real', 'integer'):
        raise ValueError('Unsupported variable type for IOH')
    lower = evaluator.lower_bounds
    upper = evaluator.upper_bounds
    if kind == 'integer':
        if any(int(v) != v for v in lower + upper):
            raise ValueError('Integer IOH bounds must be integral')
        lower, upper = list(map(int, lower)), list(map(int, upper))
    problem = ioh.wrap_problem(evaluator, name=evaluator.ioh_name,
        problem_class=ioh.ProblemClass.INTEGER if kind == "integer" else ioh.ProblemClass.REAL, dimension=evaluator.dimension, instance=1,
        optimization_type=ioh.OptimizationType.MIN,
        lb=min(lower), ub=max(upper))
    # wrap_problem accepts scalar bounds. Set the actual vector box afterwards
    # so optimizers see each variable's declared bounds, including mixed units.
    problem.bounds.lb = lower
    problem.bounds.ub = upper
    return problem


@contextmanager
def analyzer_logger(problem, output, *, algorithm_name="FlowField", algorithm_info=""):
    """Always log evaluations and coordinates; close/flush before returning.

    Destination must be new, avoiding IOH's automatic suffixing of existing logs.
    """
    ioh = _ioh()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"IOH output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    logger = ioh.logger.Analyzer(root=str(output.parent), folder_name=output.name,
        algorithm_name=algorithm_name, algorithm_info=algorithm_info,
        triggers=[ioh.logger.trigger.ALWAYS], store_positions=True)
    problem.attach_logger(logger)
    try:
        yield logger
    finally:
        problem.detach_logger()
        logger.close()
