"""Read-only inspection of a configured benchmark, with no optimizer imports."""

from .problem import CFDProblem
from .registry import create_problem, problem_info


def describe_problem(identifier, **options):
    """Report the selected instance without preflight, files, or solver calls.

    Configuration validation still applies. Requirements describe what must be
    installed/supplied; this function does not assert execution readiness.
    """
    problem = create_problem(identifier, **options)
    result = problem_info(identifier).to_dict()
    result.update(
        name=problem.name, instance=problem.ioh_name, dimension=problem.dimension,
        lower_bounds=problem.lower_bounds, upper_bounds=problem.upper_bounds,
        variable_type=getattr(problem, "variable_type", "real"),
        initial_design=problem.initial_design(), objective_direction="minimize",
        constraint_convention="g(x) <= 0", supports_cases=isinstance(problem, CFDProblem),
    )
    config = getattr(problem, "config", None)
    if config is not None:
        result["config"] = config.to_dict()
        result["config_sha256"] = config.fingerprint
    return result
