"""Explicit benchmark registry; research folders are not implicitly executable."""

from .base import FlowFieldProblem
from .problems import P000Problem
from .problems.level2_medium_parametric.p001 import P001Problem
from .problems.level2_medium_parametric.p002 import P002Problem
from .problems.level4_high_dimensional_topology.p003 import P003Problem

from .problems.level3_discrete_combinatorial.p004 import P004Problem, P004HydraulicProblem

_FACTORIES = {"P000": P000Problem, "P001": P001Problem, "P002": P002Problem, "P003": P003Problem, "P004": P004Problem, "P004-hydraulic": P004HydraulicProblem}


def list_problems() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def register_problem(identifier, factory):
    """Register another factory without replacing an existing benchmark."""
    if not isinstance(identifier, str) or not identifier or not callable(factory):
        raise ValueError("Provide a nonempty identifier and callable problem factory")
    if identifier in _FACTORIES:
        raise ValueError(f"Problem already registered: {identifier}")
    _FACTORIES[identifier] = factory


def get_problem(identifier="P000", **kwargs) -> FlowFieldProblem:
    try:
        factory = _FACTORIES[identifier]
    except KeyError as exc:
        raise ValueError(f"Unknown problem {identifier!r}; available: {', '.join(list_problems())}") from exc
    problem = factory(**kwargs)
    if not isinstance(problem, FlowFieldProblem):
        raise TypeError(f"Factory for {identifier} must return a FlowFieldProblem")
    return problem
