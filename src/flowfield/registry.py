"""Explicit benchmark registry; research folders are not implicitly executable."""

import re

from .problem import AbstractFlowFieldProblem
from .config import load_config
from .problems import P000Problem
from .problems.level2_medium_parametric.p001 import P001Problem
from .problems.level2_medium_parametric.p002 import P002Problem
from .problems.level4_high_dimensional_topology.p003 import P003Problem

from .problems.level3_discrete_combinatorial.p004 import P004Problem, P004HydraulicProblem

from .problems.level1_small_continuous.p005 import P005Problem
from .problems.level1_small_continuous.p006 import P006Problem
from .problems.level4_high_dimensional_topology.p007 import P007Problem

from .problems.level1_small_continuous.p008 import P008Problem
from .problems.level1_small_continuous.p009 import P009Problem

from .problems.level1_small_continuous.p010 import P010Problem

_FACTORIES = {"P010": P010Problem, "P009": P009Problem, "P008": P008Problem, "P007": P007Problem, "P006": P006Problem, "P005": P005Problem, "P000": P000Problem, "P001": P001Problem, "P002": P002Problem, "P003": P003Problem, "P004": P004Problem, "P004-hydraulic": P004HydraulicProblem}

_CONFIG_TYPES = {name: factory.config_type for name, factory in _FACTORIES.items()
                 if hasattr(factory, 'config_type')}


def list_problems() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def register_problem(identifier, factory, *, config_type=None):
    """Register another factory without replacing an existing benchmark."""
    if (not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', identifier)
            or not callable(factory)):
        raise ValueError("Provide a safe problem identifier and callable problem factory")
    if identifier in _FACTORIES:
        raise ValueError(f"Problem already registered: {identifier}")
    _FACTORIES[identifier] = factory
    config_type = config_type or getattr(factory, 'config_type', None)
    if config_type is not None:
        _CONFIG_TYPES[identifier] = config_type


def create_problem(identifier="P000", **kwargs) -> AbstractFlowFieldProblem:
    try:
        factory = _FACTORIES[identifier]
    except KeyError as exc:
        raise ValueError(f"Unknown problem {identifier!r}; available: {', '.join(list_problems())}") from exc
    if 'config' in kwargs and identifier in _CONFIG_TYPES:
        kwargs['config'] = load_config(_CONFIG_TYPES[identifier], kwargs['config'])
    problem = factory(**kwargs)
    if not isinstance(problem, AbstractFlowFieldProblem):
        raise TypeError(f"Factory for {identifier} must return a AbstractFlowFieldProblem")
    return problem


def problem_config_type(identifier):
    """Configuration constructor, or None for a problem without a CFD config."""
    if identifier not in _FACTORIES:
        raise ValueError(f'Unknown problem {identifier!r}')
    return _CONFIG_TYPES.get(identifier)


# Kept for existing callers. Unlike ioh.get_problem this creates CFD resources
# and configuration; IOH adapts this instance for logging rather than owning it.
get_problem = create_problem
