"""Lazy problem construction, independent of optimizers and CFD execution."""

from dataclasses import dataclass
from importlib import import_module
import re

from .catalog import BUILTINS, ProblemInfo
from .config import load_config
from .problem import FlowFieldProblem


@dataclass(frozen=True)
class _Registration:
    factory: object
    info: ProblemInfo
    config_type: type | None = None

    def resolve(self):
        if isinstance(self.factory, str):
            module, name = self.factory.split(":")
            return getattr(import_module(module), name)
        return self.factory


_FACTORIES = {info.identifier: _Registration(factory, info) for factory, info in BUILTINS}


def _registration(identifier):
    try:
        return _FACTORIES[identifier]
    except KeyError as exc:
        raise ValueError(f"Unknown problem {identifier!r}; available: {', '.join(list_problems())}") from exc


def list_problems(*, level=None, variable_type=None, evaluation_kind=None) -> tuple[str, ...]:
    """Discover IDs without constructing problems or loading optional packages."""
    if level is not None and (type(level) is not int or level not in range(1, 5)):
        raise ValueError("level must be 1, 2, 3, or 4")
    if variable_type not in (None, "real", "integer"):
        raise ValueError("variable_type must be real or integer")
    if evaluation_kind not in (None, "cfd", "data", "custom"):
        raise ValueError("evaluation_kind must be cfd, data, or custom")
    return tuple(sorted(name for name, entry in _FACTORIES.items()
                        if (level is None or entry.info.level == level)
                        and (variable_type is None or entry.info.variable_type == variable_type)
                        and (evaluation_kind is None or entry.info.evaluation_kind == evaluation_kind)))


def problem_info(identifier) -> ProblemInfo:
    """Return static requirements and scope, without checking the local runtime."""
    return _registration(identifier).info


def register_problem(identifier, factory, *, config_type=None, info=None):
    """Register an external problem; metadata is optional for existing callers."""
    if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", identifier)
            or not callable(factory)):
        raise ValueError("Provide a safe problem identifier and callable problem factory")
    if identifier in _FACTORIES:
        raise ValueError(f"Problem already registered: {identifier}")
    if info is None:
        info = ProblemInfo(identifier, identifier, variable_type=getattr(factory, "variable_type", "real"))
    if not isinstance(info, ProblemInfo) or info.identifier != identifier:
        raise ValueError("ProblemInfo.identifier must match the registered identifier")
    _FACTORIES[identifier] = _Registration(factory, info, config_type)


def problem_config_type(identifier):
    """Resolve only the selected implementation's validated configuration type."""
    entry = _registration(identifier)
    return entry.config_type or getattr(entry.resolve(), "config_type", None)


def create_problem(identifier="P000", **kwargs) -> FlowFieldProblem:
    entry = _registration(identifier)
    factory = entry.resolve()
    config_type = entry.config_type or getattr(factory, "config_type", None)
    if "config" in kwargs and config_type is not None:
        kwargs["config"] = load_config(config_type, kwargs["config"])
    problem = factory(**kwargs)
    if not isinstance(problem, FlowFieldProblem):
        raise TypeError(f"Factory for {identifier} must return a FlowFieldProblem")
    return problem


get_problem = create_problem
