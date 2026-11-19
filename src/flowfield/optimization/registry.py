"""Optimizer registration independent of the problem catalog."""

from dataclasses import dataclass
import re

from .samplers import Sampler, RandomSampler, CMAESSampler
from .nsga2 import NSGAIISampler
from .bo import BayesianOptimizationSampler
from .turbo import TuRBOSampler
from .cobyla import COBYLASampler
from .lqcmaes import LQCMAESSampler
from .repair import evaluation_repair_settings


@dataclass(frozen=True)
class OptimizerDefinition:
    factory: object
    label: str
    variable_types: tuple[str, ...]
    revision: str
    multi_objective: bool = False


_OPTIMIZERS = {}


def register_optimizer(name, factory, *, label=None, variable_types=('real',), revision='1', multi_objective=False):
    """Factory receives (problem, seed, initial, population, sigma).

    Change revision when an implementation or its settings change, so resumed
    runs cannot silently mix algorithms. Install optional dependencies inside
    the factory, never when importing the FlowField package.
    """
    if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', name) or name == 'auto':
        raise ValueError('Optimizer name must be a lowercase identifier other than auto')
    if name in _OPTIMIZERS:
        raise ValueError(f'Optimizer already registered: {name}')
    if not callable(factory) or not variable_types or not set(variable_types) <= {'real', 'integer'}:
        raise ValueError('Provide a callable factory and supported variable types')
    if not isinstance(revision, str) or not revision:
        raise ValueError('Optimizer revision must be a nonempty string')
    _OPTIMIZERS[name] = OptimizerDefinition(factory, label or name, tuple(variable_types), revision, multi_objective)


def list_optimizers():
    return tuple(sorted(_OPTIMIZERS))


def optimizer_definition(name):
    try:
        return _OPTIMIZERS[name]
    except KeyError as exc:
        raise ValueError(f'Unknown optimizer {name!r}; available: {", ".join(list_optimizers())}') from exc


def resolve_optimizer(problem, name='auto'):
    kind = getattr(problem, 'variable_type', 'real')
    if name == 'auto':
        name = 'random' if kind == 'integer' else 'cmaes'
    spec = optimizer_definition(name)
    if kind not in spec.variable_types:
        raise ValueError(f'{spec.label} does not support {kind} variables; use a compatible optimizer')
    if kind == 'integer' and name == 'random' and not hasattr(problem, 'sample_design'):
        raise ValueError('Discrete random search requires a feasible sample_design method')
    return name


def optimizer_label(name, problem):
    if name == 'cmaes' and problem.dimension > 100:
        return 'Sep-CMA-ES'
    if name == 'lqcmaes' and problem.dimension > 100:
        return 'Sep-lq-CMA-ES'
    if name == 'random' and getattr(problem, 'variable_type', 'real') == 'integer':
        return 'FeasiblePathRandomSearch'
    return optimizer_definition(name).label


def optimizer_revision(name, problem):
    revision = optimizer_definition(name).revision
    if evaluation_repair_settings(problem, name):
        revision += '+candidate-repair-1'
    return revision


def create_sampler(name, problem, **options):
    name = resolve_optimizer(problem, name)
    sampler = optimizer_definition(name).factory(problem, **options)
    if not isinstance(sampler, Sampler):
        raise TypeError(f'Factory for {name} must return a Sampler')
    return sampler


register_optimizer('random', RandomSampler, label='RandomSearch', variable_types=('real', 'integer'),
                   revision='2-volume-repair')
register_optimizer('cmaes', CMAESSampler, label='CMA-ES')
register_optimizer('lqcmaes', LQCMAESSampler, label='lq-CMA-ES', revision='1-pycma-population-replay')
register_optimizer('nsga2', NSGAIISampler, label='NSGA-II', multi_objective=True)
register_optimizer('bo', BayesianOptimizationSampler, label='Vanilla BO', revision='2')
register_optimizer('turbo', TuRBOSampler, label='TuRBO-1', revision='2')
register_optimizer('cobyla', COBYLASampler, label='COBYLA', revision='1-scipy-replay-paper-stop')
