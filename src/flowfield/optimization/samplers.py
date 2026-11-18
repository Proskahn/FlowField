"""Stateful proposal algorithms; none of these classes execute simulations."""

from abc import ABC, abstractmethod
import random


class Sampler(ABC):
    """Batch ask/tell interface for BO and population-based algorithms.

    tell receives complete evaluation dictionaries, including named objectives,
    constraints, status and metadata. A scalar optimizer uses result['objective'];
    a Pareto optimizer can use result['objectives'] and result['constraints'].
    Batch size belongs to the algorithm. The runner may stop partway through a
    batch when the evaluation budget is exhausted and journals every observation.
    Implementations must be deterministic for a fixed seed and observation stream
    if they want to support journal-based resume.
    Adaptive evaluation algorithms may set manages_generations=True and supply
    complete_generations/partial_generation_evaluations in diagnostics; a serial
    request then need not represent a complete algorithm generation.
    """
    generation = None
    stop_reason = None
    manages_generations = False

    def initialize(self, x, result):
        """Observe the problem's baseline; model-based algorithms can train on it."""

    @abstractmethod
    def ask(self) -> list[list[float]]:
        """Return the next batch, or [] when the algorithm has terminated."""
        raise NotImplementedError

    @abstractmethod
    def tell(self, designs, results):
        """Receive one result for every design in the last complete batch."""
        raise NotImplementedError


class RandomSampler(Sampler):
    def __init__(self, problem, *, seed, initial, **options):
        self.problem = problem
        self.rng = random.Random(seed)
        self.seen = {tuple(initial)}

    def ask(self):
        for _ in range(1000):
            x = (list(self.problem.sample_design(self.rng)) if hasattr(self.problem, 'sample_design') else
                 [self.rng.uniform(lo, hi) for lo, hi in zip(self.problem.lower_bounds, self.problem.upper_bounds)])
            if tuple(x) not in self.seen:
                self.seen.add(tuple(x))
                return [x]
        self.stop_reason = 'no_new_design_found'
        return []

    def tell(self, designs, results):
        pass


class CMAESSampler(Sampler):
    def __init__(self, problem, *, seed, initial, population=4, sigma=.1):
        try:
            import cma
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('Install CMA-ES with: python -m pip install -e ".[optim]"') from exc
        self.np = np
        self.lower = np.asarray(problem.lower_bounds)
        self.span = np.asarray(problem.upper_bounds) - self.lower
        normalized = (np.asarray(initial) - self.lower) / self.span
        self.es = cma.CMAEvolutionStrategy(normalized.tolist(), sigma, dict(
            bounds=[0, 1], popsize=population, CMA_diagonal=problem.dimension > 100,
            CMA_mirrors=0, randn=np.random.RandomState(seed).randn, seed=float('nan'),
            verbose=-9, verb_log=0, verb_disp=0))
        self.pending = None
        self.diagnostics = {}

    def ask(self):
        if self.pending is not None:
            raise RuntimeError('tell must complete the previous CMA-ES batch before ask')
        stopped = self.es.stop()
        if stopped:
            self.stop_reason = 'cma_termination'
            self.diagnostics = {'cma_stop': {str(k): str(v) for k, v in stopped.items()}}
            return []
        self.pending = self.es.ask()
        self.generation = int(self.es.countiter) + 1
        self.designs = [(self.lower + self.span * self.np.asarray(x)).tolist() for x in self.pending]
        return self.designs

    def tell(self, designs, results):
        if self.pending is None or designs != self.designs or len(results) != len(designs):
            raise ValueError('CMA-ES feedback must match the complete pending batch')
        self.es.tell(self.pending, [r['objective'] for r in results])
        self.pending = None
