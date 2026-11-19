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
    # A batch adapter may supply one metadata object per design; serial
    # adapters continue to use proposal_metadata. The runner journals each row.
    batch_proposal_metadata = None

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


def random_settings(problem):
    """Record the proposal distribution separately from feasibility repair."""
    repair = getattr(problem, 'candidate_repair', None)
    return dict(proposal_source='problem.sample_design' if hasattr(problem, 'sample_design')
                else 'independent uniform coordinates over declared bounds',
                candidate_repair=repair,
                sampling_policy='repair proposals before duplicate checks and evaluation' if repair is not None
                else 'evaluate unique proposals without repair',
                rng='private Python random.Random',
                duplicate_policy='exclude baseline and all previously proposed evaluation designs',
                maximum_draws_per_candidate=1000, objective_adaptation=False)


class RandomSampler(Sampler):
    """Seeded random proposals with optional problem-owned feasibility repair.

    Repaired uniform draws are not uniform over the feasible domain. Feedback
    never changes the proposal distribution; failed designs also remain seen.
    """
    def __init__(self, problem, *, seed, initial, **options):
        self.problem = problem
        self.rng = random.Random(seed)
        self.seen = {tuple(initial)}
        self.repair = None
        if getattr(problem, 'candidate_repair', None) is not None:
            from .repair import CandidateRepair
            try:
                self.repair = CandidateRepair(problem)
                self.repair.check_initial(initial)
            except ImportError as exc:
                raise RuntimeError('Install random-search repair dependencies with: '
                                   'python -m pip install -e ".[random]"') from exc
        self.proposal_metadata = None
        self.diagnostics = dict(random_draws=0, random_repaired_draws=0, random_duplicate_draws=0)

    def ask(self):
        self.proposal_metadata = None
        for _ in range(1000):
            x = (list(self.problem.sample_design(self.rng)) if hasattr(self.problem, 'sample_design') else
                 [self.rng.uniform(lo, hi) for lo, hi in zip(self.problem.lower_bounds, self.problem.upper_bounds)])
            self.diagnostics['random_draws'] += 1
            metadata = None
            if self.repair is not None:
                repair = self.repair
                proposed = (repair.np.asarray(x)-repair.lower)/repair.span
                evaluated = repair.apply(proposed[None, :])[0]
                metadata = repair.metadata(proposed, evaluated)
                metadata['proposed_x'] = list(x)
                if metadata['applied']:
                    self.diagnostics['random_repaired_draws'] += 1
                    x = (repair.lower+repair.span*evaluated).tolist()
            if tuple(x) not in self.seen:
                self.seen.add(tuple(x))
                self.proposal_metadata = metadata
                return [x]
            self.diagnostics['random_duplicate_draws'] += 1
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
        from .repair import evaluation_repair
        self.evaluation_repair = evaluation_repair(problem, initial)
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
        if self.evaluation_repair is not None:
            # Keep the sampled search points for pycma's distribution update.
            # Their fitness is the composed objective f(repair(x)).
            self.designs, self.batch_proposal_metadata = self.evaluation_repair.prepare_batch(self.designs)
        return self.designs

    def tell(self, designs, results):
        if self.pending is None or designs != self.designs or len(results) != len(designs):
            raise ValueError('CMA-ES feedback must match the complete pending batch')
        self.es.tell(self.pending, [r['objective'] for r in results])
        self.pending = None
