"""Shared ask/tell interface for stateful proposal algorithms."""

from abc import ABC, abstractmethod


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
