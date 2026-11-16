"""Shared contract for real-valued scalar-minimization flow-field problems."""

from abc import ABC, abstractmethod
import math
from numbers import Real
from typing import Any, TypedDict


class EvaluationResult(TypedDict):
    """JSON-compatible evaluation: constraints use g(x) <= 0 for feasibility.

    `objective` is the scalar consumed by IOH. `objectives` contains named raw
    physical minimization objectives (negate quantities to maximize them);
    `constraints` contains optimization constraints (not
    solver residuals). Metadata contains provenance and numerical diagnostics.
    """
    objective: float
    objectives: dict[str, float]
    constraints: dict[str, float]
    metadata: dict[str, Any]
    status: str


class FlowFieldProblem(ABC):
    """Implement decode and evaluate_raw to add a benchmark to the suite.

    All problems minimize the scalar returned by __call__. A constrained or
    multi-objective problem defines its own explicit scalarization/penalty for
    scalar optimizers and IOH, while preserving raw objectives and constraints
    for Pareto optimizers such as NSGA-II.

    Problems may provide candidate_repair metadata and a repair_candidates(rows,
    anchor=...) hook for cheap geometric constraints. BO and TuRBO apply this
    before acquisition scoring. The hook consumes physical coordinates and
    preserves feasible rows and anchor-to-input bounds; evaluate_raw stays pure
    with respect to the supplied design (it must not silently repair it).
    """
    name: str
    dimension: int
    lower_bounds: list[float]
    upper_bounds: list[float]

    def __init__(self, name, dimension, lower_bounds, upper_bounds):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Problem name must be a nonempty string")
        if type(dimension) is not int or dimension < 1:
            raise ValueError("Problem dimension must be a positive integer")
        lower, upper = list(lower_bounds), list(upper_bounds)
        if len(lower) != dimension or len(upper) != dimension:
            raise ValueError("Bound lengths must match the problem dimension")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
               for v in lower + upper):
            raise ValueError("Bounds must be finite real numbers")
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise ValueError("Each lower bound must be below its upper bound")
        self.name = name
        self.dimension = dimension
        self.lower_bounds = [float(v) for v in lower]
        self.upper_bounds = [float(v) for v in upper]
        self.last_result = None

    @property
    def ioh_name(self):
        """Override to distinguish configurations that define different objectives."""
        return self.name

    def validate_design(self, x):
        values = list(x)
        if len(values) != self.dimension:
            raise ValueError(f"{self.name} requires {self.dimension} design variables")
        for i, (value, lo, hi) in enumerate(zip(values, self.lower_bounds, self.upper_bounds)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{self.name}: x[{i}] must be finite and within [{lo}, {hi}]")
        return [float(value) for value in values]

    @abstractmethod
    def decode(self, x) -> Any:
        """Convert optimizer variables to physical geometry, without running CFD."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_raw(self, x) -> EvaluationResult:
        """Return CFD objectives, constraints and metadata for one evaluation."""
        raise NotImplementedError

    def __call__(self, x) -> float:
        self.last_result = None
        result = self.evaluate_raw(self.validate_design(x))
        objective = float(result["objective"])
        if not math.isfinite(objective):
            raise ValueError(f"{self.name} returned a nonfinite scalar objective")
        self.last_result = result
        return objective

    def initial_design(self):
        """Default starting point; a problem can supply a physical baseline."""
        return [(lo + hi) / 2 for lo, hi in zip(self.lower_bounds, self.upper_bounds)]

    def preflight(self):
        """Check fixed dependencies before optimization spends any evaluations.

        Raise on setup errors even when failed CFD evaluations will be penalized.
        This check must not generate cases or run CFD; geometry-only use remains
        available without optional simulation dependencies.
        """


class CFDProblem(FlowFieldProblem):
    """Optional simulation interface for problems backed by OpenFOAM.

    Data/surrogate problems only need FlowFieldProblem. Algorithms consume the
    vector-to-result interface and never need these simulation methods.
    """

    @property
    @abstractmethod
    def case_definition(self):
        """Return the problem's CaseDefinition (including its result extractor)."""
        raise NotImplementedError

    def case_arguments(self, x):
        return self.decode(x), self.config

    def write_case(self, x, destination):
        """Decode variables and write a standalone case without running CFD."""
        return self.case_definition.generate(destination, *self.case_arguments(x))

    def run_case(self, case):
        from .evaluation import CaseEvaluator
        return CaseEvaluator(self.case_definition, self.runtime).run_case(case)

    def extract_results(self, case):
        from .evaluation import CaseEvaluator
        return CaseEvaluator(self.case_definition).extract_results(case)


# Compatibility with the earlier abstract-class name.
AbstractFlowFieldProblem = FlowFieldProblem
