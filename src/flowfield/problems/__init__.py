"""Concrete benchmarks organized into the four agreed optimization levels.

Each implementation records its numerical and paper-validation status. See
docs/problem-levels-report.md for classification and extension rules.
"""

from .level1_small_continuous import P000Problem
from .level2_medium_parametric import P001Problem, P002Problem

from .level4_high_dimensional_topology import P003Problem

from .level3_discrete_combinatorial import P004Problem, P004HydraulicProblem

__all__ = ["P000Problem", "P001Problem", "P002Problem", "P003Problem", "P004Problem", "P004HydraulicProblem"]
