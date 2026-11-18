"""Concrete benchmarks organized into the four agreed optimization levels.

Each implementation records its numerical and paper-validation status. See
docs/problem-levels-report.md for classification and extension rules.
"""

from .p000 import P000Problem
from .level1_small_continuous import P005Problem
from .level2_medium_parametric import P001Problem, P002Problem

from .level4_high_dimensional_topology import P003Problem, P007Problem

from .level3_discrete_combinatorial import P004Problem, P004HydraulicProblem

__all__ = ["P005Problem", "P000Problem", "P001Problem", "P002Problem", "P003Problem", "P004Problem", "P004HydraulicProblem", "P007Problem"]

from .level1_small_continuous.p008 import P008Problem, AEMConfig
__all__ += ["P008Problem", "AEMConfig"]

from .level1_small_continuous.p009 import P009Problem, ChenConfig
__all__ += ["P009Problem", "ChenConfig"]
