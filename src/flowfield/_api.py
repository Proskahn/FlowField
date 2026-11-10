"""One public API for installed packages and imports from the repository root."""

from .base import EvaluationResult, FlowFieldProblem
from .model import ChannelConfig, decode_design, reference_pressure_drop
from .evaluation import Evaluator, evaluate
from .problems import P000Problem, P001Problem, P002Problem, P003Problem
from .problems.level2_medium_parametric import PPFCConfig, PEMConfig
from .problems.level4_high_dimensional_topology import TopologyConfig
from .problems.level3_discrete_combinatorial import P004Problem,P004HydraulicProblem,HydraulicConfig
from .registry import get_problem, list_problems, register_problem

__all__ = ["FlowFieldProblem", "EvaluationResult", "P000Problem", "P001Problem", "PPFCConfig", "P002Problem", "PEMConfig", "P003Problem", "TopologyConfig", "P004Problem", "P004HydraulicProblem", "HydraulicConfig", "get_problem",
           "list_problems", "register_problem", "ChannelConfig", "decode_design",
           "reference_pressure_drop", "Evaluator", "evaluate"]
__version__ = "0.7.0"
