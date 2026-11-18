"""FlowField problems, simulation interfaces, and problem construction."""

from .problem import EvaluationResult, FlowFieldProblem, AbstractFlowFieldProblem, CFDProblem
from .evaluation import CaseEvaluator, CaseDefinition
from .mesh import PolyMesh, MeshGenerator
from .config import load_config
from .registry import create_problem, get_problem, list_problems, register_problem
from .problems.level4_high_dimensional_topology.p007 import P007Problem
from .problems import P000Problem, P001Problem, P002Problem, P003Problem, P004Problem, P004HydraulicProblem, P005Problem
from .problems.p000.config import ChannelConfig, decode_design, reference_pressure_drop
from .problems.level1_small_continuous import DualLayerConfig
from .problems.level2_medium_parametric import PPFCConfig, PEMConfig
from .problems.level3_discrete_combinatorial import HydraulicConfig
from .problems.level4_high_dimensional_topology import TopologyConfig

__version__ = "0.8.0"
__all__ = [
    "FlowFieldProblem", "AbstractFlowFieldProblem", "CFDProblem", "EvaluationResult",
    "CaseEvaluator", "CaseDefinition", "PolyMesh", "MeshGenerator", "load_config",
    "create_problem", "get_problem", "list_problems", "register_problem",
    "P000Problem", "P001Problem", "P002Problem", "P003Problem", "P004Problem", "P007Problem",
    "P004HydraulicProblem", "P005Problem", "ChannelConfig", "PPFCConfig", "PEMConfig",
    "TopologyConfig", "HydraulicConfig", "DualLayerConfig", "decode_design",
    "reference_pressure_drop",
]

from .problems.level1_small_continuous.p008 import P008Problem, AEMConfig
__all__ += ["P008Problem", "AEMConfig"]

from .problems.level1_small_continuous.p009 import P009Problem, ChenConfig
__all__ += ["P009Problem", "ChenConfig"]
