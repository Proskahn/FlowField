"""Discover and construct benchmarks without importing the simulation stack."""
from flowfield._lazy import resolve_export
from .problem import EvaluationResult, FlowFieldProblem, AbstractFlowFieldProblem, CFDProblem
from .config import load_config
from .catalog import ProblemInfo
from .registry import create_problem, get_problem, list_problems, register_problem, problem_info

__version__ = "0.8.0"

_EXPORTS = {'P000Problem': 'flowfield.problems.p000:P000Problem',
 'P001Problem': 'flowfield.problems.level2_medium_parametric.p001:P001Problem',
 'P002Problem': 'flowfield.problems.level2_medium_parametric.p002:P002Problem',
 'P003Problem': 'flowfield.problems.level4_high_dimensional_topology.p003:P003Problem',
 'P004Problem': 'flowfield.problems.level3_discrete_combinatorial.p004:P004Problem',
 'P004HydraulicProblem': 'flowfield.problems.level3_discrete_combinatorial.p004:P004HydraulicProblem',
 'P005Problem': 'flowfield.problems.level1_small_continuous.p005:P005Problem',
 'P006Problem': 'flowfield.problems.level1_small_continuous.p006:P006Problem',
 'P007Problem': 'flowfield.problems.level4_high_dimensional_topology.p007:P007Problem',
 'P008Problem': 'flowfield.problems.level1_small_continuous.p008:P008Problem',
 'P009Problem': 'flowfield.problems.level1_small_continuous.p009:P009Problem',
 'P010Problem': 'flowfield.problems.level1_small_continuous.p010:P010Problem',
 'P011Problem': 'flowfield.problems.level1_small_continuous.p011:P011Problem',
 'ChannelConfig': 'flowfield.problems.p000.config:ChannelConfig',
 'PPFCConfig': 'flowfield.problems.level2_medium_parametric.p001:PPFCConfig',
 'PEMConfig': 'flowfield.problems.level2_medium_parametric.p002:PEMConfig',
 'TopologyConfig': 'flowfield.problems.level4_high_dimensional_topology.p003:TopologyConfig',
 'HydraulicConfig': 'flowfield.problems.level3_discrete_combinatorial.p004:HydraulicConfig',
 'DualLayerConfig': 'flowfield.problems.level1_small_continuous.p005:DualLayerConfig',
 'ManifoldConfig': 'flowfield.problems.level1_small_continuous.p006:ManifoldConfig',
 'AEMConfig': 'flowfield.problems.level1_small_continuous.p008:AEMConfig',
 'ChenConfig': 'flowfield.problems.level1_small_continuous.p009:ChenConfig',
 'PEMFCConfig': 'flowfield.problems.level1_small_continuous.p010:PEMFCConfig',
 'TaperedConfig': 'flowfield.problems.level1_small_continuous.p011:TaperedConfig',
 'CaseEvaluator': 'flowfield.evaluation:CaseEvaluator',
 'CaseDefinition': 'flowfield.evaluation:CaseDefinition',
 'PolyMesh': 'flowfield.mesh:PolyMesh',
 'MeshGenerator': 'flowfield.mesh:MeshGenerator',
 'decode_design': 'flowfield.problems.p000.config:decode_design',
 'reference_pressure_drop': 'flowfield.problems.p000.config:reference_pressure_drop',
 'describe_problem': 'flowfield.inspection:describe_problem'}

__all__ = ['FlowFieldProblem', 'AbstractFlowFieldProblem', 'CFDProblem', 'EvaluationResult', 'load_config', 'ProblemInfo', 'create_problem', 'get_problem', 'list_problems', 'register_problem', 'problem_info'] + list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())


def __dir__():
    return sorted(set(globals()) | set(__all__))
