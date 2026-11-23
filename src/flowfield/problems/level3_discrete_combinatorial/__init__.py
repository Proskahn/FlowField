"""level3_discrete_combinatorial: lazy compatibility exports for problem implementations."""
from flowfield._lazy import resolve_export

_EXPORTS = {'P004Problem': 'flowfield.problems.level3_discrete_combinatorial.p004:P004Problem',
 'P004HydraulicProblem': 'flowfield.problems.level3_discrete_combinatorial.p004:P004HydraulicProblem',
 'HydraulicConfig': 'flowfield.problems.level3_discrete_combinatorial.p004:HydraulicConfig'}

__all__ = [] + list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())


def __dir__():
    return sorted(set(globals()) | set(__all__))
