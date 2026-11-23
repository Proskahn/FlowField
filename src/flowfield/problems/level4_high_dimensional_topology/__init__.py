"""level4_high_dimensional_topology: lazy compatibility exports for problem implementations."""
from flowfield._lazy import resolve_export

_EXPORTS = {'P003Problem': 'flowfield.problems.level4_high_dimensional_topology.p003:P003Problem',
 'P007Problem': 'flowfield.problems.level4_high_dimensional_topology.p007:P007Problem',
 'TopologyConfig': 'flowfield.problems.level4_high_dimensional_topology.p003:TopologyConfig'}

__all__ = [] + list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())


def __dir__():
    return sorted(set(globals()) | set(__all__))
