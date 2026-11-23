"""level2_medium_parametric: lazy compatibility exports for problem implementations."""
from flowfield._lazy import resolve_export

_EXPORTS = {'P001Problem': 'flowfield.problems.level2_medium_parametric.p001:P001Problem',
 'P002Problem': 'flowfield.problems.level2_medium_parametric.p002:P002Problem',
 'PPFCConfig': 'flowfield.problems.level2_medium_parametric.p001:PPFCConfig',
 'PEMConfig': 'flowfield.problems.level2_medium_parametric.p002:PEMConfig'}

__all__ = [] + list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())


def __dir__():
    return sorted(set(globals()) | set(__all__))
