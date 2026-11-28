"""level1_small_continuous: lazy compatibility exports for problem implementations."""
from flowfield._lazy import resolve_export

_EXPORTS = {'P005Problem': 'flowfield.problems.level1_small_continuous.p005:P005Problem',
 'P006Problem': 'flowfield.problems.level1_small_continuous.p006:P006Problem',
 'P008Problem': 'flowfield.problems.level1_small_continuous.p008:P008Problem',
 'P009Problem': 'flowfield.problems.level1_small_continuous.p009:P009Problem',
 'P010Problem': 'flowfield.problems.level1_small_continuous.p010:P010Problem',
 'P011Problem': 'flowfield.problems.level1_small_continuous.p011:P011Problem',
 'DualLayerConfig': 'flowfield.problems.level1_small_continuous.p005:DualLayerConfig',
 'ManifoldConfig': 'flowfield.problems.level1_small_continuous.p006:ManifoldConfig',
 'AEMConfig': 'flowfield.problems.level1_small_continuous.p008:AEMConfig',
 'ChenConfig': 'flowfield.problems.level1_small_continuous.p009:ChenConfig',
 'PEMFCConfig': 'flowfield.problems.level1_small_continuous.p010:PEMFCConfig',
 'TaperedConfig': 'flowfield.problems.level1_small_continuous.p011:TaperedConfig'}

__all__ = [] + list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())


def __dir__():
    return sorted(set(globals()) | set(__all__))
