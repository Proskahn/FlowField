"""Transcribed Table 5 and printed Eq.7; reference data are never CFD output."""
from .config import BOUNDS

TABLE5 = (
    (70, 1.2, 1.92, .0287), (92.5, .3, 1.6, .0364), (92.5, .75, 1.92, .0321),
    (115, .3, 1.92, .0380), (92.5, 1.2, 1.6, .0298), (70, .75, 2.24, .0317),
    (92.5, 1.2, 2.24, .0281), (92.5, 1.2, 1.6, .0298), (70, .3, 1.92, .0388),
    (92.5, 1.2, 1.6, .0298), (115, .75, 1.6, .0327), (70, .75, 1.6, .0337),
    (115, .75, 2.24, .0310), (70, .3, 1.92, .0388), (115, .75, 1.6, .0327),
    (92.5, .75, 1.92, .0321), (115, .75, 1.6, .0327), (92.5, .3, 2.24, .0350),
    (115, 1.2, 1.92, .0279), (92.5, .75, 1.92, .0321),
)
PRODUCT_BASELINE = (92.5, .66, 2.44)
REPORTED_OPTIMUM = (70., .3, 1.92)


def coded(values):
    return [(2*v-lo-hi)/(hi-lo) for v, (lo, hi) in zip(values, BOUNDS)]


def printed_equation(values, *, coded_variables=True):
    """Eq.7 as printed. Coding is an explicit inference, not stated by authors."""
    a, b, c = coded(values) if coded_variables else values
    return (.0321-.000413*a-.0042*b-.000856*c+.000617*a*a
            +.000645*b*b-.00045*c*c+.00009*a*c)
