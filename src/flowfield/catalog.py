"""Static benchmark descriptions; discovery never imports simulation code.

Descriptions are selection aids, not paper-validation claims. Actual bounds,
dimension, configuration and instance identity come from the selected problem.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProblemInfo:
    identifier: str
    title: str
    level: int | None = None
    variable_type: str = "real"
    evaluation_kind: str = "custom"
    description: str = ""
    objective: str = "Problem-defined scalar minimization"
    requirements: tuple[str, ...] = ()
    extra: str | None = None
    reference: str | None = None
    documentation: str | None = None

    def __post_init__(self):
        if not self.identifier or not self.title:
            raise ValueError("A problem description needs an identifier and title")
        if self.level is not None and (type(self.level) is not int or self.level not in range(1, 5)):
            raise ValueError("Problem level must be 1, 2, 3, 4, or None")
        if self.variable_type not in ("real", "integer"):
            raise ValueError("variable_type must be real or integer")
        if self.evaluation_kind not in ("cfd", "data", "custom"):
            raise ValueError("evaluation_kind must be cfd, data, or custom")
        object.__setattr__(self, "requirements", tuple(self.requirements))

    def to_dict(self):
        return asdict(self)


_CFD = ("OpenCFD OpenFOAM v2312 (native or Docker)",)


def _info(identifier, title, level, description, objective, *, extra=None,
          reference=None, requirements=(), variable_type="real", kind="cfd"):
    return ProblemInfo(
        identifier, title, level, variable_type, kind, description, objective,
        (_CFD if kind == "cfd" else ()) + requirements, extra, reference,
        f"benchmarks/{identifier.split('-')[0]}/",
    )


# Resolve import paths only on selection. New built-ins go here; user interfaces
# and registry code need no new problem-specific branches.
BUILTINS = (
    ("flowfield.problems.p000:P000Problem", _info(
        "P000", "Rectangular channel verification", 1,
        "Two normalized channel dimensions; analytical pressure-drop verification, not an electrochemical model.",
        "Pressure drop in Pa")),
    ("flowfield.problems.level2_medium_parametric.p001:P001Problem", _info(
        "P001", "Partitioned point flow channel", 2,
        "Six geometry variables; provisional alkaline flow-channel reconstruction.",
        "Negative flow uniformity", extra="p001", reference="10.1016/j.ijhydene.2025.03.237")),
    ("flowfield.problems.level2_medium_parametric.p002:P002Problem", _info(
        "P002", "Coupled PEM electrolyzer", 2,
        "Seven geometry variables; provisional coupled electrochemical reconstruction.",
        "Explicit power scalarization of voltage and pressure loss",
        reference="10.1016/j.ijhydene.2023.08.026")),
    ("flowfield.problems.level4_high_dimensional_topology.p003:P003Problem", _info(
        "P003", "PEMFC cathode topology", 4,
        "Configurable nodal field (2145 variables by default); forward model, no replicated adjoint optimization.",
        "Weighted power/uniformity with channel-volume penalty", extra="p003",
        reference="10.1016/j.ijheatmasstransfer.2019.01.050")),
    ("flowfield.problems.level3_discrete_combinatorial.p004:P004Problem", _info(
        "P004", "Discrete fuel-cell path library", 3,
        "49 integer path entries; external current labels or an explicitly trained surrogate, no coupled CFD.",
        "Negative mean current density", variable_type="integer", kind="data",
        requirements=("Supply dataset= or model=; author current labels are not bundled",))),
    ("flowfield.problems.level3_discrete_combinatorial.p004:P004HydraulicProblem", _info(
        "P004-hydraulic", "Discrete path hydraulics", 3,
        "49 integer path entries; auxiliary channel-only flow, not fuel-cell current prediction.",
        "Pressure drop in Pa", variable_type="integer")),
    ("flowfield.problems.level1_small_continuous.p005:P005Problem", _info(
        "P005", "Dual-layer bubble detachment", 1,
        "Two channel dimensions; provisional transient VOF reconstruction with censored events.",
        "Censored bubble-detachment time in seconds", reference="10.1016/j.egyai.2024.100411")),
    ("flowfield.problems.level1_small_continuous.p006:P006Problem", _info(
        "P006", "Two-fluid PEM manifold", 1,
        "Three channel-width variables; steady two-fluid reconstruction with geometric constraints.",
        "Temperature standard deviation with explicit constraint penalty", extra="p006")),
    ("flowfield.problems.level4_high_dimensional_topology.p007:P007Problem", _info(
        "P007", "Flow and heat topology", 4,
        "Configurable design field (128 variables by default); estimated parameters and unreplicated final layouts.",
        "Weighted heat transfer/dissipation with volume penalty", extra="p007",
        reference="10.1016/j.fuel.2025.136144")),
    ("flowfield.problems.level1_small_continuous.p008:P008Problem", _info(
        "P008", "AEM channel performance", 1,
        "Two geometry variables; provisional 3D reconstruction with unresolved closure and unit ambiguities.",
        "Negative frozen entropy-weighted score; uncalibrated evaluations are current diagnostics only",
        extra="p008", requirements=("Optimization requires frozen EWM calibration from qualified CFD cases",))),
    ("flowfield.problems.level1_small_continuous.p009:P009Problem", _info(
        "P009", "Porous oxygen-side channel", 1,
        "Three variables; single-phase reconstruction with unresolved CAD/resistance details.",
        "Negative flow performance (mean speed per pressure drop)", extra="p009", reference="10.1002/ese3.70173")),
    ("flowfield.problems.level1_small_continuous.p010:P010Problem", _info(
        "P010", "ESCFF PEM fuel cell", 1,
        "Four variables; coupled 3D reconstruction, quantitatively unvalidated against the paper.",
        "Negative net power in W", extra="p010", reference="10.1016/j.ijheatmasstransfer.2025.127792")),
    ("flowfield.problems.level1_small_continuous.p011:P011Problem", _info(
        "P011", "Tapered-channel PEM fuel cell", 1,
        "Four endpoint widths with reciprocal heights; provisional coupled 3D two-phase PEMFC reconstruction.",
        "100000 times cathode pumping power divided by electrical output",
        extra="p011", reference="10.1016/j.renene.2026.125531")),
)
