"""P000 implementation of the shared FlowFieldProblem contract."""

from dataclasses import replace
import math
from pathlib import Path

from ...base import EvaluationResult, FlowFieldProblem
from ...model import ChannelConfig, decode_design
from ...runtime import Runtime, SimulationError


class P000Problem(FlowFieldProblem):
    """Fixed-flow rectangular channel; scalar objective is pressure drop in Pa.

    Coordinates are normalized width/height in [0,1]^2. Numerical health checks
    are execution requirements, not engineering optimization constraints. P000
    therefore returns an empty `constraints` mapping.
    """
    def __init__(self, output_root="runs", *, config=None, runtime=None, failure_penalty=None):
        super().__init__("P000", 2, [0.0, 0.0], [1.0, 1.0])
        if failure_penalty is not None and (not math.isfinite(failure_penalty) or failure_penalty <= 0):
            raise ValueError("failure_penalty must be positive and finite")
        self.config = config or ChannelConfig()
        self.runtime = runtime or Runtime()
        self.output_root = Path(output_root)
        self.failure_penalty = failure_penalty

    @property
    def ioh_name(self):
        policy = "raise" if self.failure_penalty is None else format(self.failure_penalty, ".12g")
        return f"P000_CFD_{self.config.fingerprint[:12]}_failure_{policy}"

    def decode(self, x) -> dict[str, float]:
        config = decode_design(self.validate_design(x), self.config)
        return {"width_m": config.width_m, "height_m": config.height_m, "length_m": config.length_m}

    def evaluate_raw(self, x) -> EvaluationResult:
        # Import here so the legacy evaluation module can re-export this class.
        from ...evaluation import evaluate, write_json

        self.last_result = None
        design = self.validate_design(x)
        geometry = self.decode(design)
        config = replace(self.config, **geometry)
        metadata = {"problem_id": self.name, "problem_name": self.ioh_name,
                    "dimension": self.dimension, "design": design, "geometry": geometry,
                    "objective_unit": "Pa", "objective_direction": "minimize",
                    "config": config.to_dict(), "config_sha256": config.fingerprint}
        try:
            cfd = evaluate(config, output_root=self.output_root, runtime=self.runtime)
        except SimulationError as exc:
            if self.failure_penalty is None:
                raise
            metadata.update(error=str(exc), valid_cfd=False,
                            failure_penalty=self.failure_penalty, case_directory=exc.case_directory)
            result = {"status": "penalized", "objective": float(self.failure_penalty),
                      "objectives": {}, "constraints": {}, "metadata": metadata,
                      "error": str(exc)}
        else:
            metadata.update({key: cfd[key] for key in ("case_directory", "runtime", "wall_time_s",
                             "input_sha256", "metrics", "created_utc") if key in cfd})
            metadata["valid_cfd"] = True
            result = {**cfd, "objectives": {"pressure_drop_pa": float(cfd["objective"])},
                      "constraints": {}, "metadata": metadata}
        # Keep the raw CFD record unchanged and persist the framework result beside it.
        if metadata.get("case_directory"):
            write_json(Path(metadata["case_directory"]) / "evaluation.json", result)
        self.last_result = result
        return result
