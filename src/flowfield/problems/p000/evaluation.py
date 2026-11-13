"""P000 simulation recipe for the shared case evaluator."""
from ...evaluation import CaseDefinition, CaseEvaluator
from ...runtime import SimulationError
from ...mesh import block_mesh
from .case import generate_case
from .config import ChannelConfig
from .metrics import extract_metrics

INPUTS = frozenset({"0/U", "0/p", "constant/transportProperties", "constant/turbulenceProperties",
    "system/blockMeshDict", "system/controlDict", "system/fvSchemes", "system/fvSolution"})


def _validate(case, manifest, config):
    pass


def _execute(case, config, manifest, runtime, info, result):
    result.update(objective_unit="Pa", objective_direction="minimize")
    block_mesh(case, runtime, info)
    log = runtime.run(case, "checkMesh", ("-allGeometry", "-allTopology"), info=info)
    text = log.read_text(encoding="utf-8")
    if "Mesh OK." not in text or "Failed " in text:
        raise SimulationError("checkMesh did not report Mesh OK; solver was not started")
    runtime.run(case, "simpleFoam", info=info)


def extract_results(case, config):
    metrics = extract_metrics(case, config)
    return dict(objective=metrics['pressure_drop_pa'], metrics=metrics,
                objectives={'pressure_drop_pa': metrics['pressure_drop_pa']}, constraints={})


CASE = CaseDefinition(
    name="P000", config_type=ChannelConfig, inputs=INPUTS,
    commands=("blockMesh", "checkMesh", "simpleFoam"), validate=_validate,
    execute=_execute, generate=generate_case, extract=extract_results, timeout_s=600,
    execution_outputs=("constant/polyMesh", "log.blockMesh"))


def run_case(case, runtime=None):
    return CaseEvaluator(CASE, runtime).run_case(case)


def evaluate(config=None, *, output_root="runs", runtime=None):
    return CaseEvaluator(CASE, runtime).evaluate(config or ChannelConfig(), output_root=output_root)
