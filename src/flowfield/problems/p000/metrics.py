"""Read native surfaceFieldValue output; pressure p is kinematic, not Pa."""

import math
from pathlib import Path
import re

from .config import reference_pressure_drop
from ...runtime import SimulationError


from ...openfoam import read_surface_history


def extract_metrics(case, config):
    case = Path(case)
    mesh_log = (case / "log.checkMesh").read_text(encoding="utf-8")
    if not re.search(r"\bMesh OK\.", mesh_log) or "Failed " in mesh_log:
        raise SimulationError("checkMesh did not report a valid mesh")
    solver_log = (case / "log.simpleFoam").read_text(encoding="utf-8")
    if not re.search(r"SIMPLE solution converged in \d+ iterations", solver_log):
        raise SimulationError("simpleFoam did not satisfy residualControl before its iteration limit")
    if not re.search(r"^End\s*$", solver_log, re.MULTILINE):
        raise SimulationError("simpleFoam log does not contain a normal End")
    times = re.findall(r"^Time = ([0-9.eE+\-]+)\s*$", solver_log, re.MULTILINE)
    if not times:
        raise SimulationError("Missing solver iteration history")
    last_solver_time = float(times[-1])
    histories = {}
    for name, column in (("p_inlet", "areaAverage(p)"), ("p_outlet", "areaAverage(p)"),
                         ("q_inlet", "sum(phi)"), ("q_outlet", "sum(phi)")):
        files = list((case / "postProcessing" / name).glob("*/surfaceFieldValue.dat"))
        if len(files) != 1:
            raise SimulationError(f"Expected one fresh-run table for {name}, found {len(files)}")
        histories[name] = read_surface_history(files[0], column)
    latest = [max(history) for history in histories.values()]
    if len(set(latest)) != 1 or latest[0] != last_solver_time:
        raise SimulationError(f"Stale or misaligned surface data: {latest}, solver {last_solver_time}")
    values = {name: history[latest[0]] for name, history in histories.items()}
    qi, qo = values["q_inlet"], values["q_outlet"]
    if qi >= 0 or qo <= 0:
        raise SimulationError("Inlet/outlet fluxes have incorrect signs")
    balance = abs(qi + qo) / config.flow_rate_m3_s
    flow_error = max(abs(-qi / config.flow_rate_m3_s - 1), abs(qo / config.flow_rate_m3_s - 1))
    if max(balance, flow_error) > config.mass_balance_tolerance:
        raise SimulationError(f"Flow conservation failed: balance={balance:g}, flow error={flow_error:g}")
    dp = config.density_kg_m3 * (values["p_inlet"] - values["p_outlet"])
    if not math.isfinite(dp) or dp <= 0:
        raise SimulationError("Pressure drop must be positive and finite")
    reference = reference_pressure_drop(config)
    error = abs(dp / reference - 1)
    return {"pressure_drop_pa": dp, "reference_pressure_drop_pa": reference,
            "reference_relative_error": error, "reference_check_passed": error <= config.reference_tolerance,
            "inlet_kinematic_pressure_m2_s2": values["p_inlet"],
            "outlet_kinematic_pressure_m2_s2": values["p_outlet"],
            "inlet_flux_m3_s": qi, "outlet_flux_m3_s": qo,
            "mass_balance_relative_error": balance, "flow_rate_relative_error": flow_error,
            "reynolds_number": config.reynolds_number, "iterations": int(last_solver_time),
            "cell_count": math.prod(config.cells), "mesh_ok": True, "converged": True}
