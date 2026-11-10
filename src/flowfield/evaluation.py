"""Fresh evaluation directories and explicit failure records for every CFD call."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import uuid

from .case import generate_case
from .metrics import extract_metrics
from .model import ChannelConfig
from .runtime import Runtime, SimulationError


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_case(case, runtime=None):
    """Run a generated, never-executed case. Re-running requires a fresh case."""
    case = Path(case).resolve()
    runtime = runtime or Runtime()
    metadata = json.loads((case / "case.json").read_text(encoding="utf-8"))
    config = ChannelConfig(**metadata["config"])
    if metadata.get("problem") != "P000" or metadata.get("config_sha256") != config.fingerprint:
        raise ValueError("Not a valid P000 case manifest")
    input_hashes = metadata.get("input_sha256", {})
    expected_inputs = {"0/U", "0/p", "constant/transportProperties", "constant/turbulenceProperties",
                       "system/blockMeshDict", "system/controlDict", "system/fvSchemes", "system/fvSolution"}
    if set(input_hashes) != expected_inputs:
        raise ValueError("Case manifest does not list all P000 inputs")
    for name, checksum in input_hashes.items():
        if hashlib.sha256((case / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Case input changed: {name}; regenerate with an updated configuration")
    if any((case / name).exists() for name in ("postProcessing", "constant/polyMesh", "result.json", "log.blockMesh")):
        raise FileExistsError("Case already contains run data; generate a fresh destination")
    # Exclusive marker prevents concurrent execution and resuming failed cases.
    with (case / ".execution-started").open("x", encoding="utf-8") as stream:
        stream.write(datetime.now(timezone.utc).isoformat() + "\n")
    started = time.monotonic()
    result = {"schema_version": 1, "problem": "P000", "suite_version": "0.2.0",
              "status": "failed", "objective_unit": "Pa", "objective_direction": "minimize",
              "config": config.to_dict(), "config_sha256": config.fingerprint,
              "input_sha256": input_hashes,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "python_version": platform.python_version(), "case_directory": str(case)}
    try:
        info = runtime.resolve()
        result["runtime"] = info
        runtime.run(case, "blockMesh", info=info)
        log = runtime.run(case, "checkMesh", ("-allGeometry", "-allTopology"), info=info)
        if "Mesh OK." not in log.read_text(encoding="utf-8"):
            raise SimulationError("checkMesh did not report Mesh OK; solver was not started")
        runtime.run(case, "simpleFoam", info=info)
        metrics = extract_metrics(case, config)
        result["metrics"] = metrics
        # Reference comparison is reported separately; numerical health determines evaluation validity.
        result.update(status="ok", objective=metrics["pressure_drop_pa"])
    except (SimulationError, OSError, ValueError) as exc:
        result["error"] = str(exc)
        raise SimulationError(f"P000 evaluation failed in {case}: {exc}", case_directory=case) from exc
    except KeyboardInterrupt:
        result["error"] = "Interrupted by user"
        raise
    finally:
        result["wall_time_s"] = time.monotonic() - started
        write_json(case / "result.json", result)
    return result


def evaluate(config=None, *, output_root="runs", runtime=None):
    config = config or ChannelConfig()
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    case = root / ("P000-" + uuid.uuid4().hex)
    generate_case(case, config)
    return run_case(case, runtime)


# Preserve imports/call signatures from the initial scaffold.
from .problems import P000Problem

Evaluator = P000Problem
