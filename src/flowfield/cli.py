"""Command-line interface; errors exit nonzero and never overwrite prior work."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .case import generate_case
from .evaluation import evaluate, run_case, write_json
from .experiments import add_search_arguments, search_from_args
from .ioh_adapter import analyzer_logger, make_ioh_problem
from .model import ChannelConfig, MESHES, reference_pressure_drop
from .runtime import Runtime, SimulationError, DEFAULT_IMAGE
from .registry import get_problem, list_problems


def _config(args):
    config = ChannelConfig.from_json(args.config) if args.config else ChannelConfig()
    changes = {}
    for arg, name in (("width_mm", "width_m"), ("height_mm", "height_m")):
        if getattr(args, arg, None) is not None:
            changes[name] = getattr(args, arg) * 1e-3
    if args.mesh:
        changes["cells"] = MESHES[args.mesh]
    return replace(config, **changes)


def _runtime(args):
    return Runtime(args.backend, args.image, args.timeout)


def smoke(output, config, runtime, with_ioh=False):
    """One CFD call, including IOH logging if requested; assert physical checks."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    evaluator = get_problem("P000", output_root=output / "cases", config=config, runtime=runtime)
    # The smoke geometry is the 1 mm square baseline regardless of config geometry.
    x = [1 / 3, 1 / 3]
    if with_ioh:
        problem = make_ioh_problem(evaluator)
        with analyzer_logger(problem, output / "ioh", algorithm_name="P000-smoke",
                             algorithm_info="One deterministic CFD evaluation; no optimizer"):
            objective = problem(x)
    else:
        objective = evaluator(x)
    result = evaluator.last_result
    passed = result["metrics"]["reference_check_passed"]
    summary = {"status": "passed" if passed else "failed", "objective_pa": objective,
               "design": x, "case_directory": result["case_directory"], "metrics": result["metrics"],
               "ioh_enabled": with_ioh, "config_sha256": result["config_sha256"]}
    summary["interface"] = {"class": type(evaluator).__name__, "name": evaluator.name,
                            "dimension": evaluator.dimension,
                            "raw_result_file": str(Path(result["case_directory"]) / "evaluation.json")}
    if with_ioh:
        dat_files = list((output / "ioh").rglob("*.dat"))
        info_files = list((output / "ioh").glob("IOHprofiler*.json"))
        if not dat_files or not info_files:
            raise SimulationError("IOH logger did not produce native data and metadata files")
        summary["ioh_files"] = [str(p.relative_to(output)) for p in info_files + dat_files]
    write_json(output / "smoke.json", summary)
    if not passed:
        raise SimulationError(f"Pressure drop exceeds {config.reference_tolerance:.1%} reference tolerance; see {output}/smoke.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="FlowField P000: v2312 straight-channel verification")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List registered benchmark IDs")
    add_search_arguments(sub.add_parser("search", help="Run a seeded benchmark experiment"))
    for name in ("generate", "evaluate", "reference", "smoke"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, help="JSON ChannelConfig; unknown keys are rejected")
        cmd.add_argument("--mesh", choices=MESHES)
        if name != "smoke":
            cmd.add_argument("--width-mm", type=float)
            cmd.add_argument("--height-mm", type=float)
        if name == "generate":
            cmd.add_argument("--case", required=True, type=Path)
        elif name in ("smoke", "evaluate"):
            cmd.add_argument("--output", required=True, type=Path)
        if name == "smoke":
            cmd.add_argument("--ioh", action="store_true", help="Include the real IOH adapter/logger")
        if name in ("smoke", "evaluate"):
            _runtime_args(cmd)
    run = sub.add_parser("run", help="Execute a freshly generated case once")
    run.add_argument("case", type=Path)
    _runtime_args(run)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            result = {"problems": list(list_problems())}
        elif args.command == "search":
            result = search_from_args(args)
        elif args.command == "run":
            result = run_case(args.case, _runtime(args))
        else:
            config = _config(args)
            if args.command == "generate":
                result = {"case_directory": str(generate_case(args.case, config))}
            elif args.command == "reference":
                result = {"backend": "analytical_reference_only", "pressure_drop_pa": reference_pressure_drop(config),
                          "reynolds_number": config.reynolds_number, "config": config.to_dict()}
            elif args.command == "evaluate":
                result = evaluate(config, output_root=args.output, runtime=_runtime(args))
            else:
                result = smoke(args.output, config, _runtime(args), args.ioh)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"flowfield: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def _runtime_args(parser):
    parser.add_argument("--backend", choices=("auto", "native", "docker"), default="auto")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--timeout", type=float, default=600, help="Seconds allowed per OpenFOAM command")
