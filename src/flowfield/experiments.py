"""Reusable experiment runner over the FlowFieldProblem interface."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time

from .base import FlowFieldProblem
from .evaluation import write_json
from .ioh_adapter import analyzer_logger, make_ioh_problem
from .model import ChannelConfig, MESHES
from .registry import get_problem, list_problems
from .runtime import DEFAULT_IMAGE, Runtime


def random_search(problem: FlowFieldProblem, *, budget=3, seed=42, output, with_ioh=True):
    """Seeded serial search. One call per sample; preserve complete raw records.

    IOH counts and raw records refer to the same evaluations; querying
    evaluate_raw separately would perform another CFD run and is avoided here.
    """
    if not isinstance(problem, FlowFieldProblem):
        raise TypeError("random_search requires a FlowFieldProblem")
    if type(budget) is not int or budget < 1:
        raise ValueError("budget must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", {"schema_version": 1, "algorithm": "RandomSearch",
        "problem": problem.name, "problem_name": problem.ioh_name,
        "dimension": problem.dimension, "lower_bounds": problem.lower_bounds,
        "upper_bounds": problem.upper_bounds, "budget": budget, "seed": seed,
        "ioh_enabled": with_ioh, "objective_direction": "minimize"})
    started = time.monotonic()
    summary = {"status": "failed", "problem": problem.name, "evaluations": 0,
               "budget": budget, "seed": seed, "best_objective": None, "best_design": None}
    rng = random.Random(seed)
    try:
        objective = make_ioh_problem(problem) if with_ioh else problem
        context = analyzer_logger(objective, output / "ioh", algorithm_name="RandomSearch",
            algorithm_info=f"seed={seed}; budget={budget}") if with_ioh else nullcontext()
        with context, (output / "evaluations.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(budget):
                x = (problem.sample_design(rng) if hasattr(problem, "sample_design") else
                     [rng.uniform(lo, hi) for lo, hi in zip(problem.lower_bounds, problem.upper_bounds)])
                value = float(objective(x))
                record = {"evaluation": index + 1, "x": x, "objective": value,
                          "result": problem.last_result}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                summary["evaluations"] = index + 1
                if summary["best_objective"] is None or value < summary["best_objective"]:
                    summary.update(best_objective=value, best_design=x)
        summary["status"] = "completed"
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    except KeyboardInterrupt:
        summary["error"] = "Interrupted by user"
        raise
    finally:
        summary["wall_time_s"] = time.monotonic() - started
        write_json(output / "summary.json", summary)
    return summary


def add_search_arguments(parser):
    parser.add_argument("--problem", choices=list_problems(), default="P000")
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ioh", action="store_true", help="Also write native IOH Analyzer logs")
    parser.add_argument("--backend", choices=("auto", "native", "docker"), default="auto")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--mesh", choices=MESHES, default="coarse")


def search_from_args(args):
    if args.problem == "P004":
        raise ValueError("P004 needs current data/model: use the P004 search command with --dataset or --model")
    elif args.problem == "P004-hydraulic":
        from .problems.level3_discrete_combinatorial.p004 import HydraulicConfig
        config = HydraulicConfig(mesh=args.mesh)
    elif args.problem == "P001":
        from .problems.level2_medium_parametric.p001 import PPFCConfig
        config = PPFCConfig(mesh=args.mesh)
    elif args.problem == "P002":
        from .problems.level2_medium_parametric.p002 import PEMConfig
        config = PEMConfig(mesh=args.mesh)
    elif args.problem == "P003":
        from .problems.level4_high_dimensional_topology.p003 import TopologyConfig
        config = TopologyConfig(mesh=args.mesh)
    else:
        config = ChannelConfig(cells=MESHES[args.mesh])
    problem = get_problem(args.problem, output_root=args.output / "cases",
        config=config,
        runtime=Runtime(args.backend, args.image, args.timeout))
    return random_search(problem, budget=args.budget, seed=args.seed,
                         output=args.output, with_ioh=args.ioh)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Seeded FlowField benchmark experiment")
    add_search_arguments(parser)
    args = parser.parse_args(argv)
    try:
        result = search_from_args(args)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"flowfield search: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
