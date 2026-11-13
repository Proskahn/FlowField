"""Repeated-run benchmarking and IOHanalyzer export, separate from algorithms."""
from .ioh import make_ioh_problem, analyzer_logger
from .analysis import compare_runs


def run_benchmark(**kwargs):
    from .runner import run_benchmark as run
    return run(**kwargs)

__all__ = ["make_ioh_problem", "analyzer_logger", "run_benchmark", "compare_runs"]
