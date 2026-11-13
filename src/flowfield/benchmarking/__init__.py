"""Repeated-run benchmarking and IOHanalyzer export, separate from algorithms."""
from .ioh import make_ioh_problem, analyzer_logger
from .analysis import compare_runs


def __getattr__(name):
    if name in ('plot_benchmark', 'plot_convergence', 'plot_fixed_budget', 'plot_ert', 'plot_ecdf'):
        from . import plotting
        return getattr(plotting, name)
    raise AttributeError(name)


def run_benchmark(**kwargs):
    from .runner import run_benchmark as run
    return run(**kwargs)

__all__ = ["make_ioh_problem", "analyzer_logger", "run_benchmark", "compare_runs",
           "plot_benchmark", "plot_convergence", "plot_fixed_budget", "plot_ert", "plot_ecdf"]
