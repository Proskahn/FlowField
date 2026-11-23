# FlowField as a benchmark suite

FlowField exposes a problem as a callable with declared bounds, a baseline,
named physical objectives, constraints, status and provenance. An external
optimizer needs only this interface; it does not need to assemble OpenFOAM
cases, choose a solver command, or import the built-in optimization algorithms.

This structure follows the design principles in Olarte Rodríguez et al.,
*MECHBench: A Set of Black-Box Optimization Benchmarks originated from Structural
Mechanics*, arXiv:2511.10821v1, especially Sections 2.1–2.3 and Figure 1. The paper
is a design reference. Its crash models, physical units and solver installation
instructions are not requirements for FlowField.

## Start without CFD

Install from a checkout with Python 3.10 or later:

```bash
python -m pip install -e .
flowfield list
flowfield list --level 1 --kind cfd --details
flowfield describe P002
flowfield decode --problem P002 --x .5 .5 .5 .5 .5 .5 .5
```

`python -m flowfield` is equivalent to `flowfield`. Discovery loads only static
metadata. Inspection constructs only the selected problem and validates its
configuration; it does not check the runtime, create directories or execute
CFD. The catalog reports installation extras, required external inputs and
model limitations. `describe` reports actual configured dimensions, bounds,
baseline, configuration fingerprint and objective instance identity.

```python
from flowfield import create_problem, describe_problem, list_problems, problem_info

ids = list_problems(level=2)              # ('P001', 'P002')
print(problem_info('P002').objective)     # Static catalog description
print(describe_problem('P002')['dimension'])
problem = create_problem('P002', output_root='runs/my-cases')
x = problem.initial_design()
geometry = problem.decode(x)            # Pure, no files or solver
```

A configuration mapping or JSON file can be supplied to both `create_problem`
and `describe_problem`. For P003/P007 it also controls the design-field dimension.
Each problem's `config.py` groups fields by geometry, physical parameters,
initial conditions, boundary conditions, mesh/discretization, solver controls,
objectives/constraints, and diagnostics/output. Topology problems also separate
filtering and projection settings. Comments identify conditions defined in case
generators or reference data. These are source sections; configuration JSON
remains flat. Construct configuration objects with keyword arguments, since
dataclass field order follows these sections.
Fixed-dimensional problems retain their physical parameterization. Continuous
built-ins retain their existing [0,1] bounds, and integer paths retain their
integrality and graph constraints. MECHBench's [-5,5] coordinates are a convention,
not a reason to change the meaning of existing FlowField experiments.

## Evaluate independently of an optimizer

Install the extra reported by `problem_info(...).extra`, if any, and provide
OpenCFD OpenFOAM v2312 through the existing native or Docker runtime. For example:

```bash
flowfield evaluate --problem P000 --x .5 .5 --output runs/designs --backend native
flowfield generate --problem P002 --case runs/p002-inspect
flowfield run runs/p002-inspect --backend native
flowfield extract runs/p002-inspect
```

Omit `--x` to use the declared baseline. Use `--design design.json` with a JSON
array for large fields. `--config config.json` selects a validated configuration.
`run` and `extract` read the problem ID and configuration from the case manifest;
`--problem` can assert the expected ID. A mismatched case is rejected by the
shared immutable-input checks. Runtime overrides are optional: omitting them
preserves the selected problem's timeout and backend defaults.

CLI results are JSON, so other programming languages can submit designs and
consume results without importing Python classes. `evaluate` makes one call to
the problem. The scalar, vector objectives and constraints all come from that
same evaluation. `extract` parses a saved case without executing the solver or
consuming another evaluation. Evaluation roots may contain multiple uniquely
named cases; standalone case destinations must be new.

```python
# Run only after the selected problem's simulation/data requirements are met.
value = problem(x)                       # One evaluation, scalar to minimize
raw = problem.last_result               # Same call: no extra simulation
print(raw['objectives'], raw['constraints'], raw['status'])
```

P004 requires `--dataset` or `--model`; it never substitutes hydraulic scores
for current density. P004-hydraulic is a separate CFD benchmark. P008 can run
baseline diagnostics without calibration, but optimization requires a frozen
qualified EWM calibration. Setup errors are rejected before starting an
experiment or creating its output directory.

## Plan experiments, then execute them

```bash
flowfield benchmark --problems P000 --optimizers random --seeds 1 2 3 \
  --budget 10 --output runs/comparison --no-ioh --dry-run

# Remove --dry-run when the plan and simulation environment are ready.
flowfield benchmark --problems P000 --optimizers random --seeds 1 2 3 \
  --budget 10 --output runs/comparison --no-ioh
```

The plan validates selection, configuration, required objective calibration,
optimizer compatibility and option values. It reports run count, maximum
attempts, bounds, baselines, configurations and instance identities. It creates
no files and does not resolve runtime availability. Baselines and failed
attempts consume the same budget. Each problem/optimizer/seed combination gets
a fresh instance and an independent trace.

```python
from flowfield.benchmarking import plan_benchmark, run_benchmark

experiment = dict(problems=['P000'], optimizers=['random'], seeds=[1, 2, 3],
                  budget=10, with_ioh=False)
plan = plan_benchmark(**experiment)
# run_benchmark(**experiment, output='runs/comparison')
```

`benchmark.json` schema 2 retains the configured plan and relative run paths.
Individual experiment manifests and journals retain their existing identity,
optimizer revision, dependency versions and recovery semantics. Analysis and
plotting consume saved traces; they do not call the problem again. Comparisons
remain restricted to matching objective identities and dimensions unless an
existing explicit analysis override is used.

## Plot objective histories

Plot the scalar score and every named raw objective against evaluation attempts:

```bash
python -m pip install -e '.[plot]'
flowfield plot runs/my-run --kind objectives --output runs/plots/objectives.png
```

```python
from flowfield.benchmarking import plot_objectives

figure = plot_objectives('runs/my-run', output='runs/plots/objectives.pdf')
```

The returned Matplotlib Figure has a scalar panel followed by one panel per raw
objective in alphabetical order, each with its own y scale. This supports scalar
and multi-objective optimizers, including scalarized problems whose raw
components have different units. Raw values retain their saved names, signs and
units; the scalar includes any constraint penalties. Journals without named
objectives produce just the scalar panel.

An iteration here is one evaluation attempt, including the baseline and failed
attempts, rather than an optimizer generation. Curves show candidate values,
not best-so-far values. Circles mark feasible candidates, crosses mark constraint
violations, and failed or missing values leave gaps. A missing component never
falls back to the scalar score. Lines stop at the last actual attempt.

Pass one run directory, a parent directory, or a list of directories to compare
completed runs of the same problem/configuration and dimension. Different
budgets and optimizer failure policies are labeled in the legends. PNG, SVG and
PDF are supported. Plotting only reads saved journals; it never reruns CFD.

## Plot an NSGA-II Pareto front

Use the saved `pareto.json` archive to plot all observed feasible nondominated
designs, including designs from a partial final generation:

```bash
flowfield plot runs/p002-nsga2-42 --kind pareto --output runs/plots/pareto.png
```

```python
from flowfield.benchmarking import plot_pareto

ax = plot_pareto('runs/p002-nsga2-42',
                 objectives=('pressure_sum_Pa', 'voltage_V'),
                 output='runs/plots/pareto.pdf')
```

The function returns a Matplotlib Axes and accepts an existing `ax=`. Pass run
directories, parents containing NSGA-II runs, or `pareto.json` files with their
adjacent `experiment.json` manifests. Multiple runs get separate series; their
objective/configuration, dimension and failure policy must match. Different
budgets are allowed and labeled. PNG, SVG and PDF output are supported.

Two raw objectives are selected automatically in alphabetical order. Override
the axes with `objectives=(x_name, y_name)` or `--objectives X Y`. For more than
two objectives, an explicit pair is required: the plot projects the full Pareto
set without recomputing dominance on just those two axes. Two-objective points
are connected in x order as a visual guide; higher-dimensional projections use
markers only. All objectives retain their saved signs and units and are
minimized. Scalar diagnostic scores are not used as coordinates.

The plot reads the archive directly without replaying journals or rerunning CFD.
It also supports snapshots from interrupted runs and labels the result as a
saved front, not a completed optimization. An empty archive with known objective
names is shown explicitly; an archive without at least two known objectives
cannot define a two-dimensional plot.

## Responsibility boundaries

| Concern | Owner | Contract |
| --- | --- | --- |
| Discovery and requirements | `catalog.py`, `registry.py` | Metadata and lazy factory selection |
| Configured inspection | `inspection.py` | Bounds, baseline, identity; no execution |
| Mathematical problem | `problem.py`, `problems/<level>/<id>/` | Design → objectives, constraints, status, evidence |
| Geometry and physical model | Each problem's config, geometry, mesh and case modules | Design → immutable case inputs |
| Simulation lifecycle | `evaluation.py`, `runtime.py` | Validate inputs, run once, record failures and timing |
| Physical metrics and scalarization | Each problem's metrics/extraction modules | Saved solver outputs → declared formulation |
| One optimization run | `optimization/runner.py`, samplers and state | Proposal/feedback, budget, journal and resume |
| Optimizer CLI policy | `optimization/cli.py` | CLI settings and multi-problem orchestration |
| Experiment planning | `benchmarking/planning.py` | Problem × optimizer × seed specification |
| Experiment execution | `benchmarking/runner.py` | Fresh independent runs and matrix manifest |
| Reporting | `benchmarking/analysis.py`, `plotting.py`, `ioh.py` | Saved traces and optional export |
| User entry points | `cli.py`, problem-specific CLIs | Argument transport and dispatch |

The execution core imports no concrete benchmark implementation. The top-level
CLI imports neither an optimizer nor a simulation implementation for discovery.
Importing a concrete problem through an existing public class name remains
supported and loads only that problem's implementation package.

The paper motivates flexible formulations from multiple physical outputs.
FlowField retains named raw objectives and constraints alongside the explicit
scalarization owned by each problem. Changing an objective or its weights must
change its instance identity. The refactor does not introduce silent
reweighting, fake cheap CFD, new surrogate models, BBOB transformations or HPC
scheduling. MECHBench itself describes instance transformations and faster
surrogate/multi-fidelity evaluations as future work (Section 4).

## Add a benchmark without changing an optimizer

Implement `FlowFieldProblem.decode` and `evaluate_raw`, declare bounds, and
provide `initial_design` when the midpoint is not a physical baseline. Use
`CFDProblem` plus `CaseDefinition` only when standalone simulation cases apply.
Keep fixed-input checks in `preflight` and optimization-specific setup checks
in `validate_optimization` if needed. Constructors, inspection and decoding must
not create cases or execute solvers.

```python
from flowfield import ProblemInfo, register_problem

register_problem('MyProblem', MyProblem,
    info=ProblemInfo('MyProblem', 'My physical design problem', level=2,
                     evaluation_kind='cfd', requirements=('My solver',)))
```

Registration is explicit and process-local. Custom problems can participate in
filtered discovery, inspection and the same experiment runners. To ship a new
built-in, add its lazy factory path and description to `catalog.BUILTINS`, bundle
its necessary resources, and verify its mathematical and physical contract.
Optional dependencies should load where their functionality is used.

Existing `create_problem`, `get_problem`, problem-class imports, dedicated
problem commands, `flowfield-optimize`, `flowfield-benchmark`, `flowfield-plot`
and `flowfield-check` remain supported. The old P000 physical-unit commands
still work; their implementation now lives in `problems/p000/cli.py` and can
also be called explicitly through `flowfield p000` or `flowfield-p000`.
