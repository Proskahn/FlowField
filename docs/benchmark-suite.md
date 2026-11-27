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

## P006 search resolution and reference rechecks

P006 supports explicitly cheaper search configurations. Its default remains
`mesh=coarse`, `field_tolerance=1e-5`; existing case and optimizer identities are
unchanged. These presets all use a 3500-iteration cap and two MPI ranks:

| Configuration in `benchmarks/P006/` | Mesh | Baseline cells | Field-change limit |
| --- | --- | ---: | ---: |
| `reference-config.json` | `coarse` | 13,104 | `1e-5` |
| `search-tolerance-config.json` | `coarse` | 13,104 | `1e-4` |
| `search-config.json` | `screening` | 9,072 | `1e-4` |
| `search-coarse-config.json` | `screening-coarse` | 7,056 | `1e-4` |

The recorded local OpenCFD v2312/ARM64 baseline checks with two MPI ranks passed full coupled
convergence and independent saved-field verification:

| Mesh / field limit | Solver iterations | Temperature spread | Difference from reference |
| --- | ---: | ---: | ---: |
| `coarse` / `1e-5` | 1,973 | 0.813470 K | reference |
| `screening` / `1e-4` | 2,734 | 0.798144 K | -1.88% |
| `screening-coarse` / `1e-4` | 2,053 | 0.803560 K | -1.22% |

See [recorded validation evidence](../benchmarks/P006/search-verification.json).
The difference is not monotonic with mesh size, and these baseline comparisons
do not establish accuracy across optimized designs or grid independence.
The tolerance-only preset also passed a continuation from the completed
reference state; that continuation is not a cold-start speed benchmark.
Validation jobs overlapped initially and one was paused, so their elapsed
times are not a controlled runtime comparison. Fewer cells can require more
iterations, as the `screening` baseline demonstrates.

A subsequent native run on Rivium stopped during momentum continuation with
`Gas-mass preconditioner failed: 0.00182144741417`. This is an inner linear
solve failure, not exhaustion of the outer 3500-iteration budget. The local
baseline checks therefore do not establish portable startup robustness.
The gas-mass correction now retries a failed GMRES(80), limited to 800 steps,
once with GMRES(320), limited to 3200 additional steps, using the same operator
and the previous approximation. The true linear residual is recomputed, the
`1e-4` failure limit is retained, and a failed retry still rejects the case.
Normal successful solves keep the original path. Retry messages appear in
the console and solver log. This bounded fallback needs a fresh pilot on the
target machine before a campaign; existing cases contain their own solver
sources and do not pick up the change. Keep the failed run for diagnosis and
use a new output directory rather than resuming its failed baseline.
The [retry verification record](../benchmarks/P006/gmres-retry-verification.json)
covers manufactured stagnation recovery, singular-system rejection, analytic
phase flow in serial/MPI including an empty fluid rank, and a qualified coupled
continuation. It does not reproduce the failed native cold-start matrix.

A later Rivium baseline reached the 3500-iteration cap with full momentum but
heat still off (`field_change=0.146`, local phase balances `0.113` and `0.0468`).
That state is not close to final acceptance, and BO had not started evaluating
new designs. Raising only the final field-change tolerance cannot bypass the
earlier momentum and heat activation checks.

Continuation now preserves a completed partial momentum/heat stage when the
final history and completion status agree at the target voltage. In particular,
a completed state at `momentum_fraction=1`, `thermal_active=false` keeps those
values instead of repeating the entire momentum ramp. At a budget boundary
where the solver announced a new stage after saving the history, continuation
conservatively restores the last stage actually solved. No convergence status
is inherited: all five final samples and saved-field checks must pass again.
Older states without stage evidence and explicit lower starting voltages keep
normal initialization. Use a new same-geometry case with `--initial-state`;
optimizer `--resume` does not retry a failed baseline or import this continuation.

A local experiment partitioned two ranks through the membrane to keep each
fluid network local. It reduced flow linear iterations and passed a full
coupled warm-start check, but its cold start lingered at an intermediate
momentum stage. It has not established faster cold-start qualification and is
not adopted as the default; generated cases retain Scotch decomposition.
The [continuation verification record](../benchmarks/P006/continuation-stage-verification.json)
separates the five-step partial-stage restart smoke test from the qualified
warm partition test and the deliberately interrupted cold experiment.

A subsequent fully coupled serial recovery on Rivium also exhausted 400 steps.
The user confirmed that its bundled solver used the proposed `1e-14` final-stage
Anderson regularization. Its final liquid-velocity and pressure changes were
`4.06e-4` and `1.44e-4`, respectively, against a `1e-4` field limit. A local
106-step pass with that setting therefore did not establish portable recovery.
Further direct-QR experiments passed independent least-squares checks in serial
and MPI, but full CFD remained sensitive: variants that qualified the original
saved state failed a `1e-8 Pa` starting-pressure perturbation with the identical
solver binary and otherwise identical inputs. These candidates are not adopted
as recovery fixes. The [sensitivity verification record](../benchmarks/P006/anderson-sensitivity-verification.json)
records completed failures separately from deliberately stopped experiments.
The nonlinear convergence problem remains unresolved; the observed objective
stability is insufficient to qualify the case or establish campaign readiness.

Cell counts refer to the 1 mm manifold. Search meshes coarsen only the channel
length direction (nominal spacing 1.5 or 2 mm). They preserve the transverse
grid, all material layers, manifold interfaces, the branch sampling plane,
inlet flow and fixed topology across channel-width designs. Reduced grids
start at the existing `1e-6` momentum continuation stage, where numerical
pseudo-storage can regularize the initial pressure/saturation solve. This
avoids the zero-storage startup used by the reference grid; it does not
change the required final momentum fraction of 1.

The field-change limit measures change between nonlinear iterations, not
physical prediction error. `1e-4` is the maximum supported search limit.
Conservation and momentum limits remain at `1e-3` or tighter, charge acceptance
remains at `1e-6`, boundedness violations must be zero, and five consecutive
iterations must pass with full momentum and energy coupling at the requested
voltage. Independent saved-field checks still run. A 3500-iteration cap does
not guarantee convergence; failed or timed-out cases remain unqualified.

Cases, evaluation metadata and optimizer plans record `fidelity`, including
`purpose=search` and `reference_recheck_required=true` for reduced meshes or
relaxed field-change limits. A passing search case qualifies only under its
recorded settings. Mesh independence and preservation of design rankings
require comparison against reference results. Pilot the tolerance-only preset
first, then compare reduced meshes on the baseline and several varied feasible
designs before committing to a large optimization campaign.

Run a pilot on the machine with OpenCFD v2312 loaded:

```bash
.venv/bin/flowfield-check run --problem P006 \
  --config P006=benchmarks/P006/search-tolerance-config.json \
  --backend native --timeout 28800 \
  --output runs/p006/search-tolerance-check-001
```

Select `search-config.json` or `search-coarse-config.json` to test fewer cells.
Keep the same configuration for each optimizer and seed being compared. For
example, after qualifying the pilot:

```bash
.venv/bin/flowfield-optimize --problem P006 --optimizer bo --seed 42 \
  --budget 200 --config benchmarks/P006/search-tolerance-config.json \
  --backend native --timeout 28800 \
  --output runs/p006/search-tolerance/p006-bo-42
```

`random`, `turbo` and `cmaes` use the same options. The budget counts attempted
design evaluations, including the baseline and failures; each accepted
evaluation still requires a full coupled solve under the selected settings.
Use new output directories when changing mesh or tolerance. Resume verifies
the configuration fingerprint and refuses to mix different fidelities in one
optimizer history. `flowfield-p006 generate/evaluate` also accepts
`--mesh screening` and `--field-tolerance 1e-4` directly; existing cases cannot
be relabeled by overriding these flags on `run` or `extract`.

Re-evaluate the best feasible designs with reference settings. For example,
export the winning normalized design and run it as a new case:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
run = Path('runs/p006/search-tolerance/p006-bo-42/P006')
winner = json.loads((run/'best-feasible.json').read_text())
(run/'reference-design.json').write_text(json.dumps(winner['x']))
PY
.venv/bin/flowfield evaluate --problem P006 \
  --config benchmarks/P006/reference-config.json \
  --design runs/p006/search-tolerance/p006-bo-42/P006/reference-design.json \
  --backend native --timeout 28800 --output runs/p006/reference-recheck-001
```

Repeat this for several finalists to check ranking. For mesh sensitivity,
create a reference configuration with `mesh=medium` and retain `1e-5`, then
compare the physical observables. Reference here means the original numerical
settings, not a claim of grid independence. Same-geometry, same-mesh completed
fields may initialize a new stricter case through `--initial-state`, but never
transfer qualification. Changing mesh requires a fresh solve; cross-mesh
field copying is rejected.

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

## P002 GCMMA with coupled discrete adjoints

`flowfield-p002 gcmma` minimizes the same scalar objective used by P002 CMA-ES,
BO and TuRBO. With the default current and flow rate it is
`1.21 * voltage_V + 2.5e-7 * pressure_sum_Pa` in watts. It uses the existing
GCMMA conservative subproblem solver and a separate, checked linearization of
the original OpenFOAM primal solver. No alternative physical model is used.

```sh
.venv/bin/python -m pip install -e '.[gcmma]'
flowfield-p002 gcmma --output runs/p002/gcmma/paper-start \
  --initial paper --iterations 30 --max-evaluations 100 --backend docker

# Plan without creating output or invoking Docker.
flowfield-p002 gcmma --output runs/p002/gcmma/paper-start --dry-run

# Increase an exhausted CFD budget, preserving sources/configuration/options.
flowfield-p002 gcmma --output runs/p002/gcmma/paper-start \
  --resume --max-evaluations 150 --backend docker

# Inspect both objective gradients at an existing, verified serial case.
flowfield-p002 adjoint runs/path/to/P002-case --output runs/adjoint.json
```

The state includes velocity, pressure, face mass flux, gas concentration,
temperature, electronic and ionic potentials, and catalyst current. Sparse
automatic differentiation forms the complete coupled residual Jacobian. Two
transpose solves give the individual objective gradients; their weighted sum
drives GCMMA. Only mesh-coordinate derivatives use finite differences, with no
perturbed CFD solves and with connectivity/material checks at two step sizes.
The discrete-adjoint identity is described in the
[pyadjoint gradient documentation](https://pyadjoint.readthedocs.io/en/latest/documentation/maths/3-gradients.html).

Eight independent CFD probes check both objective gradients at the first
iterate by default. They are validation calls, count against the CFD budget,
and are never substituted for the adjoint gradient. `--no-check-gradients`
explicitly skips these probes. Every initial evaluation, trial and failure
counts; adjoint time is recorded separately. Compare optimizers using wall
time as well as CFD counts. A completed iteration schedule or a small step is
not a global-optimum certificate.

The generated mesh is only piecewise differentiable. The 1 mm baseline is a
connectivity transition, so the default start is the paper Table 5 geometry;
`--x` accepts any smooth, in-bounds seven-variable initial design. Derivative
evaluation rejects topology changes. Trial points with invalid meshes or CFD
failures cause smaller GCMMA moves without fabricated penalty gradients.
Successful CFD probes and trials contribute to `best.json`; accepted GCMMA
iterates are recorded separately in `current.json`.

This implementation supports the pinned serial P002 discretization and checks
source/input hashes, saved-field conservation, the coupled primal residual,
and transpose-solve accuracy. It defaults to a 150,000-unknown resource guard
(coarse mesh); larger meshes require an explicit `--max-unknowns` override and
substantial memory. The tested 111,180-unknown coarse case used roughly 186
million LU entries, so several GB of free memory are needed. PyMETIS ordering
is included in the `gcmma` optional dependencies. The original P002 convergence
controls remain unchanged; a tighter outer tolerance alone can hit the linear
solvers' numerical noise floor.

`gcmma-run.json`, `evaluations.jsonl`, `attempts/`, `adjoints/`, and
`gradient-check.json` retain the experiment and validation evidence. Resume
replays completed evaluations and adjoints under identical sources, settings
and dependency versions. An interrupted CFD request without a saved result
stops replay for inspection; it is not silently re-executed. P002 remains a
provisional paper reconstruction: these gradient checks validate derivatives
of the implemented solver, not agreement of its physics with the paper.

## Scalable P003 on a fixed medium CFD mesh

Specify the dimension directly when launching optimization; no configuration
file is needed:

```bash
.venv/bin/flowfield-optimize --problem P003 --optimizer turbo \
    --dimension 500 --seed 42 --budget 200 --backend native --timeout 14400 \
    --output runs/p003/scalable/n500-turbo-42
```

Change `--dimension` to select another size and `--optimizer` to select an
algorithm, such as `cmaes`. Supported examples include **500, 512, 1000, 2000,
2500, 3000, 4000, 5000, and 6000**. Other exact counts are accepted when they
fit a supported rectangular design grid; unsupported counts are rejected
before any CFD starts. The grid is selected automatically and shown with the
actual CFD cell count in `--dry-run` output and saved run metadata.

The multi-seed benchmark command accepts the same option:

```bash
.venv/bin/flowfield-benchmark --problem P003 --optimizers turbo cmaes \
    --dimension 500 --seeds 42 43 44 45 46 --budget 200 \
    --backend native --timeout 14400 \
    --output runs/p003/scalable/n500-benchmark
```

`--dimension` selects the medium mesh automatically. It applies to P003 only
and cannot be combined with `--config`. Omitting it preserves the original
default or an explicitly supplied configuration. An optimization run saves
the dimension and complete effective configuration; resume using just
`--resume --output RUN`, optionally with `--extend-budget TOTAL`.

These instances vary the number of nodal design variables while retaining
the same 8,320-cell medium CFD mesh, physical
plate, 1.2 mm inlet/outlet widths, 0.35 mm filter radius, beta 50 projection,
power weight 1, uniformity weight 0, and material stage 1.
The maximum CFD iteration count is 10,000: the medium-mesh uniform pilot did
not meet its balance tolerances within the legacy limit of 3,000. The original
convergence tolerances are unchanged, and the solver stops when they are met.

| Optional configuration | Variables | Design points, rows × columns |
| --- | ---: | ---: |
| `n500.json` | 500 | 25 × 20 |
| `n1000.json` | 1,000 | 40 × 25 |
| `n2000.json` | 2,000 | 50 × 40 |
| `n3000.json` | 3,000 | 75 × 40 |
| `n4000.json` | 4,000 | 80 × 50 |
| `n5000.json` | 5,000 | 100 × 50 |
| `n6000.json` | 6,000 | 100 × 60 |

These instances explicitly select `design_mapping="bilinear_cell_filter_v1"`:
bilinear interpolation from the control nodes to CFD cell centers, followed by
an area-weighted radial cone filter on those cells, then the original projection
and material interpolation. This makes low-dimensional grids possible without
enlarging the physical filter radius. It is a new parameterization, not an exact
copy of the paper's direct nodal filtering. Use the same mapping at every
dimension being compared. The original `nodal_filter` mapping and its default
configuration fingerprint and numerical outputs are preserved.

For advanced control, the optional configurations in `benchmarks/P003/scalable/`
remain available. To export a configuration or specify points on both axes
explicitly, use the following commands. They generate configurations only,
run no CFD or optimization, and refuse to overwrite an existing file.

```bash
.venv/bin/flowfield-p003 scalable-config --dimension 2500 \
    --output benchmarks/P003/scalable/n2500.json

# Nearly equal physical point spacings; exactly 512 variables.
.venv/bin/flowfield-p003 scalable-config --shape 32 16 \
    --output benchmarks/P003/scalable/n512.json
```

`--mesh medium`, `--stage 1`, and `--filter-radius 0.00035` are the defaults.
`--iterations` changes the maximum CFD solver iteration count (default 10,000),
not the optimizer budget.
`--dry-run` inspects the mapping without writing files. The report lists the
actual dimension, grid shape, cell count, point spacings and variable support.
Exact counts must factor into a rectangular grid with at least three points
per axis. Among the supported factorizations the generator chooses the closest
physical point spacings; it never rounds the dimension or pads unused variables.
For more even spacing across the dimension sweep, use explicit shapes and
accept nearby counts such as 512 rather than 500. Either axis is rejected if
it has more design points than CFD cells, which would force redundancy in this
bilinear mapping. Support checks do not establish numerical rank or CFD accuracy.

Use `--config PATH` instead of `--dimension N` only when supplying a custom
configuration. `flowfield-p003 gcmma --dimension 500 --output RUN` also accepts
the direct dimension option, but that runner applies its own continuation schedule and uses
one finite-difference CFD probe per variable per gradient. Compare fixed-stage
instances separately from continuation campaigns; include gradient probes in
CFD budgets. Geometry plotting reconstructs the selected mapping automatically.

New mesh, dimension or mapping settings require new runs. Archived runs keep
their original settings. GCMMA additionally pins the source code, so retain the
original checkout for resuming a pre-update GCMMA campaign even if it uses the
unchanged legacy mapping.

The medium mesh is a common analysis resolution, not a claim of mesh independence.
Forward CFD checks passed for a 500-variable uniform design (3,833 solver
iterations) and a 6,000-variable parallel-channel design (1,228 iterations),
including the existing independent balance checks. These check the implementation
at the dimension endpoints; they do not establish mesh independence or compare
optimization performance, since the two designs differ.
Check representative designs on finer meshes before using small differences in
power to rank algorithms. A fixed physical filter produces more smoothing
relative to point spacing at higher dimensions, so record this setting and the
control-grid shape when reporting results. Individual variable influence does
not guarantee equal sensitivity or difficulty across dimensions.

## P003 GCMMA with material continuation

`flowfield-p003 gcmma` optimizes every raw topology node (2,145 variables on
the default 65 × 33 grid). It uses the optional
[mmapy 0.3.1 implementation of GCMMA](https://github.com/arjendeetman/GCMMA-MMA-Python)
(GPL-3.0), including conservative inner iterations. The PDE objective gradient
uses finite differences; the channel-volume constraint uses the exact derivative
of the filter and projection. This is not an adjoint implementation.

The default objective is negative power divided by initial power, with power
weight 1 and current-uniformity weight 0. Initial normalization is calibrated
once and frozen. Volume is an explicit inequality, not an objective penalty in
GCMMA. Derivative probes can exceed the volume limit and are never repaired;
accepted steps must satisfy it.

At the default 5 Pa, the material continuation is `(p,q) = (2,1000), (3,100),
(5,10)`, with 60 accepted iterations per stage and projection beta 50. The
optimized design carries forward and the CFD is recomputed at every stage.
These parameters penalize gray material through diffusivity and Brinkman drag;
there is no additional arbitrary gray-pixel term in the objective. Below or at
4 Pa, the default schedule also includes stage 0, `(1,10000)`.

From the repository root, with native OpenCFD OpenFOAM v2312 already sourced:

```bash
.venv/bin/python -m pip install -e '.[gcmma,plot]'
mkdir -p runs/p003
output="runs/p003/gcmma-continuation-42"

nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    .venv/bin/flowfield-p003 gcmma \
    --output "$output" \
    --backend native --timeout 14400 --workers 8 \
    --stages 1 2 3 --iterations-per-stage 60 \
    --fd-scheme forward --fd-step 0.0001 --move 0.01 \
    --check-gradients --discard-probe-cases \
    --max-evaluations 390000 \
    > "${output}.log" 2>&1 < /dev/null &
echo $! > "${output}.pid"
```

`--workers 8` runs up to eight independent serial CFD probes; set it to suit the
server's available CPU and memory. The schedule needs **386,320 CFD evaluations**
if every first trial is accepted, including calibration and directional checks.
With all 15 inner trials used at every step it needs 388,840, before any explicit
retries. The 390,000 cap leaves room for retries. Add `--dry-run` to inspect a
command without creating files or running CFD. GCMMA starts from the uniform
design deterministically; looping over seeds 42–46 repeats the same experiment.

The first gradient in every stage is checked against three centered directional
differences at two step sizes. Failed checks or CFD solves stop the run. Completed
requests are saved individually, including partial gradients. Resume the same
campaign after its process has stopped:

```bash
output="runs/p003/gcmma-continuation-42"
nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    .venv/bin/flowfield-p003 gcmma --resume --output "$output" \
    --backend native --timeout 14400 --workers 8 \
    --retry-interrupted \
    >> "${output}.log" 2>&1 < /dev/null &
echo $! > "${output}.pid"
```

Saved settings are reused. `--retry-interrupted` counts each unfinished CFD
request again; completed requests are reused. After correcting a runtime error,
`--retry-failed` similarly retries failed requests and preserves the old failure
record. `--extend-budget 1000` adds 1,000 to the existing total cap. Changing
model settings, continuation, numerical dependencies or implementation requires
a new campaign. TuRBO output cannot be resumed as GCMMA.

`--discard-probe-cases` deletes only successful gradient/check CFD folders after
saving their JSON results; it retains all baseline, trial and failed cases.
Designs, metrics, configurations and attempt records remain available for
replay and geometry plotting, including when a probe is the best feasible
design. Omitting this flag keeps all cases (about 2 TB for the default schedule
at the locally measured 6.2 MB per case); JSON records alone still require
substantial storage. A deleted probe's flow fields require a new CFD evaluation
to inspect. Each stage has its own `best-feasible.json`, `current.json`, and
`accepted-iterations.json`. The latter records accepted outer iterations;
generic black-box convergence journals are not produced. Stage objective
values should not be compared as if they used the same material model.

Plot the final stage's best saved geometry without CFD or ParaView:

```bash
.venv/bin/flowfield-plot "$output/stage-3" --kind geometry \
    --output "$output/geometry.png" \
    --pixels-output "$output/geometry-pixels.png"
```

The pixel image has 65 rows and 33 columns. `completed_schedule` means that all
requested iterations ran, not that an optimality tolerance was met or the paper
was reproduced. The forward physics remains provisionally validated.

## BoTorch multi-objective Bayesian optimization

Two optional optimizers use the named physical `objectives` and explicit
`constraints`, independently of the scalar diagnostic `objective`:

| Optimizer ID | BoTorch acquisition | Search criterion |
| --- | --- | --- |
| `qlognehvi` | `qLogNoisyExpectedHypervolumeImprovement` | Log noisy expected hypervolume improvement |
| `qlognparego` | `qLogNParEGO` | Log noisy expected improvement of a new random augmented Chebyshev scalarization |

Install the optional dependency and run either method directly:

```bash
.venv/bin/python -m pip install -e '.[mobo,p003,plot]'
.venv/bin/flowfield-optimize --problem P003 --dimension 500 \
    --optimizer qlognehvi --seed 42 --budget 100 \
    --backend native --timeout 14400 \
    --output runs/p003/mobo/qlognehvi-d500-42
```

Replace `qlognehvi` with `qlognparego` and choose a fresh output directory for
the second method. The existing `--dimension` option selects the design grid
and medium CFD mesh. No configuration file or scalar objective weights are
needed. Both power and current-density uniformity remain active objectives.

The benchmark runner also accepts both methods across seeds:

```bash
.venv/bin/flowfield-benchmark --problem P003 --dimension 500 \
    --optimizers qlognehvi qlognparego --seeds 42 43 44 45 46 \
    --budget 100 --backend native --timeout 14400 --no-ioh \
    --output runs/p003/mobo/d500-seeds42-46
```

Add `--dry-run` to either command to inspect settings without installing
BoTorch, writing files or running CFD. Actual execution uses BoTorch 0.16.1
with a CPU, double-precision `SingleTaskGP`, independent standardized outputs,
the default dimension-scaled RBF priors and learned observation noise.
All FlowField objectives are minimized: the adapter negates them for BoTorch's
maximization convention. Named constraints retain their `g(x) <= 0` direction.
At least two finite named objectives are required; there is no scalar fallback.

Both adapters propose one evaluation at a time (`q=1`). They select the largest
BoTorch acquisition value over a fresh uniform candidate pool. This is a finite
candidate approximation, not BoTorch's continuous multistart acquisition
optimizer. Problem-owned repair is applied before novelty checks and acquisition
scoring, so the GP observes the coordinates actually evaluated. A constraint
explicitly enforced by that repair (P003's channel volume) is checked on returned
results and excluded from the GP; remaining constraints are modeled as separate
outputs and used in the acquisition's feasibility weighting.

`--bo-initial-points` controls initialization attempts including the baseline
(default 5). `--bo-candidates` controls pool size (default 2048), and
`--mobo-mc-samples` controls Sobol Monte Carlo samples (default 128; power of two,
at least 16). Initial candidates use a seeded Latin hypercube. Failed attempts
still count against the budget and are excluded from the GP and Pareto archive;
their scalar failure penalties are never treated as physical measurements.
If failures leave fewer than two valid observations, initialization continues
within the original attempt budget. A numerical model/acquisition failure stops
the run rather than silently switching algorithms.

By default qLogNEHVI infers its acquisition reference point once, at its first
model step, from feasible valid observations (or all valid observations if none
are feasible). It freezes that point for the rest of the run. To set a common
point explicitly, pass `--mobo-reference-point` as a JSON object whose keys
exactly match the raw objective names and whose values use the original
minimization signs and units. For example, the *acquisition* setting
`'{"negative_power_domain_W":0,"current_std_A_m2":1000}'` specifies zero power
and 1,000 A/m² standard deviation; it is not a validated benchmark scoring
reference. qLogNParEGO draws a new seeded Dirichlet(1) scalarization vector at
each model step and requires no hypervolume reference point. Its weights are
independent of P003's scalar power/uniformity weights.

Complete observations and acquisition metadata are journaled. The effective
reference point and qLogNParEGO weights are recorded in proposal metadata;
`pareto.json` contains all observed feasible nondominated designs. Resume and
budget extension use the existing commands and reconstruct proposals without
rerunning CFD. Resume verifies settings and the original Python, NumPy, SciPy,
PyTorch, BoTorch, GPyTorch and linear-operator versions. Model/acquisition work
uses one PyTorch CPU thread and restores the caller's thread count and CPU RNG.

These adapters use the multi-objective/Pareto execution path. Select
`--regime multi` explicitly and use `flowfield-analyze` below for shared
hypervolume scoring. Scalar `best_objective` and IOH traces remain diagnostics;
do not use them to rank these algorithms' Pareto-front quality. Acquisition
reference points inferred separately per run are not a common scoring reference.
The adapters accept full P003 dimensions, but correctness tests at 500 and
6,000 variables do not establish sample efficiency at those dimensions.

The implementations use the upstream [BoTorch 0.16.1 acquisition APIs](https://botorch.readthedocs.io/en/v0.16.1/acquisition.html)
for [qLogNEHVI](https://botorch.readthedocs.io/en/v0.16.1/_modules/botorch/acquisition/multi_objective/logei.html)
and [qLogNParEGO](https://botorch.readthedocs.io/en/v0.16.1/_modules/botorch/acquisition/multi_objective/parego.html).

## Objective regimes and IOHinspector analysis

`flowfield-optimize` and `flowfield-benchmark` accept `--regime single|multi`.
The Python `optimize`, `plan_benchmark`, and `run_benchmark` APIs accept the same
`regime` keyword. Explicit selections reject incompatible optimizers before
preflight or CFD. Existing commands that omit it retain per-optimizer inference.

| Regime | Optimizers | Benchmark result |
| --- | --- | --- |
| `single` | random, CMA-ES, lq-CMA-ES, BO, TuRBO, COBYLA | Best feasible scalar objective |
| `multi` | random, NSGA-II, qLogNEHVI, qLogNParEGO | Feasible Pareto archive and hypervolume history |

Random search uses the same proposals in both regimes. In `multi` it also
maintains `pareto.json`. Explicit multi-objective execution requires at least
two finite named raw objectives; scalar values remain diagnostic. The regime
and validation policy are saved and cannot change on resume. P003 uses
`negative_power_domain_W` and `current_std_A_m2`, both minimized, regardless
of its scalar power/uniformity weights. Its channel-volume constraint remains
`g(x) <= 0`, with feasibility tolerance `1e-12`.

P002 supports the same regime with its original seven variables and the raw
objectives `voltage_V` and `pressure_sum_Pa`, both minimized. Its configured
scalarization (`total_power_w`, `voltage_v`, or `pressure_sum_pa`) only affects
scalar diagnostics during a multi-objective run.

`flowfield-benchmark --population N` sets the population for selected NSGA-II,
CMA-ES, and lq-CMA-ES runs (default 4); it leaves random and BO methods alone.
`--failure-penalty VALUE` applies the same explicit failure policy to every
selected run. Rejected meshes and failed solves still count as attempts;
their scalar penalties never become physical objective vectors or contribute
to feasible hypervolume. Without this option, CFD failures stop the campaign.
Both controls are validated by `--dry-run` and recorded in the campaign plan.

### P002 multi-objective benchmark

Install the optimizer, logging, and analysis dependencies, then run this small
proof of concept on the server after loading OpenFOAM v2312:

```bash
.venv/bin/python -m pip install -e '.[mobo,ioh,inspector,plot]'

.venv/bin/flowfield-benchmark \
    --problem P002 --regime multi \
    --optimizers random nsga2 qlognehvi qlognparego \
    --seeds 42 --budget 6 --population 4 \
    --bo-initial-points 3 --bo-candidates 64 --mobo-mc-samples 32 \
    --failure-penalty 1000000 --backend native --timeout 1800 \
    --output runs/p002/multi-poc-42

.venv/bin/flowfield-analyze \
    --input runs/p002/multi-poc-42 --regime multi \
    --output runs/p002/multi-poc-42-analysis
```

For local use, select `--backend docker` with Docker Desktop and the v2312
image ready. The four methods run sequentially, spending at most 24 CFD
attempts including baselines and failures. This is a pipeline check, not a
comparison of optimizer performance. `--population 4` applies to NSGA-II;
random and BO retain their own sampling policies. Add `--dry-run` to validate
the plan without running CFD or creating files. Always choose a new output.

The scalar failure penalty is only a continuation policy and diagnostic. A
failed solve has no physical voltage/pressure vector, is excluded from GP
training and Pareto archives, and contributes no hypervolume improvement.
The campaign manifest records the objective names and links each run's
`P002/<optimizer>/seed-<seed>/pareto.json`. The analysis produces a
`convergence.png`, `indicators.csv`, and IOHinspector dataset. Higher feasible
hypervolume is better. Keep `analysis-settings.json` to freeze the shared
normalization/reference across later comparisons with `--settings`.

For a longer run that survives disconnecting SSH, choose a fresh campaign path:

```bash
mkdir -p runs/p002
campaign="runs/p002/multi-b100-seeds42-46"
nohup bash -c '
    set -e
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    .venv/bin/flowfield-benchmark \
        --problem P002 --regime multi \
        --optimizers random nsga2 qlognehvi qlognparego \
        --seeds 42 43 44 45 46 --budget 100 --population 10 \
        --failure-penalty 1000000 --backend native --timeout 1800 \
        --output "$1"
    .venv/bin/flowfield-analyze --input "$1" --regime multi \
        --output "${1}-analysis"
' _ "$campaign" > "${campaign}.log" 2>&1 < /dev/null &
echo $! > "${campaign}.pid"
```

This runs 20 optimizer/seed combinations sequentially, with a maximum of 2,000
attempts, using the default BO settings. It does not modify P002's geometry,
seven-variable dimension, coarse mesh policy, or convergence tolerances.
`tail -f "${campaign}.log"` follows the log. Existing results can be analyzed
separately without new CFD, for example:

```bash
.venv/bin/flowfield-analyze --input runs/p002/nsga2 --regime multi \
    --output runs/p002/nsga2-hypervolume
```

### P003 multi-objective proof of concept

For a small **local** P003 proof of concept, start Docker Desktop and use the
existing OpenFOAM v2312 image. If the image is absent, pull it first with
`docker pull opencfd/openfoam-default:2312`. No SSH or external server is used:

```bash
cd /Users/zhuang/FlowField
.venv/bin/python -m pip install -e '.[mobo,p003,ioh,inspector]'

.venv/bin/flowfield-benchmark \
    --problem P003 --dimension 500 --regime multi \
    --optimizers random qlognehvi qlognparego --seeds 42 --budget 6 \
    --bo-initial-points 3 --bo-candidates 64 --mobo-mc-samples 32 \
    --backend docker --timeout 14400 \
    --output runs/p003/multi-poc-d500-42

.venv/bin/flowfield-analyze \
    --input runs/p003/multi-poc-d500-42 --regime multi \
    --output runs/p003/multi-poc-d500-42-analysis
```

This spends at most 18 CFD attempts: six per optimizer, including each baseline.
The medium mesh and 500-variable parameterization are unchanged. Six attempts
and one seed verify the pipeline; they do not establish algorithm performance.
Use new output paths for another campaign. Add `--dry-run` to the benchmark
command to validate the plan without creating files or launching CFD. Native
Linux execution can instead use `--backend native` after sourcing OpenFOAM.

Analysis only reads existing journals and never instantiates a problem or
executes a solver. It also works on older scalar-optimizer runs that retained
both raw objectives:

```bash
.venv/bin/flowfield-analyze \
    --input runs/p003-turbo-42 --regime multi \
    --output runs/p003/turbo42-hypervolume
```

Inputs may be run directories, parent/benchmark directories, or individual
`evaluations.jsonl` paths. Resume-session copies are excluded; overlapping
input paths are deduplicated. Duplicate algorithm/seed runs are rejected.
Analyze each problem/configuration, dimension and failure policy separately.
Unfinished and failed attempts remain in the evaluation clock. Only valid,
feasible objective vectors contribute hypervolume. A recovered attempt is
counted once; a run without a feasible solution has zero hypervolume. If no
selected run has feasible observations, provide an explicit reference point.
Incomplete/truncated JSON lines are rejected; retry once the writer finishes.

The new analysis directory contains:

- `evaluations.csv`, `mapping.json`, `columns.json`: all attempts, raw objective
  vectors, constraints, validity, feasibility and recorded evaluation times.
- `iohinspector/`: the converted IOHinspector Parquet dataset, ready for
  `DataManager.add_dataset(...)`. No upload occurs.
- `analysis.json`, `indicators.csv`, `convergence.png`: per-run anytime scores
  at every recorded attempt, without extrapolating to unspent budget.
- `analysis-settings.json` in multi mode: shared objective normalization and
  the hypervolume reference, plus the problem identity for reuse.

Hypervolume is computed by IOHinspector 0.2.1's `add_indicator` and `HyperVolume`,
using moocore 0.3.2. Full histories are imported with `monotonic=False`; filtering
by scalar improvements would lose genuine Pareto improvements. Invalid and
infeasible rows are represented by zero-volume reference points only in the
temporary scoring frame; the exported physical objectives are retained.

By default, each objective is normalized using the minimum and maximum over
**all selected feasible observations, across algorithms and seeds**. The shared
reference is `(1.1, ..., 1.1)` in normalized coordinates. A constant axis uses
`max(abs(value), 1)` as its recorded scale. These are empirical analysis
settings, not a known Pareto front or optimal hypervolume. Reuse the generated
settings for later comparisons; independently normalized reports have different
scales and their scores must not be compared directly:

```bash
.venv/bin/flowfield-analyze --input runs/p003/multi-poc-d500-42 \
    --settings runs/p003/multi-poc-d500-42-analysis/analysis-settings.json \
    --output runs/p003/multi-poc-d500-42-analysis-repeat
```

Alternatively, `--reference-point` accepts a JSON map with **raw minimization
values** for every objective. This disables normalization and produces
hypervolume in the product of the raw objective units. For example,
`'{"negative_power_domain_W":0,"current_std_A_m2":1000}'` is a user-selected
reference, not a physically validated standard. Designs outside that reference
box contribute no volume. This scoring reference is independent of BoTorch's
`--mobo-reference-point` acquisition setting.

Recorded cumulative time is the sum of evaluation durations, excluding BO
fitting/proposal overhead. It becomes unknown when a duration is missing; it
is not reported as total campaign wall time. Source journal hashes and analysis
dependency versions are included in `analysis.json`.

For single-objective runs, use `flowfield-analyze --regime single` to export
the same IOHinspector dataset and compute best feasible scalar histories.
`flowfield analyze` is an equivalent entry point. To access the exported data
directly (without discarding scalar non-improvements):

```python
from iohinspector import DataManager

manager = DataManager()
manager.add_dataset("runs/p003/multi-poc-d500-42-analysis/iohinspector")
data = manager.load(monotonic=False, include_meta_data=True)
# columns.json maps objective_0, objective_1, ... to physical objective names.
```

See the [IOHinspector project](https://github.com/IOHprofiler/IOHinspector) and
its [multi-objective tutorial](https://github.com/IOHprofiler/IOHinspector/blob/main/examples/MO_Examples.ipynb).

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

## Plot a multi-objective Pareto front

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
directories, parents containing NSGA-II or BoTorch multi-objective runs, or `pareto.json` files with their
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
