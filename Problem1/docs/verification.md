# Verification record

## 64-opening update, 4 November 2026

New default cases use 64 openings at 0.5 mm pitch, diagnostics every 10 solver
steps plus the final step, and the unchanged 0.01 s field-write interval.
Total Faraday-law oxygen feed is unchanged; local opening velocity is four
times the legacy value. All three generated medium meshes contain 192,512 cells
and pass `checkMesh -allTopology -allGeometry`. Coarse meshes contain 32,352 cells.

Eleven Python tests pass, including conservation of the imposed gas feed when
the opening count changes, geometric invariants across the three layouts,
independent diagnostic/field-write intervals, and rejection of invalid sampling
intervals and attempts to overwrite an existing generated case.

Three separate four-process OpenCFD v2312 startup runs completed 23 steps at a
fixed 1 µs step: coarse with every-step diagnostics, coarse with every-10-step
diagnostics, and medium with every-10-step diagnostics. The sampled runs record
steps **10, 20, and 23**. The last sample is retained even though step 23 is not
a multiple of 10. Every-step sampling records 23 rows, without duplicating the
final sample. All three pass the existing 10⁻⁷ sampled cell-alpha bound tolerance.

At matching times, every diagnostic column in the two coarse runs is identical.
All twelve reconstructed field files (`alpha.water`, `U`, `p`, `p_rgh`, `phi`,
and `alphaPhi0.water` at 10 and 20 µs) are byte-identical. Thus the sampling change
did not change the computed solution in this test. Sparse diagnostics miss
intervening extrema; their lower maximum residual is not evidence of improved
numerical conservation.

Details: [openings64_verification.json](../results/openings64_verification.json).
Short-run evidence is in `runs/openings64-check/`; new 1 s cases are clean initial
conditions in `cases/openings64/`. These tests do not establish 1 s stability,
stationarity, convergence, or runtime speedup. Production cases retain adaptive
time stepping; only the sampling tests used a fixed step for exact comparison.
The previous 256-opening results and their verification record are preserved.

## Production-case audit, 4 November 2026

The server copy of `cases/w1mm` now contains saved fields and diagnostics through
1 s. The [production audit](w1mm_stability_2026-11-04.md) found closely bounded
fields and small instantaneous conservation errors, but also overlapping solver
histories, an incomplete log, and six missing local processor meshes. The existing
10⁻⁷ alpha tolerance is exceeded slightly during startup. This is not a full-run
stability or convergence sign-off. The audit includes a diagnostic figure and
machine-readable evidence, plus a tested ParaView subset-export workflow.

The sections below record the earlier, separate startup tests. Statements about
clean `cases/` directories describe their state at that time.

## Original startup verification

Verified on 23 October 2026 using the installed ARM64 Docker image
`opencfd/openfoam-default:2312` and OpenCFD OpenFOAM v2312. The existing image was
used; no solver installation or image download was needed.

## Checks completed

- Five Python tests pass: equal physical volumes and oxygen feed across all three
  arrangements, mesh refinement preserving pore geometry, Faraday-law scaling,
  rejection of invalid inputs, and physical-time weighting on nonuniform samples.
- All three medium meshes pass `checkMesh -allTopology -allGeometry`, with zero
  nonorthogonality, positive volumes, and valid topology. Coarse meshes also pass.
- `interFoam` runs on all three coarse meshes to approximately 0.001 s.
- All three medium meshes run to 0.000020 s; the 1 mm medium case runs on four MPI
  processes, including successful decomposition and field reconstruction.
- The compiled diagnostic function writes channel-only volume fraction, surface
  coverages, pressure drop, raw cell alpha extrema, and phase-conservation checks
  in both serial and parallel runs.
- Serial restart from the latest saved state and clean wrapper completion were
  exercised. The 1 mm coarse run was extended after its first 0.000020 s test.

The test runs are separate from `cases/`, which contains clean initial conditions
and the configured 1 s runs. No background solver jobs are left running.

Launcher update verified on 24 October 2026 after moving the project into
`Problem1`: dictionary paths resolve from both `FlowField` and `Problem1`, and
project-relative case paths also work when launched from `FlowField`. A separate
four-process test in `runs/wrapper-check-20261024/w1mm` completed two 10⁻⁶ s steps
and reconstructed successfully. Both `Time` and `deltaT` appeared live and in the
saved solver log. The launcher now preserves command arguments while sourcing
OpenFOAM's environment, then streams solver output through `tee` with `pipefail`.

## Coarse startup results

Each coarse mesh contains 53,408 cells. These meshes retain two cells across each
oxygen passage but use coarser channel and manifold cells.

| Channel width | Last diagnostic time (s) | Maximum cell alpha | Largest oxygen residual / imposed O2 feed |
|---|---:|---:|---:|
| 2 mm | 0.001000000 | 1.00000001699 | 7.3224 × 10⁻⁶ |
| 1 mm | 0.001001818 | 1.00000001809 | 1.2739 × 10⁻⁵ |
| 0.5 mm | 0.001000000 | 1.00000001802 | 7.6902 × 10⁻⁶ |

The largest alpha overshoot in these **cell** values is 1.81 × 10⁻⁸. The diagnostic
acceptance tolerance is 10⁻⁷. Contact-angle boundary extrapolations are not included
in these cell extrema. Surface coverage is clipped to [0,1], while raw internal
alpha is retained for volume and conservation calculations.

The largest relative oxygen residual is approximately **0.001274%** of the imposed
oxygen feed (electrolysis feed plus gas entering with the main inlet). This is an
instantaneous discrete balance check, not a demonstrated convergence error bound.
The small endpoint overshoot of the restarted 1 mm case is recorded rather than
rounded away. Its summary integrates the actual available time samples.

![Actual startup diagnostics](../results/startup_diagnostics.png)

Machine-readable details: `results/startup_summary.json`. Every value is computed
from the corresponding solver CSV; none is digitized from or fitted to the paper.

## Medium-mesh startup results

| Channel width | Cells | Execution | Final physical time |
|---|---:|---|---:|
| 2 mm | 198,868 | Serial | 0.000020 s |
| 1 mm | 202,888 | Four MPI processes, reconstructed | 0.000020 s |
| 0.5 mm | 210,928 | Serial | 0.000020 s |

All three pass the same 10⁻⁷ cell-alpha bound tolerance. Maximum relative oxygen
balance residuals are below 4.7 × 10⁻⁶ over these short tests. Detailed results are
in `results/medium_summary.json`. This verifies that the intended production meshes
and parallel workflow execute; it is not a serial/parallel equivalence study.

## Evidence retained

`results/verification_logs/coarse/` and `results/verification_logs/medium/` contain
each case's mesh log, solver log, metadata, and diagnostic CSV. The parallel case
also includes decomposition and reconstruction logs. The full field snapshots are
available in this workspace under `runs/check/` and `runs/medium/`; these larger
temporary runs are excluded from the portable case archive.

The source-only reproduction, from `/Users/zhuang/FlowField/Problem1`, is:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/generate_cases.py --output runs/new-check --mesh coarse \
  --end-time 0.001 --write-interval 0.0002
./scripts/run_case.sh runs/new-check/w2mm run
./scripts/run_case.sh runs/new-check/w1mm run
./scripts/run_case.sh runs/new-check/w0.5mm run
python3 scripts/analyze.py runs/new-check/w2mm runs/new-check/w1mm \
  runs/new-check/w0.5mm --allow-short --output results/new-check.json
```

## What these checks do not establish

The longest original startup test is about 1 ms. The paper assesses bubble evolution up to 1 s and
selects a quasi-steady comparison after 0.2 s. The completed tests do **not** cover
that interval, resolved bubble detachment cycles, long-time stability, mesh/time
independence, or the paper's coverage/current-density rankings. Startup pressure
drops are dominated by initialization and should not be used as operating values.

At the time of the startup checks, the 1 s production runs and grid/time-step
sensitivity studies remained to be run. The later `w1mm` copy is assessed above;
grid/time-step sensitivity and clean full-run verification remain outstanding.
The estimates for pore geometry and oxygen generation also prevent a claim of
exact reproduction even after those runs finish. The case metadata and model
specification record these limitations explicitly.
