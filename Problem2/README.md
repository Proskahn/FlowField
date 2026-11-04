# Problem2: coupled PEM electrolyzer model in OpenFOAM

OpenFOAM reconstruction of the numerical model in Zhuang et al.,
*Multi-objective optimization of channel structure for a proton exchange
membrane water electrolysis cell*, IJHE 49 (2024) 337-352,
[DOI 10.1016/j.ijhydene.2023.08.026](https://doi.org/10.1016/j.ijhydene.2023.08.026).
The optimization part is excluded.

The custom `pemElectrolyzerFoam` solver couples laminar flow, porous resistance,
oxygen/hydrogen transport, electronic and ionic potentials, Butler-Volmer
reaction rates, and heat conduction/convection. Gas is generated inside the
catalyst layers from the **local solved current**. This paper uses a steady
homogeneous mixture model, so this case does not run `interFoam` or VOF.

**Reproduction status:** the paper omits several inputs and contains ambiguous
or dimensionally inconsistent expressions. The implementation uses documented
estimates/corrections; it is not a claim of matching the published curves.
Read [the model specification](docs/model.md) and [verification](docs/verification.md)
before interpreting quantitative results.

## Run on this Mac

Requirements: Docker Desktop running, Python 3.9+, and the installed
`opencfd/openfoam-default:2312` image. All OpenFOAM commands execute inside
that image. No COMSOL, MATLAB, or Python scientific packages are needed.

From a terminal:

```bash
cd /Users/zhuang/FlowField/Problem2
./scripts/build.sh
./scripts/run_case.sh cases/singleChannel
```

Run the original five-channel parallel model:

```bash
./scripts/run_case.sh cases/parallelFlow
```

The supplied cases use the medium mesh and include converged results from
verification. The commands above resume those results. To start a fresh
medium run, generate into a new directory:

```bash
python3 scripts/generate_cases.py --mesh medium --output runs/fresh
./scripts/run_case.sh runs/fresh/singleChannel
```

For a smaller practice run:

```bash
python3 scripts/generate_cases.py --geometry single --mesh coarse --output runs/practice
./scripts/run_case.sh runs/practice/singleChannel
```

Paths also work from the parent directory:

```bash
cd /Users/zhuang/FlowField
./Problem2/scripts/run_case.sh Problem2/cases/singleChannel
```

The scripts locate their own project folder, so moving the entire `Problem2`
folder does not require editing hard-coded paths. Runs do not modify Problem1.

## Live progress, stopping and restarting

Each outer iteration prints `Iteration = ...`, linear-solver residuals and a
`PEM:` line with voltage, both currents, both pressure drops, maximum
temperature, field change, charge residual and gas balance error. Output is
also saved to a timestamped `log.solver.*` file in the case directory.

This is a **steady solver**: time-folder names such as `100` denote iteration
numbers, not 100 seconds of electrolysis. `deltaT=1` in `controlDict` only
advances that iteration counter. There is no physical VOF time step here.

Press Ctrl-C to interrupt. For a serial case, the same run command resumes the
most recent saved iteration. For a parallel case, use `resume N` with the original
process count. Only the latest two periodic processor result directories are retained;
the initial `0` directory and all CSV histories remain. The default maximum
is 1000 iterations. To extend a run, edit `endTime` in that case's
`system/controlDict`, for example to 2000, then run the same command again.
The solver returns status 2 if it reaches the limit without convergence.

**CPU count:** the solver supports serial and MPI runs. The coupled electrical
potentials, flow, gas transport, and heat equations are all partitioned between
processes. The launcher handles decomposition and reconstructs the latest saved
iteration for analysis and ParaView. There is no GPU backend.

## Run on multiple CPU cores on the university server

Copy the updated `Problem2` source and scripts to the server, then **rebuild
there**. An older solver binary rejects MPI; the Mac's binary is not portable
to the server's CPU architecture.

```bash
module load mpi/openmpi-x86_64
source "$HOME/OpenFOAM/OpenFOAM-v2312/etc/bashrc"
export FLOWFIELD_NATIVE=1

cd ~/FlowField/Problem2
./scripts/build.sh

# Generate a separate case once, preserving previous results.
python3 scripts/generate_cases.py --geometry both --mesh medium --output runs/mpi16

# Use 16 MPI processes for one simulation.
./scripts/run_case.sh runs/mpi16/singleChannel parallel 16
```

Run the five-channel geometry with the same number of processes:

```bash
./scripts/run_case.sh runs/mpi16/parallelFlow parallel 16
```

`parallelFlow` is the geometry name; `parallel 16` selects the process count.
Use a CPU count allocated to your job. Having 256 logical CPUs visible on a
shared server does not mean they are all assigned to you. Performance depends
on mesh size and MPI overhead; 16 processes need not outperform 4 or 8. The
validation record checks numerical agreement, not speedup on your server.

Reload the MPI module and OpenFOAM environment in each new login or batch job.
The updated launcher automatically selects a loaded v2312 environment;
`FLOWFIELD_NATIVE=1` explicitly requires native execution, while `0` forces Docker.

| Task | Command from `Problem2` |
|---|---|
| Check the original mesh | `./scripts/run_case.sh CASE check` |
| Serial run or restart | `./scripts/run_case.sh CASE run` |
| Create a decomposition and run | `./scripts/run_case.sh CASE parallel N` |
| Continue an existing parallel run | `./scripts/run_case.sh CASE resume N` |
| Continue using the existing partition count | `./scripts/run_case.sh CASE resume` |

Replace `CASE` with the case directory and `N` with your process count. Omitted
`N` defaults to 4 for a new parallel run. To restart the example above:

```bash
./scripts/run_case.sh runs/mpi16/singleChannel resume 16
```

Do not regenerate the case when resuming. A mismatched process count, missing
checkpoint fields, or differing latest iteration numbers between ranks is rejected
before starting the solver. To change process count while retaining progress,
stop the solver, reconstruct its latest checkpoint, and decompose a separate
copy; `resume N` does not repartition it. `parallel N` can also decompose the
latest saved iteration of a serial case that has no processor directories.

Decomposition defaults to Scotch, as recorded in `system/decomposeParDict`.
The launcher uses uncollated `processor0`, `processor1`, ... directories. Global
histories and status remain in the case's `postProcessing/pem/` directory, and
only one process writes them. An iteration-limit stop returns status 2 and also
reconstructs the latest checkpoint; raise `endTime` before resuming that run.

To compare a reconstructed MPI run with a converged serial run using the same
mesh and inputs, run `python3 scripts/compare_runs.py SERIAL_CASE MPI_CASE`.
This repeats the conservation checks and compares full cell fields as well as
voltage, pressure drops, production rates, and heat.

## Cases and mesh sizes

| Geometry | Coarse | Medium (supplied) | Fine |
|---|---:|---:|---:|
| 50 x 2 mm straight-channel cell | 9,600 | 73,600 | 244,800 |
| 11 x 11 mm, original five-channel cell | 11,616 | 89,056 | 296,208 |

Counts include all fluid and solid zones. The paper used COMSOL FEM meshes of
76,800 and 251,563 elements for the corresponding models; those are not
identical to these OpenFOAM finite-volume meshes. The CLs and membrane are
explicitly resolved at every level. `checkMesh` runs before each simulation.
Pressure drop is still sensitive to mesh refinement; these meshes are not
claimed to be mesh independent. The completed checks are in `docs/verification.md`.

Regenerate into a new directory to preserve previous results:

```bash
python3 scripts/generate_cases.py --geometry both --mesh fine --output runs/fine
```

## Change operating conditions or geometry

Edit a copy of `config/reference.json` and generate new cases:

```bash
cp config/reference.json config/my_case.json
# Edit config/my_case.json, then:
python3 scripts/generate_cases.py --config config/my_case.json --mesh medium --output runs/my_case
./scripts/run_case.sh runs/my_case/singleChannel
```

The baseline uses 60 C, 1 atm, 15 mL/min water **per electrode**, and
10000 A/m2 (1 A/cm2). Total current is 1 A for the straight model and 1.21 A
for the parallel model. Channel width/depth and layer thicknesses are mesh
parameters. The parallel template reconstructs the original layout, not
the optimized dimensions from Table 5. Geometry assumptions are listed in
`docs/model.md`.

Operating/closure parameters are written to `constant/pemProperties`; boundary
conditions to `0/U`, `0/T`, etc. Generating from JSON keeps them consistent.
The generator refuses to overwrite a case that already has result folders.
If Cgas is bounded or the solver does not converge after changing inputs,
inspect the history and model assumptions before treating the result as valid.

## Results

On successful completion, `summary.json` contains voltage, pressure drops,
electrical/pump power, production rates and conservation errors. You can
also summarize a stopped run (exit status 2 indicates it has not passed):

```bash
python3 scripts/analyze.py cases/singleChannel
```

`postProcessing/pem/history_*.csv` records every iteration and can be opened
in a spreadsheet or plotting tool. `postProcessing/pem/status.json` records
whether the convergence criteria were met.

The run script also checks saved fields independently and writes
`field_verification.json`. These checks compare current through the membrane
with terminal current, electrical power with conduction/reaction losses,
heat-source units and signs, and gas leaving the outlet patches with Faraday
production. They can be rerun with:

```bash
python3 scripts/verify_results.py cases/singleChannel
```

Open `cases/singleChannel/singleChannel.foam` or
`cases/parallelFlow/parallelFlow.foam` in ParaView. Select the latest iteration
and the relevant cell zones. Useful fields:

| Field | Meaning |
|---|---|
| `U`, `p` | Intrinsic fluid velocity (m/s), gauge pressure (Pa) |
| `T` | Temperature (K), including solids |
| `phiS`, `phiE` | Electronic / ionic potential (V); use their conducting zones |
| `iVol` | Signed volumetric current (A/m3), positive ACL, negative CCL |
| `Cgas`, `Cwater` | Gas and liquid-water molarity (mol/m3 of mixture) |
| `gasVolumeFraction`, `gasMoleFraction` | Distinct volume and mole fractions |
| `gasMolarSource`, `massSource` | Reaction molar and total mass source densities |
| `coverage` | Empirical catalyst bubble coverage, not a VOF interface |
| `heatSource` | Reaction plus Joule plus reversible heat (W/m3) |
| `material`, `rho` | Material IDs (see model document), mixture density |

Fields outside their physical domains are placeholders: do not average `U`
or `Cgas` across solid cells, or `phiE` outside the membrane/CLs.

## Build and verify on Linux

Either use the same Docker commands or source a native **OpenCFD v2312**
installation and run:

```bash
export FLOWFIELD_NATIVE=1
./scripts/build.sh
./scripts/run_case.sh cases/singleChannel
```

Other OpenFOAM forks/versions are not tested. Rebuild after editing C++ source.
The binary in `bin/` is architecture-specific and is regenerated by `wmake`.

Run geometry/invariant tests using the Python standard library:

```bash
python3 -m unittest discover -s tests -v
```

The implementation and estimates are fully exposed in `applications/`,
`scripts/` and `config/`. Model calibration, mesh-independence studies and the
optimization workflow can build on this baseline without hidden fitted data.
