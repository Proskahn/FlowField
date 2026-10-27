# Verification record

Verification performed on 2026-10-25 with OpenCFD OpenFOAM v2312 in
`opencfd/openfoam-default:2312`, Linux ARM64, double precision, 32-bit labels.
The custom application was compiled with `wmake`. Simulations used one CPU
per case; the existing Problem1 simulation was not modified or interrupted.

## What is checked

- Python tests check reaction area, CL/PTL/membrane volumes, channel volumes,
  inlet areas, separation of the two fluid domains, mesh-refinement geometry,
  and rejection of invalid physical inputs.
- `checkMesh` checks the actual generated OpenFOAM meshes, including face
  orientation, cell volumes, topology and orthogonality.
- Each simulation must settle velocity, pressure, temperature and gas
  concentration for five consecutive iterations, with normalized field
  changes below 1e-5, charge residual below 1e-6, mass and gas balance errors
  below 1e-3, and no active gas-concentration bounds.
- `scripts/verify_results.py` reads the saved OpenFOAM fields independently
  of the solver's CSV output. It integrates ionic current through internal
  membrane planes, reconstructs electrical face dissipation, checks terminal
  power against reaction/conduction losses, checks the integrated heat
  sources, confirms H2:O2 molar stoichiometry, and evaluates outlet gas fluxes.

The independent identities are:

```
I_terminal = integral_membrane(-sigma_m grad(phiE) dot n dA)
I_terminal V_cell = P_joule + integral_CL[i (phiS-phiE) dV]
I_terminal V_cell = integral(Q dV) + integral_CL(i Eeq dV)
                    - integral_ACL(i T dErev/dT dV)
nDot_H2 = 2 nDot_O2 = I/(2 F)
integral_cell(Smass dV) = 0
```

These test current conservation and the dimensions/signs of source terms.
They do not establish that the missing electrochemical/material inputs match
the authors' COMSOL settings.

## Coarse-case results

Both cases passed the coupled convergence criteria and the independent field
checks. Full logs and field outputs are retained in `runs/verified/`.

| Quantity | Straight channel | Original parallel field |
|---|---:|---:|
| Cells | 9,600 | 11,616 |
| Converged outer iteration | 75 | 88 |
| Terminal current | 1 A | 1.21 A |
| Cell voltage | 2.277808 V | 2.248044 V |
| Anode inlet-to-outlet pressure drop | 160.631 Pa | 99.482 Pa |
| Cathode inlet-to-outlet pressure drop | 180.394 Pa | 124.216 Pa |
| Maximum temperature | 334.887535 K | 336.127899 K |
| O2 production at inlet T,p | 4.24998 mL/min | 5.14248 mL/min |
| H2 production at inlet T,p | 8.49997 mL/min | 10.28496 mL/min |
| Maximum relative gas-outlet error | 7.91e-8 | 1.94e-8 |
| Relative membrane-current error | 3.02e-10 | 3.54e-9 |
| Relative electrical-power identity error | 2.17e-8 | 3.23e-10 |
| Relative heat-source identity error | 2.18e-8 | 1.14e-10 |
| Bounded concentration cells at convergence | 0 | 0 |

## Supplied medium cases

Both supplied cases passed `checkMesh`, coupled convergence, and the
independent saved-field checks. Results are retained in `cases/` and can be
opened directly in ParaView.

| Quantity | Straight channel | Original parallel field |
|---|---:|---:|
| Cells | 73,600 | 89,056 |
| Converged outer iteration | 108 | 96 |
| Terminal current | 1 A | 1.21 A |
| Cell voltage | 2.277679 V | 2.251873 V |
| Anode inlet-to-outlet pressure drop | 206.981 Pa | 106.831 Pa |
| Cathode inlet-to-outlet pressure drop | 232.064 Pa | 130.880 Pa |
| Maximum temperature | 334.941706 K | 335.580986 K |
| Maximum gas volume fraction | 0.437561 | 0.787022 |
| Maximum relative gas-outlet error | 3.60e-7 | 9.72e-8 |
| Relative membrane-current error | 1.03e-9 | 5.84e-9 |
| Relative electrical-power identity error | 8.14e-11 | 4.45e-10 |
| Relative heat-source identity error | 6.12e-10 | 9.74e-10 |
| Bounded concentration cells at convergence | 0 | 0 |

Medium simulations took approximately six minutes per case on this machine
while Problem1 was also running. This is a measured example, not a runtime
guarantee.

The summed pressure drop changes by **22.3% for the straight cell and 5.9%
for the parallel cell** from coarse to medium, relative to the medium value.
The corresponding voltage changes are only about 0.0056% and 0.17%.
Consequently these tests **do not establish mesh independence**, particularly
for pressure drop. The fine mesh options are provided for a subsequent mesh
study; they have not been run here.

## Reproduce the checks

From `Problem2`:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/generate_cases.py --mesh coarse --output runs/new_verification
./scripts/run_case.sh runs/new_verification/singleChannel
./scripts/run_case.sh runs/new_verification/parallelFlow
python3 scripts/verify_results.py runs/new_verification/singleChannel
python3 scripts/verify_results.py runs/new_verification/parallelFlow
```

`run_case.sh` already performs the field checks; the last two commands show
how to repeat them without rerunning the solver. Build and run logs retain
the native OpenFOAM version, iteration output, and final convergence status.

## What remains unvalidated

This package has **not** reproduced the paper's reported 0.9% agreement with
experiment, the complete polarization curves, or its grid-independence
threshold. No experimental points have been digitized or fitted. Coarse
cases are numerical smoke/conservation tests, not resolved predictions.
Even agreement of voltage between two meshes would not prove that pressure
or gas distribution is mesh independent.

Pressure drops here are the inlet patch area-average gauge pressure relative
to the zero-pressure outlet, including manifolds and entrance effects. They
must not be compared to a pressure difference sampled only across a branch
without matching the measurement locations. The inferred manifold geometry
and material/kinetic closures must also be reconciled before a paper-level
quantitative comparison.

The thermal check above verifies source power, not a separately benchmarked
conjugate-heat-transfer solution. There is no MPI test because this version
deliberately supports serial solves only. Any changed operating condition,
geometry or closure requires its own convergence and conservation checks.
