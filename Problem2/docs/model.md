# Model specification and provenance

Reference: Y. Zhuang, P. Cui, R. Long, W. Liu and Z. Liu, **Multi-objective
optimization of channel structure for a proton exchange membrane water
electrolysis cell**, International Journal of Hydrogen Energy 49 (2024),
337-352. DOI: https://doi.org/10.1016/j.ijhydene.2023.08.026.

This is a reproducible OpenFOAM reconstruction of the numerical model, not
the authors' COMSOL model or an independently validated reproduction of their
curves. Sections 2, 3.1, 3.2 and the original geometry in 3.3.1 are in scope.
No GA, optimizer, surrogate, MATLAB coupling, or optimized geometry is included.

## Domain and geometry

The mesh contains nine cell zones, bottom to top: `CBP`, `CCH`, `CPTL`, `CCL`,
`PEM`, `ACL`, `APTL`, `ACH`, `ABP`. Channels are voids inside the plate region;
ribs, backing plates and the entire membrane-electrode assembly are resolved.
Interfaces are conformal orthogonal faces. No mesh is extruded as a 2-D case.

| Quantity | Value | Provenance |
|---|---:|---|
| Straight model reaction footprint | 50 x 2 mm | Table 1 |
| Parallel model footprint | 11 x 11 mm | Table 4 |
| Parallel branches | 5, each 8 mm long | Fig. 11, Table 4 |
| Original channel width and height | 1 x 1 mm | Table 5 |
| PTL / CL / membrane thickness | 0.3 / 0.02 / 0.178 mm | Tables 1, 4 |
| Plate total thickness | 1.5 mm | Interpretation of Table 1 |

Geometry uncertainties are explicit:

- Table 1 also lists a **BP width of 1.5 mm**, inconsistent with the 2 mm wide
  reaction footprint and an unspecified rib layout. Here each straight plate
  spans the 2 mm footprint, with a centered 1 mm channel and two 0.5 mm ribs.
- The plate height is interpreted as its total depth, leaving 0.5 mm backing
  at the baseline channel depth. A channel cannot equal or exceed this depth.
- Fig. 11 is not dimensioned CAD. The five parallel branch centers are at
  x = 1.5, 3.5, 5.5, 7.5, 9.5 mm. Their ends are y = 1.5 and 9.5 mm.
  Each manifold is 1 mm wide and 10 mm long: x = 0..10 at y = 0.5..1.5,
  and x = 1..11 at y = 9.5..10.5. These port positions/manifold widths are
  reconstruction assumptions. They affect pressure drop.
- Both electrode feeds use the same flow direction. The paper does not fully
  specify the plumbing orientation. There is no gas crossover through the PEM.

## Equations actually solved

The paper assumes steady laminar **homogeneous mixture flow**, not VOF.
`U` is intrinsic pore velocity; the volumetric face flux is
`phi = epsilon * U dot Sf`. Water remains liquid and gases are ideal at the
specified absolute operating pressure. There is no resolved bubble interface,
surface tension, bubble nucleation, slip velocity, or turbulent closure.

### Charge and reaction

Electronic potential exists in the plates, PTLs and CLs. Ionic potential
exists in both CLs and the membrane. Both potentials coexist in the CL cells.
They are solved together using a Newton Jacobian with a Butler-Volmer link at
each CL cell. OpenFOAM `lduMatrix`, PCG and DIC solve that system. Electronic
and ionic conductances use distance-weighted harmonic face values. Inactive
interfaces have exactly zero conductance.

With signed local Faradaic current `i` positive at the anode and negative at
the cathode, the implemented sign convention is

```
div(-sigma_s grad(phiS)) = -i
div(-sigma_m grad(phiE)) = +i
i = av * (1-theta) * i0 * (Cgas/Cref)
    * [exp(alpha_a F eta/(R T)) - exp(-alpha_c F eta/(R T))]
eta = phiS - phiE - Eelectrode
sigma_m = 6.94 exp[1268 (1/303 - 1/T)] S/m
```

This sign convention conserves current between the two conducting phases.
It is the opposite definition of signed `iv` to the source labels in Table 3;
the anode and cathode production rates below are always positive in normal
electrolysis operation. No additional electron number multiplies the
exponential: Eqs. (1)-(2) print `F`, not `n F`.

The anode terminal receives a uniform current density; the cathode terminal
is grounded. Voltage is the area average of the anode face potential,
including its half-cell conduction drop. It is an output, not a prescribed
guess. The initial 1.8 V field is only a Newton starting value.

The charge graph applies these terminal conditions explicitly using patch
names and `currentDensity` in `pemProperties`. The generic patch entries in
`phiS`/`phiE` are storage fields, not the equations used by a standard OpenFOAM
electric-potential solver. Use the reported voltage for the terminal value;
cell-centered potential plots exclude the terminal half-cell drop.

For the concentration correction, Eqs. (9)-(11) are represented as local
equilibrium shifts:

```
Ea = 1.229 - 0.0009 (T-298.15) + R T/(4 F) ln(CO2/CrefO2)
Ec = -R T/(2 F) ln(CH2/CrefH2)
```

Their difference contains both printed diffusion terms. They are not added
again to the computed terminal voltage. `includeNernstConcentration=false`
disables these shifts for a sensitivity study. The paper does not supply the
reference concentrations or a complete interface activity prescription; using
local CL concentrations here is a closure assumption. The concentration
multiplier in the printed Butler-Volmer expressions is retained. Because it
also changes activation losses, one must not assume this reconstruction will
reproduce the paper's geometry trends without checking them.

The default specific active area is **1e5 m^-1 at each electrode**, an
uncalibrated estimate. Multiplying it by the 20 micrometre CL thickness gives
a roughness factor of 2. The paper supplies `i0` but omits `av`; therefore it
does not uniquely determine the polarization curve. Reference concentrations
are approximately the ideal-gas molarity at 60 C and 1 atm (36.579 mol/m3),
also an explicit estimate. A ratio floor of 1e-6 keeps the log and exchange
current finite near a pure-water inlet. The floor is not a gas source.

### Mixture flow and porous resistance

```
div(epsilon rho U) = Sm
div(epsilon rho U U) = -epsilon grad(p)
                      + div(epsilon mu grad(U))
                      - mu epsilon^2/K U
rho = rho_liquid (1-fgas) + rho_gas fgas
mu  = mu_liquid  (1-fgas) + mu_gas  fgas
rho_gas = p_absolute Mgas/(R T)
```

SIMPLE pressure correction conserves the mass face flux. A pseudo-transient
diagonal stabilizes iteration; its source uses the previous velocity, so it
vanishes at convergence. `flowPseudoDt` is a numerical parameter, not a time
step for physical transient results. Pressure in the output is **gauge Pa**;
the gas equation of state uses `operatingPressure` (101325 Pa), as appropriate
to the paper's incompressible-gas approximation.

Internal fluid/solid faces apply no slip at the face, with the correct
fluid-center-to-wall distance. Momentum, pressure and species are not solved
inside solid cells. PTL and CL resistance follows Table 3, using porosities
0.5 and 0.25 and permeabilities 1e-12 and 1e-13 m2. Gravity is omitted, since
the supplied momentum equation does not contain it.

### Species and gas production

`Cgas` is oxygen molarity in the anode fluid and hydrogen molarity in the
cathode fluid, in mol/m3 of mixture. They occupy disconnected fluid domains.

```
div(epsilon U Cgas) = div(Deff grad(Cgas)) + Sgas
Deff = epsilon^1.5 D0 (T/Tref) (101325/p_absolute)
fgas = Cgas R T/p_absolute
Cwater = rho_liquid (1-fgas)/Mwater
gasMoleFraction = Cgas/(Cgas+Cwater)

ACL: SO2 =  i/(4 F)                 mol/m3/s
     SH2O = -i/(2 F)                mol/m3/s
     Sm = i [MO2/(4 F)-MH2O/(2 F)]  kg/m3/s
CCL: SH2 = -i/(2 F)                 mol/m3/s (i is negative here)
     Sm = -i MH2/(2 F)              kg/m3/s
```

The mass change at each side accounts for transfer of hydrogen mass through
the membrane. Water is recovered from total mass minus gas mass rather than
solving a redundant independent water equation. The gas diffusive mass flux
has an equal/opposite water contribution in that total-mass formulation.
This is an explicit closure of the paper's unprovided phase-fraction relation.
Faradaic efficiency is 100%; dissolved gas, water drag/back diffusion,
evaporation, and membrane dehydration are absent, consistent with the
saturated-membrane/no-crossover idealization.

Pure-water inlet convection sets Cgas=0, with zero diffusive gas flux there.
Outlet diffusion is zero. Gas cannot diffuse into plates or the membrane.
Concentrations are bounded during iteration; **a case with active bounds
cannot pass convergence**, and the exported CSV counts those cells. This
prevents silently declaring a clipped, nonconservative solution valid.

At 1 A total current the exact integrated targets are approximately
2.5911e-6 mol/s O2 and 5.1821e-6 mol/s H2 (4.25 and 8.50 mL/min at 60 C,
1 atm). They follow from the solved catalyst current, not a prescribed
gas-inlet velocity. The parallel baseline has 1.21 A and proportionally
larger gas production.

### Thermal equation

```
rho Cp_fluid epsilon U dot grad(T) = div(k_eff grad(T)) + Q
k_eff = (1-epsilon) k_solid + epsilon k_fluid
Q = Qjoule + i*eta + [i*T*dErev/dT in ACL]
dErev/dT = -0.0009 V/K
```

Fluid Cp is mass weighted; gas/liquid conductivity is volume weighted.
Stationary solids only conduct heat. Reaction and Joule heating are coupled
to the solved potentials; internal face Joule power `G*(deltaPhi)^2` is split
between neighboring cells and terminal half-cell dissipation is included.
Reversible electrolysis heat is cooling, so the entropy contribution has the
sign required by `Q = I*(V-Ethermoneutral)`. `reversibleHeatSign=-1` allows a
sign sensitivity check against the paper's printed expression.

For the steady problem, solid density and solid heat capacity do not enter
the stationary solid energy balance. This avoids needing to invent missing
COMSOL database values for them. The paper's effective-density/effective-Cp
form of advective heat transfer is interpreted using moving fluid only;
advecting solid heat capacity with pore velocity would be unphysical.

## Corrections and unresolved ambiguities in the printed model

| Printed item | Implementation and reason |
|---|---|
| Eq. (12) transports mol/m3, but Table 3 multiplies sources by molecular mass | Molar sources for Cgas, mass sources for continuity; no unit mixing |
| Table 3 uses `iv^2/sigma` while describing `iv` as A/m3 | Joule heat from spatial current density, equivalently face conductance times voltage drop squared; W/m3 |
| Table 3 entropy term is `-T dVeq/dT` without a current factor | Multiply by local current to obtain W/m3; use cooling sign for electrolysis |
| Eq. (3) appears to apply the full cell equilibrium voltage to both electrodes | Anode reversible potential plus a zero cathode reference, with opposite Nernst shifts |
| Eq. (6) prints `1/(p-psat)` with no pressure-unit normalization | Default uses dimensionless `p/(p-psat)`; `coveragePressureMode paperSI` reproduces the printed factor with Pa |
| Eq. (11) lists ohmic and diffusion losses separately | Spatial charge transport supplies ohmic loss; equilibrium shifts supply diffusion terms, each counted once |

The dimensionless coverage-pressure correction is supported by the original
alkaline-electrolyzer model of Abdin et al., Eq. (31):
https://research-repository.griffith.edu.au/bitstream/10072/355192/1/AbdinPUB4982.pdf.
It remains an empirical correlation transferred to a PEM electrode. Coverage
is bounded to 0..0.95, reference temperature is assumed 298.15 K, and saturation
pressure uses the water Antoine correlation. The exact convention used in
the authors' COMSOL file is unknown.

## Parameter provenance

| Parameter | Default | Status |
|---|---:|---|
| Inlet temperature, pressure, flow per side | 333.15 K, 101325 Pa, 15 mL/min | Table 2, Sec. 2.3 |
| Current density | 10000 A/m2 | Sec. 3 baseline |
| Electron conductivity | 1000 S/m | Table 2; applied to all electron-conducting zones |
| i0 anode / cathode | 0.1 / 10000 A/m2 | Table 2 |
| Transfer coefficients | 0.5 / 0.5 | Table 2 |
| av anode / cathode | 1e5 / 1e5 m^-1 | Estimated, not fitted |
| Reference gas concentrations | 36.579 mol/m3 each | Estimated ideal-gas reference |
| O2 / H2 D0 | 2e-5 / 6e-5 m2/s | Estimated effective gas-mixture transport, not liquid dissolved-gas diffusivity |
| Water density / viscosity / Cp / k | 983.2 / 0.000466 / 4184 / 0.65 SI | Approximate 60 C properties |
| O2 viscosity / Cp / k | 2.2e-5 / 920 / 0.028 SI | Approximate properties |
| H2 viscosity / Cp / k | 9.5e-6 / 14300 / 0.20 SI | Approximate properties |
| Titanium / carbon thermal conductivity | 21.9 / 10 W/(m K) | Estimates, carbon isotropic |
| ACL / CCL / membrane thermal conductivity | 2 / 2 / 0.2 W/(m K) | Estimates |

The paper says remaining thermodynamic properties came from COMSOL's material
database but does not provide the actual entries. Effective diffusivities,
specific active areas and concentration references are particularly important
uncertainties. Replace them with measured or author-supplied values before
claiming quantitative agreement. No parameters have been fitted to Fig. 4.

## Numerical checks and limits

Charge convergence uses the sum of absolute node current residuals divided by
the terminal current. Flow and gas outlet balances are tracked independently.
An outer iteration must also settle U, p, T and Cgas, with no active
concentration bounds, for five consecutive iterations. The solver returns
exit status 2 when the iteration budget ends without meeting these criteria.
The zero-gradient gas outlet assumes outflow; backflow is outside this setup.

First-order upwind advection is used for robust startup. Orthogonal diffusion
is second order. Mesh refinement, discretization sensitivity and calibration
are required before comparison with the published FEM results. The supplied
mesh levels are not asserted to reproduce the paper's element counts or its
reported 0.9% experimental agreement. See `verification.md` for tests actually
performed on this implementation.
