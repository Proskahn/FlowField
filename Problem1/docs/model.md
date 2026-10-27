# Model specification and provenance

## Scope and equations

Reference: Li, Chen, Song and Tian (2025), DOI
10.1016/j.ijheatmasstransfer.2025.127446. The supplied PDF has 16 pages. The
flow-field VOF submodel is described on pages 3–7 and its comparison on pages 9–13.

The computational region is the anode channels, inlet/outlet manifolds, and short
oxygen feed passages. There is no porous-medium region in this VOF domain: the
paper's PTL is represented by gas inlet openings beneath the channel. No cathode,
membrane, catalyst layer, charge transport, or energy equation is solved here.

With `alpha.water = α`, oxygen fraction is `1−α`. The solver advances the
incompressible mixture continuity and momentum equations and a conservative phase
transport equation with numerical interface compression. Mixture density and
dynamic viscosity are linearly weighted by phase fraction. Surface tension uses
the continuum-surface-force formulation with wall contact-angle correction.
The numerical interface-compression term is an OpenFOAM discretization choice,
not an additional physical source. No evaporation, gas dissolution, nucleation
model, turbulence, or compressible gas expansion is included.

`interFoam` solves `p_rgh = p − ρ(g·x)`, in Pa. The outlet uses `prghPressure`
with static gauge pressure zero; hence pressure remains correct across its height
and changing phase composition. The physical reference pressure is assumed to be
101325 Pa when calculating constant oxygen density and oxygen inlet volume.

## Reported values and reconstruction decisions

| Quantity | Implemented value | Provenance / limitation |
|---|---:|---|
| Channel width = rib width | 2, 1, 0.5 mm | Section 2.1 and Fig. 3 |
| Channel count | 1, 2, 4 | Fig. 3; not four channels for every design |
| Channel length | 8 mm | Fig. 3 and Table 1 |
| Channel height | 1 mm | Table 1 |
| MEA dimensions | 8 × 4 mm | Table 1 |
| Each manifold | 4 mm long × 1 mm wide × 1 mm high | Fig. 3 and channel height |
| Outlet-manifold offset | `−w` relative to inlet manifold | Reconstructed from Fig. 3; see coordinates below |
| Oxygen opening size | 0.1 mm square width | Table 1 calls this a pore diameter; Fig. 3 depicts squares. Square cross-section is an explicit approximation, not a diameter-to-area identity |
| Pore pitch | 0.25 mm | Estimated; 256 regularly arranged openings, not an extracted CAD layout |
| Pore passage length | 0.1 mm | Estimated; not equated to the 0.2 mm physical PTL thickness |
| Water/oxygen inlet speed | 0.1 m/s | Section 2.4; treated as mixture velocity |
| Inlet water fraction | 0.9 | Section 2.4; uniform fraction BC, not resolved inlet bubbles |
| Initial condition | Channels/manifolds water-filled; pore passages oxygen-filled | Liquid-filled channel reported in Section 3.2; gas-filled passages are a startup assumption |
| Inlet temperature / property temperature | 343.15 K | Section 2.4; fixed properties in the isothermal VOF submodel |
| Contact angle | 80° through water | Table 2; applied uniformly to all solid walls, including passage walls. No hysteresis reported |
| Surface tension | 0.0644 N/m | Table 2 |
| Water density | 977.8 kg/m³ | Engineering estimate at 70°C, not tabulated in the paper |
| Water dynamic viscosity | 4.04 × 10⁻⁴ Pa·s | Engineering estimate at 70°C, not tabulated in the paper |
| Oxygen density | 1.13637 kg/m³, approximately | Ideal gas evaluated at 343.15 K and assumed 1 atm; metadata records exact value |
| Oxygen dynamic viscosity | 2.29 × 10⁻⁵ Pa·s | Engineering estimate near 343 K, not tabulated in the paper |
| Gravity | (0, 0, −9.81) m/s² | Orientation interpreted from Fig. 7: toward channel bottom |
| Walls | No slip | Section 2.4 |
| Oxygen generation | Uniform Faraday-law estimate | Replaces missing ACH–APTL oxygen-flux map; detailed below |
| Full comparison duration | 1 s; analyze 0.2–1 s | Fig. 11 and Section 3.3 |
| Time stepping | Adaptive, initially 10⁻⁶ s; capillary and advective limits | Paper reports 10⁻⁶–10⁻³ s. This implementation permits smaller steps if stability requires them |

Water viscosity is converted to kinematic viscosity (`nu = mu/rho`) when writing
the OpenFOAM transport dictionary. Oxygen density is held constant once evaluated.
These estimates should be replaced if the authors provide their actual properties.
The oxygen-viscosity magnitude is consistent with the 343 K parameter table in
[VTT Publication 772, Appendix A](https://sarjaweb.vtt.fi/pdf/publications/2011/P772.pdf).

## Geometry and coordinates

All geometry generation uses mm internally and writes OpenFOAM coordinates in m.
The channel height direction is `z`; the gas inlet velocity points along `+z`.

For channel width `w`, `N = 4/(2w)`. Channel `c = 0…N−1` occupies:

```text
x: 2cw … (2c+1)w
y: 0 … 8 mm
z: 0 … 1 mm
```

The inlet manifold spans `x = 0…4`, `y = 8…9`; the inlet is on its `x = 4` face
with velocity `(−0.1, 0, 0) m/s`. The outlet manifold spans `x = −w…4−w`,
`y = −1…0`; its outlet is on `x = −w`. Both have `z = 0…1 mm`. This preserves
the opposed, offset manifold arrangement in Fig. 3.

Pore centers in each channel are spaced at 0.25 mm, starting 0.125 mm from the
channel edges. There are 32 rows along the length and a total of 8 columns across
all channels: 8×32, 2×4×32, or 4×2×32. Each passage extends from `z = −0.1` to
`z = 0 mm`. The generator builds a conformal mesh connecting every passage to the
channel and records its mouth as an internal face zone. Channel walls have separate
bottom, top, and side patches; manifold and pore walls have separate patches.

The mesh uses axis-aligned hexahedra with local segmentation at every pore edge.
Coarse/medium/fine maximum spacing is 0.1/0.05/0.025 mm, with at least two cells
across every pore even on the coarse mesh. The exact pore geometry remains fixed
through refinement. Fine resolution has four cells across a pore; a meaningful
interface/curvature convergence study may require further refinement. Similarity
to the paper's cell count is not proof of equivalent resolution.

## Estimated oxygen boundary condition

At the anode, Faraday's law gives

```text
I       = j × A_MEA
nDot_O2 = eta_F × I / (4F)
Q_O2    = nDot_O2 × R T / P_abs
U_pore  = Q_O2 / A_open
```

Default `j = 2.02 A/cm² = 20200 A/m²`, `A_MEA = 32×10⁻⁶ m²`,
`eta_F = 1`, and `A_open = 256 × (0.1 mm)² = 2.56 mm²`. Thus:

- Total cell current: 0.6464 A.
- Oxygen volume flow: approximately 4.71609 × 10⁻⁸ m³/s = 47.1609 µL/s.
- Oxygen inlet velocity: approximately 0.0184222 m/s in each pore.
- Main inlet mixture volume flow: 100 µL/s, of which 10 µL/s is oxygen at
  the uniform inlet fraction of 0.9. This oxygen is additional to electrolysis gas.

Generation is based on the **full MEA area**, including the area underneath ribs.
All produced oxygen is assumed to emerge through the openings beneath the channels.
Using only the 16 mm² channel footprint would halve the intended current-based feed.

The 2.02 A/cm² value is the paper's **coupled 1 mm design result at 2.15 V**.
It is not the conventional-model oxygen feed map used to initialize the paper's VOF
calculation. It is used here as an explicit common-feed estimate for all three
designs. Therefore width-dependent results are a controlled hydrodynamic study at
equal gas production. They do not establish which design produces the highest
current at a fixed voltage. No claim is made to reproduce the reported 22% current
advantage. The current can be changed per case or in a sensitivity sweep.

## Boundary conditions

| Patch | Velocity `U` | Water fraction | `p_rgh` |
|---|---|---|---|
| `waterInlet` | Fixed (−0.1,0,0) m/s | Fixed 0.9 | `fixedFluxPressure` |
| `oxygenInlets` | Fixed (0,0,U_pore) | Fixed 0 | `fixedFluxPressure` |
| `outlet` | `pressureInletOutletVelocity` | `inletOutlet`; backflow water fraction 1 | `prghPressure`; static gauge p=0 |
| All solid walls | `noSlip` | `constantAlphaContactAngle`, theta0=80°, limit gradient | `fixedFluxPressure` |

The water fraction is the first phase, so the contact-angle convention is through
water in v2312's `grad(alpha.water)` interface-normal formulation. The 0.9 inlet
condition supplies a mixed VOF fraction, not a uniquely specified population of
resolved bubbles. The paper does not describe the inlet bubble size, distribution,
or method. Pure-water inlet runs (`--inlet-alpha 1`) and specified resolved-bubble
injection are appropriate sensitivity studies before comparing inlet-manifold gas
clusters quantitatively.

## Numerics and time-step control

Euler time discretization, van Leer phase advection with MULES and interface
compression, linear-upwind momentum advection, and PIMPLE pressure/velocity
coupling are used. There are two outer corrections, two pressure corrections,
and two alpha corrections. The mesh is orthogonal, so no nonorthogonal correction
loop is required. Linear-solver tolerances are recorded in `fvSolution`.

The maximum time step is the smaller of 10⁻³ s and

```text
0.25 × sqrt((rho_water + rho_oxygen) × h_min³ / (pi × sigma))
```

This is a conservative capillary-wave stability estimate; the safety factor is an
implementation choice. Advective limits `maxCo=0.25` and `maxAlphaCo=0.15` can
reduce it further. Do not enforce a minimum time step at 10⁻⁶ s if that violates
capillary or advection constraints. Record time-step sensitivity in addition to
grid sensitivity. Time averages are integrated in physical time, not averaged
over unequal time steps.

## Metrics and coupling interface

The `channels` cell zone excludes both manifolds and all inlet passages. Channel
gas fraction is `sum((1−alpha)*V)/sum(V)` over this zone. Raw alpha is used for
volume/conservation diagnostics; phase fractions are clipped only when computing
surface coverage, and raw extrema are reported separately.

Bottom-wall coverage is the area average of gas fraction on the solid bottom wall.
A second measure includes the pore-mouth area using linear face interpolation
between channel and passage cells. A third uses `gas fraction >= 0.5` on that
complete bottom plane. Top and side coverage use the area-weighted wall phase
fraction. These definitions are explicit because the paper does not specify its
coverage quadrature or phase threshold. The initial gas-filled passages can cause
nonzero interpolated mouth coverage even with an initially water-filled channel.

Gas inventory includes the full fluid domain. The oxygen flux is computed as
`phi − alphaPhi0.water`, using the solver's compressed water flux, rather than
approximating phase flux as `phi*(1−alpha)`. The per-step residual is
`(V_O2(t)−V_O2(t−dt))/dt + sum_boundary(phi−alphaPhi0.water)`. Processor boundaries
are excluded from external flux sums. Internal mouth faces duplicated by domain
decomposition are counted with half weight. These formulas assume the supplied
Euler time scheme, static mesh, and single alpha subcycle; revise the diagnostics
if changing those choices.

Equations (44) and (45) are available as postprocessing:

```text
eta_bubble = R T / (n alpha_a F) × [−omega1 ln(1−beta)]
a_eff/a   = (1−beta)^omega2
omega1    = 1.5
omega2    = 2
alpha_a   = 0.5
n         = 4 (OER stoichiometry; explicit choice)
```

The default analysis uses complete-bottom mean coverage. A nonlinear expression
evaluated at mean beta differs from the mean instantaneous expression; the JSON
labels this distinction. Coupling these outputs back into a cell-voltage calculation
would additionally require the charge, two-phase porous transport, and energy
equations, the authors' electrochemical parameters and flux map, and an iteration
convergence criterion. Those are outside this flow-field VOF reconstruction.

## Quantitative validation still required

1. Obtain the authors' oxygen evolution map, actual pore layout, fluid properties,
   pressure reference, contact-angle convention, and coverage definition if possible.
2. Repeat the bubble detachment validation in Fig. 5 using the experimental
   geometry and conditions from reference [42]; those conditions are not specified
   sufficiently in the supplied paper to claim that validation here.
3. Run each width through at least 1 s and inspect rolling means and cycle lengths.
   The paper selects `t > 0.2 s`, but this must be checked for the reconstructed case.
4. Compare coarse, medium, and fine meshes and a reduced time-step cap using the
   same physical geometry/feed. Track beta, gas holdup, bubble size, and detachment.
5. Compare Fig. 11 only after resolving input differences. Reported targets include
   0.5 mm bottom coverage 95% above 1 mm, 0.5 mm gas fraction 12.5% above 1 mm,
   and 0.5 mm side coverage 60% above 2 mm. These are validation targets, not output
   values produced by this package.

Table 3 reports gas fractions 0.2504, 0.2465, and 0.2457 at 144705, 206026, and
242395 cells. It does not identify every necessary input or averaging detail for
that comparison; reproducing its mesh count alone is insufficient.
