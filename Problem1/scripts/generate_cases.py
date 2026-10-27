#!/usr/bin/env python3
"""Reconstruct Li et al.'s anode VOF domains; Python standard library only.

Writes conformal, orthogonal hexahedral polyMesh directly. Every physical
dimension is independent of resolution. SI units are used in OpenFOAM files.
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = 8.31446261815324
F = 96485.33212
LEVELS = {"coarse": 0.1, "medium": 0.05, "fine": 0.025}
PATCHES = ("waterInlet", "outlet", "oxygenInlets", "bottomWall", "topWall",
           "sideWalls", "manifoldWalls", "poreWalls")


def header(name, cls="dictionary", location=None):
    loc = f'    location "{location}";\n' if location else ""
    return ("FoamFile\n{\n    version 2.0;\n    format ascii;\n"
            f"    class {cls};\n{loc}    object {name};\n}}\n\n")


def write(path, body, cls="dictionary", location=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header(path.name, cls, location) + body)


def subdivide(knots, max_spacing, refine_ranges=(), fine_spacing=None):
    knots = sorted(set(round(x, 10) for x in knots))
    values = [knots[0]]
    for a, b in zip(knots, knots[1:]):
        spacing = max_spacing
        if any(lo < (a+b)/2 < hi for lo, hi in refine_ranges):
            spacing = min(spacing, fine_spacing)
        n = max(1, math.ceil((b-a)/spacing-1e-8))
        values.extend(a+(b-a)*j/n for j in range(1, n+1))
    return values


def validate(p, width, h):
    for key in ("temperature_K", "absolute_pressure_Pa", "channel_length_mm",
                "channel_height_mm", "mea_width_mm", "manifold_width_mm",
                "manifold_length_mm", "pore_width_mm", "pore_pitch_mm",
                "pore_length_mm", "water_density_kg_m3", "water_dynamic_viscosity_Pa_s",
                "oxygen_dynamic_viscosity_Pa_s", "oxygen_molar_mass_kg_mol",
                "surface_tension_N_m", "end_time_s", "write_interval_s"):
        if p[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 <= p["water_inlet_alpha"] <= 1 or not 0 < p["faradaic_efficiency"] <= 1:
        raise ValueError("Invalid inlet fraction or Faradaic efficiency")
    if p["current_density_A_cm2"] < 0 or p["water_inlet_velocity_m_s"] < 0:
        raise ValueError("Inlet flow/current must be nonnegative")
    if not 0 < p["contact_angle_water_deg"] < 180:
        raise ValueError("Contact angle must lie between 0 and 180 degrees")
    if p["pore_width_mm"] >= p["pore_pitch_mm"] or width < p["pore_pitch_mm"]:
        raise ValueError("Pores must be separated and fit inside a channel")
    if h <= 0:
        raise ValueError("Mesh spacing must be positive")
    for length in (width, p["channel_length_mm"]):
        if abs(length/p["pore_pitch_mm"]-round(length/p["pore_pitch_mm"])) > 1e-8:
            raise ValueError("Channel width and length must be multiples of pore pitch")
    n = p["mea_width_mm"]/(2*width)
    if abs(n-round(n)) > 1e-8 or n < 1:
        raise ValueError("MEA width must contain an integer number of channel/rib pairs")
    if abs(p["manifold_length_mm"]-p["mea_width_mm"]) > 1e-8:
        raise ValueError("This Fig. 3 topology requires manifold length = MEA width")


def build_mesh(p, width, h):
    validate(p, width, h)
    L, H, W = p["channel_length_mm"], p["channel_height_mm"], p["mea_width_mm"]
    M, d, pitch, stem = (p[k] for k in ("manifold_width_mm", "pore_width_mm",
                                      "pore_pitch_mm", "pore_length_mm"))
    nchan = round(W/(2*width))
    channels = [(2*c*width, (2*c+1)*width) for c in range(nchan)]
    px = [(a+(i+0.5)*pitch-d/2, a+(i+0.5)*pitch+d/2)
          for a, b in channels for i in range(round(width/pitch))]
    py = [((i+0.5)*pitch-d/2, (i+0.5)*pitch+d/2) for i in range(round(L/pitch))]
    # At least two cells across a pore avoid rank-deficient single-cell tubes.
    x = subdivide([-width, 0, W-width, W]+[v for t in channels+px for v in t], h, px, d/2)
    y = subdivide([-M, 0, L, L+M]+[v for t in py for v in t], h, py, d/2)
    z = subdivide([-stem, 0, H], h, [(-stem,0)], stem/2)
    xc, yc, zc = [[(a+b)/2 for a, b in zip(v, v[1:])] for v in (x, y, z)]
    in_px = [any(a < c < b for a, b in px) for c in xc]
    in_py = [any(a < c < b for a, b in py) for c in yc]
    chan_id = [next((i for i, (a, b) in enumerate(channels) if a < c < b), -1) for c in xc]
    cells, labels = [], {}
    zones = {"channels": [], "poreStems": [], **{f"Ch{i+1}": [] for i in range(nchan)}}
    for k, cz in enumerate(zc):
        for j, cy in enumerate(yc):
            for i, cx in enumerate(xc):
                in_ch = 0 < cy < L and chan_id[i] >= 0
                inside = ((cz > 0 and (in_ch or (L < cy < L+M and 0 < cx < W)
                            or (-M < cy < 0 and -width < cx < W-width)))
                          or (cz < 0 and in_px[i] and in_py[j]))
                if not inside:
                    continue
                cid = len(cells)
                labels[i, j, k] = cid
                cells.append((i, j, k))
                if cz < 0:
                    zones["poreStems"].append(cid)
                elif in_ch:
                    zones["channels"].append(cid)
                    zones[f"Ch{chan_id[i]+1}"].append(cid)
    return dict(x=x, y=y, z=z, cells=cells, labels=labels, zones=zones,
                channels=channels, pore_count=len(px)*len(py), width=width, parameters=p)


def write_mesh(case, m):
    x, y, z = (m[k] for k in ("x", "y", "z"))
    p, width = m["parameters"], m["width"]
    L, H, W = p["channel_length_mm"], p["channel_height_mm"], p["mea_width_mm"]
    vertices, vlabels, internal = [], {}, []
    boundary = {name: [] for name in PATCHES}
    mouths = []
    # Outward vertex order for x-/x+/y-/y+/z-/z+.
    dirs = [((-1,0,0), ((0,0,0),(0,0,1),(0,1,1),(0,1,0))),
            ((1,0,0), ((1,0,0),(1,1,0),(1,1,1),(1,0,1))),
            ((0,-1,0), ((0,0,0),(1,0,0),(1,0,1),(0,0,1))),
            ((0,1,0), ((0,1,0),(0,1,1),(1,1,1),(1,1,0))),
            ((0,0,-1), ((0,0,0),(0,1,0),(1,1,0),(1,0,0))),
            ((0,0,1), ((0,0,1),(1,0,1),(1,1,1),(0,1,1)))]
    for cid, (i,j,k) in enumerate(m["cells"]):
        cx, cy, cz = (x[i]+x[i+1])/2, (y[j]+y[j+1])/2, (z[k]+z[k+1])/2
        for side, ((di,dj,dk), corners) in enumerate(dirs):
            neighbor = m["labels"].get((i+di,j+dj,k+dk), -1)
            if 0 <= neighbor < cid:
                continue
            face = []
            for a,b,c in corners:
                key = (i+a,j+b,k+c)
                if key not in vlabels:
                    vlabels[key] = len(vertices)
                    vertices.append(key)
                face.append(vlabels[key])
            if neighbor >= 0:
                if side == 5 and abs(z[k+1]) < 1e-9:
                    mouths.append(len(internal))
                internal.append((face,cid,neighbor))
                continue
            in_ch = 0 < cy < L
            if cz < 0:
                patch = "oxygenInlets" if side == 4 else "poreWalls"
            elif side == 1 and L < cy and abs(x[i+1]-W) < 1e-9:
                patch = "waterInlet"
            elif side == 0 and cy < 0 and abs(x[i]+width) < 1e-9:
                patch = "outlet"
            elif in_ch:
                patch = "bottomWall" if side == 4 else "topWall" if side == 5 else "sideWalls"
            else:
                patch = "manifoldWalls"
            boundary[patch].append((face,cid,-1))
    poly = case/"constant/polyMesh"
    poly.mkdir(parents=True, exist_ok=True)
    with (poly/"points").open("w") as f:
        f.write(header("points","vectorField","constant/polyMesh")+f"{len(vertices)}\n(\n")
        for i,j,k in vertices:
            f.write(f"({x[i]*1e-3:.12g} {y[j]*1e-3:.12g} {z[k]*1e-3:.12g})\n")
        f.write(")\n")
    nfaces = len(internal)+sum(map(len,boundary.values()))
    with (poly/"faces").open("w") as ff, (poly/"owner").open("w") as fo, (poly/"neighbour").open("w") as fn:
        for stream,name,cls,count in [(ff,"faces","faceList",nfaces),(fo,"owner","labelList",nfaces),
                                      (fn,"neighbour","labelList",len(internal))]:
            stream.write(header(name,cls,"constant/polyMesh")+f"{count}\n(\n")
        for group in [internal]+list(boundary.values()):
            for face,owner,neighbor in group:
                ff.write("4("+" ".join(map(str,face))+")\n")
                fo.write(f"{owner}\n")
                if neighbor >= 0:
                    fn.write(f"{neighbor}\n")
        for stream in (ff,fo,fn):
            stream.write(")\n")
    offset = len(internal)
    btext = f"{len(PATCHES)}\n(\n"
    for patch, faces in boundary.items():
        typ = "patch" if patch in PATCHES[:3] else "wall"
        btext += f"{patch}\n{{ type {typ}; nFaces {len(faces)}; startFace {offset}; }}\n"
        offset += len(faces)
    write(poly/"boundary", btext+")\n", "polyBoundaryMesh", "constant/polyMesh")
    ztext = f'{len(m["zones"])}\n(\n'
    for name, ids in m["zones"].items():
        ztext += f"{name}\n{{ type cellZone; cellLabels List<label> {len(ids)}\n(\n"
        ztext += "\n".join(map(str,ids))+"\n); }\n"
    write(poly/"cellZones", ztext+")\n", "regIOobject", "constant/polyMesh")
    write(poly/"faceZones", "1\n(\nporeMouths\n{ type faceZone; faceLabels List<label> "
          +str(len(mouths))+"\n(\n"+"\n".join(map(str,mouths))+"\n);\nflipMap List<bool> "
          +str(len(mouths))+"\n(\n"+"\n".join("0" for _ in mouths)+"\n);\n}\n)\n",
          "regIOobject", "constant/polyMesh")
    return dict(cells=len(m["cells"]), points=len(vertices), faces=nfaces,
                patch_faces={n:len(v) for n,v in boundary.items()}, mouth_faces=len(mouths))


def properties(p, m):
    area = p["channel_length_mm"]*p["mea_width_mm"]*1e-6
    rho_o2 = p["absolute_pressure_Pa"]*p["oxygen_molar_mass_kg_mol"]/(R*p["temperature_K"])
    current = p["current_density_A_cm2"]*1e4*area
    q = p["faradaic_efficiency"]*current/(4*F)*R*p["temperature_K"]/p["absolute_pressure_Pa"]
    pore_area = m["pore_count"]*p["pore_width_mm"]**2*1e-6
    return dict(mea_area_m2=area, current_A=current, oxygen_density_kg_m3=rho_o2,
                oxygen_flow_m3_s=q, pore_area_m2=pore_area, oxygen_velocity_m_s=q/pore_area,
                inlet_mixture_flow_m3_s=p["water_inlet_velocity_m_s"]*p["manifold_width_mm"]
                                      *p["channel_height_mm"]*1e-6)


def fields_and_controls(case, m, p, derived, end_time, write_interval):
    u = derived["oxygen_velocity_m_s"]
    U = f"dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField\n{{\n"
    U += f"waterInlet {{ type fixedValue; value uniform (-{p['water_inlet_velocity_m_s']:.12g} 0 0); }}\n"
    U += "outlet { type pressureInletOutletVelocity; value uniform (0 0 0); }\n"
    U += f"oxygenInlets {{ type fixedValue; value uniform (0 0 {u:.12g}); }}\n"
    U += "\n".join(f"{patch} {{ type noSlip; }}" for patch in PATCHES[3:])+"\n}\n"
    write(case/"0/U",U,"volVectorField","0")
    ptext = "dimensions [1 -1 -2 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n"
    for patch in PATCHES:
        ptext += (f"{patch} {{ type prghPressure; p uniform 0; rho rho; value uniform 0; }}\n" if patch=="outlet"
                  else f"{patch} {{ type fixedFluxPressure; value uniform 0; }}\n")
    write(case/"0/p_rgh",ptext+"}\n","volScalarField","0")
    stem_ids = set(m["zones"]["poreStems"])
    atext = "dimensions [0 0 0 0 0 0 0];\ninternalField nonuniform List<scalar> "
    atext += str(len(m["cells"]))+"\n(\n"+"\n".join("0" if i in stem_ids else "1" for i in range(len(m["cells"])))+"\n);\nboundaryField\n{\n"
    atext += f"waterInlet {{ type fixedValue; value uniform {p['water_inlet_alpha']}; }}\n"
    atext += "oxygenInlets { type fixedValue; value uniform 0; }\n"
    atext += "outlet { type inletOutlet; inletValue uniform 1; value uniform 1; }\n"
    # v2312 interface normal is grad(alpha.water); theta0 is the water-side angle.
    atext += "\n".join(f"{patch} {{ type constantAlphaContactAngle; theta0 {p['contact_angle_water_deg']}; limit gradient; value uniform 1; }}" for patch in PATCHES[3:])+"\n}\n"
    write(case/"0/alpha.water",atext,"volScalarField","0")
    rho_w, rho_g = p["water_density_kg_m3"], derived["oxygen_density_kg_m3"]
    write(case/"constant/transportProperties",f"phases (water oxygen);\nwater\n{{ transportModel Newtonian; rho {rho_w}; nu {p['water_dynamic_viscosity_Pa_s']/rho_w:.12g}; }}\noxygen\n{{ transportModel Newtonian; rho {rho_g:.12g}; nu {p['oxygen_dynamic_viscosity_Pa_s']/rho_g:.12g}; }}\nsigma {p['surface_tension_N_m']};\n")
    write(case/"constant/turbulenceProperties","simulationType laminar;\n")
    write(case/"constant/g","dimensions [0 1 -2 0 0 0 0];\nvalue ("+" ".join(map(str,p["gravity_m_s2"]))+");\n","uniformDimensionedVectorField","constant")
    spacings = [b-a for name in ("x","y","z") for a,b in zip(m[name],m[name][1:])]
    # Conservative explicit capillary-wave step limit, independent of advective Co.
    dt_cap = 0.25*math.sqrt((rho_w+rho_g)*(min(spacings)*1e-3)**3/(math.pi*p["surface_tension_N_m"]))
    max_dt = min(1e-3, dt_cap)
    derived.update(capillary_max_delta_t_s=max_dt, minimum_cell_spacing_mm=min(spacings))
    write(case/"system/controlDict",f"application interFoam;\nstartFrom latestTime;\nstartTime 0;\nstopAt endTime;\nendTime {end_time};\ndeltaT {min(1e-6,max_dt):.12g};\nadjustTimeStep yes;\nmaxCo {p['max_Co']};\nmaxAlphaCo {p['max_alpha_Co']};\nmaxDeltaT {max_dt:.12g};\nwriteControl adjustableRunTime;\nwriteInterval {write_interval};\npurgeWrite 0;\nwriteFormat ascii;\nwritePrecision 10;\nwriteCompression off;\ntimeFormat general;\ntimePrecision 12;\nrunTimeModifiable true;\nfunctions\n{{\n#include \"bubbleMetrics\"\n}}\n")
    write(case/"system/fvSchemes","""ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(rhoPhi,U) Gauss linearUpwind grad(U);
    div(phi,alpha) Gauss vanLeer;
    div(phirb,alpha) Gauss interfaceCompression;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
fluxRequired { default no; p_rgh; alpha.water; }
""")
    write(case/"system/fvSolution","""solvers
{
    "alpha.water.*"
    {
        nAlphaCorr 2;
        nAlphaSubCycles 1;
        cAlpha 1;
        MULESCorr yes;
        nLimiterIter 5;
        solver smoothSolver;
        smoother symGaussSeidel;
        tolerance 1e-9;
        relTol 0;
    }
    "pcorr.*" { solver PCG; preconditioner DIC; tolerance 1e-9; relTol 0; }
    p_rgh { solver GAMG; tolerance 1e-8; relTol 0.01; smoother DICGaussSeidel; }
    p_rghFinal { $p_rgh; relTol 0; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0; }
    UFinal { $U; relTol 0; }
}
PIMPLE
{
    momentumPredictor yes;
    nOuterCorrectors 2;
    nCorrectors 2;
    nNonOrthogonalCorrectors 0;
}
relaxationFactors { equations { ".*" 1; } }
""")
    write(case/"system/decomposeParDict","numberOfSubdomains 4;\nmethod scotch;\n")
    metrics = (ROOT/"templates/bubbleMetrics").read_text()
    (case/"system/bubbleMetrics").write_text(metrics)
    (case/"flowfield.foam").touch()


def generate(case, width, level, p, end_time=None, write_interval=None):
    if case.exists() and any(case.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {case}; use a new --output directory")
    m = build_mesh(p,width,LEVELS[level])
    stats = write_mesh(case,m)
    d = properties(p,m)
    fields_and_controls(case,m,p,d,end_time or p["end_time_s"],write_interval or p["write_interval_s"])
    def volume(ids):
        return sum((m['x'][i+1]-m['x'][i])*(m['y'][j+1]-m['y'][j])*(m['z'][k+1]-m['z'][k])*1e-9
                   for i,j,k in (m['cells'][cid] for cid in ids))
    stats.update(channel_volume_m3=volume(m['zones']['channels']),
                 total_volume_m3=volume(range(len(m['cells']))), pore_count=m['pore_count'],
                 channel_count=len(m['channels']),channel_width_mm=width,mesh_level=level,
                 end_time_s=end_time or p['end_time_s'],write_interval_s=write_interval or p['write_interval_s'])
    meta = dict(paper_doi="10.1016/j.ijheatmasstransfer.2025.127446",parameters=p,derived=d,mesh=stats,
                status="Reconstructed VOF submodel; estimated oxygen feed; not a validated reproduction")
    (case/"case_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(f"{case}: {stats['cells']:,} cells, {m['pore_count']} pores, Q(O2)={d['oxygen_flow_m3_s']:.6g} m3/s")
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/reference.json")
    parser.add_argument("--output",type=Path,default=ROOT/"cases")
    parser.add_argument("--widths",type=float,nargs="+",choices=[0.5,1,2],default=[2,1,0.5])
    parser.add_argument("--mesh",choices=LEVELS,default="medium")
    parser.add_argument("--end-time",type=float)
    parser.add_argument("--write-interval",type=float)
    parser.add_argument("--current-density",type=float,help="A/cm2; uniform Faraday-law feed for all selected cases")
    parser.add_argument("--inlet-alpha",type=float,help="Water fraction at mixture inlet; paper 0.9, pure-water sensitivity 1")
    args=parser.parse_args()
    if args.end_time is not None and args.end_time <= 0 or args.write_interval is not None and args.write_interval <= 0:
        parser.error("Time settings must be positive")
    p=json.loads(args.config.read_text())
    if args.current_density is not None: p['current_density_A_cm2']=args.current_density
    if args.inlet_alpha is not None: p['water_inlet_alpha']=args.inlet_alpha
    for w in args.widths:
        generate(args.output/f"w{w:g}mm",w,args.mesh,p,args.end_time,args.write_interval)


if __name__ == "__main__":
    main()
