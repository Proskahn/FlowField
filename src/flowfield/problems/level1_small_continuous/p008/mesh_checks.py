"""Explicit, narrow allowance for orthogonal micrometre-thick catalyst cells."""
import math
import re
from .config import AEMConfig
from .geometry import build_geometry

def assess_mesh(case,log,manifest):
    if 'Mesh OK.' in log and not re.search(r'Failed \d+ mesh checks',log):
        return dict(accepted=True,openfoam_all_checks_passed=True,warnings=[])
    if (not re.search(r'^Failed 1 mesh checks\.$',log,re.M) or len(re.findall(r'^\s*\*\*\*',log,re.M))!=1
        or '***High aspect ratio cells found' not in log):raise ValueError('Unexpected P008 mesh-quality failure')
    def number(pattern):
        match=re.search(pattern,log)
        if not match:raise ValueError('Missing mesh diagnostic')
        value=float(match[1].rstrip('.,'))
        if not math.isfinite(value):raise ValueError('Nonfinite mesh diagnostic')
        return value
    aspect=number(r'Max aspect ratio:\s*([^,]+)')
    nonorth=number(r'Mesh non-orthogonality Max:\s*(\S+)');skew=number(r'Max skewness = (\S+)')
    volume=number(r'Min volume = (\S+)')
    if not (aspect<=1400 and nonorth<1e-6 and skew<1e-6 and volume>1e-20):raise ValueError('Outside the declared thin, orthogonal CL mesh policy')
    body=(case/'constant/polyMesh/sets/highAspectRatioCells').read_text().split('}',1)[1]
    match=re.search(r'(\d+)\s*\((.*?)\)',body,re.S)
    if not match:raise ValueError('Missing high-aspect cell addressing')
    cells=[int(v) for v in match[2].split()];m=build_geometry(manifest['design_geometry'],AEMConfig(**manifest['config']))
    if not cells or len(cells)!=int(match[1]) or len(set(cells))!=len(cells) or any(c<0 or c>=len(m['ids']) or m['ids'][c] not in (3,5) for c in cells):raise ValueError('High aspect ratio is not confined to CL cells')
    # Independently check the actual prescribed lengths of every flagged cell.
    actual=[]
    for c in cells:
        widths=[m[axis][i+1]-m[axis][i] for axis,i in zip(('x','y','z'),m['cells'][c])]
        actual.append(max(widths)/min(widths))
    if not math.isclose(max(actual),aspect,rel_tol=1e-8):raise ValueError('Aspect-ratio report disagrees with prescribed mesh')
    return dict(accepted=True,openfoam_all_checks_passed=False,max_aspect_ratio=aspect,min_volume_m3=volume,
        max_nonorthogonality=nonorth,max_skewness=skew,warnings=[dict(kind='orthogonal_thin_catalyst_layer_aspect_ratio',cells=len(cells),
            policy='Only CL cells; aspect <= 1400, positive volumes, nonorthogonality/skewness < 1e-6; every other OpenFOAM mesh test must pass')])
