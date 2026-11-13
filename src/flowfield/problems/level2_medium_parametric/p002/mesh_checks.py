"""Explicit policy for orthogonal, thin catalyst layers; retain every diagnostic."""

import math
import re

from .geometry import geometry


def assess_mesh(case, log, manifest):
    if 'Mesh OK.' in log and not re.search(r'Failed \d+ mesh checks',log):
        return dict(accepted=True, openfoam_all_checks_passed=True, warnings=[])
    # OpenFOAM's optional determinant metric flags highly anisotropic, even
    # exactly orthogonal cells. Accept this one diagnostic only, in CL zones.
    if not re.search(r'^Failed 1 mesh checks\.$',log,re.M):
        failures = '; '.join(line.strip() for line in log.splitlines() if line.lstrip().startswith('***'))
        raise ValueError('Mesh failed checks other than the thin-layer determinant diagnostic: '
                         f'{failures or "see log.checkMesh"}')
    if len(re.findall(r'^\s*\*\*\*',log,re.M)) != 1 or '***Cells with small determinant' not in log:
        raise ValueError('Unexpected mesh quality failure')
    def number(pattern):
        match = re.search(pattern,log)
        if not match:
            raise ValueError('Missing mesh diagnostic')
        value = float(match.group(1))
        if not math.isfinite(value):
            raise ValueError('Nonfinite mesh diagnostic')
        return value
    determinant = number(r'Cell determinant \(wellposedness\) : minimum:\s*(\S+)')
    aspect = number(r'Max aspect ratio = (\S+)')
    nonorth = number(r'Mesh non-orthogonality Max:\s*(\S+)')
    skew = number(r'Max skewness = (\S+)')
    if not (determinant > 1e-8 and aspect <= 100 and nonorth < 1e-6 and skew < 1e-6):
        raise ValueError('Mesh falls outside the orthogonal catalyst-layer exception')
    text = (case/'constant/polyMesh/sets/underdeterminedCells').read_text().split('}',1)[1]
    match = re.search(r'(\d+)\s*\((.*?)\)',text,re.S)
    if not match:
        raise ValueError('Cannot inspect underdetermined cell set')
    cells = [int(v) for v in match.group(2).split()]
    m = geometry(manifest['parameters']['geometry'],'parallel',manifest['mesh_level'])
    if (not cells or len(cells) != int(match.group(1)) or len(set(cells)) != len(cells)
            or any(c<0 or c>=len(m['ids']) or m['ids'][c] not in (3,5) for c in cells)):
        raise ValueError('Determinant warning is not confined to catalyst layers')
    return dict(accepted=True, openfoam_all_checks_passed=False, min_determinant=determinant,
                max_aspect_ratio=aspect, max_nonorthogonality=nonorth, max_skewness=skew,
                warnings=[dict(kind='orthogonal_thin_catalyst_layer_determinant', cells=len(cells),
                    policy='Only CL cells; determinant > 1e-8, aspect <= 100, nonorthogonality/skewness < 1e-6')])
