"""Copy a finished same-geometry state as a new, independently verified guess."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from .config import ManifoldConfig
from .metrics import values, vectors
from .geometry import PATCHES

VOLUME_FIELDS=('Ul','Ug','p','T','sg','lambda','material','phiS','phiE')
SURFACE_FIELDS=('phiL','phiG')
FIELDS=VOLUME_FIELDS+SURFACE_FIELDS
NUMERICAL={'iterations','write_interval','processes','pseudo_dt','phase_numerical_courant','field_tolerance',
           'balance_tolerance','minimum_iterations','initial_voltage'}


def completed_coupling(source, status, config):
    """Restore a completed full-model stage, never its convergence certificate.

    Older/partially initialized guesses still use the normal continuation.
    Explicitly requesting a lower initial voltage also keeps that behavior.
    """
    if status.get('momentum_fraction') != 1 or status.get('thermal_active') is not True:
        return dict(restored=False,reason='Source has not recorded the full coupled stage')
    path=source/'postProcessing/p006/history.csv'
    text=path.read_text()
    reader=csv.DictReader(text.splitlines())
    if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames):
        raise ValueError('Initial-state history has missing/duplicate columns')
    last=None
    for row in reader:last=row
    try:
        final={k:float(v) for k,v in last.items()}
        if (not all(math.isfinite(v) for v in final.values())
                or final['iteration']!=status['iteration']
                or final['momentum_fraction']!=1 or final['thermal_active']!=1
                or not math.isclose(final['applied_voltage_V'],config.cell_voltage,rel_tol=0,abs_tol=1e-12)):
            raise ValueError('Completion status and final coupled history disagree')
    except (AttributeError,KeyError,TypeError,ValueError) as exc:
        raise ValueError('Initial-state full-coupling evidence is inconsistent') from exc
    if min(config.initial_voltage,config.cell_voltage)!=config.cell_voltage:
        return dict(restored=False,reason='Requested a lower initial voltage')
    return dict(restored=True,momentum_fraction=1.,thermal_active=True,voltage_V=final['applied_voltage_V'],
                history_sha256=hashlib.sha256(text.encode()).hexdigest())


def same_geometry(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(same_geometry(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(same_geometry(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
    return a==b


def load(source, geometry, config, required_inputs, materials):
    source=Path(source).resolve()
    manifest=json.loads((source/'case.json').read_text())
    if manifest.get('problem')!='P006' or not same_geometry(manifest['design_geometry'],geometry):
        raise ValueError('Initial state must have the same P006 geometry')
    old=ManifoldConfig(**manifest['config'])
    if old.fingerprint!=manifest['config_sha256']:
        raise ValueError('Initial-state configuration fingerprint differs')
    if any(old.to_dict()[key]!=value for key,value in config.to_dict().items() if key not in NUMERICAL):
        raise ValueError('Initial state must have the same physical configuration and mesh')
    hashes=manifest.get('input_sha256',{})
    # Early development cases initialized face fluxes inside the solver. Their
    # completed output fluxes can still be verified and copied into new inputs.
    legacy_inputs=set(required_inputs)-{'0/phiL','0/phiG'}
    if set(hashes) not in (set(required_inputs),legacy_inputs) or any(
            hashlib.sha256((source/name).read_bytes()).hexdigest()!=sha for name,sha in hashes.items()):
        raise ValueError('Initial-state case inputs were altered')
    status_path=source/'postProcessing/p006/status.json'
    log_path=source/'log.flowfieldP006Foam'
    result_path=source/'result.json'
    if not status_path.is_file() or not log_path.is_file() or not result_path.is_file():
        raise ValueError('Initial state must come from a finished run; running or interrupted cases are not copied')
    result=json.loads(result_path.read_text())
    if result.get('status') not in ('ok','failed') or result.get('input_sha256')!=hashes:
        raise ValueError('Initial-state execution record is inconsistent')
    status=json.loads(status_path.read_text())
    log=log_path.read_text()
    endings=list(re.finditer(r'^End\s*$',log,re.M))
    starts=list(re.finditer(r'^(?:Iteration = |Create time)',log,re.M))
    if not endings or (starts and starts[-1].start()>endings[-1].start()):
        raise ValueError('Initial-state run has no completion marker')
    iteration=status.get('iteration')
    if (type(iteration) is not int or iteration<1 or status.get('cells')!=manifest['cells']
            or status.get('processes')!=old.processes):
        raise ValueError('Initial-state completion metadata is inconsistent')
    directory=source/str(iteration)
    n=manifest['cells'];snapshots={};data={}
    if n!=len(materials):raise ValueError('Initial-state cell count differs')
    for name in VOLUME_FIELDS:
        # Read every complete field before modifying any newly generated input.
        snapshots[name]=(directory/name).read_text()
        data[name]=(vectors if name in ('Ul','Ug') else values)(snapshots[name],'internalField',n)
    neighbour=(source/'constant/polyMesh/neighbour').read_text().split('}',1)[1]
    nfaces=int(re.search(r'(\d+)\s*\(',neighbour).group(1))
    boundary=(source/'constant/polyMesh/boundary').read_text()
    for name in SURFACE_FIELDS:
        snapshots[name]=(directory/name).read_text()
        values(snapshots[name],'internalField',nfaces)
        for patch in PATCHES:
            meshblock=re.search(r'\b'+patch+r'\s*\{(.*?)\}',boundary,re.S).group(1)
            count=int(re.search(r'nFaces\s+(\d+)',meshblock).group(1))
            block=re.search(r'\b'+patch+r'\s*\{(.*?)\}',snapshots[name],re.S)
            if not block:raise ValueError('Initial-state face flux is missing a boundary patch')
            values(block.group(1),'value',count)
    if data['material']!=materials:
        raise ValueError('Initial-state material addressing differs')
    if (any(not 273.15<t<373.15 for t in data['T'])
            or any(not .64<w<=30 for w in data['lambda'])):
        raise ValueError('Initial-state temperature or hydration is outside closure bounds')
    for i,material in enumerate(materials):
        if material in (1,2,3,5,6,7):
            if not 1e-6<=data['sg'][i]<=1-1e-6 or data['p'][i]<=-101325:
                raise ValueError('Initial-state phase fraction or pressure is outside bounds')
    metadata=dict(kind='same_geometry_initial_guess',source_case=str(source),source_iteration=iteration,
        source_converged=status.get('converged') is True,
        source_config_sha256=manifest['config_sha256'],
        source_field_sha256={name:hashlib.sha256(text.encode()).hexdigest() for name,text in snapshots.items()},
        preserves_target_boundary_conditions=True,
        preserves_face_fluxes=True,
        coupling=completed_coupling(source,status,config),
        qualification_inherited=False)
    return snapshots,metadata


def write(case, state):
    snapshots,metadata=state
    pattern=r'\binternalField\b[^;]+;'
    for name in FIELDS:
        block=re.search(pattern,snapshots[name]).group(0)
        target=Path(case)/'0'/name
        text,count=re.subn(pattern,lambda _:block,target.read_text(),count=1)
        if count!=1:raise ValueError('Generated initial field is malformed')
        if name in SURFACE_FIELDS:
            for patch in PATCHES:
                # Keep freshly prescribed inlet flow; retain conservative
                # outlet/wall fluxes and all internal face fluxes from the guess.
                if patch.endswith('Inlet'):continue
                patch_pattern=r'\b'+patch+r'\s*\{(.*?)\}'
                saved=re.search(patch_pattern,snapshots[name],re.S).group(1)
                value=re.search(r'\bvalue\b[^;]+;',saved).group(0)
                text=re.sub(patch_pattern,lambda _:f'{patch} {{ type calculated; {value} }}',text,count=1,flags=re.S)
        target.write_text(text)
    if metadata['coupling']['restored']:
        # These are already supported solver inputs. Write before case hashes
        # are generated, so both activation flags are immutable/provenanced.
        props=Path(case)/'constant/modelProperties'
        with props.open('a') as stream:
            stream.write('\ninitial_momentum_fraction 1;\ninitial_thermal_active true;\n')
    return metadata
