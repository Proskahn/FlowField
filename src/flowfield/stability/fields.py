"""Supplement problem-owned audits with finite ASCII field and bound checks."""
import gzip
import math
from pathlib import Path
import re


def final_fields(case, name, config, expected_step, cells, max_bytes=128*1024*1024):
    times=[]
    for p in case.iterdir():
        if not p.is_dir():continue
        try:value=float(p.name)
        except ValueError:continue
        if math.isfinite(value) and value>0:times.append((value,p))
    if not times:return dict(status='incomplete',reason='No reconstructed final fields available.',fields={})
    t,directory=max(times)
    if expected_step is not None and not math.isclose(t,expected_step,rel_tol=1e-8,abs_tol=1e-12):
        return dict(status='failed',reason=f'Saved time {t} differs from solver step {expected_step}.',fields={})
    results={};errors=[];unchecked=[]
    required={'P000':('p','U'),'P001':('p','U'), 'P004-hydraulic':('p','U'),
        'P002':('p','U','T','Cgas'),'P003':('p','oxygen','water','nitrogen'),
        'P005':('alpha.water','p_rgh','U'),'P006':('p','sg','T','Ul','Ug','qL','qG'),
        'P008':('p','T','sg','U','Ul','Ug','phiS','phiE','phiL','phiG','iVol','heatSource'),
        'P007':('p','U','T','gamma','brinkman','phi')}[name]
    for p in sorted(directory.iterdir()):
        if not p.is_file():continue
        key=p.stem if p.suffix=='.gz' else p.name
        opener=gzip.open if p.suffix=='.gz' else open
        try:
            with opener(p,'rb') as stream:raw=stream.read(max_bytes+1)
            if len(raw)>max_bytes:unchecked.append(f'{key}: exceeds field byte limit');continue
            header=raw[:2048].decode('ascii',errors='replace')
            cls=re.search(r'\bclass\s+((?:vol|surface)(?:Scalar|Vector|SphericalTensor|SymmTensor|Tensor)Field)\s*;',header)
            if not cls:continue
            if re.search(r'\bformat\s+binary\s*;',header):unchecked.append(f'{key}: binary fields are not supported');continue
            text=raw.decode('ascii');text=re.sub(r'/\*.*?\*/|//[^\n]*','',text,flags=re.S)
            if re.search(r'(?<![\w.])[+-]?(?:nan|inf(?:inity)?)(?![\w.])',text,re.I):raise ValueError('Nonfinite field or boundary value')
            # Overflow such as 1e999 is nonfinite too, including in a patch
            # value that is not part of the internal-field list below.
            for number in re.finditer(r'(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])',text):
                if not math.isfinite(float(number[0])):raise ValueError('Overflowing field or boundary value')
            part=re.search(r'\binternalField\s+(.+?);',text,re.S)
            if not part:raise ValueError('Missing internal field')
            part=part[1];components=3 if 'Vector' in cls[1] else 6 if 'SymmTensor' in cls[1] else 9 if 'Tensor' in cls[1] and 'Spherical' not in cls[1] else 1
            match=re.fullmatch(r'\s*nonuniform\s+List<\w+>\s+(\d+)\s*\((.*)\)\s*',part,re.S)
            if match:
                count=int(match[1]);body=match[2]
                if cls[1].startswith('vol') and cells is not None and count!=cells:raise ValueError('Field count differs from mesh cells')
            else:
                match=re.fullmatch(r'\s*uniform\s+(.+)\s*',part,re.S)
                if not match:raise ValueError('Unsupported internal field encoding')
                count=1;body=match[1]
            vals=[float(v) for v in body.replace('(',' ').replace(')',' ').split()]
            if len(vals)!=count*components or any(not math.isfinite(v) for v in vals):raise ValueError('Truncated or nonfinite internal field')
            if not vals:
                results[key]=dict(count=0);continue
            low,high=min(vals),max(vals);bounds=None
            if key in ('sg','oxygen','water','nitrogen','alpha.water','gamma'):bounds=(0,1)
            elif key=='T':bounds=(273.15,373.15) if name in ('P006','P008') else (0,1) if name=='P007' else (0,None)
            elif key=='brinkman':bounds=(0,None)
            elif key in ('k','epsilon','nut','Cgas'):bounds=(0,None)
            tol=getattr(config,'alpha_tolerance',1e-8) if key=='alpha.water' else 1e-8
            if bounds and (low<bounds[0]-tol or (bounds[1] is not None and high>bounds[1]+tol)):
                raise ValueError(f'Outside physical bounds {bounds}: [{low}, {high}]')
            results[key]=dict(count=count,min_component=low,max_component=high,bounds=bounds)
        except (OSError,ValueError,UnicodeError,EOFError) as exc:errors.append(f'{key}: {exc}')
    missing=set(required)-set(results)
    if missing and not unchecked:errors.append('Missing required fields: '+', '.join(sorted(missing)))
    return dict(status='failed' if errors else 'inconclusive' if unchecked else 'passed',
        directory=str(directory),time=t,fields=results,errors=errors,unchecked=unchecked,
        scope='Finite saved internal/boundary numbers and declared bounds; model-owned extraction supplies conservation checks.')
