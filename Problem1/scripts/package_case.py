#!/usr/bin/env python3
"""Package source, clean production cases, and verification evidence; no PDF."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'output'
    output.mkdir(exist_ok=True)
    archive=output/'li-vof-openfoam-v2312.zip'
    files=[ROOT/'README.md',ROOT/'.gitignore']
    for folder in ['scripts','config','templates','tests','docs','results']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix!='.pyc')
    for case in sorted((ROOT/'cases').glob('w*mm')):
        for folder in ['0','system','constant']:
            files.extend(p for p in (case/folder).rglob('*') if p.is_file()
                         and 'sets' not in p.parts)
        files.extend(case/name for name in ['case_metadata.json','flowfield.foam','log.checkMesh'])
    if len(list((ROOT/'cases').glob('w*mm')))!=3:
        raise RuntimeError('Expected all three clean production cases')
    with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(set(files)):
            z.write(path,Path('li-vof-openfoam')/path.relative_to(ROOT))
    with ZipFile(archive) as z:
        bad=z.testzip()
        if bad: raise RuntimeError(f'Archive integrity failure: {bad}')
        members=len(z.namelist())
    manifest=dict(file=archive.name,bytes=archive.stat().st_size,members=members,
                  sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  scope='Three clean OpenFOAM v2312 production cases plus source and short-run verification evidence')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
