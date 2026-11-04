#!/usr/bin/env python3
"""Package source, clean production cases, and verification evidence; no PDF."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root',type=Path,default=ROOT/'cases/openings64')
    parser.add_argument('--output',type=Path,
                        default=ROOT/'output/li-vof-openfoam-v2312-64-openings.zip')
    args=parser.parse_args()
    case_root=args.case_root.resolve()
    if not case_root.is_relative_to(ROOT):
        parser.error('--case-root must be inside Problem1')
    output=args.output.parent
    output.mkdir(parents=True,exist_ok=True)
    archive=args.output
    files=[ROOT/'README.md',ROOT/'.gitignore']
    for folder in ['scripts','config','templates','tests','docs','results']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix!='.pyc')
    cases=sorted(case_root.glob('w*mm'))
    if len(cases)!=3:
        raise RuntimeError(f'Expected three generated production cases under {case_root}')
    for case in cases:
        for folder in ['0','system','constant']:
            files.extend(p for p in (case/folder).rglob('*') if p.is_file()
                         and 'sets' not in p.parts)
        files.extend(case/name for name in ['case_metadata.json','flowfield.foam','log.checkMesh'])
    with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(set(files)):
            z.write(path,Path('Problem1')/path.relative_to(ROOT))
    with ZipFile(archive) as z:
        bad=z.testzip()
        if bad: raise RuntimeError(f'Archive integrity failure: {bad}')
        members=len(z.namelist())
    manifest=dict(file=archive.name,bytes=archive.stat().st_size,members=members,
                  sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  cases=[dict(directory=str(c.relative_to(ROOT)),
                              mesh=json.loads((c/'case_metadata.json').read_text())['mesh'])
                         for c in cases],
                  scope='Initial fields, meshes and controls for three OpenFOAM v2312 cases; source and verification evidence; no transient snapshots')
    archive.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
