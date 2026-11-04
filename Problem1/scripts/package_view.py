#!/usr/bin/env python3
"""Package selected OpenFOAM snapshots for ParaView, without changing the case.

Run on the machine holding the COMPLETE, stopped case. Standard library only.
The archive is for visualization, not for restarting interFoam.
"""
import argparse
import io
from pathlib import Path
import re
import tarfile

MESH_FILES = ('points', 'faces', 'owner', 'neighbour', 'boundary')


def collect(case, times, fields):
    processors = sorted((p for p in case.iterdir()
                         if p.is_dir() and re.fullmatch(r'processor\d+', p.name)),
                        key=lambda p: int(p.name[9:]))
    roots = processors or [case]
    if processors and [int(p.name[9:]) for p in processors] != list(range(len(processors))):
        raise ValueError('Processor numbers are not contiguous from zero.')
    dictionary = case / 'system/decomposeParDict'
    if processors and dictionary.is_file():
        count = re.search(r'\bnumberOfSubdomains\s+(\d+)\s*;', dictionary.read_text())
        if count and int(count[1]) != len(processors):
            raise ValueError('Processor folder count differs from numberOfSubdomains.')
    selected = []
    missing = []
    # Keep the original mesh too: ParaView probes it before the user changes
    # the reader from its default reconstructed mode to decomposed mode.
    for root in ([case] + roots if processors else roots):
        for name in MESH_FILES:
            f = root / 'constant/polyMesh' / name
            f = f if f.is_file() else f.with_suffix(f.suffix + '.gz')
            (selected if f.is_file() else missing).append(f)
    for root in roots:
        for time in times:
            for field in fields:
                f = root / time / field
                f = f if f.is_file() else f.with_suffix(f.suffix + '.gz')
                (selected if f.is_file() else missing).append(f)
    if missing:
        details = '\n'.join(str(p.relative_to(case)) for p in missing[:40])
        raise ValueError(f'Incomplete case: {len(missing)} required files are missing.\n'
                         f'{details}\nCopy the original mesh/results from the server; '
                         'do not create a replacement decomposition over these fields.')
    return selected, bool(processors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--times', default='0.2,0.4,0.6,0.8,1',
                        help='Comma-separated exact saved directory names (default: five snapshots).')
    parser.add_argument('--fields', default='alpha.water,U,p')
    parser.add_argument('--output', required=True, type=Path, help='New .tar.gz archive.')
    args = parser.parse_args()
    case = args.case.resolve()
    times = args.times.split(',')
    fields = args.fields.split(',')
    if not all(re.fullmatch(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', t) for t in times):
        parser.error('Times must be nonnegative numeric directory names.')
    if not all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', f) for f in fields):
        parser.error('Invalid field name.')
    if not case.is_dir():
        parser.error(f'Case not found: {case}')
    try:
        files, parallel = collect(case, times, fields)
    except ValueError as exc:
        parser.exit(2, f'{exc}\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prefix = case.name + '-view'
    # Exclusive creation protects existing archives. All source files are read only.
    with args.output.open('xb') as output:
        try:
            with tarfile.open(fileobj=output, mode='w:gz', compresslevel=5) as archive:
                for f in files:
                    archive.add(f, arcname=f'{prefix}/{f.relative_to(case)}', recursive=False)
                content = {
                    'flowfield.foam': '',
                    'system/controlDict': (
                        'FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n'
                        f'application interFoam; startFrom startTime; startTime {min(map(float,times))};\n'
                        f'stopAt endTime; endTime {max(map(float,times))}; deltaT 1e-6;\n'
                        'writeControl timeStep; writeInterval 1;\n'),
                    'VIEWING.txt': (
                        'Visualization subset; not a solver restart case.\n'
                        'Open flowfield.foam in ParaView.\n'
                        f'Case Type: {"Decomposed Case" if parallel else "Reconstructed Case"}.\n'
                        'Disable List timesteps according to controlDict if present.\n'
                        'Select internalMesh and alpha.water; Apply.\n'
                        'Select a saved time, then Contour: alpha.water = 0.5.\n'
                        f'Saved times: {", ".join(times)}\nFields: {", ".join(fields)}\n'
                        'Packaged files were checked for presence; numerical validity is not certified.\n')}
                for name, value in content.items():
                    data = value.encode()
                    info = tarfile.TarInfo(f'{prefix}/{name}')
                    info.size = len(data)
                    info.mode = 0o644
                    archive.addfile(info, io.BytesIO(data))
        except BaseException:
            args.output.unlink(missing_ok=True)
            raise
    print(f'{args.output}: {args.output.stat().st_size / 1024**2:.1f} MiB; '
          f'{len(times)} snapshots. Source case unchanged.')


if __name__ == '__main__':
    main()
