"""Shared OpenFOAM dictionary formatting and native output parsing."""
import math
from pathlib import Path
from .runtime import SimulationError


def foam_file(name, body, cls="dictionary"):
    return f"""/* FlowField P000; OpenCFD OpenFOAM v2312; SI units. */
FoamFile
{{
    version 2.0;
    format ascii;
    class {cls};
    object {name};
}}
{body.strip()}
"""


def surface_function(name, patch, operation, field):
    return f"""
    {name}
    {{
        type surfaceFieldValue;
        libs ("libfieldFunctionObjects.so");
        writeControl timeStep;
        writeInterval 1;
        log false;
        writeFields false;
        regionType patch;
        name {patch};
        operation {operation};
        fields ({field});
    }}
"""


def read_surface_history(path, column):
    """Read one fresh-run scalar table, validating its header and all data rows."""
    rows = {}
    header_seen = False
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SimulationError(f"Missing surface data: {path}") from exc
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            columns = line[1:].split()
            if columns and columns[0] == "Time":
                header_seen = columns == ["Time", column]
                if not header_seen:
                    raise SimulationError(f"Unexpected columns in {path}: {columns}")
            continue
        if not header_seen:
            raise SimulationError(f"Missing Time/{column} header in {path}")
        try:
            time, value = map(float, line.split())
        except ValueError as exc:
            raise SimulationError(f"Malformed scalar row in {path}: {line}") from exc
        if not math.isfinite(time) or not math.isfinite(value) or time < 0:
            raise SimulationError(f"Nonfinite or invalid surface data in {path}")
        if rows and time <= next(reversed(rows)):
            raise SimulationError(f"Non-increasing surface times in {path}")
        rows[time] = value
    if not rows:
        raise SimulationError(f"No surface data in {path}")
    return rows


