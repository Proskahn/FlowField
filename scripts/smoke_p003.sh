#!/usr/bin/env bash
# One real P003 CFD evaluation; output directory must be new.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# -lt 1 ]]; then
    echo 'Usage: smoke_p003.sh NEW_OUTPUT_DIRECTORY [--ioh] [--backend docker|native|auto]' >&2
    exit 2
fi
output="$1"
shift
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m flowfield.problems.level4_high_dimensional_topology.p003.cli smoke --output "$output" "$@"
