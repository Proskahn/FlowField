#!/usr/bin/env bash
# One real P004 auxiliary hydraulic CFD evaluation; output directory must be new.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# -lt 1 ]]; then
    echo 'Usage: smoke_p004.sh NEW_OUTPUT_DIRECTORY [--ioh] [--backend docker|native|auto]' >&2
    exit 2
fi
output="$1"
shift
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m flowfield.problems.level3_discrete_combinatorial.p004.cli hydraulic-smoke --output "$output" "$@"
