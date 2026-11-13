#!/usr/bin/env bash
# Real 50 microsecond VOF execution/IOH test; not a detachment-time validation.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# -lt 1 ]]; then
    echo 'Usage: smoke_p005.sh NEW_OUTPUT_DIRECTORY [--ioh] [--backend docker|native|auto]' >&2
    exit 2
fi
output="$1"
shift
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m flowfield.problems.level1_small_continuous.p005.cli smoke --output "$output" "$@"
