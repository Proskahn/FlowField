#!/usr/bin/env bash
# Works from any directory; an explicit new output path is required.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# -lt 1 ]]; then
    echo 'Usage: smoke_p000.sh NEW_OUTPUT_DIRECTORY [--ioh] [--backend docker|native|auto]' >&2
    exit 2
fi
output="$1"
shift
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m flowfield smoke --output "$output" "$@"
