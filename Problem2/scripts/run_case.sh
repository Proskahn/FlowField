#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
target="${1:-cases/singleChannel}"
if [[ -d "$target" ]]; then
    case_dir="$(cd "$target" && pwd -P)"
elif [[ -d "$root/$target" ]]; then
    case_dir="$(cd "$root/$target" && pwd -P)"
else
    echo "Case not found: $target. Run scripts/generate_cases.py first." >&2
    exit 2
fi
if [[ ! -x "$root/bin/pemElectrolyzerFoam" ]]; then
    "$root/scripts/build.sh"
fi
export PATH="$root/bin:$PATH"
"$root/scripts/foam.sh" checkMesh -case "$case_dir" 2>&1 | tee "$case_dir/log.checkMesh"
if ! grep -q 'Mesh OK' "$case_dir/log.checkMesh"; then
    echo 'Mesh validation did not pass.' >&2
    exit 1
fi
# This solver currently uses one CPU. Its coupled charge graph is serial.
# Log names are timestamped so restarting never overwrites the previous run.
log="$case_dir/log.solver.$(date +%Y%m%d-%H%M%S)"
"$root/scripts/foam.sh" pemElectrolyzerFoam -case "$case_dir" 2>&1 | tee "$log"
python3 "$root/scripts/analyze.py" "$case_dir"
python3 "$root/scripts/verify_results.py" "$case_dir"
