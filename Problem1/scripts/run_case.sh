#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# -lt 1 ]]; then
    echo 'Usage: scripts/run_case.sh CASE [check|run|parallel|resume] [NPROCS]' >&2
    exit 2
fi
if [[ -d "$1" ]]; then
    case_dir="$(cd "$1" && pwd -P)"
elif [[ "$1" != /* && -d "$root/$1" ]]; then
    case_dir="$(cd "$root/$1" && pwd -P)"
else
    echo "Case directory not found: $1 (also checked relative to $root)" >&2
    exit 2
fi
action="${2:-run}"
foam="$root/scripts/foam.sh"
case "$case_dir" in "$root"/*) ;; *) echo 'Case must be within the workspace.' >&2; exit 2;; esac
case "$action" in check|run|parallel|resume) ;; *) echo "Unknown action: $action" >&2; exit 2;; esac
run_solver() {
    echo "Running $case_dir"
    echo 'Live output: Time = simulation time (s); deltaT = time-step size (s).'
    echo "Full log: $case_dir/log.interFoam"
    # pipefail preserves solver failures even when tee succeeds.
    "$foam" "$@" 2>&1 | tee -a "$case_dir/log.interFoam"
}
echo "Checking mesh: $case_dir"
if ! "$foam" checkMesh -case "$case_dir" -allTopology -allGeometry > "$case_dir/log.checkMesh" 2>&1; then
    tail -60 "$case_dir/log.checkMesh" >&2
    exit 1
fi
if ! grep -q 'Mesh OK' "$case_dir/log.checkMesh"; then
    tail -60 "$case_dir/log.checkMesh" >&2
    exit 1
fi
if [[ "$action" == check ]]; then echo "Mesh OK: $case_dir"; exit 0; fi
if [[ "$action" == parallel ]]; then
    nprocs="${3:-4}"
    [[ "$nprocs" =~ ^[1-9][0-9]*$ ]] || { echo 'NPROCS must be a positive integer' >&2; exit 2; }
    if [[ -d "$case_dir/processor0" ]]; then echo 'Existing processor data: use resume or a new case.' >&2; exit 2; fi
    "$foam" foamDictionary "$case_dir/system/decomposeParDict" -entry numberOfSubdomains -set "$nprocs" > "$case_dir/log.decomposeSettings" 2>&1
    echo "Decomposing mesh for $nprocs processes..."
    "$foam" decomposePar -case "$case_dir" > "$case_dir/log.decomposePar" 2>&1
    run_solver mpirun -np "$nprocs" interFoam -parallel -case "$case_dir"
    echo 'Reconstructing parallel results...'
    "$foam" reconstructPar -case "$case_dir" > "$case_dir/log.reconstructPar" 2>&1
elif [[ "$action" == resume && -d "$case_dir/processor0" ]]; then
    nprocs="${3:-4}"
    [[ "$nprocs" =~ ^[1-9][0-9]*$ ]] || { echo 'NPROCS must be a positive integer' >&2; exit 2; }
    run_solver mpirun -np "$nprocs" interFoam -parallel -case "$case_dir"
    echo 'Reconstructing parallel results...'
    "$foam" reconstructPar -case "$case_dir" > "$case_dir/log.reconstructPar" 2>&1
else
    if [[ -d "$case_dir/processor0" ]]; then echo 'Parallel case: use resume with the original processor count.' >&2; exit 2; fi
    run_solver interFoam -case "$case_dir"
fi
echo "Solver completed: $case_dir"
