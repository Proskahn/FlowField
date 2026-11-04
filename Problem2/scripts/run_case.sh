#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# -gt 3 ]]; then
    echo 'Usage: scripts/run_case.sh CASE [run|check|parallel|resume] [NPROCS]' >&2
    exit 2
fi
target="${1:-cases/singleChannel}"
action="${2:-run}"
case "$action" in run|check|parallel|resume) ;; *) echo "Unknown action: $action" >&2; exit 2;; esac
if [[ -d "$target" ]]; then
    case_dir="$(cd "$target" && pwd -P)"
elif [[ -d "$root/$target" ]]; then
    case_dir="$(cd "$root/$target" && pwd -P)"
else
    echo "Case not found: $target. Run scripts/generate_cases.py first." >&2
    exit 2
fi
case "$case_dir" in "$root"/*) ;; *) echo 'Case must be within Problem2.' >&2; exit 2;; esac
foam="$root/scripts/foam.sh"
shopt -s nullglob
processors=()
for path in "$case_dir"/processor[0-9]*; do
    [[ -d "$path" && "${path##*/}" =~ ^processor[0-9]+$ ]] && processors+=("$path")
done
for path in "$case_dir"/processors[0-9]*; do
    if [[ -d "$path" ]]; then
        echo 'This launcher requires uncollated processor0, processor1, ... directories.' >&2
        exit 2
    fi
done
existing="${#processors[@]}"
for ((i=0; i<existing; ++i)); do
    [[ -d "$case_dir/processor$i" ]] || { echo 'Incomplete processor directory sequence.' >&2; exit 2; }
done
nprocs="${3:-}"
if [[ -n "$nprocs" && ! "$nprocs" =~ ^[1-9][0-9]*$ ]]; then
    echo 'NPROCS must be a positive integer.' >&2; exit 2
fi
case "$action" in
    parallel)
        nprocs="${nprocs:-4}"
        ((nprocs>=2)) || { echo 'Use run for one process.' >&2; exit 2; }
        ((existing==0)) || { echo "Existing decomposition: use resume $existing, or generate a separate case." >&2; exit 2; }
        ;;
    resume)
        if ((existing>0)); then
            nprocs="${nprocs:-$existing}"
            ((nprocs==existing)) || { echo "Case has $existing partitions; resume requires $existing processes. Repartition before changing this count." >&2; exit 2; }
        else
            nprocs="${nprocs:-1}"
            ((nprocs==1)) || { echo 'No processor directories: use parallel N to create the decomposition.' >&2; exit 2; }
        fi
        ;;
    run)
        ((existing==0)) || { echo "Parallel case: use resume $existing." >&2; exit 2; }
        [[ -z "$nprocs" || "$nprocs" == 1 ]] || { echo 'Use parallel N for multiple processes.' >&2; exit 2; }
        nprocs=1
        ;;
    check)
        [[ -z "$nprocs" ]] || { echo 'check does not take a process count.' >&2; exit 2; }
        ;;
esac
if [[ "$action" == resume && "$existing" -gt 0 ]]; then
    # Every rank must restart the same complete checkpoint. An interrupted
    # write must not silently combine fields from different iterations.
    if ! python3 - "$case_dir" "$existing" <<'PY'
import math
from pathlib import Path
import sys

case, count = Path(sys.argv[1]), int(sys.argv[2])
latest = []
for rank in range(count):
    times = []
    for path in (case / f'processor{rank}').iterdir():
        try:
            value = float(path.name)
        except ValueError:
            continue
        if path.is_dir() and math.isfinite(value) and value >= 0:
            times.append((value, path))
    if not times:
        sys.exit(f'processor{rank} has no saved iteration; incomplete decomposition.')
    value, path = max(times, key=lambda item: item[0])
    missing = [name for name in ('U', 'p', 'T', 'Cgas', 'phiS', 'phiE', 'material')
               if not (path / name).is_file() and not (path / (name + '.gz')).is_file()]
    if missing:
        sys.exit(f'Incomplete checkpoint {path}: missing {", ".join(missing)}. Restore a complete checkpoint before resuming.')
    latest.append(value)
if len(set(latest)) != 1:
    sys.exit(f'Processor checkpoints disagree: {latest}. Restore a common checkpoint before resuming.')
PY
    then
        exit 2
    fi
fi
if [[ "$action" != check && ( ! -x "$root/bin/pemElectrolyzerFoam" ||
      "$root/applications/pemElectrolyzerFoam/pemElectrolyzerFoam.C" -nt "$root/bin/pemElectrolyzerFoam" ||
      "$root/applications/pemElectrolyzerFoam/ChargeNetwork.H" -nt "$root/bin/pemElectrolyzerFoam" ) ]]; then
    "$root/scripts/build.sh"
fi
export PATH="$root/bin:$PATH"
"$foam" checkMesh -case "$case_dir" 2>&1 | tee "$case_dir/log.checkMesh"
if ! grep -q 'Mesh OK' "$case_dir/log.checkMesh"; then
    echo 'Mesh validation did not pass.' >&2
    exit 1
fi
[[ "$action" != check ]] || exit 0
if [[ "$action" == parallel ]]; then
    if [[ ! -f "$case_dir/system/decomposeParDict" ]]; then
        cat > "$case_dir/system/decomposeParDict" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }
numberOfSubdomains 4;
method scotch;
EOF
    fi
    "$foam" foamDictionary "$case_dir/system/decomposeParDict" -entry numberOfSubdomains -set "$nprocs" > "$case_dir/log.decomposeSettings" 2>&1
    echo "Decomposing the latest saved iteration into $nprocs parts..."
    "$foam" decomposePar -case "$case_dir" -latestTime -fileHandler uncollated 2>&1 | tee "$case_dir/log.decomposePar"
fi
# Log names are timestamped so restarting never overwrites the previous run.
log="$case_dir/log.solver.$(date +%Y%m%d-%H%M%S)"
solver_command=(pemElectrolyzerFoam -case "$case_dir" -fileHandler uncollated)
if ((nprocs>1)); then
    solver_command=(mpirun -np "$nprocs" "${solver_command[@]}" -parallel)
fi
echo "Running the coupled model on $nprocs process(es); live output shows outer iterations."
solver_status=0
"$foam" "${solver_command[@]}" 2>&1 | tee "$log" || solver_status=$?
# Status 2 is a clean iteration-limit stop. Reconstruct that checkpoint too.
if [[ "$solver_status" != 0 && "$solver_status" != 2 ]]; then
    exit "$solver_status"
fi
if ((nprocs>1)); then
    echo 'Reconstructing the latest saved iteration...'
    "$foam" reconstructPar -case "$case_dir" -latestTime -fileHandler uncollated 2>&1 | tee "$case_dir/log.reconstructPar"
fi
if ((solver_status!=0)); then
    echo "Iteration limit reached. Extend endTime and use resume $nprocs to continue." >&2
    exit "$solver_status"
fi
python3 "$root/scripts/analyze.py" "$case_dir"
python3 "$root/scripts/verify_results.py" "$case_dir"
