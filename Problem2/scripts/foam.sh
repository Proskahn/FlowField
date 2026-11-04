#!/usr/bin/env bash
# Prefer a loaded native OpenCFD v2312 environment; otherwise use Docker.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# -eq 0 ]]; then
    echo 'Usage: foam.sh COMMAND [ARGS...]' >&2
    exit 2
fi
# Resolve paths from the caller first, then from the relocated project folder.
# Accept paths from FlowField, Problem2, or any other working directory.
args=()
previous=""
for arg in "$@"; do
    resolved="$arg"
    if [[ "$arg" != /* && ( "$arg" == */* || "$arg" == . || "$arg" == .. || "$previous" == -case ) ]]; then
        if [[ -e "$arg" ]]; then
            resolved="$(cd "$(dirname "$arg")" && pwd -P)/$(basename "$arg")"
        elif [[ -e "$root/$arg" ]]; then
            resolved="$root/$arg"
        fi
    fi
    args+=("$resolved")
    previous="$arg"
done
runtime="${FLOWFIELD_NATIVE:-auto}"
case "$runtime" in
    auto)
        if [[ "${WM_PROJECT_VERSION:-}" == v2312 || "${WM_PROJECT_VERSION:-}" == 2312 ]]; then
            runtime=1
        else
            runtime=0
        fi
        ;;
    0|1) ;;
    *) echo 'FLOWFIELD_NATIVE must be auto (default), 1 (native), or 0 (Docker).' >&2; exit 2 ;;
esac
if [[ "$runtime" == 1 ]]; then
    if [[ "${WM_PROJECT_VERSION:-}" != v2312 && "${WM_PROJECT_VERSION:-}" != 2312 ]]; then
        echo 'Source OpenCFD OpenFOAM v2312 before using FLOWFIELD_NATIVE=1.' >&2
        exit 2
    fi
    export FOAM_USER_APPBIN="$root/bin"
    export PATH="$root/bin:$PATH"
    if ! command -v "${args[0]}" >/dev/null 2>&1; then
        echo "Native command '${args[0]}' is unavailable. Load the OpenFOAM and MPI environment in this terminal." >&2
        exit 2
    fi
    exec "${args[@]}"
fi
if ! command -v docker >/dev/null 2>&1; then
    echo 'Load MPI and source OpenCFD OpenFOAM v2312 in this terminal, then export FLOWFIELD_NATIVE=1. Docker is not available.' >&2
    exit 2
fi
# Mount Problem2 itself at /work, regardless of the caller's working directory.
container_args=()
for arg in "${args[@]}"; do
    case "$arg" in
        "$root") container_args+=(/work) ;;
        "$root"/*) container_args+=("/work/${arg#"$root/"}") ;;
        *) container_args+=("$arg") ;;
    esac
done
exec docker run --rm --init --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env USER=foam --workdir /work \
    --mount "type=bind,source=$root,target=/work" \
    --entrypoint /bin/bash "${OPENFOAM_IMAGE:-opencfd/openfoam-default:2312}" \
    -c 'flowfield_command=("$@"); set --; source /usr/lib/openfoam/openfoam2312/etc/bashrc || exit $?; export FOAM_USER_APPBIN=/work/bin; export PATH="/work/bin:$PATH"; exec "${flowfield_command[@]}"' bash "${container_args[@]}"
