#!/usr/bin/env bash
# Use native OpenCFD v2312, or the installed ARM64-compatible Docker image.
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
if [[ "${FLOWFIELD_NATIVE:-0}" == 1 ]]; then
    if [[ "${WM_PROJECT_VERSION:-}" != v2312 && "${WM_PROJECT_VERSION:-}" != 2312 ]]; then
        echo 'Source OpenCFD OpenFOAM v2312 before using FLOWFIELD_NATIVE=1.' >&2
        exit 2
    fi
    exec "${args[@]}"
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
