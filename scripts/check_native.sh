#!/usr/bin/env bash
# Run the numerical checker with Python, MPI and OpenFOAM loaded in that order.
# Example: bash scripts/check_native.sh --problem P008 --output runs/check-p008-002
# Optional: FLOWFIELD_OPENFOAM_BASHRC, FLOWFIELD_MPI_MODULE, FLOWFIELD_VENV.
set -o pipefail
if [[ $# == 0 || $1 == --help || $1 == -h ]]; then
    echo 'Usage: bash scripts/check_native.sh --problem P008 --output NEW_DIRECTORY [--study mesh numerics]'
    echo 'Loads the project .venv, mpi/openmpi-x86_64 (when modules are available), and OpenFOAM v2312.'
    echo 'Override locations with FLOWFIELD_VENV, FLOWFIELD_MPI_MODULE, FLOWFIELD_OPENFOAM_BASHRC.'
    exit 0
fi
flowfield_check_args=("$@")
set --  # OpenFOAM bashrc treats positional arguments as configuration overrides.
flowfield_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)" || exit 1
flowfield_venv="${FLOWFIELD_VENV:-$flowfield_root/.venv}"
flowfield_bashrc="${FLOWFIELD_OPENFOAM_BASHRC:-$HOME/OpenFOAM/OpenFOAM-v2312/etc/bashrc}"
if [[ ! -r "$flowfield_venv/bin/activate" || ! -x "$flowfield_venv/bin/python" ]]; then
    echo "Missing Python environment: $flowfield_venv. Create/install the project's .venv first." >&2
    exit 1
fi
if [[ ! -r "$flowfield_bashrc" ]]; then
    echo "Missing OpenFOAM bashrc: $flowfield_bashrc. Set FLOWFIELD_OPENFOAM_BASHRC to the installed v2312 file." >&2
    exit 1
fi
# Re-activation can restore an older PATH. It must precede MPI/OpenFOAM setup.
source "$flowfield_venv/bin/activate" || exit 1
if ! type module >/dev/null 2>&1; then
    for flowfield_module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [[ -r "$flowfield_module_init" ]]; then
            source "$flowfield_module_init" || exit 1
            break
        fi
    done
fi
if type module >/dev/null 2>&1; then
    flowfield_mpi_module="${FLOWFIELD_MPI_MODULE:-mpi/openmpi-x86_64}"
    # Refresh paths even if the module is marked loaded from an earlier shell.
    module unload "$flowfield_mpi_module" || exit 1
    module load "$flowfield_mpi_module" || exit 1
fi
source "$flowfield_bashrc" || exit 1
hash -r
if [[ ${WM_PROJECT_VERSION:-} != v2312 && ${WM_PROJECT_VERSION:-} != 2312 ]]; then
    echo "Expected OpenCFD OpenFOAM v2312; loaded: ${WM_PROJECT_VERSION:-none}. No cases started." >&2
    exit 1
fi
for flowfield_tool in checkMesh wmake; do
    if ! command -v "$flowfield_tool"; then
        echo "Missing $flowfield_tool after loading $flowfield_bashrc. No cases started." >&2
        exit 1
    fi
done
if ! checkMesh -help >/dev/null; then
    echo 'checkMesh exists but cannot execute. Check the loader/library error above. No cases started.' >&2
    exit 1
fi
cd "$flowfield_root" || exit 1
exec "$flowfield_venv/bin/python" -m flowfield.stability run "${flowfield_check_args[@]}" --backend native
