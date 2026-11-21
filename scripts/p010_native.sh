#!/usr/bin/env bash
# Load Python before MPI/OpenFOAM; stop before creating cases on setup errors.
set -o pipefail
flowfield_args=("$@")
set --
flowfield_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)" || exit 1
flowfield_venv="${FLOWFIELD_VENV:-$flowfield_root/.venv}"
flowfield_bashrc="${FLOWFIELD_OPENFOAM_BASHRC:-$HOME/OpenFOAM/OpenFOAM-v2312/etc/bashrc}"
if [[ ! -r "$flowfield_venv/bin/activate" ]]; then
    echo "Missing Python environment: $flowfield_venv" >&2
    exit 1
fi
source "$flowfield_venv/bin/activate" || exit 1
if [[ ${flowfield_args[0]:-} == reference-audit || ${flowfield_args[0]:-} == generate || ${flowfield_args[0]:-} == extract || ${flowfield_args[0]:-} == --help ]]; then
    cd "$flowfield_root" || exit 1
    exec "$flowfield_venv/bin/python" -m flowfield.problems.level1_small_continuous.p010.cli "${flowfield_args[@]}"
fi
if [[ ! -r "$flowfield_bashrc" ]]; then
    echo "Missing OpenFOAM bashrc: $flowfield_bashrc. Set FLOWFIELD_OPENFOAM_BASHRC to the installed v2312 path." >&2
    exit 1
fi
if ! type module >/dev/null 2>&1; then
    for flowfield_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [[ -r "$flowfield_init" ]]; then source "$flowfield_init" || exit 1; break; fi
    done
fi
if type module >/dev/null 2>&1; then
    flowfield_mpi="${FLOWFIELD_MPI_MODULE:-mpi/openmpi-x86_64}"
    module unload "$flowfield_mpi" || exit 1
    module load "$flowfield_mpi" || exit 1
fi
source "$flowfield_bashrc" || exit 1
hash -r
if [[ ${WM_PROJECT_VERSION:-} != v2312 && ${WM_PROJECT_VERSION:-} != 2312 ]]; then
    echo 'Expected OpenCFD OpenFOAM v2312; no cases started.' >&2
    exit 1
fi
for flowfield_command in checkMesh wmake; do
    command -v "$flowfield_command" >/dev/null || { echo "Missing $flowfield_command; no cases started." >&2; exit 1; }
done
checkMesh -help >/dev/null || exit 1
cd "$flowfield_root" || exit 1
exec "$flowfield_venv/bin/python" -m flowfield.problems.level1_small_continuous.p010.cli "${flowfield_args[@]}" --backend native
