#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
mkdir -p "$root/bin"
if [[ "${FLOWFIELD_NATIVE:-0}" == 1 ]]; then
    export FOAM_USER_APPBIN="$root/bin"
fi
"$root/scripts/foam.sh" wmake "$root/applications/pemElectrolyzerFoam" 2>&1 | tee "$root/log.build"
