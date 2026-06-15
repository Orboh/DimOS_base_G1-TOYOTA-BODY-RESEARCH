#!/usr/bin/env bash
# Boot wrapper for the teleimager image_server on the G1 NX.
#
# Activates the teleimager conda env and serves the head D435i over ZMQ
# (config REQ-REP :60000, frames PUB :55555). Used by g1-teleimager.service.
#
# Override the env name with TELEIMAGER_CONDA_ENV (default: teleimager_relobot).
set -euo pipefail

CONDA_ENV="${TELEIMAGER_CONDA_ENV:-teleimager_relobot}"

# Locate conda.sh (miniconda / miniforge / system) and activate the env.
for c in "$HOME/miniconda3/etc/profile.d/conda.sh" \
         "$HOME/miniforge3/etc/profile.d/conda.sh" \
         "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$c" ]; then
        # shellcheck disable=SC1090
        source "$c"
        break
    fi
done

conda activate "$CONDA_ENV"

# --rs enables the RealSense (head D435i). Extra args pass through.
exec teleimager-server --rs "$@"
