#!/usr/bin/env bash
# Send a natural-language command to the agentic blueprint running in the
# GPU container (started by docker-gpu/run.sh in another terminal).
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 \"<command>\"" >&2
    echo "Example: $0 \"前に進んで\"" >&2
    exit 1
fi

exec docker exec -i go2-agentic-gpu-g1 dimos agent-send "$*"
