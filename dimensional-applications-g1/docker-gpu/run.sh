#!/usr/bin/env bash
# Launch the G1 agentic blueprint inside the SAME GPU image built for Go2
# (go2-agentic-gpu:latest). The image already contains the unitree-g1-* blueprints
# because they ship in the same dimos package — only the .env (ROBOT_IP_G1) and
# the dimos subcommand differ.
#
# Usage:
#   docker-gpu/run.sh                  # dimos run unitree-g1-agentic
#   docker-gpu/run.sh humancli         # dimos humancli (Textual TUI)
#   docker-gpu/run.sh shell            # bash inside the container
#   docker-gpu/run.sh -- dimos run ... # pass-through
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Required keys: ROBOT_IP_G1, OPENAI_API_KEY" >&2
    exit 1
fi

# Pull ROBOT_IP_G1 from .env and re-export it as ROBOT_IP for the container.
set -a
. "$ENV_FILE"
set +a
if [ -z "${ROBOT_IP_G1:-}" ]; then
    echo "ERROR: ROBOT_IP_G1 is not set in $ENV_FILE." >&2
    exit 1
fi

IMAGE="go2-agentic-gpu:latest"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image '$IMAGE' not found. Build it from dimensional-applications-g1/docker-gpu/" >&2
    echo "       or load the shipped tar: gunzip -c go2-agentic-gpu.tar.gz | docker load" >&2
    exit 1
fi

COMMON_ARGS=(
    --rm -it
    --name go2-agentic-gpu-g1
    --network host
    --cap-add NET_ADMIN
    --gpus all
    --env-file "$ENV_FILE"
    -e ROBOT_IP="$ROBOT_IP_G1"
)

case "${1:-}" in
    shell|bash)
        exec docker run "${COMMON_ARGS[@]}" "$IMAGE" bash
        ;;
    humancli)
        exec docker run "${COMMON_ARGS[@]}" "$IMAGE" dimos humancli
        ;;
    --)
        shift
        exec docker run "${COMMON_ARGS[@]}" "$IMAGE" "$@"
        ;;
    "")
        exec docker run "${COMMON_ARGS[@]}" "$IMAGE" dimos run unitree-g1-agentic
        ;;
    *)
        exec docker run "${COMMON_ARGS[@]}" "$IMAGE" "$@"
        ;;
esac
