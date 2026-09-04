#!/usr/bin/env bash
# Container entrypoint -- prepares LCM multicast on the (host-shared) loopback,
# verifies WebRTC reachability, sanity-checks GPU, then execs the command.
set -euo pipefail

: "${ROBOT_IP:?ROBOT_IP must be set (e.g. 192.168.123.161)}"

# LCM multicast on lo (idempotent — shared with host because of --network host).
if ! ip -d link show lo 2>/dev/null | grep -q '\<MULTICAST\>'; then
    ip link set lo multicast on || true
fi
if ! ip route show 224.0.0.0/4 2>/dev/null | grep -q '\<dev lo\>'; then
    ip route add 224.0.0.0/4 dev lo 2>/dev/null || true
fi

# GPU presence check — warn loudly if container was launched without --gpus.
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "WARN: no NVIDIA GPU visible in the container."
    echo "      Launch with '--gpus all' (run.sh does this by default)."
    echo "      onnxruntime will fall back to CPU and the agentic stack will be slow."
fi

# Pre-flight: warn (don't fail) if the robot's WebRTC port is unreachable.
if ! nc -z -w 2 "${ROBOT_IP}" 9991 2>/dev/null; then
    echo "WARN: ${ROBOT_IP}:9991 is unreachable — check LAN cable / G1 power."
    echo "      Continuing anyway; dimos will fail to connect if the port stays closed."
fi

cd /app/dimensional-applications-g1
exec "$@"
