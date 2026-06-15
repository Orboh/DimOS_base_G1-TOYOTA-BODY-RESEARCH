#!/usr/bin/env bash
# Install the teleimager image_server as the boot-time camera publisher on the
# NX, and DISABLE the old GEAR-SONIC publisher (g1-cam-publisher). The head
# D435i is single-holder, so the two cannot coexist.
#
# Run FROM THE LAPTOP; prompts for the NX password (ssh + sudo).
# Usage: scripts/install_nx_teleimager_service.sh [user@host]   (default: unitree@192.168.123.164)
#
# Prereq on the NX: the teleimager conda env (default name: teleimager_relobot)
# with `teleimager-server` on PATH, and cam_config_server.yaml configured for
# the head D435i (480x640). See docs / Stage A notes.
set -euo pipefail

NX="${1:-unitree@192.168.123.164}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> Copying wrapper + unit file to $NX"
scp "$HERE/g1-teleimager-run.sh" "$HERE/g1-teleimager.service" "$NX:/tmp/"

echo "==> Installing teleimager service + disabling the GEAR-SONIC publisher (sudo on the NX)"
ssh -t "$NX" '
  set -e
  install -m 755 /tmp/g1-teleimager-run.sh "$HOME/g1-teleimager-run.sh"
  sudo install -m 644 /tmp/g1-teleimager.service /etc/systemd/system/g1-teleimager.service
  sudo systemctl daemon-reload
  # D435i is single-holder: stop + disable the old GEAR-SONIC publisher first.
  sudo systemctl disable --now g1-cam-publisher 2>/dev/null || true
  sudo systemctl enable --now g1-teleimager
  sleep 4
  systemctl status g1-teleimager --no-pager -l | head -15
'

echo "==> Verifying teleimager ports from the laptop"
HOST="${NX#*@}"
ok=1
for port in 60000 55555; do
    if timeout 3 bash -c "echo > /dev/tcp/$HOST/$port" 2>/dev/null; then
        echo "  ✅ tcp://$HOST:$port open"
    else
        echo "  ❌ tcp://$HOST:$port closed"; ok=0
    fi
done
if [ "$ok" != 1 ]; then
    echo "Check logs: ssh $NX journalctl -u g1-teleimager -n 30 --no-pager"
    exit 1
fi
echo "✅ g1-teleimager is up (head camera served via teleimager)"
