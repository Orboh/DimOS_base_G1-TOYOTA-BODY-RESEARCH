#!/bin/bash
# One-shot launcher for the G1 IK-reach + SCRIPTED grasp PoC (NO ACT).
#
#   bash start_okra_ik_only_grasp.sh            # DRY-RUN (no motion) — overlay view + click check
#   bash start_okra_ik_only_grasp.sh --live     # LIVE: arm moves + gripper closes (e-stop in hand)
#
# Same operational glue as dimos/robot/unitree/g1/examples/start_ik_reach.sh
# (Jetson camera kick, laptop network prep, viewer) — identical infrastructure
# since it's the same physical camera/robot/laptop setup. The blueprint itself
# lives at dimos/robot/unitree/g1/blueprints/manipulation/
# unitree_g1_okra_ik_only_grasp.py and is registered in
# dimos/robot/all_blueprints.py as unitree-g1-okra-ik-only-grasp, so the final
# line below runs it the same way start_ik_reach.sh runs unitree-g1-ik-reach.
#
# Does everything in one go:
#   [1] Jetson NX: free the D435i + LCM route/buffers + start the standalone cloud
#       publisher (no dimos coordinator) — see ik_camera_standalone.py
#   [2] Laptop: wired multicast route + loopback multicast + buffers (sudo ONCE)
#   [3] Auto-opens the NATIVE dimos-viewer once the app is up (RGB image-plane +
#       RGB-colored point cloud overlaid in one 3D view; click the okra to reach)
#   [4] Runs oda/run_okra_ik_only_grasp.py in this terminal (Ctrl-C stops everything)
#
# === SITE CONFIG (override via env, or edit for your deployment) ===========
#   G1_NX        : Jetson NX ssh target. Use the WIRED eth0 (.164) — the WiFi
#                  wlan0 (.0.211) drops mid-session and the bring-up then fails.
#   ROBOT_NIC    : laptop NIC on the robot subnet (DDS for rt/arm_sdk + LCM).
#   LAPTOP_IP    : this laptop's IPv4 on ROBOT_NIC (multicast join + preflight).
#   G1_NX_PW     : Jetson ssh password (sshpass).
#   CYCLONEDDS_HOME : Unitree's no-shm cyclonedds 0.10.2 prefix (DDS interop).
G1_NX="${G1_NX:-unitree@192.168.123.164}"
ROBOT_NIC="${ROBOT_NIC:-enp46s0}"
LAPTOP_IP="${LAPTOP_IP:-192.168.123.222}"
G1_NX_PW="${G1_NX_PW:-123}"
CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/home/sota/cyclonedds-noshm}"
MCAST="239.255.76.67"; PORT="7667"
# Repo root is auto-derived from this script's location (…/oda, one level below root).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# ===========================================================================

set -u
LIVE="${1:-}"
export LCM_DEFAULT_URL="udpm://${MCAST}:${PORT}?ttl=1"

echo "== [1/4] Jetson ($G1_NX): kicking detached camera publisher (run_ik_camera.sh) =="
# Kick the Jetson-side bring-up DETACHED so an SSH timeout can't kill the launch.
timeout 15 sshpass -p "$G1_NX_PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$G1_NX" \
  'setsid nohup bash ~/run_ik_camera.sh >/dev/null 2>&1 & echo "  kicked (pid $!)"' \
  || echo "  WARNING: couldn't reach Jetson — check the robot / use the wired .164."
echo "  waiting ~24s for D435i release + publisher startup ..."
sleep 24
timeout 15 sshpass -p "$G1_NX_PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$G1_NX" \
  'p=$(pgrep -f ik_camera_standalone | head -1); echo "  publisher pid=${p:-DEAD}"; grep -m1 "publishing on" ~/ik_cam_standalone.log 2>/dev/null || tail -2 ~/ik_cam_standalone.log' \
  || echo "  (could not verify publisher)"

echo "== [2/4] Laptop: wired multicast route + loopback multicast + buffers (sudo) =="
sudo ip route replace 224.0.0.0/4 dev "$ROBOT_NIC"
# dimos' startup MulticastConfigurator (critical) needs lo to carry the MULTICAST
# flag for inter-module LCM. Set it here so the cached sudo cred is reused and the
# in-app check passes without a second prompt (resets on reboot).
sudo ip link set lo multicast on
sudo sysctl -w net.core.rmem_max=67108864 net.core.rmem_default=67108864 >/dev/null
echo "  route -> $(ip route get $MCAST | head -1)"

cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }
export CYCLONEDDS_HOME
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"
export PYTEST_VERSION=1
export ROBOT_INTERFACE="$ROBOT_NIC"
export DISPLAY="${DISPLAY:-:0}"
export DIMOS_SKIP_COORDINATOR_RPC=1
if [ "$LIVE" = "--live" ]; then
  export IK_REACH_LIVE=1
  echo "*** LIVE MODE: the arm WILL move and the gripper WILL close (scripted, no ACT). Keep the e-stop in hand, area clear. ***"
else
  unset IK_REACH_LIVE 2>/dev/null || true
  echo "  (DRY-RUN: no motion. Pass --live to drive the arm+gripper.)"
fi

LOGFILE="${IK_REACH_LOG:-/tmp/okra_ik_only_grasp_app.log}"
echo "== [3/4] pre-flight: is the cloud reaching this laptop's NIC? (logged to $LOGFILE) =="
"$REPO/.venv/bin/python" - "$LAPTOP_IP" "$MCAST" "$PORT" <<'PYEOF' 2>&1 | tee "$LOGFILE"
import socket, struct, sys, time
laptop_ip, group, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(f"[preflight] joining {group}:{port} on {laptop_ip}, 4s ...")
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 67108864)
    s.bind(("", port))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                 struct.pack("4s4s", socket.inet_aton(group), socket.inet_aton(laptop_ip)))
    s.settimeout(4); n = big = 0; t = time.time()
    while time.time() - t < 4:
        d, _ = s.recvfrom(70000); n += 1
        if len(d) > 1400: big += 1
    print(f"[preflight] RESULT: {n} pkts in 4s ({big} cloud-fragments). >0 => cloud reaches laptop.")
except socket.timeout:
    print("[preflight] RESULT: 0 pkts (TIMEOUT) -> cloud NOT reaching laptop NIC (route/rmem/publisher).")
except Exception as e:
    print(f"[preflight] recv error: {e!r}")
PYEOF

echo "== [4/4] native dimos-viewer auto-opens in ~12s; starting oda IK-only-grasp app (Ctrl-C to stop) =="
( sleep 12; "$REPO/.venv/bin/dimos-viewer" \
    --connect rerun+http://127.0.0.1:9877/proxy --ws-url ws://127.0.0.1:3030/ws >/dev/null 2>&1 ) &
"$REPO/.venv/bin/dimos" run unitree-g1-okra-ik-only-grasp 2>&1 | tee -a "$LOGFILE"
