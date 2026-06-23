#!/bin/bash
# One-shot launcher for the G1 IK->ACT okra-harvest pipeline (first stage, NO MCP).
#
#   bash start_okra_harvest.sh            # DRY-RUN (no motion) — click check + ACT action log
#   bash start_okra_harvest.sh --live     # LIVE: arm moves + right Dex1 closes (e-stop in hand)
#
# Flow: click okra in the viewer -> IkReachBridge reaches a pre-grasp pose ->
# on settle it fires reach_done -> ActBridge runs the okra ACT grasp for
# OKRA_GRASP_DURATION_S then stops. One blueprint, one coordinator on the laptop.
#
# Does everything in one go:
#   [1] Jetson NX: free the D435i + start the standalone cloud publisher (the same
#       /camera/* streams feed BOTH the click cloud and ACT's color input)
#   [2] Laptop: wired multicast route + loopback multicast + buffers (sudo ONCE)
#   [3] Pre-flight: confirm the cloud reaches this laptop's NIC
#   [4] Start the ACT inference service (ZMQ :5701) and wait until it is serving
#   [5] Auto-open the native dimos-viewer, then run the okra-harvest app (Ctrl-C stops all)
#
# === SITE CONFIG (override via env, or edit for your deployment) ===========
#   G1_NX        : Jetson NX ssh target (WIRED eth0 .164; WiFi drops mid-session).
#   ROBOT_NIC    : laptop NIC on the robot subnet (DDS for rt/arm_sdk+rt/dex1 + LCM).
#   LAPTOP_IP    : this laptop's IPv4 on ROBOT_NIC (multicast join + preflight).
#   G1_NX_PW     : Jetson ssh password (sshpass).
#   CYCLONEDDS_HOME : Unitree's no-shm cyclonedds 0.10.2 prefix (DDS interop).
#   ACT_VENV_PY  : python in the lerobot ACT venv that runs scripts/act_service.py.
#   OKRA_GRASP_DURATION_S : how long ACT drives the grasp after reach_done [s].
G1_NX="${G1_NX:-unitree@192.168.123.164}"
ROBOT_NIC="${ROBOT_NIC:-enp46s0}"
LAPTOP_IP="${LAPTOP_IP:-192.168.123.222}"
G1_NX_PW="${G1_NX_PW:-123}"
CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/home/sota/cyclonedds-noshm}"
ACT_VENV_PY="${ACT_VENV_PY:-$HOME/act-okura/.venv_act/bin/python}"
export OKRA_GRASP_DURATION_S="${OKRA_GRASP_DURATION_S:-4.0}"
# ACT model: 8-dim right-only, 2-camera (cam_high + cam_right_wrist) tree-right.
export ACT_REPO_ID="${ACT_REPO_ID:-sotata/act-okura-pick-tree-right-06162026}"
export ACT_DATASET_REPO="${ACT_DATASET_REPO:-sotata/okura-pick-tree-right-20260616}"
MCAST="239.255.76.67"; PORT="7667"
# Repo root is auto-derived from this script's location (…/dimos/robot/unitree/g1/examples).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
# ===========================================================================

set -u
LIVE="${1:-}"
export LCM_DEFAULT_URL="udpm://${MCAST}:${PORT}?ttl=1"

ACT_PID=""
cleanup() {
  [ -n "$ACT_PID" ] && kill "$ACT_PID" 2>/dev/null && echo "  stopped ACT service (pid $ACT_PID)"
}
trap cleanup EXIT INT TERM

echo "== [1/5] Jetson ($G1_NX): kicking detached camera publishers (head D435i + wrist UVC) =="
timeout 15 sshpass -p "$G1_NX_PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$G1_NX" \
  'setsid nohup bash ~/run_ik_camera.sh >/dev/null 2>&1 & echo "  head kicked (pid $!)"' \
  || echo "  WARNING: couldn't reach Jetson — check the robot / use the wired .164."
echo "  + wrist UVC publisher (run_wrist_camera.sh, /dev/video6 -> /camera/right_wrist_color)"
timeout 15 sshpass -p "$G1_NX_PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$G1_NX" \
  'setsid nohup bash ~/run_wrist_camera.sh >/dev/null 2>&1 & echo "  wrist kicked (pid $!)"' \
  || echo "  WARNING: couldn't start wrist publisher (re-plug the wrist USB if /dev/video6 missing)."
echo "  waiting ~24s for D435i release + publisher startup ..."
sleep 24
timeout 15 sshpass -p "$G1_NX_PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$G1_NX" \
  'p=$(pgrep -f ik_camera_standalone | head -1); echo "  publisher pid=${p:-DEAD}"; grep -m1 "publishing on" ~/ik_cam_standalone.log 2>/dev/null || tail -2 ~/ik_cam_standalone.log' \
  || echo "  (could not verify publisher)"

echo "== [2/5] Laptop: wired multicast route + loopback multicast + buffers (sudo) =="
sudo ip route replace 224.0.0.0/4 dev "$ROBOT_NIC"
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
  echo "*** LIVE MODE: arm WILL move via rt/arm_sdk and the right Dex1 WILL close. E-stop in hand. ***"
else
  unset IK_REACH_LIVE 2>/dev/null || true
  echo "  (DRY-RUN: no motion. Pass --live to drive the arm + gripper.)"
fi

LOGFILE="${OKRA_HARVEST_LOG:-/tmp/okra_harvest_app.log}"
echo "== [3/5] pre-flight: is the cloud reaching this laptop's NIC? (logged to $LOGFILE) =="
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

echo "== [4/5] starting ACT inference service (ZMQ :5701) =="
if [ ! -x "$ACT_VENV_PY" ]; then
  echo "  ERROR: ACT venv python not found at $ACT_VENV_PY (set ACT_VENV_PY). ActBridge will time out."
else
  # Free a stale ACT service holding :5701 (e.g. from a previous crashed run),
  # else the new one fails to bind (ZMQError: Address already in use).
  pkill -f "act_service.py --serve" 2>/dev/null
  fuser -k 5701/tcp 2>/dev/null
  sleep 1
  ( cd "$REPO" && "$ACT_VENV_PY" scripts/act_service.py --serve ) >/tmp/act_service.log 2>&1 &
  ACT_PID=$!
  echo "  act_service.py launched (pid $ACT_PID, log /tmp/act_service.log); waiting for 'serving on' ..."
  for _ in $(seq 1 60); do
    kill -0 "$ACT_PID" 2>/dev/null || { echo "  ERROR: ACT service died on startup — see /tmp/act_service.log"; break; }
    grep -qm1 "serving on tcp://127.0.0.1:5701" /tmp/act_service.log && { echo "  ACT service ready."; break; }
    sleep 1
  done
fi

echo "== [5/5] native dimos-viewer auto-opens in ~12s; starting okra-harvest app (Ctrl-C to stop) =="
( sleep 12; "$REPO/.venv/bin/dimos-viewer" \
    --connect rerun+http://127.0.0.1:9877/proxy --ws-url ws://127.0.0.1:3030/ws >/dev/null 2>&1 ) &
"$REPO/.venv/bin/dimos" run unitree-g1-okra-harvest 2>&1 | tee -a "$LOGFILE"
