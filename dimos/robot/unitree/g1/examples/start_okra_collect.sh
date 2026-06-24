#!/bin/bash
# One-shot launcher for KINESTHETIC ACT data collection (design-plan §10).
#
#   bash start_okra_collect.sh            # DRY-RUN (no motion) — wiring/click check
#   bash start_okra_collect.sh --live     # LIVE: arm moves; on reach_done the RIGHT arm
#                                         #       goes compliant (hand-guide it). E-stop!
#
# Flow: click okra -> IkReachBridge reaches pre-grasp (stiff) -> on settle reach_done ->
# G1ArmSdkConnection(collection_mode) makes the RIGHT arm compliant (kp->0 + gravity tau)
# so the operator hand-guides the grasp (left arm + waist stay stiff). Next click re-stiffens.
#
# This script starts the COLLECT BLUEPRINT (owns rt/arm_sdk) AND auto-starts the data
# RECORDER (read-only LCM) in this terminal. All artifacts live under ONE base dir
# (OKRA_COLLECT_DIR, default ~/okra_collect): raw/ captures, app log. No ACT service.
#
# === SITE CONFIG (override via env) ===
G1_NX="${G1_NX:-unitree@192.168.123.164}"
ROBOT_NIC="${ROBOT_NIC:-enp46s0}"
LAPTOP_IP="${LAPTOP_IP:-192.168.123.222}"
G1_NX_PW="${G1_NX_PW:-123}"
CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/home/sota/cyclonedds-noshm}"
MCAST="239.255.76.67"; PORT="7667"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
# ======================================

set -u
LIVE="${1:-}"
export LCM_DEFAULT_URL="udpm://${MCAST}:${PORT}?ttl=1"
# All collection artifacts under ONE base dir.
OKRA_COLLECT_DIR="${OKRA_COLLECT_DIR:-$HOME/okra_collect}"
RAW_OUT="$OKRA_COLLECT_DIR/raw"
# CLEAR every run (fresh dataset each session): wipe raw + derived lerobot/view so
# numbering starts at episode_000 with no stale frames. Set OKRA_KEEP_RAW=1 to append instead.
if [ "${OKRA_KEEP_RAW:-0}" = "1" ]; then
  echo "  OKRA_KEEP_RAW=1: keeping existing raw (new episodes append)."
else
  echo "  clearing previous collection ($OKRA_COLLECT_DIR: raw/ lerobot/ view/) — fresh session."
  rm -rf "$RAW_OUT" "$OKRA_COLLECT_DIR/lerobot" "$OKRA_COLLECT_DIR/view"
fi
mkdir -p "$RAW_OUT"

COLLECT_PID=""; VIEWER_PID=""
cleanup() {
  [ -n "$VIEWER_PID" ] && kill "$VIEWER_PID" 2>/dev/null
  [ -n "$COLLECT_PID" ] && kill "$COLLECT_PID" 2>/dev/null && echo "  stopped collect app (pid $COLLECT_PID)"
}
trap cleanup EXIT INT TERM

echo "== [1/4] Jetson ($G1_NX): kicking camera publishers (head D435i + wrist UVC) =="
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
  'p=$(pgrep -f ik_camera_standalone | head -1); echo "  head pid=${p:-DEAD}"; w=$(pgrep -f wrist_camera_standalone | head -1); echo "  wrist pid=${w:-DEAD}"' \
  || echo "  (could not verify publishers)"

echo "== [2/4] Laptop: wired multicast route + loopback multicast + buffers (sudo) =="
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
  echo "*** LIVE: arm moves; on reach_done the RIGHT arm goes COMPLIANT. SUPPORT IT. E-stop in hand. ***"
else
  unset IK_REACH_LIVE 2>/dev/null || true
  echo "  (DRY-RUN: no motion. Pass --live to drive the arm.)"
fi

LOGFILE="$OKRA_COLLECT_DIR/okra_collect_app.log"
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

echo "== [4/4] starting okra-collect app (background, log $LOGFILE) + viewer + recorder (foreground) =="
# Collect app in the BACKGROUND so the recorder owns this terminal's keyboard ('s'/'q').
"$REPO/.venv/bin/dimos" run unitree-g1-okra-collect >>"$LOGFILE" 2>&1 &
COLLECT_PID=$!
( sleep 12; "$REPO/.venv/bin/dimos-viewer" \
    --connect rerun+http://127.0.0.1:9877/proxy --ws-url ws://127.0.0.1:3030/ws >/dev/null 2>&1 ) &
VIEWER_PID=$!
echo "  waiting for the collect app to come up ..."
for _ in $(seq 1 40); do
  kill -0 "$COLLECT_PID" 2>/dev/null || { echo "  ERROR: collect app exited — see $LOGFILE"; tail -5 "$LOGFILE"; exit 1; }
  grep -qm1 "G1ArmSdkConnection started" "$LOGFILE" && { echo "  collect app up."; break; }
  sleep 1
done
echo "================================================================"
echo "  RECORDER (this terminal): 's' = compliant + start/stop episode (one key), 'q' = quit."
echo "  Per episode: click okra in viewer -> arm REACHES & HOLDS at pre-grasp ->"
echo "               's' (arm compliant + record) -> hand-guide the grasp -> 's' stop. Next click re-stiffens."
echo "  Raw episodes -> $RAW_OUT   (auto-converted to lerobot on quit)"
echo "================================================================"
"$REPO/.venv/bin/python" scripts/okra_kinesthetic_capture.py --out "$RAW_OUT"

# On quit: ALWAYS (re)build the persisted LeRobot dataset from ALL raw episodes,
# so the lerobot-format data is always saved under the base dir (rebuilt from raw,
# which is the source of truth). Needs the ACT venv (lerobot).
ACT_VENV_PY="${ACT_VENV_PY:-$HOME/act-okura/.venv_act/bin/python}"
OKRA_REPO_ID="${OKRA_REPO_ID:-sotata/okura-kinesthetic-wrist-7d}"
if ls "$RAW_OUT"/episode_* >/dev/null 2>&1; then
  echo "== converting raw -> LeRobot (persisted under $OKRA_COLLECT_DIR/lerobot) =="
  if [ -x "$ACT_VENV_PY" ]; then
    "$ACT_VENV_PY" "$REPO/scripts/okra_lerobot_writer.py" --raw "$RAW_OUT" --repo-id "$OKRA_REPO_ID" \
      || echo "  convert failed — raw is safe at $RAW_OUT; rerun okra_lerobot_writer.py manually."
  else
    echo "  ACT venv not found ($ACT_VENV_PY); raw saved at $RAW_OUT — convert manually:"
    echo "    \$ACT_VENV_PY scripts/okra_lerobot_writer.py --raw $RAW_OUT --repo-id $OKRA_REPO_ID"
  fi
fi
