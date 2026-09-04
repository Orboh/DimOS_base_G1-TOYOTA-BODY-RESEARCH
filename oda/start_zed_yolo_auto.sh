#!/bin/bash
# One-shot launcher for the ZED x YOLO x IK AUTONOMOUS okra grasp (NO ACT).
#
#   bash start_zed_yolo_auto.sh            # DRY-RUN : arm frozen + bridge prints coords only
#   bash start_zed_yolo_auto.sh --hover    # arm reaches to 5cm SHORT (no grasp) + bridge LIVE
#   bash start_zed_yolo_auto.sh --live     # FULL: arm reaches + gripper closes + bridge LIVE
#
# Same operational glue as oda/start_okra_ik_only_grasp.sh, but for the ZED
# chest-camera pipeline (PC-direct, NO Jetson) and with the YOLO bridge replacing
# the human click. It does in one go:
#   [1] Laptop network prep: wired multicast route + loopback multicast + buffers (sudo ONCE)
#   [2] Start the ZED+IK app (unitree-g1-okra-ik-only-grasp-zed) in the background
#   [3] Auto-open the native dimos-viewer (~12s) so you can watch detection + reach
#   [4] Run oda/yolo_click_bridge.py in THIS terminal — press Enter to detect+harvest
#       one okra; q+Enter (or Ctrl-C) tears everything down cleanly.
#
# The three modes map to the staged escalation in RUN_ZED_YOLO_IK_AUTO.md:
#   default -> stage 1 (verify coords), --hover -> stage 2 (verify depth/offset),
#   --live -> stage 3 (real grasp). ALWAYS run DRY-RUN first on site.
#
# === SITE CONFIG (override via env, or edit for your deployment) ============
#   ROBOT_NIC       : laptop NIC on the robot subnet (DDS rt/arm_sdk + LCM).
#   CYCLONEDDS_HOME : Unitree's no-shm cyclonedds prefix (DDS interop).
#   Grasp tuning (see RUN_CHEATSHEET.md §4) is exported below with the field
#   values proven on 2026-07-23; override any of them via env.
ROBOT_NIC="${ROBOT_NIC:-enp46s0}"
CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/home/sota/cyclonedds-noshm}"
MCAST="239.255.76.67"; PORT="7667"
# Repo root is auto-derived from this script's location (…/oda, one level below root).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- grasp / arm tuning (override via env) ---
OKRA_NOACT_KP_ARM="${OKRA_NOACT_KP_ARM:-160}"
OKRA_NOACT_KD_ARM="${OKRA_NOACT_KD_ARM:-6.0}"
OKRA_GRIP_KP="${OKRA_GRIP_KP:-20}"
OKRA_NOACT_CLOSE_Q="${OKRA_NOACT_CLOSE_Q:-1.7}"   # 刃全閉q(≈1.8)-0.1。ゼロ点不安時は再測
OKRA_OPEN_Q="${OKRA_OPEN_Q:-3.7}"
OKRA_TIP_OFFSET_XYZ="${OKRA_TIP_OFFSET_XYZ:-0.25,-0.003,0}"
OKRA_APPROACH_ABOVE_M="${OKRA_APPROACH_ABOVE_M:-0.08}"  # 上からコの字。空にすると直行リーチ
# --- YOLO bridge tuning (ZED画角の調整値) ---
OKRA_YOLO_CONF="${OKRA_YOLO_CONF:-0.25}"
YOLO_BRIDGE_PX_RADIUS="${YOLO_BRIDGE_PX_RADIUS:-20}"
# ===========================================================================

set -u
MODE="${1:-}"
export LCM_DEFAULT_URL="udpm://${MCAST}:${PORT}?ttl=1"

case "$MODE" in
  --live)  ARM_LIVE=1; BRIDGE_LIVE=1; STANDOFF=""      ;;
  --hover) ARM_LIVE=1; BRIDGE_LIVE=1; STANDOFF="0.05"  ;;
  "")      ARM_LIVE=0; BRIDGE_LIVE=0; STANDOFF=""      ;;
  *) echo "usage: $0 [--hover|--live]   (no flag = DRY-RUN)"; exit 2 ;;
esac

echo "== [1/4] Laptop: wired multicast route + loopback multicast + buffers (sudo) =="
sudo ip route replace 224.0.0.0/4 dev "$ROBOT_NIC"
sudo ip link set lo multicast on
sudo sysctl -w net.core.rmem_max=67108864 net.core.rmem_default=67108864 >/dev/null
echo "  route -> $(ip route get $MCAST | head -1)"

# quick ZED presence check (the app fails hard without it; catch it early)
if ! lsusb 2>/dev/null | grep -qi stereolabs; then
  echo "  WARNING: no Stereolabs ZED on USB. Plug into USB3 port 4-4 (4-1 is bad), re-seat if only HID shows."
fi

cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }
export CYCLONEDDS_HOME
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"
export PYTEST_VERSION=1
export ROBOT_INTERFACE="$ROBOT_NIC"
export DISPLAY="${DISPLAY:-:0}"
export DIMOS_SKIP_COORDINATOR_RPC=1
export OKRA_NOACT_KP_ARM OKRA_NOACT_KD_ARM OKRA_GRIP_KP OKRA_NOACT_CLOSE_Q \
       OKRA_OPEN_Q OKRA_TIP_OFFSET_XYZ
export OKRA_NOACT_GRIP_LIVE=1
[ -n "$OKRA_APPROACH_ABOVE_M" ] && export OKRA_APPROACH_ABOVE_M
[ -n "$STANDOFF" ] && export OKRA_NOACT_STANDOFF_M="$STANDOFF"
if [ "$ARM_LIVE" = "1" ]; then
  export IK_REACH_LIVE=1
  if [ -n "$STANDOFF" ]; then
    echo "*** HOVER MODE: arm reaches to ${STANDOFF}m SHORT (no grasp). e-stop in hand, area clear. ***"
  else
    echo "*** LIVE MODE: the arm WILL move and the gripper WILL close. Keep the e-stop (L2+B) in hand, area clear. ***"
  fi
else
  unset IK_REACH_LIVE 2>/dev/null || true
  echo "  (DRY-RUN: arm frozen, bridge prints coords only. Pass --hover then --live to escalate.)"
fi

LOGFILE="${ZED_YOLO_LOG:-/tmp/zed_yolo_auto_app.log}"
VIEWER_PID=""; APP_PGID=""
cleanup() {
  echo ""
  echo "== stopping: SIGINT -> ZED+IK app, closing viewer =="
  [ -n "$VIEWER_PID" ] && kill "$VIEWER_PID" 2>/dev/null
  if [ -n "$APP_PGID" ]; then
    kill -INT -"$APP_PGID" 2>/dev/null
    # wait up to 8s for a clean arm hand-back (G1ArmSdkConnection disconnected)
    for _ in $(seq 1 16); do kill -0 -"$APP_PGID" 2>/dev/null || break; sleep 0.5; done
  fi
  if grep -q "G1ArmSdkConnection disconnected" "$LOGFILE" 2>/dev/null; then
    echo "  arm handed back (disconnected). safe to power off."
  else
    echo "  WARNING: no 'disconnected' in log. If the gripper is stuck: L2+B, or"
    echo "    DEX1_NIC=$ROBOT_NIC DEX1_OPEN_Q=$OKRA_OPEN_Q .venv/bin/python oda/gripper_open.py"
  fi
}
trap cleanup EXIT INT TERM

echo "== [2/4] starting ZED+IK app in background (log: $LOGFILE) =="
setsid "$REPO/.venv/bin/dimos" run unitree-g1-okra-ik-only-grasp-zed >"$LOGFILE" 2>&1 &
APP_PGID="$(ps -o pgid= "$!" | tr -d ' ')"
echo "  app pgid=$APP_PGID — waiting ~20s for ZED init + camera topics ..."
sleep 20

echo "== [3/4] native dimos-viewer auto-opens (watch detection + reach) =="
( "$REPO/.venv/bin/dimos-viewer" \
    --connect rerun+http://127.0.0.1:9877/proxy --ws-url ws://127.0.0.1:3030/ws >/dev/null 2>&1 ) &
VIEWER_PID=$!

echo "== [4/4] YOLO bridge (this terminal): Enter=detect+harvest 1 okra / q+Enter=stop =="
[ "$BRIDGE_LIVE" = "1" ] && export YOLO_BRIDGE_LIVE=1
YOLO_BRIDGE_BODY_FRAME=1 \
OKRA_YOLO_CONF="$OKRA_YOLO_CONF" \
YOLO_BRIDGE_PX_RADIUS="$YOLO_BRIDGE_PX_RADIUS" \
"$REPO/.venv/bin/python" oda/yolo_click_bridge.py
