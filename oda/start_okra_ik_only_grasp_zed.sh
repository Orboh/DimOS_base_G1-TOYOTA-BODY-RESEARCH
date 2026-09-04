#!/bin/bash
# One-shot launcher for the G1 IK-reach + SCRIPTED grasp PoC on the AGX Orin,
# chest ZED Mini (NO ACT). See dimos/robot/unitree/g1/blueprints/manipulation/
# unitree_g1_okra_ik_only_grasp_zed.py for the blueprint and its two
# unresolved blockers (chest-ZED hand-eye calibration, missing enable_disconnect).
#
#   bash oda/start_okra_ik_only_grasp_zed.sh            # DRY-RUN (no motion)
#   bash oda/start_okra_ik_only_grasp_zed.sh --live     # LIVE: arm moves + gripper closes
#
# Unlike the laptop + head-D435i original (oda/start_okra_ik_only_grasp.sh),
# there is NO Jetson-NX camera kick and NO laptop multicast-route prep here:
# the ZED Mini is a direct USB3 attachment to this machine, and the dimos
# coordinator runs on this same machine, so ZEDCamera is just another module
# in the blueprint's autoconnect (no external standalone publisher, no
# multicast route to another host).
#
# === SITE CONFIG (override via env, or edit for your deployment) ===========
#   ROBOT_NIC       : this machine's NIC on the robot subnet (DDS for
#                     rt/arm_sdk + LCM). Must be the wired G1-backpack
#                     Ethernet, NOT the WiFi AP/Tailscale interface.
#   IK_ZED_CAM_XYZ  : "x,y,z" [m], torso_link -> zed_link (URDF fixed-joint
#                     convention). REQUIRED for a correct LIVE reach -- see
#                     blocker 1 in the blueprint's docstring. Leave unset to
#                     stay obviously-uncalibrated (falls back to the WRONG
#                     D435i default; a loud error logs at LIVE launch).
#   IK_ZED_CAM_RPY  : "roll,pitch,yaw" [rad], paired with IK_ZED_CAM_XYZ.
# Repo root is auto-derived from this script's location (…/oda, one level below root).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_NIC="${ROBOT_NIC:-}"
# ===========================================================================

set -u
LIVE="${1:-}"

if [ -z "$ROBOT_NIC" ]; then
  echo "ERROR: ROBOT_NIC is not set. Pass the wired NIC on the G1 backpack subnet, e.g.:"
  echo "  ROBOT_NIC=eno1 bash oda/start_okra_ik_only_grasp_zed.sh"
  exit 1
fi

cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }
export ROBOT_INTERFACE="$ROBOT_NIC"
export DIMOS_SKIP_COORDINATOR_RPC=1

if [ "$LIVE" = "--live" ]; then
  export IK_REACH_LIVE=1
  echo "*** LIVE MODE: the arm WILL move (scripted, no ACT). Keep the e-stop in hand, area clear. ***"
  if [ -z "${IK_ZED_CAM_XYZ:-}" ] || [ -z "${IK_ZED_CAM_RPY:-}" ]; then
    echo "*** WARNING: IK_ZED_CAM_XYZ/IK_ZED_CAM_RPY are unset -- the reach will use the WRONG"
    echo "*** (head-D435i) hand-eye extrinsic. See AGX_ORIN_PORT_SPEC.md blocker 1. ***"
  fi
else
  unset IK_REACH_LIVE 2>/dev/null || true
  echo "  (DRY-RUN: no motion. Pass --live to drive the arm+gripper.)"
fi

LOGFILE="${IK_REACH_LOG:-/tmp/okra_ik_only_grasp_zed_app.log}"
echo "== dimos-viewer auto-opens in ~12s; starting unitree-g1-okra-ik-only-grasp-zed (Ctrl-C to stop) =="
( sleep 12; "$REPO/.venv/bin/dimos-viewer" \
    --connect rerun+http://127.0.0.1:9877/proxy --ws-url ws://127.0.0.1:3030/ws >/dev/null 2>&1 ) &
"$REPO/.venv/bin/dimos" run unitree-g1-okra-ik-only-grasp-zed 2>&1 | tee -a "$LOGFILE"
