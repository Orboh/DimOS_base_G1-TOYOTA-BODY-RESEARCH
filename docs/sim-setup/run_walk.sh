#!/bin/bash
# 公式 sim2sim 歩行の一括起動: MuJoCoシム + g1_ctrl(C++ deploy) + orchestrator。
# ゲームパッド不要（wireless_remote をファイル注入）・elastic band 自動制御。
set -u
REPO=/home/kota-ueda/Desktop/dimos-hackathon
PY=$REPO/.venv/bin/python
DEPLOY=/home/kota-ueda/Desktop/unitree_rl_lab/deploy/robots/g1_29dof
G1CTRL=$DEPLOY/build/g1_ctrl
ORT=/home/kota-ueda/Desktop/unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib
export LD_LIBRARY_PATH=$HOME/.local/lib:$ORT
LOG=${WALK_LOG_DIR:-/tmp/walk_logs}
mkdir -p "$LOG"
rm -f /tmp/sim_joy.bin /tmp/sim_band.txt "$LOG"/*.log

cleanup() {
  echo "[run_walk] cleanup"
  kill "${CTRL_PID:-}" "${SIM_PID:-}" 2>/dev/null
  sleep 1
  kill -9 "${CTRL_PID:-}" "${SIM_PID:-}" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "[run_walk] 1) MuJoCo シム起動"
$PY "$REPO/docs/sim-setup/sim_walk_run.py" > "$LOG/sim.log" 2>&1 &
SIM_PID=$!
sleep 7   # sim 起動 + lowstate publish 開始待ち

if ! kill -0 "$SIM_PID" 2>/dev/null; then
  echo "[run_walk] ❌ sim 起動失敗。log:"; tail -20 "$LOG/sim.log"; exit 1
fi

echo "[run_walk] 2) g1_ctrl (C++ deploy) 起動"
$G1CTRL --network lo < /dev/null > "$LOG/g1ctrl.log" 2>&1 &
CTRL_PID=$!
sleep 3

if ! kill -0 "$CTRL_PID" 2>/dev/null; then
  echo "[run_walk] ❌ g1_ctrl 起動失敗。log:"; tail -20 "$LOG/g1ctrl.log"; exit 1
fi

echo "[run_walk] 3) orchestrator（FixStand→Velocity→前進）"
WALK_BOOT_DELAY=${WALK_BOOT_DELAY:-2} WALK_SECS=${WALK_SECS:-12} WALK_VX=${WALK_VX:-0.4} \
  $PY "$REPO/docs/sim-setup/sim_walk_joy.py" > "$LOG/joy.log" 2>&1

sleep 2
echo "[run_walk] 完了。ログ: $LOG/{sim,g1ctrl,joy}.log"
