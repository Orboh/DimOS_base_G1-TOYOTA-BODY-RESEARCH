#!/bin/bash
# 重力補償のみでの姿勢保持テスト起動スクリプト(IK/カメラ/グリッパー無し)。
#
#   bash start_gravity_hold_test.sh            # DRY-RUN(rt/arm_sdkへ書き込まない)
#   bash start_gravity_hold_test.sh --live     # LIVE: 別ターミナルの
#                                               #   scripts/gravity_hold_toggle.py で
#                                               #   'c' を押すと右腕がCOMPLIANTになる
#
# 起動直後は右腕を現在の実測姿勢でSTIFF保持するだけ(何もしなければ動かない)。
# 重力モデルURDFの切り替えは OKRA_GRAVITY_URDF(空 = g1.urdf既定/ダミーハンド170g、
# または dimos/robot/unitree/g1/g1_dex1_1_official.urdf で公式Dex1-1込み365g)。
#
# === SITE CONFIG (override via env) ===
ROBOT_NIC="${ROBOT_NIC:-enp46s0}"
CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/home/sota/cyclonedds-noshm}"
MCAST="239.255.76.67"; PORT="7667"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
# ======================================

set -u
LIVE="${1:-}"
export LCM_DEFAULT_URL="udpm://${MCAST}:${PORT}?ttl=1"

echo "== laptop: 有線マルチキャストルート + ループバックマルチキャスト + バッファ (sudo) =="
sudo ip route replace 224.0.0.0/4 dev "$ROBOT_NIC"
sudo ip link set lo multicast on
sudo sysctl -w net.core.rmem_max=67108864 net.core.rmem_default=67108864 >/dev/null
echo "  route -> $(ip route get "$MCAST" | head -1)"

cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }
export CYCLONEDDS_HOME
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"
export ROBOT_INTERFACE="$ROBOT_NIC"
export DIMOS_SKIP_COORDINATOR_RPC=1

if [ "$LIVE" = "--live" ]; then
  export IK_REACH_LIVE=1
  echo "*** LIVE: 'c'(別ターミナルの gravity_hold_toggle.py)で右腕がCOMPLIANTになります。"
  echo "*** 押す前に右腕を手で支え、E-stopをすぐ押せる状態にしてください。 ***"
else
  unset IK_REACH_LIVE 2>/dev/null || true
  echo "  (DRY-RUN: 腕は動きません。実際に駆動するには --live を付けてください。)"
fi

echo "  重力モデルURDF     : ${OKRA_GRAVITY_URDF:-<既定 g1.urdf, ダミーハンド170g>}"
echo "  gravity_tau_scale  : ${OKRA_GRAVITY_TAU_SCALE:-1.0}"
echo "  compliant_kp_ramp_s: ${OKRA_COMPLIANT_KP_RAMP_S:-1.5}"
echo ""
echo "== 起動: unitree-g1-gravity-hold-test (Ctrl+C で weight->0 に安全ランプダウンして停止) =="
echo "   別ターミナルで: .venv/bin/python scripts/gravity_hold_toggle.py"
exec "$REPO/.venv/bin/dimos" run unitree-g1-gravity-hold-test
