#!/bin/bash
# 胸ZED クリック→IKリーチを LIVE 起動 + ビューアを開く。
# 9/3の oda/ik_up.sh のZED版（あちらは頭D435i + 別PC=/home/techshare 前提で、このPCでは動かない）。
# 既定値は 2026-09-03 の実機実績値と 2026-09-04 の推奨値。コードは変更しない（-c/-o だけで渡す）。
#
#   oda/zed_up.sh                     # 既定（ホバー5cm・計測用）
#   ZED_STANDOFF_M=0.0 oda/zed_up.sh  # 的まで行かせる（本番相当）
#   ZED_GRAV_RIGHT=1 oda/zed_up.sh    # 9/4の右腕重力FFを有効化
#
# === 軌道 ===
#   ZED_ABOVE_M   : 3段階アプローチ[m]（既定 0.001）。真上へ上げる→目標z+この値の高さで水平移動
#                   →真下に降りる。**0.001 = 目標高さで水平接近、降下は実質ゼロ（2026-09-04推奨）**。
#                   0.10 は 9/3 の実績値（目標の10cm上を通って降りる）。0 で無効。
#   ZED_FRONT_M   : 2段階アプローチ[m]（既定 0 = 無効）。ABOVE が優先。**伸ばすと肩が26〜30°
#                   振り上がってから振り戻す（9/3 実測、幾何的必然）ので基本使わない。**
#   ZED_PATH_STEP / ZED_PATH_CADENCE : 軌跡の刻み[m]/発行間隔[s]（既定 0.015/0.05 = 手先0.30 m/s）
#   ZED_WS_X_MIN  : workspaceゲートのX下限[m]（既定 -0.02。素の既定は +0.05）。
#                   **経路の中間点にも同じゲートが効くため、+0.05 のままだと駐機姿勢から
#                   1歩目で必ず弾かれ、黙って旧来の「下からすくう」動作にフォールバックする。**
#   ZED_STANDOFF_M: 目標の手前で止まる距離[m]（既定 0.05 = ホバー計測用。触れない）
#   ZED_X/Y/Z_OFFSET : クリック点をtorso系で平行移動[m]（既定 0。+X前/+Y左/+Z上）。
#                   9/3は Y=0.02 Z=0.06 を使っていたが、あれは kp160 前提のたわみ経験補正。
#                   ズレの実測をするときは 0 のままにすること（生の誤差を測るため）。
# === 位置ゲイン ===
#   ZED_KP_ARM/KD_ARM/KP_WRIST/KD_WRIST : 既定 120/4.5/60/2.25（9/3終了時点の値）。
#                   素は 80/3/40/1.5。kp160/6/80/3 は追従が最良（9/1: 誤差1/5.6）だが
#                   **戻り動作で異音**（9/3 §8②、未解明）。2026-09-07 に kp120 で静かなことを確認。
# === 右腕の重力FF（2026-09-04 実装、既定OFF）===
#   ZED_GRAV_RIGHT : 1 で有効（**既定1**。2026-09-07 実機で効果確認: 追従誤差 0.075→0.017 rad、
#                   到達残差 4.5cm→1.3cm、肩pitchのFFトルク -7.17 N*m。OFFだと 12cm 下で
#                   止まり「最後にすくい上げる」動きになる）。0 で無効
#   ZED_GRAV_JOINTS: 対象関節 0..6 のカンマ区切り（既定 0 = 肩pitchのみ。
#                   9/3 §8①「**shoulder_pitch の1軸だけから始めてください**」に従う。
#                   距離で7.5倍変動するのはこの軸だけ。ik_up.sh の既定は全7関節）
#   ZED_GRAV_SCALE / ZED_GRAV_LIMIT / ZED_GRAV_RAMP_S : 既定 1.0 / 12.0 N*m / 5.0 s
#   ZED_GRAV_URDF  : 重力モデルURDF（既定 = Dex1-1校正550g版）
# === カメラ ===
#   ZED_MOUNT      : torso_link -> ZED左レンズ "x,y,z,roll,pitch,yaw"（既定は7/23の目測値。**未校正**）
#   ZED_DEPTH_MODE / ZED_PC_VOXEL / ZED_DEPTH_TRUNC : 既定 PERFORMANCE / 0.004 / 0.8
#                   （QUALITY+ビューア+YOLO@1280 で 2026-09-07 18:04 にVRAM枯渇でPC電源断）
#                   （このPCはVRAM 8GB。NEURALはVRAM枯渇でフリーズ実績あり 2026-07-20）
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
NIC="${ROBOT_NIC:-enp2s0}"
CDDS="${CYCLONEDDS_HOME:-$HOME/cyclonedds-noshm}"

if ps -eo pid,args | grep -q "[b]in/dimos run unitree-g1-okra-ik-only-grasp-zed"; then
  echo "!! アプリが既に起動中です。二重起動は rt/arm_sdk を取り合って異音＋暴走します。"
  echo "   落とすなら: oda/zed_arm_down.sh してから oda/zed_down.sh"; exit 1
fi
ip link show lo | head -1 | grep -q MULTICAST || {
  echo "!! lo に MULTICAST がありません。点群・クリックが全滅します。自分の端末で:"
  echo "   sudo ip link set lo multicast on"
  echo "   sudo ip route replace 224.0.0.0/4 dev $NIC"
  echo "   sudo sysctl -w net.core.rmem_max=67108864 net.core.rmem_default=67108864"; exit 1; }

ABOVE="${ZED_ABOVE_M:-0.001}"; FRONT="${ZED_FRONT_M:-0.0}"
PSTEP="${ZED_PATH_STEP:-0.015}"; PCAD="${ZED_PATH_CADENCE:-0.05}"
WSXMIN="${ZED_WS_X_MIN:--0.02}"; STANDOFF="${ZED_STANDOFF_M:-0.05}"
X="${ZED_X_OFFSET:-0.0}"; Y="${ZED_Y_OFFSET:-0.0}"; Z="${ZED_Z_OFFSET:-0.0}"
KPA="${ZED_KP_ARM:-120}"; KDA="${ZED_KD_ARM:-4.5}"; KPW="${ZED_KP_WRIST:-60}"; KDW="${ZED_KD_WRIST:-2.25}"
MOUNT="${ZED_MOUNT:-0.159,0.06,0.348,0.0,-0.0209,0.0}"
DMODE="${ZED_DEPTH_MODE:-PERFORMANCE}"; VOXEL="${ZED_PC_VOXEL:-0.004}"; TRUNC="${ZED_DEPTH_TRUNC:-0.8}"
TIPN="${ZED_TIP_LOG_N:-50}"; TRACKN="${ZED_TRACK_LOG_N:-1250}"
GRAV="${ZED_GRAV_RIGHT:-1}"; GJOINTS="${ZED_GRAV_JOINTS:-0}"
GSCALE="${ZED_GRAV_SCALE:-1.0}"; GLIMIT="${ZED_GRAV_LIMIT:-12.0}"; GRAMP="${ZED_GRAV_RAMP_S:-5.0}"
GURDF="${ZED_GRAV_URDF:-dimos/robot/unitree/g1/g1_dex1_1_calibrated_550g.urdf}"

mkdir -p oda/zed_run_out
CFG="$REPO/oda/zed_run_out/zed_cfg.json"
if [ "$GRAV" = "1" ]; then
  printf '{"ikreachbridge": {"approach_offset_xyz": [%s, %s, %s], "ws_x": [%s, 0.65]}, "g1armsdkconnection": {"stiff_gravity_right_joint_indices": [%s]}}\n' \
    "$X" "$Y" "$Z" "$WSXMIN" "$GJOINTS" > "$CFG"
  GRAV_OPTS="-o g1armsdkconnection.stiff_gravity_compensation_right=True"
  GRAV_OPTS="$GRAV_OPTS -o g1armsdkconnection.stiff_gravity_tau_scale=$GSCALE"
  GRAV_OPTS="$GRAV_OPTS -o g1armsdkconnection.stiff_gravity_tau_limit_nm=$GLIMIT"
  GRAV_OPTS="$GRAV_OPTS -o g1armsdkconnection.stiff_gravity_ramp_s=$GRAMP"
  GRAV_OPTS="$GRAV_OPTS -o g1armsdkconnection.stiff_gravity_right_urdf_path=$GURDF"
  echo "*** 右腕の重力FF ON: 関節[$GJOINTS] 倍率$GSCALE 上限${GLIMIT}N*m ランプ${GRAMP}s ***"
  echo "    重力モデル: $GURDF"
else
  printf '{"ikreachbridge": {"approach_offset_xyz": [%s, %s, %s], "ws_x": [%s, 0.65]}}\n' \
    "$X" "$Y" "$Z" "$WSXMIN" > "$CFG"
  GRAV_OPTS=""
fi

if [ "$(echo "$ABOVE > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
  echo "軌道: 3段階（真上へ上げる -> 目標z+${ABOVE}m の高さで水平移動 -> 真下に降りる）"
elif [ "$(echo "$FRONT > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
  echo "軌道: 2段階（手前${FRONT}mで高さ合わせ -> まっすぐ前へ押し込む）※肩の振り上げに注意"
else
  echo "軌道: OFF（終点1回publish = 下からすくう。9/3以前の動作）"
fi
echo "軌跡: ${PSTEP}m刻み / ${PCAD}s間隔 -> 手先 $(echo "scale=2; $PSTEP/$PCAD" | bc -l) m/s   workspace X下限: ${WSXMIN}m"
echo "ホバー: ${STANDOFF}m 手前で停止   クリック点オフセット: X=${X} Y=${Y} Z=${Z}"
echo "位置ゲイン: kp_arm=${KPA} kd_arm=${KDA} kp_wrist=${KPW} kd_wrist=${KDW}"
echo "ZED: mount=[${MOUNT}] depth=${DMODE} voxel=${VOXEL} trunc=${TRUNC}"

LOG="oda/zed_run_out/zed_reach_$(date +%m%d_%H%M).log"
CYCLONEDDS_HOME="$CDDS" LD_LIBRARY_PATH="$CDDS/lib" \
LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
DIMOS_SKIP_COORDINATOR_RPC=1 PYTEST_VERSION=1 \
ROBOT_INTERFACE="$NIC" DISPLAY="${DISPLAY:-:1}" \
ZED_DEPTH_MODE="$DMODE" ZED_PC_VOXEL="$VOXEL" ZED_DEPTH_TRUNC="$TRUNC" \
ZED_MOUNT_XYZRPY="$MOUNT" \
IK_REACH_LIVE=1 OKRA_NO_GRIPPER="${OKRA_NO_GRIPPER:-1}" OKRA_CONFIRM_CLICK=1 \
OKRA_APPROACH_ABOVE_M="$ABOVE" OKRA_APPROACH_FRONT_M="$FRONT" \
OKRA_PATH_STEP_M="$PSTEP" OKRA_PATH_CADENCE_S="$PCAD" \
OKRA_NOACT_STANDOFF_M="$STANDOFF" \
OKRA_NOACT_KP_ARM="$KPA" OKRA_NOACT_KD_ARM="$KDA" \
nohup .venv/bin/dimos run unitree-g1-okra-ik-only-grasp-zed -c "$CFG" \
  -o g1armsdkconnection.weight_ramp_s=0.05 \
  -o g1armsdkconnection.kp_wrist="$KPW" \
  -o g1armsdkconnection.kd_wrist="$KDW" \
  -o g1armsdkconnection.log_track_err_every_n="$TRACKN" \
  -o ikreachbridge.tip_log_every_n="$TIPN" \
  $GRAV_OPTS \
  > "$LOG" 2>&1 &

echo "起動中... log=$LOG   ** 起動直後は腕を注視（weightランプ中の暴れが過去に1回） **"
for i in $(seq 1 30); do
  grep -qa "Camera successfully opened" "$LOG" 2>/dev/null && grep -qa "weight=1.00" "$LOG" 2>/dev/null && break
  grep -qaE "Traceback|No LowState|\[ZED\]\[ERROR\]" "$LOG" 2>/dev/null && break
  sleep 2
done
if grep -qa "LAUNCHING \*\*LIVE" "$LOG" && grep -qa "weight=1.00" "$LOG"; then
  echo "  アプリ: LIVE起動OK"
else
  echo "  !! 起動失敗"; tail -25 "$LOG"; exit 1
fi
DISPLAY="${DISPLAY:-:1}" nohup .venv/bin/dimos-viewer \
  --connect rerun+http://127.0.0.1:9877/proxy --ws-url ws://127.0.0.1:3030/ws >/dev/null 2>&1 &
sleep 12
ps -eo pid,args | grep -q "[b]in/dimos-viewer" && echo "  ビューア: 起動OK" || echo "  !! ビューア起動失敗"
echo
echo "次: oda/zed_arm_down.sh で駐機姿勢へ -> ビューアで点群を同じ点に2回クリック"
echo "    クリック後は必ず: grep -aE 'leg done|leg stopped' $LOG   ('leg done'が3つ出れば設定が効いている)"
echo "log=$LOG"
