#!/bin/bash
# 頭部D435i クリック→IKリーチを LIVE 起動 + ビューアを開く。
# **2026-09-07 の胸ZED実験（oda/zed_up.sh）の対照実験用。カメラ以外は全部同じ値にしてある。**
#
#   oda/head_up.sh                     # 既定（ホバー5cm・計測用）
#   HEAD_STANDOFF_M=0.0 oda/head_up.sh # 的まで行かせる（本番相当）
#   HEAD_GRAV_RIGHT=0 oda/head_up.sh   # 重力FFを切る
#
# === ZED版との違いは3つだけ（ここが対照実験の独立変数）===
#   1. カメラ: 胸ZED(PC直結・アプリ内プロセス) -> 頭D435i(NXで中継 -> LCMマルチキャスト)
#      => 起動前に Jetson NX の ~/run_ik_camera.sh を叩いて中継を起こす（下の[1/3]）
#   2. 外部パラメータ: ZEDは ZED_MOUNT_XYZRPY（メジャー実測の**未校正**値）を渡していた。
#      頭D435iは **URDF の d435_joint をそのまま使う**（ik_reach_bridge.py:79-80、
#      xyz=[0.0576,0.0175,0.4299] pitch=0.8308rad=47.6度うつむき）ので mount は渡さない。
#      **これが今回の対照実験で見たい差分**: 着地点のズレがカメラ外部パラメータ由来なら、
#      未校正のZEDと URDF由来の頭カメラで**ズレ方が変わる**はず。変わらなければ外部
#      パラメータは主犯ではない（＝腕側の要因）。
#   3. クリック座標系: ZEDはTFを出すのでビューアのクリックが**ボディ系**で届いた
#      (click_in_camera_body_frame=True)。NX中継はTFを出さないので**光学系**の生値が届く
#      = ブループリント既定のまま。**YOLOブリッジ側も YOLO_BRIDGE_BODY_FRAME を立てないこと**
#      （oda/head_yolo.sh は立てていない。ZED版の oda/zed_yolo.sh とはここが逆）。
#
# === 以下の軌道/ゲイン/重力FFは zed_up.sh と同一の既定値（2026-09-07 実績）===
#   HEAD_ABOVE_M  : 3段階アプローチ[m]（既定 0.001 = 目標高さで水平接近、降下は実質ゼロ）
#   HEAD_WS_X_MIN : workspaceゲートX下限[m]（既定 -0.02。素の +0.05 だと駐機姿勢から
#                   1歩目で弾かれ、黙って「下からすくう」旧動作にフォールバックする）
#   HEAD_PATH_STEP/CADENCE : 0.015/0.05 = 手先0.30 m/s
#   HEAD_STANDOFF_M : 0.05（ホバー計測用。触れない）
#   HEAD_KP_ARM/KD_ARM/KP_WRIST/KD_WRIST : 120/4.5/60/2.25
#   HEAD_GRAV_RIGHT/JOINTS : 既定 1 / 0（肩pitchのみ。9/3 §8①に従う）
#   HEAD_X/Y/Z_OFFSET : クリック点の平行移動[m]（既定 0。**生の誤差を測るので 0 のまま**）
# === カメラ（NX側の中継に渡る値。ZEDの深度モードに相当）===
#   HEAD_CAM_VOXEL / HEAD_CAM_TRUNC : 点群ボクセル[m]/深度打ち切り[m]（既定 0.002/0.8）
#                   D435iは640x480。ZED(1280x720)より画素が粗いのでYOLOのimgszは要再測定
#                   （oda/head_yolo.sh の注記）。
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
NIC="${ROBOT_NIC:-enp2s0}"
CDDS="${CYCLONEDDS_HOME:-$HOME/cyclonedds-noshm}"
G1_NX="${G1_NX:-unitree@192.168.123.164}"
G1_NX_PW="${G1_NX_PW:-123}"
MCAST="239.255.76.67"; PORT="7667"

if ps -eo pid,args | grep "[b]in/dimos run unitree-g1-okra-ik-only-grasp" | grep -qv "grasp-zed"; then
  echo "!! アプリが既に起動中です。二重起動は rt/arm_sdk を取り合って異音＋暴走します。"
  echo "   落とすなら: oda/head_arm_down.sh してから oda/head_down.sh"; exit 1
fi
if ps -eo pid,args | grep -q "[b]in/dimos run unitree-g1-okra-ik-only-grasp-zed"; then
  echo "!! ZED版が起動中です。対照実験なので同時に動かさないこと -> oda/zed_arm_down.sh && oda/zed_down.sh"; exit 1
fi
ip link show lo | head -1 | grep -q MULTICAST || {
  echo "!! lo に MULTICAST がありません。点群・クリックが全滅します。自分の端末で:"
  echo "   sudo ip link set lo multicast on"
  echo "   sudo ip route replace 224.0.0.0/4 dev $NIC"
  echo "   sudo sysctl -w net.core.rmem_max=67108864 net.core.rmem_default=67108864"; exit 1; }
ip route show | grep -q "^224.0.0.0/4 dev $NIC" || {
  echo "!! 224.0.0.0/4 の経路が $NIC にありません。**ZED版では要らなかったが頭カメラでは必須**"
  echo "   （点群がNXから有線で飛んでくるため）:"
  echo "   sudo ip route replace 224.0.0.0/4 dev $NIC"; exit 1; }

ABOVE="${HEAD_ABOVE_M:-0.001}"; FRONT="${HEAD_FRONT_M:-0.0}"
PSTEP="${HEAD_PATH_STEP:-0.015}"; PCAD="${HEAD_PATH_CADENCE:-0.05}"
WSXMIN="${HEAD_WS_X_MIN:--0.02}"; STANDOFF="${HEAD_STANDOFF_M:-0.05}"
X="${HEAD_X_OFFSET:-0.0}"; Y="${HEAD_Y_OFFSET:-0.0}"; Z="${HEAD_Z_OFFSET:-0.0}"
KPA="${HEAD_KP_ARM:-120}"; KDA="${HEAD_KD_ARM:-4.5}"; KPW="${HEAD_KP_WRIST:-60}"; KDW="${HEAD_KD_WRIST:-2.25}"
TIPN="${HEAD_TIP_LOG_N:-50}"; TRACKN="${HEAD_TRACK_LOG_N:-1250}"
GRAV="${HEAD_GRAV_RIGHT:-1}"; GJOINTS="${HEAD_GRAV_JOINTS:-0}"
GSCALE="${HEAD_GRAV_SCALE:-1.0}"; GLIMIT="${HEAD_GRAV_LIMIT:-12.0}"; GRAMP="${HEAD_GRAV_RAMP_S:-5.0}"
GURDF="${HEAD_GRAV_URDF:-dimos/robot/unitree/g1/g1_dex1_1_calibrated_550g.urdf}"
CAMVOXEL="${HEAD_CAM_VOXEL:-0.002}"; CAMTRUNC="${HEAD_CAM_TRUNC:-0.8}"

echo "== [1/3] Jetson NX ($G1_NX): 頭D435i中継を起こす (~/run_ik_camera.sh) =="
# デタッチして起こす（SSHが切れても中継が道連れにならないように。start_okra_ik_only_grasp.sh と同じ）
timeout 15 sshpass -p "$G1_NX_PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$G1_NX" \
  "IK_CAMERA_VOXEL=$CAMVOXEL IK_CAMERA_DEPTH_TRUNC=$CAMTRUNC setsid nohup bash ~/run_ik_camera.sh >/dev/null 2>&1 & echo '  起こした (pid '\$!')'" \
  || { echo "  !! NXに届きません。有線 .164 を確認（WiFi .0.211 はセッション中に切れる）"; exit 1; }
echo "  D435iの解放 + 中継の起動待ち ~24秒 ..."
sleep 24
timeout 15 sshpass -p "$G1_NX_PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$G1_NX" \
  'p=$(pgrep -f ik_camera_standalone | head -1); echo "  中継 pid=${p:-DEAD}"; grep -m1 "publishing on" ~/ik_cam_standalone.log 2>/dev/null || tail -2 ~/ik_cam_standalone.log' \
  || echo "  (中継の状態を確認できず)"

echo "== [2/3] 点群がこのPCまで届いているか (3秒) =="
# 参加インタフェースは有線NICのIPv4を明示（INADDR_ANYだと既定経路のWiFi側を選ぶ）
NICIP=$(ip -4 -br a show "$NIC" 2>/dev/null | awk '{print $3}' | cut -d/ -f1)
"$REPO/.venv/bin/python" - "$MCAST" "$PORT" "${NICIP:-0.0.0.0}" <<'PYEOF'
import socket, struct, sys, time
group, port, nicip = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 67108864)
    s.bind(("", port))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                 struct.pack("4s4s", socket.inet_aton(group), socket.inet_aton(nicip)))
    s.settimeout(3); n = big = 0; t = time.time()
    while time.time() - t < 3:
        d, _ = s.recvfrom(70000); n += 1
        if len(d) > 1400: big += 1
    print(f"  結果: {n} パケット (うち大 {big} = 点群断片)。>0 なら中継はこのPCに届いている")
except socket.timeout:
    print("  結果: **0パケット** — 中継/経路/rmem のどれかが死んでいる。head_status.sh で切り分け")
except Exception as e:
    print(f"  受信エラー: {e!r}")
PYEOF

mkdir -p oda/head_run_out
CFG="$REPO/oda/head_run_out/head_cfg.json"
if [ "$GRAV" = "1" ]; then
  printf '{"ikreachbridge": {"approach_offset_xyz": [%s, %s, %s], "ws_x": [%s, 0.65]}, "g1armsdkconnection": {"stiff_gravity_right_joint_indices": [%s]}}\n' \
    "$X" "$Y" "$Z" "$WSXMIN" "$GJOINTS" > "$CFG"
  GRAV_OPTS="-o g1armsdkconnection.stiff_gravity_compensation_right=True"
  GRAV_OPTS="$GRAV_OPTS -o g1armsdkconnection.stiff_gravity_tau_scale=$GSCALE"
  GRAV_OPTS="$GRAV_OPTS -o g1armsdkconnection.stiff_gravity_tau_limit_nm=$GLIMIT"
  GRAV_OPTS="$GRAV_OPTS -o g1armsdkconnection.stiff_gravity_ramp_s=$GRAMP"
  GRAV_OPTS="$GRAV_OPTS -o g1armsdkconnection.stiff_gravity_right_urdf_path=$GURDF"
  echo "*** 右腕の重力FF ON: 関節[$GJOINTS] 倍率$GSCALE 上限${GLIMIT}N*m ランプ${GRAMP}s ***"
else
  printf '{"ikreachbridge": {"approach_offset_xyz": [%s, %s, %s], "ws_x": [%s, 0.65]}}\n' \
    "$X" "$Y" "$Z" "$WSXMIN" > "$CFG"
  GRAV_OPTS=""
fi
echo "軌道: 3段階（真上へ上げる -> 目標z+${ABOVE}m の高さで水平移動 -> 真下に降りる）"
echo "軌跡: ${PSTEP}m刻み / ${PCAD}s間隔 -> 手先 $(echo "scale=2; $PSTEP/$PCAD" | bc -l) m/s   workspace X下限: ${WSXMIN}m"
echo "ホバー: ${STANDOFF}m 手前で停止   クリック点オフセット: X=${X} Y=${Y} Z=${Z}"
echo "位置ゲイン: kp_arm=${KPA} kd_arm=${KDA} kp_wrist=${KPW} kd_wrist=${KDW}"
echo "カメラ: 頭D435i（外部パラメータは **URDF既定** = mount上書きなし）"

LOG="oda/head_run_out/head_reach_$(date +%m%d_%H%M).log"
echo "== [3/3] アプリ起動 =="
CYCLONEDDS_HOME="$CDDS" LD_LIBRARY_PATH="$CDDS/lib" \
LCM_DEFAULT_URL="udpm://${MCAST}:${PORT}?ttl=1" \
DIMOS_SKIP_COORDINATOR_RPC=1 PYTEST_VERSION=1 \
ROBOT_INTERFACE="$NIC" DISPLAY="${DISPLAY:-:1}" \
IK_REACH_LIVE=1 OKRA_NO_GRIPPER="${OKRA_NO_GRIPPER:-1}" OKRA_CONFIRM_CLICK=1 \
OKRA_APPROACH_ABOVE_M="$ABOVE" OKRA_APPROACH_FRONT_M="$FRONT" \
OKRA_PATH_STEP_M="$PSTEP" OKRA_PATH_CADENCE_S="$PCAD" \
OKRA_NOACT_STANDOFF_M="$STANDOFF" \
OKRA_NOACT_KP_ARM="$KPA" OKRA_NOACT_KD_ARM="$KDA" \
nohup .venv/bin/dimos run unitree-g1-okra-ik-only-grasp -c "$CFG" \
  -o g1armsdkconnection.weight_ramp_s=0.05 \
  -o g1armsdkconnection.kp_wrist="$KPW" \
  -o g1armsdkconnection.kd_wrist="$KDW" \
  -o g1armsdkconnection.log_track_err_every_n="$TRACKN" \
  -o ikreachbridge.tip_log_every_n="$TIPN" \
  $GRAV_OPTS \
  > "$LOG" 2>&1 &

echo "起動中... log=$LOG   ** 起動直後は腕を注視（weightランプ中の暴れが過去に1回） **"
for i in $(seq 1 30); do
  grep -qa "weight=1.00" "$LOG" 2>/dev/null && break
  grep -qaE "Traceback|No LowState" "$LOG" 2>/dev/null && break
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
echo "次: oda/head_arm_down.sh で駐機姿勢へ -> ビューアで点群を同じ点に2回クリック"
echo "    クリック後は必ず: grep -aE 'leg done|leg stopped' $LOG   ('leg done'が3つ出れば設定が効いている)"
echo "log=$LOG"
