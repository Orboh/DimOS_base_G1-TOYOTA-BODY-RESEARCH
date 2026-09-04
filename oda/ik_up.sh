#!/bin/bash
# IKリーチアプリを LIVE で起動 + ビューアを開く。
#
# 使い方:
#   oda/ik_up.sh              # Zオフセット既定値(下記 IK_Z_OFFSET)で起動
#   IK_Z_OFFSET=0.02 oda/ik_up.sh   # クリック点を2cm上へずらして起動
#   IK_Y_OFFSET=0.02 IK_Z_OFFSET=0.06 oda/ik_up.sh   # 左2cm + 上6cm
#
# クリック点をtorso座標で平行移動する量[m] (torso: +X前 / +Y左 / +Z上):
#   IK_X_OFFSET : +が前  (既定 0)
#   IK_Y_OFFSET : +がG1から見て左 (既定 0)
#   IK_Z_OFFSET : +が上  (既定 0.010)
#
# IK_FRONT_M: 2段階アプローチ[m]。0=従来(終点1回publish→手先が下から弧を描く)。
#   >0 にすると (1)目標と同じ高さ・左右で手前この距離の点まで直線 → (2)まっすぐ+Xへ押し込む。
#   各区間は path_step_m(3.5cm)刻みの密なIKウェイポイント列としてストリームされ、
#   手先が直線を追従する。速度 ≒ path_step_m/path_cadence_s = 0.19 m/s。
#   例: IK_FRONT_M=0.10 oda/ik_up.sh
#
# IK_WS_X_MIN: workspaceゲートのX下限[m] (既定 -0.02。素の既定値は +0.05)。
#   経路の中間ウェイポイントにも同じゲートが効くため、駐機姿勢(手先x≈0.006)から
#   2段階アプローチを始めると +0.05 では1歩目で必ず弾かれ、従来動作にフォールバックする。
#   下げるとクリック目標の下限も緩む(体に近い点が受理されうる)ので、必要最小限に留める。
#
# 2段階アプローチの軌跡の細かさ/速度 (コード側の下限: step>=0.005m, cadence>=0.05s):
#   IK_PATH_STEP    : ウェイポイント間隔[m] (既定 0.015)。小さい=滑らか
#   IK_PATH_CADENCE : publish間隔[s]        (既定 0.05)。小さい=速い
#   手先速度 = IK_PATH_STEP / IK_PATH_CADENCE  (既定 0.30 m/s)
#
# IK_ABOVE_M: 上から降りる3段階アプローチ[m]。0=無効(既定)。
#   >0 で (1)現在のx,yのまま真上へ上げる (2)目標の真上まで高い位置を水平移動
#        (3)目標へ真下に降りる  ... の3段階になる。
#   下から近づかないので茎を擦らない。IK_FRONT_M より優先される(コード側で above が勝つ)。
#   例: IK_ABOVE_M=0.10 oda/ik_up.sh
#
# 位置ゲイン(重力たわみ対策)。既定は素の 80/3/40/1.5。
#   IK_KP_ARM / IK_KD_ARM / IK_KP_WRIST / IK_KD_WRIST
#   右腕には重力補償フラグが存在しない(stiff_gravity_compensation_left は左腕専用、
#   collection_mode は kp を 0 にする脱力モードなので追従は悪化する)ため、
#   たわみ対策の第一手は kp 引き上げ。2026-09-01 の実機検証で 80->160 / 40->80 により
#   追従誤差が 205mm -> 36.9mm (約1/5.6) に減少。kp/kd 比は保つこと(振動防止)。
#   例: IK_KP_ARM=160 IK_KD_ARM=6 IK_KP_WRIST=80 IK_KD_WRIST=3 oda/ik_up.sh
#
# ログ量の調整（大きいほど静か）:
#   IK_TIP_LOG_N   : 実測手先位置を何 motor_states ごとに出すか (既定 100 ≒ 2.2秒間隔)
#                    0 で完全にオフ。40回計測の本番では 50 くらいに戻すと着地点が細かく追える。
#   IK_TRACK_LOG_N : 追従差を何サイクルごとに出すか (既定 1250 = 250Hz なので 5秒間隔)
#
# 右腕の重力補償 (2026-09-04 追加、既定OFF):
#   IK_GRAV_RIGHT   : 1 で有効化。位置ゲインは保ったまま g(q) をtauに上乗せする
#   IK_GRAV_JOINTS  : 対象関節を 0..6 のカンマ区切り (既定 "0" = shoulder_pitch のみ)
#                     0=肩pitch 1=肩roll 2=肩yaw 3=肘 4=手首roll 5=手首pitch 6=手首yaw
#   IK_GRAV_LIMIT   : トルク上限[N*m] (既定 2.0)。安全のため必ず有限値
#   IK_GRAV_SCALE   : 倍率 (既定 0.5)。まず控えめに入れて効きを見る
#   IK_GRAV_RAMP_S  : 立ち上がり時間[s] (既定 5.0)
#   IK_GRAV_URDF    : 重力モデルURDF (既定 = Dex1-1校正550g版)
#   例: IK_GRAV_RIGHT=1 IK_GRAV_SCALE=0.5 IK_GRAV_LIMIT=2.0 oda/ik_up.sh
#   重力たわみで指令より下に落ちる分を、狙いを上げて相殺するための経験的トリム。
#   リストを渡す必要があるため -o では指定できず(pydanticが文字列を弾く)、
#   JSON設定ファイル経由で渡している。
cd /home/techshare/DimOS_base_G1-TOYOTA-BODY-RESEARCH || exit 1
if pgrep -f "dimos ru[n] unitree-g1-ik-reach" >/dev/null; then
  echo "!! アプリが既に起動中です。二重起動は rt/arm_sdk を取り合って暴走します。"
  echo "   落とすなら: oda/arm_down.sh してから oda/ik_down.sh"; exit 1
fi
X="${IK_X_OFFSET:-0.0}"; Y="${IK_Y_OFFSET:-0.0}"; Z="${IK_Z_OFFSET:-0.010}"
WSXMIN="${IK_WS_X_MIN:--0.02}"
PSTEP="${IK_PATH_STEP:-0.015}"; PCAD="${IK_PATH_CADENCE:-0.05}"
CFG="$(pwd)/oda/ik_reach_cfg.json"
if [ "$GRAV" = "1" ]; then
  printf '{"ikreachbridge": {"approach_offset_xyz": [%s, %s, %s], "ws_x": [%s, 0.65]}, "g1armsdkconnection": {"stiff_gravity_right_joint_indices": [%s]}}\n' \
    "$X" "$Y" "$Z" "$WSXMIN" "$GJOINTS" > "$CFG"
  echo "*** 右腕の重力補償 ON: 関節[${GJOINTS}] 倍率${GSCALE} 上限${GLIMIT}N*m ランプ${GRAMP}s ***"
  echo "    重力モデル: ${GURDF}"
  GRAV_OPTS="-o g1armsdkconnection.stiff_gravity_compensation_right=True"
  GRAV_OPTS="$GRAV_OPTS -o g1armsdkconnection.stiff_gravity_tau_scale=$GSCALE"
  GRAV_OPTS="$GRAV_OPTS -o g1armsdkconnection.stiff_gravity_tau_limit_nm=$GLIMIT"
  GRAV_OPTS="$GRAV_OPTS -o g1armsdkconnection.stiff_gravity_ramp_s=$GRAMP"
  GRAV_OPTS="$GRAV_OPTS -o g1armsdkconnection.stiff_gravity_right_urdf_path=$GURDF"
else
  printf '{"ikreachbridge": {"approach_offset_xyz": [%s, %s, %s], "ws_x": [%s, 0.65]}}\n' \
    "$X" "$Y" "$Z" "$WSXMIN" > "$CFG"
  GRAV_OPTS=""
fi
FRONT="${IK_FRONT_M:-0.0}"; ABOVE="${IK_ABOVE_M:-0.0}"
GRAV="${IK_GRAV_RIGHT:-0}"
GJOINTS="${IK_GRAV_JOINTS:-0}"; GLIMIT="${IK_GRAV_LIMIT:-2.0}"
GSCALE="${IK_GRAV_SCALE:-0.5}"; GRAMP="${IK_GRAV_RAMP_S:-5.0}"
GURDF="${IK_GRAV_URDF:-dimos/robot/unitree/g1/g1_dex1_1_calibrated_550g.urdf}"
TIPN="${IK_TIP_LOG_N:-100}"; TRACKN="${IK_TRACK_LOG_N:-1250}"
KPA="${IK_KP_ARM:-80}"; KDA="${IK_KD_ARM:-3}"; KPW="${IK_KP_WRIST:-40}"; KDW="${IK_KD_WRIST:-1.5}"
echo "クリック点オフセット: X=${X} (前+) / Y=${Y} (左+) / Z=${Z} (上+)  [m]"
if [ "$(echo "$ABOVE > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
  echo "上から降りる3段階: 真上へ上げる -> 目標の ${ABOVE} m 上を水平移動 -> 真下に降りる"
  echo "軌跡: ${PSTEP} m 刻み / ${PCAD} s 間隔 -> 手先速度 $(echo "scale=2; $PSTEP/$PCAD" | bc -l) m/s"
  echo "workspace X下限: ${WSXMIN} m"
elif [ "$(echo "$FRONT > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
  echo "2段階アプローチ: 手前 ${FRONT} m で高さ合わせ -> まっすぐ前へ押し込む"
  echo "workspace X下限: ${WSXMIN} m (経路1歩目を通すため。素の既定は +0.05)"
  echo "軌跡: ${PSTEP} m 刻み / ${PCAD} s 間隔 -> 手先速度 $(echo "scale=2; $PSTEP/$PCAD" | bc -l) m/s"
else
  echo "2段階アプローチ: OFF (従来の1回publish)"
fi
echo "位置ゲイン: kp_arm=${KPA} kd_arm=${KDA} kp_wrist=${KPW} kd_wrist=${KDW}"
LOG="oda/ik_accuracy_out/ik_reach_$(date +%m%d_%H%M).log"
CYCLONEDDS_HOME=/home/techshare/cyclonedds-noshm \
LD_LIBRARY_PATH=/home/techshare/cyclonedds-noshm/lib \
LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
PYTEST_VERSION=1 DIMOS_SKIP_COORDINATOR_RPC=1 \
ROBOT_INTERFACE=enp6s0 DISPLAY=:1 IK_REACH_LIVE=1 \
nohup .venv/bin/dimos run unitree-g1-ik-reach -c "$CFG" \
  -o ikreachbridge.standoff_m=0.0 \
  -o ikreachbridge.approach_front_m="$FRONT" \
  -o ikreachbridge.approach_above_m="$ABOVE" \
  -o ikreachbridge.path_step_m="$PSTEP" \
  -o ikreachbridge.path_cadence_s="$PCAD" \
  -o ikreachbridge.confirm_click=True \
  -o ikreachbridge.fire_reach_done=False \
  -o ikreachbridge.tip_log_every_n="$TIPN" \
  -o g1armsdkconnection.log_track_err_every_n="$TRACKN" \
  -o g1armsdkconnection.weight_ramp_s=0.05 \
  -o g1armsdkconnection.kp_arm="$KPA" \
  -o g1armsdkconnection.kd_arm="$KDA" \
  -o g1armsdkconnection.kp_wrist="$KPW" \
  -o g1armsdkconnection.kd_wrist="$KDW" \
  $GRAV_OPTS \
  > "$LOG" 2>&1 &
echo "起動中... log=$LOG"; sleep 25
grep -q 'LAUNCHING \*\*LIVE\*\*' "$LOG" && echo "  アプリ: LIVE起動OK" || { echo "  !! 起動失敗"; tail -25 "$LOG"; exit 1; }
DISPLAY=:1 nohup .venv/bin/dimos-viewer --connect rerun+http://127.0.0.1:9877/proxy --ws-url ws://127.0.0.1:3030/ws >/dev/null 2>&1 &
sleep 15
pgrep -f "bin/dimos-viewe[r]" >/dev/null && echo "  ビューア: 起動OK" || echo "  !! ビューア起動失敗"
echo "log=$LOG"
