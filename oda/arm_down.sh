#!/bin/bash
# 右腕をリーチ開始姿勢へ移す。2026-09-03: 垂れ下げ -> A案 -> 「作業高さ待機」へ再変更。
#
# 現在の姿勢: 手先 torso=[0.25,-0.25,0.114] = 前方25cm / 床上0.94m
#   目標(クリック点)の平均高さが床上0.94mなので、ここから1本直線で伸ばすと
#   経路がほぼ水平になり「下からすくう」動きが消える(肩の振り上げ超過0.0度)。
#   IK_FRONT_M=0.001 との併用が前提(2段階の折れ点を廃止し単一直線にする)。
#   注意: リーチ間もこの高さで待機する。腕は下がらない。
# 過去の候補(いずれも ARM_HOME_Q_RIGHT で切替可):
#   垂れ下げ  床上0.61m 前方0.00m : "0.8,-0.25,0.0,0.4,0.0,0.0,0.0"
#   A案       床上0.70m 前方0.24m : "0.4,-0.25,0.0,0.2,0.0,0.0,0.0"
#   作業高さ  床上0.94m 前方0.25m : "0.5914,0.0274,-0.0887,-0.4362,-0.0743,-0.3618,-0.1699"
# 2026-09-03 現在: A案 (上から降りる3段階 IK_ABOVE_M=0.10 との併用が前提)
#
# アプリ(dimos run unitree-g1-ik-reach)が
# 起動している必要がある(rt/arm_sdkへの橋渡し役がアプリ内のG1ArmSdkConnectionのため)。
# グリッパは触らない(ARM_HOME_OPEN_Q="")。
cd /home/techshare/DimOS_base_G1-TOYOTA-BODY-RESEARCH || exit 1
pgrep -f "dimos ru[n] unitree-g1-ik-reach" >/dev/null || {
  echo "!! アプリが起動していません。指令が届かないので先に oda/ik_up.sh を実行してください"; exit 1; }
CYCLONEDDS_HOME=/home/techshare/cyclonedds-noshm \
LD_LIBRARY_PATH=/home/techshare/cyclonedds-noshm/lib \
LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
PYTEST_VERSION=1 DIMOS_SKIP_COORDINATOR_RPC=1 \
ARM_HOME_Q_RIGHT="${ARM_HOME_Q_RIGHT:-0.4,-0.25,0.0,0.2,0.0,0.0,0.0}" \
ARM_HOME_OPEN_Q="" \
ARM_HOME_DURATION_S=4.0 \
.venv/bin/python -u oda/arm_home.py
