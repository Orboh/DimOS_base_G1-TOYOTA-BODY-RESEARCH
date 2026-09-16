#!/bin/bash
# 右腕をリーチ開始姿勢（駐機姿勢）へ戻す。リーチの前後に毎回これ。
# oda/zed_arm_down.sh の頭カメラ版。**中身は同一**（arm_home.py は /g1/arm_target に
# 流すだけでカメラに依存しない）。違いは「起動チェックで見るアプリ名」だけ。
#
# 駐機姿勢の実績値（2026-09-03、oda/arm_down.sh より。ARM_HOME_Q_RIGHT で切替可）:
#   A案      床上0.70m 前方0.24m : "0.4,-0.25,0.0,0.2,0.0,0.0,0.0"   ← 既定
#   垂れ下げ 床上0.61m 前方0.00m : "0.8,-0.25,0.0,0.4,0.0,0.0,0.0"   ← 経路が通らないので非推奨
#   作業高さ 床上0.94m 前方0.25m : "0.5914,0.0274,-0.0887,-0.4362,-0.0743,-0.3618,-0.1699"
# 垂れ下げから始めると3段階アプローチの経路が箱ゲートで弾かれる（2026-09-07 実測: 指先x=+0.029、
# 素の ws_x 下限 +0.05 に 2.1cm 足りない）。A案なら指先x=+0.230で箱の中。
# **対照実験では ZED版と同じ A案を使うこと**（駐機姿勢を変えると独立変数が2つになる）。
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
ps -eo pid,args | grep "[b]in/dimos run unitree-g1-okra-ik-only-grasp" | grep -qv "grasp-zed" || {
  echo "!! アプリが起動していません。指令が届かないので先に oda/head_up.sh を実行してください"; exit 1; }
CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-$HOME/cyclonedds-noshm}" \
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-$HOME/cyclonedds-noshm/lib}" \
LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
PYTEST_VERSION=1 DIMOS_SKIP_COORDINATOR_RPC=1 \
ARM_HOME_Q_RIGHT="${ARM_HOME_Q_RIGHT:-0.4,-0.25,0.0,0.2,0.0,0.0,0.0}" \
ARM_HOME_OPEN_Q="" \
ARM_HOME_DURATION_S="${ARM_HOME_DURATION_S:-4.0}" \
ARM_HOME_RATE_HZ="${ARM_HOME_RATE_HZ:-20}" \
.venv/bin/python -u oda/arm_home.py
