#!/bin/bash
# YOLO検出 → /clicked_point（人間のクリックの代替）。アプリ稼働中に別ターミナルで実行。
#
#   oda/zed_yolo.sh              # DRY-RUN: 検出と3D点をログに出すだけ（腕は動かない）
#   oda/zed_yolo.sh --live       # 発火（Enterを押すたびに1回。腕が動く）
#   oda/zed_yolo.sh --live --once  # 1回だけ発火して終了
#
# 重みはリポに入らない設計（.gitignore:119 で model/ ごと除外）。このPCの実物:
#   ~/workspace/okurayolo/okra11n-seg-v5/weights/best.pt        (2026-08-18・最新, 既定)
#   ~/workspace/okurayolo/okradata/uedasmodel/okra11n-seg.pt    (2026-06-25・v5のベース)
# ZED固有の必須設定（2026-07-23 e80261d2）:
#   YOLO_BRIDGE_BODY_FRAME=1  ビューアのクリックがTF経由でボディ座標になるため
#   YOLO_BRIDGE_PX_RADIUS=20  5mmボクセル≒9px間隔に対して既定8pxでは点が拾えない
#   実は**カメラから35cm以上**に置く（ZEDの最短計測距離30cmの制約）
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
ps -eo pid,args | grep -q "[b]in/dimos run unitree-g1-okra-ik-only-grasp-zed" || {
  echo "!! アプリが起動していません。先に oda/zed_up.sh"; exit 1; }
MODEL="${OKRA_YOLO_MODEL:-$HOME/workspace/okurayolo/okra11n-seg-v5/weights/best.pt}"
[ -f "$MODEL" ] || { echo "!! モデルが無い: $MODEL"; exit 1; }
LIVE=0; ARGS=()
for a in "$@"; do case "$a" in --live) LIVE=1;; *) ARGS+=("$a");; esac; done
if [ "$LIVE" = "1" ]; then
  export YOLO_BRIDGE_LIVE=1     # ← $(...) で環境変数の前置は作れない（展開結果がコマンド名として扱われる）
  echo "*** LIVE: 検出したオクラへ腕が動きます。e-stopを手元に ***"
else
  unset YOLO_BRIDGE_LIVE
  echo "  (DRY-RUN: 検出と3D点をログに出すだけ。腕は動きません。--live で発火)"
fi
# VRAM チェック: このPCは8GB。ZED深度+点群+ビューア+YOLO推論を同時に載せると枯渇して
# **システム全体が落ちる**（2026-09-07 18:04 に実際に発生。前回bootログに
# NVRM: Out of memory [NV_ERR_NO_MEMORY]。2026-07-20 にも同じ症状の記録あり）。
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$FREE" ]; then
  echo "  GPU空きVRAM: ${FREE} MiB"
  if [ "$FREE" -lt 2500 ] 2>/dev/null; then
    echo "  !! 空きVRAMが少ないです。ビューアを閉じる / ZED_DEPTH_MODE=PERFORMANCE で起動し直す"
    echo "     / OKRA_YOLO_IMGSZ を下げる のいずれかを先に行ってください"
  fi
fi
echo "  model=$MODEL  conf=${OKRA_YOLO_CONF:-0.4}"
export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-$HOME/cyclonedds-noshm}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-$HOME/cyclonedds-noshm/lib}"
export LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1'
export PYTEST_VERSION=1 DIMOS_SKIP_COORDINATOR_RPC=1
export OKRA_YOLO_MODEL="$MODEL"
export OKRA_YOLO_CONF="${OKRA_YOLO_CONF:-0.4}"
export OKRA_YOLO_IMGSZ="${OKRA_YOLO_IMGSZ:-960}"
#   640=検出0件 / 960=conf0.57 / 1280=conf0.90 / 1920=conf0.76（2026-09-07 同一フレーム実測）。
#   1280 が最良だが VRAM 8GB では ZED深度+点群+ビューアと同時に載らず 18:04 に電源断。
#   検出が渋いときは ビューアを閉じてから OKRA_YOLO_IMGSZ=1280 にする。
export YOLO_BRIDGE_BODY_FRAME=1
export YOLO_BRIDGE_PX_RADIUS="${YOLO_BRIDGE_PX_RADIUS:-20}"
export YOLO_BRIDGE_DOUBLE_S="${YOLO_BRIDGE_DOUBLE_S:-0.6}"
export YOLO_BRIDGE_MAX_M="${YOLO_BRIDGE_MAX_M:-0.8}"
exec .venv/bin/python oda/yolo_click_bridge.py "${ARGS[@]}"
