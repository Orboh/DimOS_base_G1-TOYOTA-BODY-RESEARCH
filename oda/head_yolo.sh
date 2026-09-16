#!/bin/bash
# YOLO検出 → /clicked_point（人間のクリックの代替）。アプリ稼働中に別ターミナルで実行。
# oda/zed_yolo.sh の頭カメラ版。
#
#   oda/head_yolo.sh              # DRY-RUN: 検出と3D点をログに出すだけ（腕は動かない）
#   oda/head_yolo.sh --live       # 発火（Enterを押すたびに1回。腕が動く）
#   oda/head_yolo.sh --live --once  # 1回だけ発火して終了
#
# ============ ZED版との決定的な違い（ここを間違えると腕が明後日に飛ぶ）============
# **YOLO_BRIDGE_BODY_FRAME を立てない。** ZEDはアプリ内プロセスでTFを出すため、
# ビューアのクリックが既にボディ座標(x前/y左/z上)で届く → ブリッジ側も光学→ボディに
# 変換して合わせていた(zed_yolo.sh は =1)。頭D435iはNXの中継がTFを出さないので、
# クリックは**光学座標の生値**(x右/y下/z奥)のまま届き、IkReachBridge が内部で
# 光学回転を掛ける(click_in_camera_body_frame=False = 既定)。**ここで=1にすると
# 回転が二重に掛かる。** yolo_click_bridge.py:190 と ik_reach_bridge.py:120-131 参照。
# ================================================================================
#
# imgsz について: 9/7にZED(1280x720)で測った値は **頭D435i(640x480)にそのまま使えない**。
#   ZED実測(同一フレーム): 640=検出0件 / 960=conf0.57 / 1280=conf0.90 / 1920=conf0.76
#   学習画像はRoboflowの640px近接切り出しで被写体が画面の大部分を占めるため、
#   「入力の中でオクラが何画素か」で決まる。D435iは画角も解像度も違うので**必ず測り直す**:
#     for s in 640 960 1280; do OKRA_YOLO_IMGSZ=$s oda/head_yolo.sh --once; done   # DRY-RUNで
#   confが最大になる値を採用してから --live にすること。
#
# 重みはリポに入らない設計（.gitignore:119 で model/ ごと除外）。このPCの実物:
#   ~/workspace/okurayolo/okra11n-seg-v5/weights/best.pt        (2026-08-18・最新, 既定)
#   ~/workspace/okurayolo/okradata/uedasmodel/okra11n-seg.pt    (2026-06-25・v5のベース)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
ps -eo pid,args | grep "[b]in/dimos run unitree-g1-okra-ik-only-grasp" | grep -qv "grasp-zed" || {
  echo "!! アプリが起動していません。先に oda/head_up.sh"; exit 1; }
MODEL="${OKRA_YOLO_MODEL:-$HOME/workspace/okurayolo/okra11n-seg-v5/weights/best.pt}"
[ -f "$MODEL" ] || { echo "!! モデルが無い: $MODEL"; exit 1; }
LIVE=0; ARGS=()
for a in "$@"; do case "$a" in --live) LIVE=1;; *) ARGS+=("$a");; esac; done
if [ "$LIVE" = "1" ]; then
  export YOLO_BRIDGE_LIVE=1
  echo "*** LIVE: 検出したオクラへ腕が動きます。e-stopを手元に ***"
else
  unset YOLO_BRIDGE_LIVE
  echo "  (DRY-RUN: 検出と3D点をログに出すだけ。腕は動きません。--live で発火)"
fi
# VRAMチェック: このPCは8GB。ZED版では深度推論と食い合って 2026-09-07 18:04 に電源断したが、
# **頭D435iの深度はNX側で作るのでこのPCのGPUはYOLOだけ**。余裕は大きいはずだが一応見る。
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$FREE" ]; then
  echo "  GPU空きVRAM: ${FREE} MiB（ZED版と違い深度推論が乗らないので余裕があるはず）"
  if [ "$FREE" -lt 2500 ] 2>/dev/null; then
    echo "  !! 空きVRAMが少ない。ビューアを閉じる / OKRA_YOLO_IMGSZ を下げる"
  fi
fi
echo "  model=$MODEL  conf=${OKRA_YOLO_CONF:-0.4}  imgsz=${OKRA_YOLO_IMGSZ:-960}(**要再測定**)"
export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-$HOME/cyclonedds-noshm}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-$HOME/cyclonedds-noshm/lib}"
export LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1'
export PYTEST_VERSION=1 DIMOS_SKIP_COORDINATOR_RPC=1
export OKRA_YOLO_MODEL="$MODEL"
export OKRA_YOLO_CONF="${OKRA_YOLO_CONF:-0.4}"
export OKRA_YOLO_IMGSZ="${OKRA_YOLO_IMGSZ:-960}"
unset YOLO_BRIDGE_BODY_FRAME            # ← ZED版との唯一かつ決定的な差。上のブロック参照
# 近傍画素半径: NX中継の点群は2mmボクセル(IK_CAMERA_VOXEL既定)で、0.5m先なら1px≒1.3mm
# ＝点は密。ZEDの20（5mmボクセル対策）は不要なので素の8のまま。拾えなければ上げる。
export YOLO_BRIDGE_PX_RADIUS="${YOLO_BRIDGE_PX_RADIUS:-8}"
export YOLO_BRIDGE_DOUBLE_S="${YOLO_BRIDGE_DOUBLE_S:-0.6}"
# 距離ゲート: 頭カメラはtorso原点の43cm上・47.6度うつむき。作業域のオクラまで概ね0.5-0.7m
# なので0.8mだと余裕が少ない。遠すぎで蹴られたらここを上げる（背景誤検出の防波堤なので上げすぎ注意）。
export YOLO_BRIDGE_MAX_M="${YOLO_BRIDGE_MAX_M:-0.9}"
exec .venv/bin/python oda/yolo_click_bridge.py "${ARGS[@]}"
