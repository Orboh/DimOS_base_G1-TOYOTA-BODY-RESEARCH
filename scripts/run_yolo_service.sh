#!/usr/bin/env bash
# YOLO-seg GPU 検出サービスの起動ラッパー（Jetson AGX Orin）。
#
# 専用 venv(/mnt/ssd/yolo_gpu_venv, Python 3.10 + CUDA torch) で yolo_service.py を起動する。
# torch の GPU 動作には LD_LIBRARY_PATH が必須（2026-06-24 判明）:
#   - cublas: システムの CUDA 12.6（venv 同梱の 12.9 は torch(cu12.6) と不整合で削除済み）
#   - cudss : venv の nvidia/cu12/lib（torch 2.11 が libcudss.so.0 を要求）
#
# 使い方:
#   bash run_yolo_service.sh            # serve
#   bash run_yolo_service.sh --selftest # 速度自己測定
set -euo pipefail

VENV=/mnt/ssd/yolo_gpu_venv
CUDA_LIB=/usr/local/cuda-12.6/targets/aarch64-linux/lib
CUDSS_LIB="$VENV/lib/python3.10/site-packages/nvidia/cu12/lib"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH="$CUDA_LIB:$CUDSS_LIB:${LD_LIBRARY_PATH:-}"

if [ "${1:-}" = "--selftest" ]; then
  exec "$VENV/bin/python" "$SCRIPT_DIR/yolo_service.py" --selftest "${@:2}"
else
  exec "$VENV/bin/python" "$SCRIPT_DIR/yolo_service.py" --serve "$@"
fi
