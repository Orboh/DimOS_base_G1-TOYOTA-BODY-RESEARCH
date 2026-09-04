#!/bin/bash
# ビューアだけを立て直す（アプリは落とさない = リーチ設定・kp・オフセットは維持される）。
# ビューアを閉じてしまった / 固まった / 点群が止まった ときはこれ。
cd /home/techshare/DimOS_base_G1-TOYOTA-BODY-RESEARCH || exit 1
pgrep -f "dimos ru[n] unitree-g1-ik-reach" >/dev/null || {
  echo "!! アプリが起動していません。ビューアだけでは何もできません。"
  echo "   先に oda/ik_up.sh を実行してください"; exit 1; }
for p in $(ps -ef | grep "bin/dimos-viewer" | grep -v grep | awk '{print $2}'); do
  echo "  旧ビューア停止 (pid $p)"; kill -9 "$p"
done
sleep 3
DISPLAY=:1 nohup .venv/bin/dimos-viewer \
  --connect rerun+http://127.0.0.1:9877/proxy \
  --ws-url ws://127.0.0.1:3030/ws >/dev/null 2>&1 &
sleep 15
if pgrep -f "bin/dimos-viewe[r]" >/dev/null; then
  echo "  ビューア: 起動OK"
  L=$(ls -t oda/ik_accuracy_out/ik_reach_*.log 2>/dev/null | head -1)
  [ -n "$L" ] && grep -i "viewer connected" "$L" | tail -1
else
  echo "  !! ビューア起動失敗"
fi
