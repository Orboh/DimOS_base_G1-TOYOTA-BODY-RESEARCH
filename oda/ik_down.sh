#!/bin/bash
# アプリとビューアを止める。腕を下ろしたいなら「止める前に」oda/arm_down.sh を実行すること。
cd /home/techshare/DimOS_base_G1-TOYOTA-BODY-RESEARCH || exit 1
for p in $(pgrep -f "bin/dimos-viewe[r]"); do kill -9 $p; done
sleep 2
for p in $(pgrep -f "dimos ru[n] unitree-g1-ik-reach"); do kill $p; done
sleep 5
for p in $(pgrep -f "dimos ru[n] unitree-g1-ik-reach"); do kill -9 $p; done
sleep 2
if pgrep -f "dimos ru[n] unitree-g1-ik-reach" >/dev/null || pgrep -f "bin/dimos-viewe[r]" >/dev/null; then
  echo "!! 残留プロセスあり:"; ps -ef | grep -E "dimos run|dimos-viewer" | grep -v grep
else
  echo "クリーン(残留なし)"
fi
