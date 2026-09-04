#!/bin/bash
cd /home/techshare/DimOS_base_G1-TOYOTA-BODY-RESEARCH || exit 1
pgrep -f "dimos ru[n] unitree-g1-ik-reach" >/dev/null && echo "アプリ  : LIVE稼働中" || echo "アプリ  : 停止"
pgrep -f "bin/dimos-viewe[r]"             >/dev/null && echo "ビューア: 稼働中" || echo "ビューア: 停止"
pgrep -f "ik_accuracy_prob[e]\.py$"       >/dev/null && echo "プローブ: 稼働中" || echo "プローブ: 停止(探索中は停止でOK)"
sshpass -p 123 ssh -o ConnectTimeout=6 -o StrictHostKeyChecking=no unitree@192.168.123.164 'pgrep -f ik_camera_standalone >/dev/null' 2>/dev/null \
  && echo "カメラ  : NXで稼働中" || echo "カメラ  : NXで停止 or SSH不通"
L=$(ls -t oda/ik_accuracy_out/ik_reach_*.log 2>/dev/null | head -1)
[ -n "$L" ] && { echo "--- 最新ログ: $L ---"; grep '\[TIP\]' "$L" | tail -1; }
