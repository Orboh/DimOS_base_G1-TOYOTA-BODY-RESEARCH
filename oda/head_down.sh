#!/bin/bash
# アプリとビューアを停止する。**先に oda/head_arm_down.sh で腕を下ろすこと。**
# アプリを先に落とすと指令の届け先が消え、腕は上がったまま固まる（2026-09-03 落とし穴②）。
# pkill -f は使わない（無関係なプロセスを巻き込む。2026-09-03 禁止事項）。
# oda/zed_down.sh の頭カメラ版。**NX側の中継はここでは止めない**（残していても
# マルチキャストを流し続けるだけで腕には触らない。止めるなら HEAD_STOP_NX=1）。
#
# シグナルは INT -> TERM -> KILL の順にエスカレートする。**nohup+& で起動された
# プロセスは SIGINT を無視する**（2026-09-07 実機確認）。
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAT="[b]in/dimos run unitree-g1-okra-ik-only-grasp|[c]oordination.watchdog_main|[b]in/dimos-viewer"
pids() { ps -eo pid,args | grep -E "$PAT" | grep -v "grasp-zed" | grep -v "bash -c" | awk '{print $1}'; }

P=$(pids)
if [ -z "$P" ]; then echo "クリーン(残留なし)"; else
for sig in INT TERM KILL; do
  P=$(pids); [ -z "$P" ] && break
  echo "kill -$sig $P"
  for p in $P; do kill -"$sig" "$p" 2>/dev/null; done
  for i in $(seq 1 12); do [ -z "$(pids)" ] && break; sleep 1; done
done
fi
LOG=$(ls -t "$REPO"/oda/head_run_out/head_reach_*.log 2>/dev/null | head -1)
[ -n "$LOG" ] && grep -a "disconnected" "$LOG" | tail -1
if [ "${HEAD_STOP_NX:-0}" = "1" ]; then
  echo "NX中継を停止:"
  timeout 12 sshpass -p "${G1_NX_PW:-123}" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=6 \
    "${G1_NX:-unitree@192.168.123.164}" 'pkill -f ik_camera_standalone && echo "  停止した" || echo "  もう動いていない"' \
    2>/dev/null || echo "  (SSH不通)"
fi
if [ -z "$(pids)" ]; then echo "クリーン(残留なし)"; else
  echo "!! 残留プロセスあり（head_up.sh を実行しないこと）:"; ps -eo pid,args | grep -E "$PAT" | grep -v "bash -c"; fi
