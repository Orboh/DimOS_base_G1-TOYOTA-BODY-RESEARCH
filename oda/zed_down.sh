#!/bin/bash
# アプリとビューアを停止する。**先に oda/zed_arm_down.sh で腕を下ろすこと。**
# アプリを先に落とすと指令の届け先が消え、腕は上がったまま固まる（2026-09-03 落とし穴②）。
# pkill -f は使わない（無関係なプロセスを巻き込む。2026-09-03 禁止事項）。
#
# シグナルは INT -> TERM -> KILL の順にエスカレートする。**nohup+& で起動された
# プロセスは SIGINT を無視する**（非対話シェルの子は SIGINT が SIG_IGN で継承される。
# 2026-09-07 に実機で確認: kill -INT だけでは落ちなかった）。9/3の ik_down.sh が
# 素の kill(TERM) -> kill -9 を使っているのと同じ理由。
# ※ 腕を機体側へ丁寧に返す handback は Ctrl-C/INT 経路でしか走らない可能性がある。
#   だから「先に腕を下ろす」運用が正。TERMで落とした場合、腕は通信途絶タイムアウトで
#   機体側に戻る（脱力する）。
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAT="[b]in/dimos run unitree-g1-okra-ik-only-grasp-zed|[c]oordination.watchdog_main|[b]in/dimos-viewer"
pids() { ps -eo pid,args | grep -E "$PAT" | grep -v "bash -c" | awk '{print $1}'; }

P=$(pids)
if [ -z "$P" ]; then echo "クリーン(残留なし)"; exit 0; fi
for sig in INT TERM KILL; do
  P=$(pids); [ -z "$P" ] && break
  echo "kill -$sig $P"
  for p in $P; do kill -"$sig" "$p" 2>/dev/null; done
  for i in $(seq 1 12); do [ -z "$(pids)" ] && break; sleep 1; done
done
LOG=$(ls -t "$REPO"/oda/zed_run_out/zed_reach_*.log 2>/dev/null | head -1)
[ -n "$LOG" ] && grep -a "disconnected" "$LOG" | tail -1
if [ -z "$(pids)" ]; then echo "クリーン(残留なし)"; else
  echo "!! 残留プロセスあり（zed_up.sh を実行しないこと）:"; ps -eo pid,args | grep -E "$PAT" | grep -v "bash -c"; fi
