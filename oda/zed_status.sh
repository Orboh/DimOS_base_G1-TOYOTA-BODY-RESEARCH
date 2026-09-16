#!/bin/bash
# 胸ZEDクリック→IKリーチ 全系統の生死確認。まずこれ。
# 9/3の oda/ik_status.sh に相当（あちらは頭D435i + 別PC前提）。
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NIC="${ROBOT_NIC:-enp2s0}"
p_app=$(ps -eo pid,args | grep "[b]in/dimos run unitree-g1-okra-ik-only-grasp-zed" | grep -v "bash -c" | awk '{print $1}' | head -1)
p_vw=$(ps -eo pid,args | grep "[b]in/dimos-viewer" | grep -v "bash -c" | awk '{print $1}' | head -1)
[ -n "$p_app" ] && echo "アプリ  : LIVE稼働中 (pid $p_app)" || echo "アプリ  : 停止  -> oda/zed_up.sh"
[ -n "$p_vw" ]  && echo "ビューア: 稼働中 (pid $p_vw)"      || echo "ビューア: 停止  -> oda/zed_up.sh か手順5のコマンド"
ip link show lo 2>/dev/null | head -1 | grep -q MULTICAST \
  && echo "lo mcast: ON" || echo "lo mcast: **OFF** -> sudo ip link set lo multicast on"
ip route show 2>/dev/null | grep -q "^224.0.0.0/4 dev $NIC" \
  && echo "mcast経路: $NIC" || echo "mcast経路: **無し** -> sudo ip route replace 224.0.0.0/4 dev $NIC"
r=$(sysctl -n net.core.rmem_max 2>/dev/null)
[ "$r" -ge 67108864 ] 2>/dev/null && echo "rmem_max: $r" || echo "rmem_max: $r (**要 67108864**)"
ip -br a 2>/dev/null | grep -q "^$NIC .*UP" && echo "有線NIC : $NIC UP" || echo "有線NIC : **$NIC がDOWN/不在**"
timeout 3 ping -c1 -W2 192.168.123.161 >/dev/null 2>&1 && echo "G1      : 応答あり" || echo "G1      : **応答なし**(電源/LAN)"
"$REPO/.venv/bin/python" -c "
import pyzed.sl as sl
d = sl.Camera.get_device_list()
print('ZED     : ' + (', '.join('SN%s %s' % (x.serial_number, x.camera_state) for x in d) if d else '**未検出**'))" 2>/dev/null \
  || echo "ZED     : **pyzed読み込み失敗**"
LOG=$(ls -t "$REPO"/oda/zed_run_out/zed_reach_*.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
  echo "--- 最新ログ: $LOG ---"
  grep -aE "leg done|leg stopped|infeasible|reach #|TIP" "$LOG" | tail -4
fi
