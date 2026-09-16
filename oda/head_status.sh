#!/bin/bash
# 頭部D435iクリック→IKリーチ 全系統の生死確認。まずこれ。
# 9/7の oda/zed_status.sh の頭カメラ版。違いは「ZEDの検出」→「Jetson NX上の中継プロセス」。
# 胸ZEDはPC直結なので死活はUSBだけ見ればよかったが、頭D435iは
#   NXのD435i → ik_camera_standalone.py → LCMマルチキャスト → このPC
# という中継経路なので、**NXへの到達性とマルチキャスト経路の両方**が生きている必要がある。
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NIC="${ROBOT_NIC:-enp2s0}"
G1_NX="${G1_NX:-unitree@192.168.123.164}"
G1_NX_PW="${G1_NX_PW:-123}"
p_app=$(ps -eo pid,args | grep "[b]in/dimos run unitree-g1-okra-ik-only-grasp" | grep -v "grasp-zed" | grep -v "bash -c" | awk '{print $1}' | head -1)
p_vw=$(ps -eo pid,args | grep "[b]in/dimos-viewer" | grep -v "bash -c" | awk '{print $1}' | head -1)
[ -n "$p_app" ] && echo "アプリ  : LIVE稼働中 (pid $p_app)" || echo "アプリ  : 停止  -> oda/head_up.sh"
[ -n "$p_vw" ]  && echo "ビューア: 稼働中 (pid $p_vw)"      || echo "ビューア: 停止  -> oda/head_up.sh"
ip link show lo 2>/dev/null | head -1 | grep -q MULTICAST \
  && echo "lo mcast: ON" || echo "lo mcast: **OFF** -> sudo ip link set lo multicast on"
ip route show 2>/dev/null | grep -q "^224.0.0.0/4 dev $NIC" \
  && echo "mcast経路: $NIC" || echo "mcast経路: **無し** -> sudo ip route replace 224.0.0.0/4 dev $NIC"
r=$(sysctl -n net.core.rmem_max 2>/dev/null)
[ "$r" -ge 67108864 ] 2>/dev/null && echo "rmem_max: $r" || echo "rmem_max: $r (**要 67108864**)"
ip -br a 2>/dev/null | grep -q "^$NIC .*UP" && echo "有線NIC : $NIC UP" || echo "有線NIC : **$NIC がDOWN/不在**"
timeout 3 ping -c1 -W2 192.168.123.161 >/dev/null 2>&1 && echo "G1      : 応答あり" || echo "G1      : **応答なし**(電源/LAN)"
# NXは有線 .164 を使う（WiFi .0.211 はセッション中に切れて中継が落ちる。start_okra_ik_only_grasp.sh の注記）
timeout 4 ping -c1 -W3 192.168.123.164 >/dev/null 2>&1 && echo "NX      : 応答あり" || echo "NX      : **応答なし**(有線.164を確認)"
timeout 12 sshpass -p "$G1_NX_PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=6 "$G1_NX" \
  'p=$(pgrep -f "python.*ik_camera_standalone" | head -1); [ -n "$p" ] && echo "カメラ  : NXで稼働中 (pid $p)" || echo "カメラ  : **NXで停止** -> oda/head_up.sh が起こす"' \
  2>/dev/null || echo "カメラ  : **SSH不通** -> NXの電源/有線を確認"
# 中継が生きていてもマルチキャストがこのPCまで来ていなければ点群は出ない（経路/rmemの問題）
# 参加インタフェースは**有線NICのIPv4を明示する**。INADDR_ANY だと既定経路側(WiFi)を
# 選んでしまい、中継が生きていても 0 パケットに見える（2026-09-08 実測）。
NICIP=$(ip -4 -br a show "$NIC" 2>/dev/null | awk '{print $3}' | cut -d/ -f1)
"$REPO/.venv/bin/python" - "${NICIP:-0.0.0.0}" <<'PYEOF'
import socket, struct, sys, time
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", 7667))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                 struct.pack("4s4s", socket.inet_aton("239.255.76.67"), socket.inet_aton(sys.argv[1])))
    s.settimeout(3); n = big = 0; t = time.time()
    while time.time() - t < 3:
        d, _ = s.recvfrom(70000); n += 1
        if len(d) > 1400: big += 1
    print(f"点群受信: 3秒で {n} パケット (うち大 {big} = 点群断片)")
except socket.timeout:
    print("点群受信: **0パケット** -> 経路/rmem/NX中継のどれかが死んでいる")
except Exception as e:
    print(f"点群受信: 確認失敗 {e!r}")
PYEOF
LOG=$(ls -t "$REPO"/oda/head_run_out/head_reach_*.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
  echo "--- 最新ログ: $LOG ---"
  grep -aE "leg done|leg stopped|infeasible|reach #|TIP" "$LOG" | tail -4
fi
