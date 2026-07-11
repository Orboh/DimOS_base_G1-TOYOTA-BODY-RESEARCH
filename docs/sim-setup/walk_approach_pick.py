#!/usr/bin/env python3
"""歩行接近×摩擦把持の統合制御器（.venv）。creep接近→Δサーボ把持→運搬→カゴ投入。

bridge は SIM_WALK_POLICY=1 + SIM_TABLE=1 + SIM_GRASP_FRICTION=1 + SIM_OKRA_BREAK_N=1.0
+ SIM_WALK_SPAWN_X=-0.30 + SIM_WALK_LOG_TICKS=25 + SIM_LOG_EVERY=1 で起動しておく。
本スクリプトは bridge のログ（BRIDGE_LOG）を閉ループの観測に使う:
  - base=(x,..)            … 歩行接近の停止窓制御（0.5s 周期）
  - okra_torso=(x,y,z)     … 生の把持目標（休め姿勢時の実測）
  - Δ(okra-gap)=(dx,dy,dz) … ジョー隙間とオクラの実測ズレ（リーチ内サーボの帰還信号）

歩行モード把持の実測知見（2026-07-11）:
  - 目標 z は +0.165 補正（先端オフセットで実ジョーは指令より~10cm低い＝天板に引っかかる）
  - 目標 x は balance 上限 ~0.56（それ以上は policy が後ずさり）→ 不足分は接近で詰める
  - リーチは PP_VIA 相当の上空経由（直線補間だと机前板に衝突→転倒）
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, "/home/kota-ueda/Desktop/unitree_sdk2_python")

import numpy as np

_WEIGHT_IDX = 29
_ARM_START = 15
_Q_CLOSE = 4.4
_BASE_MOVE_FILE = os.getenv("SIM_BASE_MOVE_FILE", "/tmp/sim_base_move.txt")
_DROP_POSES = os.path.join(REPO, "docs/sim-setup/drop_poses.json")
_CANON14 = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
# 実測パラメータ（上記知見）
X_TIP = 0.19        # [m] 指令→ジョー隙間の x 先端オフセット初期値（サーボが実測で吸収）
Z_COMP = 0.165      # [m] z 補正（天板引っかかり回避）
X_CMD_MAX = 0.52    # [m] balance が許す指令 x 上限（0.54+で後退リカバリ誘発を実測）
STOP_WIN = (float(os.getenv("AP_STOP_MIN", "0.04")), float(os.getenv("AP_STOP_MAX", "0.14")))


class BridgeLog:
    """bridge ログの末尾から観測を取り出す（閉ループの観測器）。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self._mark = 0  # fresh 読みの起点（mark() 以降の行だけを観測に使う）

    def _tail(self, n: int = 4000) -> str:
        with open(self.path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200 * n))
            return f.read().decode(errors="replace")

    def mark(self) -> None:
        """これ以降のログだけを観測対象にする（古い Δ を掴まないため）。"""
        with open(self.path, "rb") as f:
            f.seek(0, 2)
            self._mark = f.tell()

    def _fresh(self) -> str:
        with open(self.path, "rb") as f:
            f.seek(self._mark)
            return f.read().decode(errors="replace")

    def base_x(self) -> float | None:
        m = re.findall(r"base=\((\+?-?[0-9.]+)", self._tail())
        return float(m[-1]) if m else None

    def okra_rest(self) -> tuple[float, float, float] | None:
        m = re.findall(r"okra_torso=\((-?[0-9.]+),(-?[0-9.]+),(-?[0-9.]+)\)", self._tail())
        return tuple(float(v) for v in m[-1]) if m else None

    def delta(self, fresh: bool = True) -> tuple[float, float, float] | None:
        m = re.findall(r"Δ\(okra-gap\)=\((-?[0-9.]+), (-?[0-9.]+), (-?[0-9.]+)\)",
                       self._fresh() if fresh else self._tail())
        return tuple(float(v) for v in m[-1]) if m else None

    def okra_z_max(self) -> float:
        """mark() 以降の okra_z の最大（持ち上げ検証: >0.82 で把持成立）。"""
        zs = [float(v) for v in re.findall(r"okra_z=([0-9.]+)", self._fresh())]
        return max(zs) if zs else 0.0


def main() -> int:
    iface = os.getenv("SIM_DDS_IFACE", "lo")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "127.0.0.1").split(",") if p.strip()]
    log = BridgeLog(os.environ["BRIDGE_LOG"])  # 必須: bridge のログパス

    # --- DDS（friction_pick_place と同一経路） ---
    import unitree_sdk2py.core.channel as ch
    if peers:
        pj = "".join(f'<Peer address="{p}"/>' for p in peers)
        ch.ChannelConfigHasInterface = (
            '<?xml version="1.0" encoding="UTF-8" ?><CycloneDDS><Domain Id="any">'
            '<General><Interfaces>'
            '<NetworkInterface name="$__IF_NAME__$" priority="default" multicast="false"/>'
            '</Interfaces><AllowMulticast>false</AllowMulticast>'
            '<EnableMulticastLoopback>false</EnableMulticastLoopback></General>'
            '<Discovery><ParticipantIndex>auto</ParticipantIndex>'
            f'<MaxAutoParticipantIndex>32</MaxAutoParticipantIndex><Peers>{pj}</Peers></Discovery>'
            '</Domain></CycloneDDS>'
        )
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import (
        unitree_go_msg_dds__MotorCmd_,
        unitree_hg_msg_dds__LowCmd_ as make_lowcmd,
    )
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

    ChannelFactoryInitialize(0, iface)
    arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
    arm_pub.Init()
    grip_pub = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
    grip_pub.Init()
    lc = make_lowcmd()
    gc = MotorCmds_()
    gc.cmds = [unitree_go_msg_dds__MotorCmd_()]
    gc.cmds[0].kp, gc.cmds[0].kd = 5.0, 0.05
    cur = [0.0] * 14

    def send(arm14, w, gq):
        lc.motor_cmd[_WEIGHT_IDX].q = w
        for j in range(14):
            lc.motor_cmd[_ARM_START + j].q = arm14[j]
        arm_pub.Write(lc)
        gc.cmds[0].q = gq
        grip_pub.Write(gc)
        cur[:] = list(arm14)

    def ramp(to14, secs, gq, w=1.0):
        frm = list(cur)
        n = max(1, int(secs * 50))
        for i in range(n):
            s = (i + 1) / n
            send([frm[j] + (to14[j] - frm[j]) * s for j in range(14)], w, gq)
            time.sleep(0.02)

    def hold(secs, gq):
        for _ in range(max(1, int(secs * 50))):
            send(cur, 1.0, gq)
            time.sleep(0.02)

    def base_pulse(vx: float, secs: float):
        """1歩ぶんの base 速度パルス（policy 歩行）。"""
        t0 = time.time()
        while time.time() - t0 < secs:
            with open(_BASE_MOVE_FILE, "w") as f:
                f.write(f"{vx},0.0")
            time.sleep(0.1)
        try:
            os.remove(_BASE_MOVE_FILE)
        except OSError:
            pass

    from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill
    ik = IkApproachSkill()

    # ============ Phase 0: 事前挙手（AP_PRERAISE=1, 既定ON） ============
    # 接近の**前に**腕を机より高い前方位置へ上げておく。リーチ（大きな重心前移動）を
    # 静止立位でなく歩行中に持ち込む＝歩行 policy が歩容として吸収し、到着後は
    # 「小さく降ろすだけ」で済む（後ずさりクリープ・机衝突の両方を回避する狙い）。
    if os.getenv("AP_PRERAISE", "1") == "1":
        pre = np.array([float(x) for x in os.getenv("AP_PRERAISE_POSE", "0.40,-0.16,0.20").split(",")])
        r0 = ik.solve(pre, [0.0] * 29)
        if r0 is not None:
            print(f"[ap] Phase 0: 事前挙手 {tuple(float(v) for v in pre)}（机より上・err={r0.err:.4f}）", flush=True)
            ramp(list(r0.arm14), 2.5, gq=0.0)
            hold(0.5, gq=0.0)

    # ============ Phase A: 歩行接近（粗→1歩ずつ・ヒステリシス停止窓） ============
    print(f"[ap] Phase A: 接近（停止窓 base_x∈[{STOP_WIN[0]},{STOP_WIN[1]}]）", flush=True)
    t0 = time.time()
    while time.time() - t0 < 30:  # 粗接近
        with open(_BASE_MOVE_FILE, "w") as f:
            f.write("0.3,0.0")
        bxv = log.base_x()
        if bxv is not None and bxv >= -0.12:
            break
        time.sleep(0.1)
    try:
        os.remove(_BASE_MOVE_FILE)
    except OSError:
        pass
    time.sleep(3.0)
    for p in range(8):  # 微調整（1歩ずつ・行き過ぎたら後退）
        bxv = log.base_x() or 0.0
        if STOP_WIN[0] <= bxv <= STOP_WIN[1]:
            print(f"[ap]   停止 base_x={bxv:.2f}（{p}手）", flush=True)
            break
        base_pulse(0.3 if bxv < STOP_WIN[0] else -0.3, 0.3)
        time.sleep(3.0)
    else:
        print(f"[ap]   停止窓に入らず base_x={log.base_x()}（続行）", flush=True)
    bxv = log.base_x() or 0.0
    if bxv > 0.18:  # 机ジャム（胸が前板に接触圏）→ 大きく後退して1回だけ再詰め
        print(f"[ap]   ジャム検出({bxv:.2f}) → 後退リカバリ", flush=True)
        base_pulse(-0.3, 0.6)
        time.sleep(3.0)
        base_pulse(0.3, 0.3)
        time.sleep(3.0)
        print(f"[ap]   リカバリ後 base_x={log.base_x()}", flush=True)
    time.sleep(2.0)

    # ============ Phase B: via 経由リーチ（初期目標＝実測式） ============
    rest = log.okra_rest()
    if rest is None:
        print("[ap] okra_torso がログに無い（bridge の SIM_GRASP_FRICTION=1 を確認）", flush=True)
        return 1
    tgt = np.array([min(rest[0] + X_TIP, X_CMD_MAX), rest[1] + 0.013, rest[2] + Z_COMP])
    print(f"[ap] Phase B: リーチ rest={tuple(round(v,3) for v in rest)} → cmd={tuple(round(float(v),3) for v in tgt)}", flush=True)
    r_via = ik.solve(np.array([tgt[0], tgt[1], tgt[2] + 0.10]), [0.0] * 29)
    if r_via is not None:
        ramp(list(r_via.arm14), 2.0, gq=0.0)
    r = ik.solve(tgt, [0.0] * 29)
    if r is None:
        print("[ap] reach IK 解けず", flush=True)
        return 1
    ramp(list(r.arm14), 2.0, gq=0.0)
    log.mark()          # ここ以降の Δ 観測だけを使う（古い行の誤帰還防止）
    hold(1.8, gq=0.0)   # 静定（新しい Δ 行が出るのを待つ）

    # ============ Phase C: Δサーボ（実測ズレを指令へ帰還 ×3） ============
    for it in range(3):
        d = log.delta()
        if d is None:
            print("[ap]   Δ観測なし → サーボ省略", flush=True)
            break
        print(f"[ap] Phase C-{it}: Δ(okra-gap)={tuple(round(v,3) for v in d)}", flush=True)
        if abs(d[0]) < 0.02 and abs(d[1]) < 0.02 and abs(d[2]) < 0.03:
            print("[ap]   収束（|Δ|<2-3cm）", flush=True)
            break
        # ゲイン0.6・±4cm クランプ（先端オフセットが姿勢依存で±10cm 揺れるため、大きく
        # 追うと深い/高い姿勢に入り policy の後退リカバリを誘発する）
        tgt = tgt + 0.6 * np.clip(np.array(d), -0.04, 0.04)
        tgt[0] = float(np.clip(tgt[0], 0.30, X_CMD_MAX + 0.001))
        tgt[1] = float(np.clip(tgt[1], -0.30, 0.05))
        tgt[2] = float(np.clip(tgt[2], 0.08, 0.16))
        if tgt[0] > X_CMD_MAX:  # balance 上限 → 不足分は半歩前進で詰める
            print(f"[ap]   x 上限。半歩前進で補う（deficit={tgt[0]-X_CMD_MAX:.3f}）", flush=True)
            tgt[0] = X_CMD_MAX
            base_pulse(0.3, 0.25)
            time.sleep(2.5)
        r = ik.solve(tgt, [0.0] * 29)
        if r is None:
            print(f"[ap]   IK 解けず cmd={tgt}", flush=True)
            break
        ramp(list(r.arm14), 1.2, gq=0.0)
        log.mark()
        hold(1.8, gq=0.0)

    # ============ Phase D: close → 検証 ============
    print("[ap] Phase D: close（摩擦把持）", flush=True)
    log.mark()  # これ以降の okra_z で持ち上げを検証
    ramp(cur, 1.5, gq=_Q_CLOSE)
    hold(0.8, gq=_Q_CLOSE)
    # lift（+18cm）
    lift = np.array([tgt[0] - 0.05, tgt[1], tgt[2] + 0.18])
    r_l = ik.solve(lift, [0.0] * 29)
    if r_l is not None:
        ramp(list(r_l.arm14), 2.0, gq=_Q_CLOSE)
    hold(0.8, gq=_Q_CLOSE)
    time.sleep(1.0)
    okz = log.okra_z_max()
    grasped = okz > 0.82
    print(f"[ap]   lift 後 okra_z(max)={okz:.3f} → {'✅ 把持成立' if grasped else '❌ 未把持'}", flush=True)

    # ============ Phase E: 運搬 → カゴ投入（記録固定角の再生） ============
    dp = json.load(open(_DROP_POSES))
    def _pose14(dr, dl):
        mg = {**dl, **dr}
        return [float(mg.get(k, 0.0)) for k in _CANON14]
    print("[ap] Phase E: 運搬→カゴ投入", flush=True)
    ramp(_pose14(dp["right_arm_body_center"], dp["left_arm_body_center"]), 3.0, gq=_Q_CLOSE)
    hold(0.4, gq=_Q_CLOSE)
    ramp(_pose14(dp["right_arm_drop_pose"], dp["left_basket_pose"]), 3.0, gq=_Q_CLOSE)
    hold(0.6, gq=_Q_CLOSE)
    hold(1.5, gq=0.0)  # open
    print(f"[ap] done（{'成功' if grasped else '把持は未成立'}）", flush=True)
    return 0 if grasped else 2


if __name__ == "__main__":
    raise SystemExit(main())
