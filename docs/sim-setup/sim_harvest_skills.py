#!/usr/bin/env python3
"""LangGraph harvest graph を sim で駆動する HarvestSkills 実装（M4 本命・.venv）。

`build_harvest_graph` の `skills` に渡すと、実オーケストレータ（graph.py）の
detect→select→grasp→verify→record→loop が **sim の仮想G1** を動かす:
  - detect_okra : 机上 GT オクラ（torso 相対座標）を返す（YOLO/ZED 不使用＝OOD回避）
  - grasp_okra  : IkApproachSkill(pinocchio)→`rt/arm_sdk` で reach→`rt/dex1/right/cmd` で閉じ
                  （bridge が対象オクラを手に固定）→ lift。把持対象 index は file で bridge へ伝える。
  - verify      : sim は把持成功＝True / record : ログ
座標系は **torso_link 相対 [X前,Y左,Z上]**（IkApproachSkill 入力系）。reach box も torso 系で渡す。
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, "/home/kota-ueda/Desktop/unitree_sdk2_python")

import numpy as np

from dimos.robot.unitree.g1.harvest.blackboard import Okra
from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill
from dimos.robot.unitree.g1.harvest.place_basket import make_place_basket_fn

_WEIGHT_IDX = 29
_ARM_START = 15
_Q_CLOSE = 4.4

# F-07 カゴ投入の教示角。外部リポ Orboh/dex1_1_service feat/drop-to-basket の
# drop_to_basket_mujoco.py の LOCKED poses（左=L字提示 2026-06-25 / 右=drop 2026-06-29更新）
# と一致させたもの。正準順 [shoulder_pitch, roll, yaw, elbow, wrist_roll, pitch, yaw] [rad]。
# ※ 参照の角度のみ移植（「閉じたまま2s保持」等のタイミング・退避復帰は移植しない）。
_PLACE_LEFT_PRESENT = [0.0037, 0.2527, -0.0478, 1.1991, -0.1714, -1.0626, -1.0144]
_PLACE_RIGHT_DROP = [-0.11606722325086594, -0.35777705907821655, 0.5581882281249830,
                     -0.23972046375274658, -0.0300226437634435, -0.2594746064536180,
                     0.8287091851234436]


class SimHarvestSkills:
    """HarvestSkills 実装: graph の判断を sim(bridge) への DDS 指令に変換する。"""

    def __init__(
        self,
        okra_torso: list[tuple[int, tuple[float, float, float]]],
        *,
        iface: str = "lo",
        peers: list[str] | None = None,
        target_file: str = "/tmp/sim_grasp_target.txt",
        basket_torso: tuple[float, float, float] = (0.230, 0.011, -0.072),
    ) -> None:
        # okra_torso: [(okra_prim_index, (X,Y,Z) torso), ...]
        self._okra = {str(idx): (idx, np.array(p, dtype=float)) for idx, p in okra_torso}
        self._picked: set[str] = set()
        self._ik = IkApproachSkill()
        self._target_file = target_file
        self.records: list[dict] = []
        # F-07 籠位置(torso, GT)。既定値は g1bag を SIM_DUMP_BASKET=1 で実測した値（左腕提示後）。
        # シーン/籠を変えたら再実測し、SIM_BASKET_TORSO="X,Y,Z" で上書きする。
        _bt = os.getenv("SIM_BASKET_TORSO")
        self._basket_torso = (
            tuple(float(x) for x in _bt.split(",")) if _bt else basket_torso
        )

        peers = peers or []
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
        self._arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._arm_pub.Init()
        self._grip_pub = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
        self._grip_pub.Init()
        self._lc = make_lowcmd()
        self._gc = MotorCmds_()
        self._gc.cmds = [unitree_go_msg_dds__MotorCmd_()]
        self._gc.cmds[0].kp = 5.0
        self._gc.cmds[0].kd = 0.05
        self._cur_arm = [0.0] * 14   # 直近の腕指令（連続性のため）
        self._grip_q = 0.0
        # DDS discovery warm-up: weight=0/grip=0 を ~1.6s 流し、最初の grasp で dex1/arm の
        # 取りこぼし（discovery 未確立）を防ぐ。weight=0 なので腕は動かない。
        for _ in range(80):
            self._send([0.0] * 14, 0.0, 0.0)
            time.sleep(0.02)
        print(f"[sim-skills] DDS up iface={iface} peers={peers or 'mcast'} (warmup done); okra={sorted(int(k) for k in self._okra)}", flush=True)
        # F-07 投入動作（本番）。送信は本クラスのランプ/ホールドへ注入＝sim も実機も同一ロジック。
        #   send_arm: 閉じ保持(_Q_CLOSE)のまま arm14 を補間（オクラを落とさない）
        #   open_gripper: 腕を保持したままグリッパを開く（=リリース）
        self._place = make_place_basket_fn(
            basket_torso=self._basket_torso,
            send_arm=lambda arm14, secs: self._ramp(arm14, secs, grip_q=_Q_CLOSE, weight_to=1.0),
            open_gripper=lambda q, secs: self._hold(secs, grip_q=q),
            get_measured=lambda: [0.0] * 15 + list(self._cur_arm),
            # C: カゴ投入は参照 dex1_1_service の教示角に合わせる（左提示＋右dropを固定角で）。
            left_present=_PLACE_LEFT_PRESENT,
            right_drop=_PLACE_RIGHT_DROP,
            # ★sim の開放角は 0.0。place_basket 既定 5.2 は実機 Dex1 の「開放」規約だが、bridge は
            #   grip_q を frac=grip_q/4.4 を[0,1]クランプして指駆動＝5.2 は frac=1.0=全閉にマップされ
            #   リリースされない。sim では 0.0 が開放（friction_pick_place も gq=0.0 で離す）。
            q_open=float(os.getenv("SIM_GRIP_OPEN_Q", "0.0")),
        )
        print(f"[sim-skills] F-07 籠位置(torso)={tuple(round(v,3) for v in self._basket_torso)}"
              " （SIM_BASKET_TORSO で上書き可）", flush=True)

    # ---- 低レベル送信 ----
    def _send(self, arm14: list[float], weight: float, grip_q: float) -> None:
        self._lc.motor_cmd[_WEIGHT_IDX].q = weight
        for j in range(14):
            self._lc.motor_cmd[_ARM_START + j].q = arm14[j]
        self._arm_pub.Write(self._lc)
        self._gc.cmds[0].q = grip_q
        self._grip_pub.Write(self._gc)
        self._cur_arm = list(arm14)
        self._grip_q = grip_q

    def _ramp(self, to14: list[float], secs: float, grip_q: float, weight_to: float = 1.0) -> None:
        frm = list(self._cur_arm)
        w0 = self._lc.motor_cmd[_WEIGHT_IDX].q
        n = max(1, int(secs * 50))
        for i in range(n):
            s = (i + 1) / n
            arm = [frm[j] + (to14[j] - frm[j]) * s for j in range(14)]
            w = w0 + (weight_to - w0) * s
            self._send(arm, w, grip_q)
            time.sleep(0.02)

    def _hold(self, secs: float, grip_q: float) -> None:
        n = max(1, int(secs * 50))
        for _ in range(n):
            self._send(self._cur_arm, 1.0, grip_q)
            time.sleep(0.02)

    # ---- HarvestSkills プロトコル ----
    def detect_okra(self) -> list[Okra]:
        out = []
        for k, (_idx, p) in sorted(self._okra.items(), key=lambda kv: int(kv[0])):
            if k in self._picked:
                continue
            out.append(Okra(id=k, pos_3d={"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                            ripeness=1.0, reachable=True))
        return out

    def grasp_okra(self, okra: Okra, force: float) -> None:
        idx, p = self._okra[okra.id]
        # bridge に「次に掴む okra prim index」を伝える
        try:
            with open(self._target_file, "w") as f:
                f.write(str(idx))
        except Exception as e:  # noqa: BLE001
            print(f"[sim-skills] target file write fail: {e}", flush=True)

        res = self._ik.solve(p, [0.0] * 29)
        if res is None:
            print(f"[sim-skills] okra {okra.id} IK 解けず（skip）", flush=True)
            return
        a_reach = list(res.arm14)
        print(f"[sim-skills] GRASP okra{okra.id} (idx={idx}) reach IK err={res.err:.4f}", flush=True)
        # reach（grip 開のまま腕を寄せる）
        self._ramp(a_reach, 2.0, grip_q=0.0, weight_to=1.0)
        # 閉じる（bridge が対象オクラを手へ固定）
        self._ramp(a_reach, 1.0, grip_q=_Q_CLOSE)
        self._hold(0.6, grip_q=_Q_CLOSE)
        # 持ち上げ（手前へ引き＋上へ）。閉じ保持＝オクラ追従
        lift = p + np.array([-0.05, 0.0, 0.18])
        r_lift = self._ik.solve(lift, [0.0] * 29)
        if r_lift is not None:
            self._ramp(list(r_lift.arm14), 1.5, grip_q=_Q_CLOSE)
        self._hold(0.5, grip_q=_Q_CLOSE)
        # F-07（本番）: 左腕で籠提示 → 右腕IKで籠上空 → 開いてリリース。
        #   重力ON+籠コライダー時は bridge 側でオクラが物理落下（テレポートではない）。
        #   重力OFF時は従来どおり bridge の world アンカーで籠位置へ収める。
        ok = self._place()
        print(f"[sim-skills] F-07 籠収納 {'OK' if ok else 'FAIL（IK解けず・要 SIM_BASKET_TORSO 調整）'}", flush=True)
        self._picked.add(okra.id)

    def verify_harvest(self) -> bool:
        return True  # sim は把持成功とみなす（実機は VLM/再観測）

    def record_harvest(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        print(f"[sim-skills] record: {record}", flush=True)

    def relative_move(self, lateral: float, forward: float = 0.0, yaw: float = 0.0) -> None:
        print(f"[sim-skills] relative_move(lat={lateral:.2f},fwd={forward:.2f}) — sim では no-op", flush=True)

    def go_to_next_station(self) -> bool:
        return False  # 単一ステーション

    def swap_basket(self) -> None:
        print("[sim-skills] swap_basket（sim no-op）", flush=True)
