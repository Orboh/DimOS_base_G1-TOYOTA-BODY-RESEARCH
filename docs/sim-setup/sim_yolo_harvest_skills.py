#!/usr/bin/env python3
"""YOLO 検出で LangGraph harvest を sim 駆動する HarvestSkills（.venv・知覚も本物）。

`sim_harvest_skills.SimHarvestSkills`（GT 座標注入版）の **detect だけ実 YOLO に置換**した版:
  - detect_okra : bridge の ego_view を実 YOLO-seg(okra11n-seg.pt) で検出→torso 3D（`sim_yolo_lib`）。
                  GT 注入なし＝知覚も本物。near クリップ修正で sim 検出が動く（exp 記録参照）。
  - grasp_okra  : IkApproachSkill(pinocchio)→`rt/arm_sdk` reach→`rt/dex1/right/cmd` close→lift。
                  把持対象 prim は bridge 側の最近傍自動把持(SIM_GRASP_NEAREST=1)が選ぶ＝index 不要。
  - verify=True / record / relative_move,swap_basket=no-op。
座標系は torso_link 相対 [X前,Y左,Z上]（IkApproachSkill 入力系・YOLO の torso 出力と同系）。
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, "/home/kota-ueda/Desktop/unitree_sdk2_python")
sys.path.insert(0, os.path.join(REPO, "docs/sim-setup"))

import numpy as np

from dimos.robot.unitree.g1.harvest.blackboard import Okra
from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill
import sim_yolo_lib

_WEIGHT_IDX = 29
_ARM_START = 15
_Q_CLOSE = 4.4


class SimYoloHarvestSkills:
    """detect=実YOLO / grasp=IK+dex1（bridge最近傍把持）/ verify=True の HarvestSkills。"""

    def __init__(self, *, iface: str = "lo", peers: list[str] | None = None,
                 cam_host: str = "127.0.0.1", cam_port: int = 5555,
                 conf: float = 0.25, dedup_m: float = 0.06,
                 save_dir: str | None = None) -> None:
        self._ik = IkApproachSkill()
        self._cam_host, self._cam_port, self._conf = cam_host, cam_port, conf
        self._dedup_m = dedup_m
        self._save_dir = save_dir
        self._picked_pos: list[np.ndarray] = []  # 収穫済みオクラの torso 位置（再検出の重複除外）
        self.records: list[dict] = []

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
        self._cur_arm = [0.0] * 14
        for _ in range(80):  # DDS discovery warm-up
            self._send([0.0] * 14, 0.0, 0.0)
            time.sleep(0.02)
        print(f"[yolo-skills] DDS up iface={iface} peers={peers or 'mcast'} (warmup); "
              f"cam={cam_host}:{cam_port} conf>={conf}", flush=True)

    def _send(self, arm14, weight, grip_q):
        self._lc.motor_cmd[_WEIGHT_IDX].q = weight
        for j in range(14):
            self._lc.motor_cmd[_ARM_START + j].q = arm14[j]
        self._arm_pub.Write(self._lc)
        self._gc.cmds[0].q = grip_q
        self._grip_pub.Write(self._gc)
        self._cur_arm = list(arm14)

    def _ramp(self, to14, secs, grip_q, weight_to=1.0):
        frm = list(self._cur_arm)
        w0 = self._lc.motor_cmd[_WEIGHT_IDX].q
        n = max(1, int(secs * 50))
        for i in range(n):
            s = (i + 1) / n
            self._send([frm[j] + (to14[j] - frm[j]) * s for j in range(14)],
                       w0 + (weight_to - w0) * s, grip_q)
            time.sleep(0.02)

    def _hold(self, secs, grip_q):
        for _ in range(max(1, int(secs * 50))):
            self._send(self._cur_arm, 1.0, grip_q)
            time.sleep(0.02)

    # ---- HarvestSkills ----
    def detect_okra(self) -> list[Okra]:
        dets = sim_yolo_lib.detect_okra_torso(
            self._cam_host, self._cam_port, conf=self._conf, secs=3.0, save_dir=self._save_dir)
        out = []
        for i, d in enumerate(dets):
            p = np.array(d.torso, dtype=float)
            # 収穫済み位置の近傍は除外（再検出の重複防止）
            if any(np.linalg.norm(p - q) < self._dedup_m for q in self._picked_pos):
                continue
            out.append(Okra(id=f"y{i}_{d.conf:.2f}",
                            pos_3d={"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                            ripeness=1.0, reachable=True))
        print(f"[yolo-skills] detect: YOLO {len(dets)}個 → 未収穫 {len(out)}個", flush=True)
        return out

    def grasp_okra(self, okra: Okra, force: float) -> None:
        p = np.array([okra.pos_3d["x"], okra.pos_3d["y"], okra.pos_3d["z"]], dtype=float)
        res = self._ik.solve(p, [0.0] * 29)
        if res is None:
            print(f"[yolo-skills] {okra.id} IK 解けず（skip）", flush=True)
            return
        print(f"[yolo-skills] GRASP {okra.id} torso={np.round(p,3)} IK err={res.err:.4f}", flush=True)
        self._ramp(list(res.arm14), 2.0, grip_q=0.0, weight_to=1.0)   # reach
        self._ramp(list(res.arm14), 1.0, grip_q=_Q_CLOSE)            # close（bridge最近傍把持）
        self._hold(0.6, grip_q=_Q_CLOSE)
        r_lift = self._ik.solve(p + np.array([-0.05, 0.0, 0.18]), [0.0] * 29)  # lift
        if r_lift is not None:
            self._ramp(list(r_lift.arm14), 1.5, grip_q=_Q_CLOSE)
        self._hold(0.4, grip_q=_Q_CLOSE)
        self._picked_pos.append(p)

    def verify_harvest(self) -> bool:
        return True

    def record_harvest(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        print(f"[yolo-skills] record: {record}", flush=True)

    def relative_move(self, lateral: float, forward: float = 0.0, yaw: float = 0.0) -> None:
        print(f"[yolo-skills] relative_move（sim no-op）", flush=True)

    def go_to_next_station(self) -> bool:
        return False

    def swap_basket(self) -> None:
        print("[yolo-skills] swap_basket（sim no-op）", flush=True)
