#!/usr/bin/env python3
"""YOLO 検出で LangGraph harvest を sim 駆動する HarvestSkills（.venv・知覚も本物）。

`sim_harvest_skills.SimHarvestSkills`（GT 座標注入版）の **detect だけ実 YOLO に置換**した版:
  - detect_okra : bridge の ego_view を実 YOLO-seg(okra11n-seg.pt) で検出→torso 3D（`sim_yolo_lib`）。
                  GT 注入なし＝知覚も本物。near クリップ修正で sim 検出が動く（exp 記録参照）。
  - grasp_okra  : IkApproachSkill(pinocchio)→`rt/arm_sdk` reach→**cut_ok VLM ゲート**→
                  `rt/dex1/right/cmd` close→lift→（把持中フレームを verify 用に保持）→籠収納。
                  把持対象 prim は bridge 側の最近傍自動把持(SIM_GRASP_NEAREST=1)が選ぶ＝index 不要。
  - verify_harvest : **moondream（Jetson, tailscale）で把持成否を判定**（既定 liveness=非空応答→True）。
  - record / relative_move,swap_basket=no-op。

VLM 統合（F-02・SS-02・計画書 §5/§8 7-C）:
  - **cut_ok ゲート**（grasp 内・閉じる直前）: ego_view を moondream に送り、応答が返れば切断/把持へ。
    ollama 停止/未到達なら **False＝切らない**（安全側, §3.1）。
  - **verify**（verify ノード）: 把持中に取得したフレームを moondream で判定。
  - sim 画像は OOD ＝ **判定の正しさは実機**（§10）。sim では「VLM が在ループで応答する」配管を確認。
  - env: ``SIM_VLM``(既定1=有効, 0で従来 verify=True/ゲート無し), ``SIM_VLM_HOST``
    (既定 http://100.113.43.64:11434), ``SIM_VLM_MODEL``(moondream),
    ``SIM_VLM_VERIFY_MODE``(liveness|caption), ``SIM_VLM_GRAB_SECS``(既定1.5)。
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

import json

import numpy as np

from dimos.robot.unitree.g1.harvest.blackboard import Okra
from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill
import sim_yolo_lib
from sim_vlm_liveness import (
    CUT_OK_PROMPT,
    make_cut_ok_liveness,
    make_verify_vlm,
    ollama_reachable,
)

# arm14 の並び（_send と一致）: 左7 + 右7、各 [shoulder_pitch, shoulder_roll, shoulder_yaw,
# elbow, wrist_roll, wrist_pitch, wrist_yaw]。drop_poses.json のキー（_joint 無し）に対応。
_CANON14 = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
_DROP_POSES = os.path.join(REPO, "docs/sim-setup/drop_poses.json")


def _pose14(d_right: dict, d_left: dict) -> list[float]:
    """右腕 dict + 左腕 dict（drop_poses.json の姿勢）→ arm14（左7+右7）。未指定キーは 0。"""
    merged = {**d_left, **d_right}
    return [float(merged.get(k, 0.0)) for k in _CANON14]


def _graph_to_ik(p_graph) -> "np.ndarray":
    """graph 系 [x=右, y=前, z=上] → IkApproachSkill 入力系 [Xf=前, Yl=左, Zu=上]。
    Xf=graph.y, Yl=-graph.x, Zu=graph.z（detect の逆変換）。"""
    return np.array([float(p_graph[1]), -float(p_graph[0]), float(p_graph[2])], dtype=float)


_WEIGHT_IDX = 29
_ARM_START = 15
_Q_CLOSE = 4.4


class SimYoloHarvestSkills:
    """detect=実YOLO / grasp=IK+dex1（cut_ok VLM ゲート付き）/ verify=moondream の HarvestSkills。"""

    def __init__(self, *, iface: str = "lo", peers: list[str] | None = None,
                 cam_host: str = "127.0.0.1", cam_port: int = 5555,
                 conf: float = 0.25, dedup_m: float = 0.06,
                 save_dir: str | None = None,
                 basket_torso: tuple[float, float, float] = (0.23, 0.011, -0.072)) -> None:
        self._ik = IkApproachSkill()
        _bt = os.getenv("SIM_BASKET_TORSO")
        self._basket_torso = tuple(float(x) for x in _bt.split(",")) if _bt else basket_torso
        self._cam_host, self._cam_port, self._conf = cam_host, cam_port, conf
        self._dedup_m = dedup_m
        self._save_dir = save_dir
        self._picked_pos: list[np.ndarray] = []   # 収穫済みオクラの torso 位置（再検出の重複除外）
        self._skipped_pos: list[np.ndarray] = []  # cut_ok ゲートで見送ったオクラ（再試行ループ防止）
        self.records: list[dict] = []
        # base 横移動（reposition）: world 累積オフセットを file で bridge(SIM_BASE_MOVE=1)へ渡す
        self._base_xy = [0.0, 0.0]
        self._base_move_file = os.getenv("SIM_BASE_MOVE_FILE", "/tmp/sim_base_move.txt")

        # ---- VLM（moondream）配線: cut_ok ゲート + 把持成否 verify（F-02 / SS-02 / §8 7-C）----
        self._vlm = os.getenv("SIM_VLM", "1") not in ("0", "false", "False", "")
        self._vlm_host = os.getenv("SIM_VLM_HOST", "http://100.113.43.64:11434")
        self._vlm_model = os.getenv("SIM_VLM_MODEL", "moondream")
        self._verify_mode = os.getenv("SIM_VLM_VERIFY_MODE", "liveness")
        self._vlm_grab_secs = float(os.getenv("SIM_VLM_GRAB_SECS", "1.5"))
        self._verify_frame = None  # grasp が把持中に確保し verify_harvest が判定するフレーム
        if self._vlm:
            self._cut_ok = make_cut_ok_liveness(
                self._grab_bgr, host=self._vlm_host, model=self._vlm_model, prompt=CUT_OK_PROMPT)
            self._verify_fn = make_verify_vlm(
                lambda: self._verify_frame, mode=self._verify_mode,
                host=self._vlm_host, model=self._vlm_model)
            reachable = ollama_reachable(self._vlm_host)
            print(f"[yolo-skills] VLM=ON host={self._vlm_host} model={self._vlm_model} "
                  f"verify={self._verify_mode} reachable={reachable}", flush=True)
            if not reachable:
                print("[yolo-skills] ⚠️ ollama 未到達: cut_ok は安全側 False＝切らない（picks=0 になる）。"
                      "tailscale と Jetson moondream を確認。VLM 無効化は SIM_VLM=0", flush=True)
        else:
            self._cut_ok = None
            self._verify_fn = None
            print("[yolo-skills] VLM=OFF（SIM_VLM=0）: cut_ok ゲート無し・verify=True 固定（従来挙動）", flush=True)

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
        # F-07 籠収納の姿勢（参照 dex1_1_service drop_to_basket_isaac と同じ「実測固定角の再生」）。
        # IK で籠を解くと手と籠が衝突したため、body_center 経由で記録済み固定角へ補間する方式に変更。
        _dp = json.load(open(_DROP_POSES))
        self._body14 = _pose14(_dp["right_arm_body_center"], _dp["left_arm_body_center"])
        self._drop14 = _pose14(_dp["right_arm_drop_pose"], _dp["left_basket_pose"])
        print(f"[yolo-skills] DDS up iface={iface} peers={peers or 'mcast'} (warmup); "
              f"cam={cam_host}:{cam_port} conf>={conf}; F-07=記録固定角(body_center→drop)", flush=True)

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

    def _grab_bgr(self):
        """bridge の ego_view を1フレーム取得し BGR(numpy) を返す（VLM 入力用）。無ければ None。"""
        fr = sim_yolo_lib.grab_frame(self._cam_host, self._cam_port, secs=self._vlm_grab_secs)
        return fr[0] if fr else None  # grab_frame は (bgr, depth, K, c2t)

    # ---- HarvestSkills ----
    def detect_okra(self) -> list[Okra]:
        dets = sim_yolo_lib.detect_okra_torso(
            self._cam_host, self._cam_port, conf=self._conf, secs=3.0, save_dir=self._save_dir)
        out = []
        for i, d in enumerate(dets):
            # YOLO の torso 出力は IK 系 (Xf=前, Yl=左, Zu=上)。graph の空間モデルは
            # x=lateral(+右), y=depth(+前), z=高さ（blackboard §spatial）。両者の軸ねじれが
            # reposition の誤移動(後退)の原因だったため、**pos_3d は graph 系で持つ**ことに統一する。
            #   graph.x(右) = -Yl,  graph.y(前) = Xf,  graph.z = Zu
            xf, yl, zu = float(d.torso[0]), float(d.torso[1]), float(d.torso[2])
            p = np.array([-yl, xf, zu], dtype=float)  # graph 系
            # 収穫済み / cut_ok 見送り位置の近傍は除外（再検出の重複・再試行ループ防止）。
            # YOLO の id は検出毎に変わるためグラフの excluded_ids では弾けず、位置で除外する。
            if any(np.linalg.norm(p - q) < self._dedup_m
                   for q in (*self._picked_pos, *self._skipped_pos)):
                continue
            out.append(Okra(id=f"y{i}_{d.conf:.2f}",
                            pos_3d={"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                            ripeness=1.0, reachable=True))
        print(f"[yolo-skills] detect: YOLO {len(dets)}個 → 未収穫 {len(out)}個", flush=True)
        return out

    def grasp_okra(self, okra: Okra, force: float) -> None:
        self._verify_frame = None  # 今回の把持で取り直す（前回フレームの取り違え防止）
        p = np.array([okra.pos_3d["x"], okra.pos_3d["y"], okra.pos_3d["z"]], dtype=float)  # graph 系
        p_ik = _graph_to_ik(p)  # IK 入力系 [前,左,上] へ変換
        res = self._ik.solve(p_ik, [0.0] * 29)
        if res is None:
            print(f"[yolo-skills] {okra.id} IK 解けず（skip）", flush=True)
            return
        print(f"[yolo-skills] GRASP {okra.id} graph={np.round(p,3)} ik={np.round(p_ik,3)} err={res.err:.4f}", flush=True)
        self._ramp(list(res.arm14), 2.0, grip_q=0.0, weight_to=1.0)   # reach（grip 開で寄せる）
        # ③ 切断可否ゲート（SS-02 §3.1）: 閉じる(=切断+把持)直前に moondream へ問う。
        #    応答が返れば把持へ。ollama 停止/未到達は False＝切らない（安全側, §10）。
        if self._cut_ok is not None:
            ok = self._cut_ok()
            print(f"[yolo-skills] {okra.id} cut_ok VLM(moondream) → {ok}", flush=True)
            if not ok:
                print(f"[yolo-skills] {okra.id} → 把持中止（安全側・このオクラは見送り）", flush=True)
                self._skipped_pos.append(p)  # 再検出時に同じオクラを再試行しない
                return
        self._ramp(list(res.arm14), 1.0, grip_q=_Q_CLOSE)            # close（bridge最近傍把持）
        self._hold(0.6, grip_q=_Q_CLOSE)
        # F-02 把持成否 verify 用フレーム: 把持中（閉じ・reach 姿勢＝カメラ中央にオクラ）に確保し、
        # verify ノードで moondream 判定する。籠投入後だとグリッパが空なので、ここで取る。
        self._verify_frame = self._grab_bgr()
        r_lift = self._ik.solve(p_ik + np.array([-0.05, 0.0, 0.18]), [0.0] * 29)  # lift（IK系: 手前へ引き＋上）
        if r_lift is not None:
            self._ramp(list(r_lift.arm14), 1.5, grip_q=_Q_CLOSE)
        self._hold(0.4, grip_q=_Q_CLOSE)
        # F-07 籠収納（参照 drop_to_basket_isaac と同方式＝記録固定角の再生・現在角から補間）:
        #   把持後(開始姿勢はオクラ毎に違う) → ① body center へ集約（手と籠の衝突回避）
        #   → ② 右腕 drop_pose / 左腕 basket_pose（固定角）へ → ③ 開いて離す（重力で籠へ落下）→ ④ center 復帰
        self._place_to_basket()
        print(f"[yolo-skills] F-07 籠収納 OK（body_center→drop 固定角再生）", flush=True)
        self._picked_pos.append(p)

    def _place_to_basket(self) -> None:
        """記録済み固定角で籠投入（IK 不使用。開始姿勢が毎回違っても body center 経由で安全）。"""
        self._ramp(self._body14, 1.8, grip_q=_Q_CLOSE)   # ① 両腕→body center（持ち上げ集約）
        self._ramp(self._drop14, 2.0, grip_q=_Q_CLOSE)   # ② 右腕drop/左腕basket（固定角）
        self._hold(0.4, grip_q=_Q_CLOSE)
        self._hold(1.2, grip_q=0.0)                       # ③ 開いて離す→重力で籠へ落下
        self._ramp(self._body14, 1.5, grip_q=0.0)        # ④ body center 復帰（次の検出を妨げない）

    def verify_harvest(self) -> bool:
        # VLM 無効時は従来どおり True 固定。有効時は把持中フレームを moondream で判定。
        if self._verify_fn is None:
            return True
        ok = self._verify_fn()  # frame 無し（cut_ok 見送り 等）→ False
        print(f"[yolo-skills] verify(VLM {self._verify_mode}) → {ok}", flush=True)
        return ok

    def record_harvest(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        print(f"[yolo-skills] record: {record}", flush=True)

    def relative_move(self, lateral: float, forward: float = 0.0, yaw: float = 0.0) -> None:
        """reposition/advance_left の base 移動。graph フレーム(lateral=+右, forward=+前)を
        world(+x=前, +y=左)へ変換し world 累積オフセットを bridge へ渡す（bridge が set_world_pose）。
        SIM_BASE_MOVE=1 の bridge が反映。base 移動後、次の detect(YOLO) が新 torso 相対で再検出する。
        """
        # graph: lateral(+右), forward(+前) → world: dx=前=forward, dy=左=-lateral
        self._base_xy[0] += float(forward)
        self._base_xy[1] += -float(lateral)
        try:
            with open(self._base_move_file, "w") as f:
                f.write(f"{self._base_xy[0]:.4f},{self._base_xy[1]:.4f}")
        except OSError as e:
            print(f"[yolo-skills] base_move file write fail: {e}", flush=True)
        # base が動くと収穫済み/見送りの torso 相対位置も変わる＝dedup をリセット（新フレームで再評価）
        self._picked_pos.clear()
        self._skipped_pos.clear()
        print(f"[yolo-skills] relative_move(lat={lateral:+.2f},fwd={forward:+.2f}) "
              f"→ base world=({self._base_xy[0]:+.2f},{self._base_xy[1]:+.2f})", flush=True)
        time.sleep(1.0)  # base 移動の整定待ち（bridge が set_world_pose→数ステップ）

    def go_to_next_station(self) -> bool:
        return False

    def swap_basket(self) -> None:
        print("[yolo-skills] swap_basket（sim no-op）", flush=True)
