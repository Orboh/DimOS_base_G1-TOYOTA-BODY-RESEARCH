#!/usr/bin/env python3
"""歩行モード（SIM_WALK_POLICY=1）用 HarvestSkills — LangGraph 収穫を「歩く仮想G1」で駆動。

SimHarvestSkills を継承し、以下を歩行対応に差し替える:
  - detect_okra    : bridge 起動時の [calib] 実測を base 移動量で補正した live torso 座標を返す
  - grasp_okra     : 事前挙手 → 位置合わせ歩行（前後 vx / 横 vy）→ Δサーボ → close → lift 検証 → 籠投入
  - relative_move  : base_move ファイルへの vy パルス（横移動収穫）

観測は bridge ログ（env BRIDGE_LOG 必須）。実測知見（メモリ g1-isaac-policy-walk-floating-base）:
  - 事前挙手（腕を机より上へ上げてから歩く）でリーチ後ずさりクリープが消える（第11知見）
  - 指令デッドバンド: vx<0.2 / vy<0.3 では歩き出さない → パルスは vx=±0.3 / vy=±0.3
  - 把持スイートスポット（okra live torso）: x∈[0.30,0.40], y∈[-0.19,-0.10]
"""
from __future__ import annotations

import os
import re
import sys
import time

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from dimos.robot.unitree.g1.harvest.blackboard import Okra
from sim_harvest_skills import SimHarvestSkills, _Q_CLOSE
from walk_approach_pick import BridgeLog, X_CMD_MAX, X_TIP, Z_COMP

_BASE_MOVE_FILE = os.getenv("SIM_BASE_MOVE_FILE", "/tmp/sim_base_move.txt")
# 把持スイートスポット（okra live torso 座標）[m]
_SWEET_X = (0.30, 0.35)  # 遠端0.4だとcap干渉でz収束せず把持ミス（実測: 成功は全て0.32-0.33）
_SWEET_Y = (-0.19, -0.10)


def _parse_calib(log_path: str) -> list[tuple[int, tuple[float, float, float]]]:
    """bridge 起動時の `[calib] Okra_i torso=(x, y, z)` を全部読む（カンマ後の空白あり形式）。"""
    txt = open(log_path, errors="replace").read()
    out = []
    for m in re.finditer(r"\[calib\] Okra_(\d+) torso=\((-?[0-9.]+), (-?[0-9.]+), (-?[0-9.]+)\)", txt):
        out.append((int(m.group(1)), (float(m.group(2)), float(m.group(3)), float(m.group(4)))))
    return out


class WalkHarvestSkills(SimHarvestSkills):
    """歩く仮想G1のための HarvestSkills（bridge ログを閉ループ観測に使う）。"""

    def __init__(self, *, iface: str = "lo", peers: list[str] | None = None,
                 pick_ids: list[int] | None = None) -> None:
        log_path = os.environ["BRIDGE_LOG"]
        self._blog = BridgeLog(log_path)
        calib = _parse_calib(log_path)
        if pick_ids is not None:
            calib = [(i, p) for i, p in calib if i in pick_ids]
        if not calib:
            raise RuntimeError("bridge ログに [calib] Okra が無い（SIM_GRASP_FRICTION=1 で起動したか）")
        # calib は spawn 位置での torso 座標。base 移動で補正するため spawn を控える。
        self._spawn = (float(os.getenv("SIM_WALK_SPAWN_X", "0.0")),
                       float(os.getenv("SIM_WALK_SPAWN_Y", "0.0")))
        super().__init__(okra_torso=calib, iface=iface, peers=peers)
        print(f"[walk-skills] 歩行モード skills 起動 pick対象={sorted(i for i,_ in calib)} spawn={self._spawn}", flush=True)

    # ---- base 移動（LocoClient 相当・file プロトコル） ----
    def _base_pose(self) -> tuple[float, float, float]:
        """base の (x, y, yaw[rad])。yaw は横歩きで漂う（-0.1rad 程度）ため座標補正に必須。"""
        m = re.findall(r"base=\((\+?-?[0-9.]+),(\+?-?[0-9.]+),[0-9.]+\) yaw=(\+?-?[0-9.]+)",
                       self._blog._tail())
        if m:
            return (float(m[-1][0]), float(m[-1][1]), float(m[-1][2]))
        return (self._spawn[0], self._spawn[1], 0.0)

    def _base_xy(self) -> tuple[float, float]:
        p = self._base_pose()
        return (p[0], p[1])

    def _pulse(self, vx: float, vy: float, secs: float) -> None:
        t0 = time.time()
        while time.time() - t0 < secs:
            with open(_BASE_MOVE_FILE, "w") as f:
                f.write(f"{vx},{vy}")
            self._hold(0.1, grip_q=self._grip_q)  # 歩行中も腕目標を保持し続ける
        try:
            os.remove(_BASE_MOVE_FILE)
        except OSError:
            pass
        self._hold(2.5, grip_q=self._grip_q)  # 静定

    def _okra_now(self, oid: str) -> np.ndarray:
        """calib 座標を base の SE(2) 移動（並進＋yaw）で補正した live torso 座標。

        okra_world = calib(spawn時 torso≒world-spawn) + spawn。live torso = R(-yaw)·(okra_world - base)。
        横歩きパルスで yaw が -0.1rad 程度漂うため回転補正が必須（無いと y に ~5cm 誤差）。
        """
        _idx, p0 = self._okra[oid]
        ox, oy = p0[0] + self._spawn[0], p0[1] + self._spawn[1]  # world 近似
        bx, by, yaw = self._base_pose()
        dx, dy = ox - bx, oy - by
        c, s_ = np.cos(-yaw), np.sin(-yaw)
        return np.array([c * dx - s_ * dy, s_ * dx + c * dy, p0[2]], dtype=float)

    # ---- HarvestSkills 差し替え ----
    def detect_okra(self) -> list[Okra]:
        out = []
        for k in sorted(self._okra, key=int):
            if k in self._picked:
                continue
            p = self._okra_now(k)
            out.append(Okra(id=k, pos_3d={"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                            ripeness=1.0, reachable=True))
        return out

    def relative_move(self, lateral: float, forward: float = 0.0, yaw: float = 0.0) -> None:
        """graph からの相対移動（lateral>0=左）。デッドバンド回避のため v=0.3 の時間パルスで実現。"""
        if abs(lateral) > 0.02:
            self._pulse(0.0, 0.3 if lateral > 0 else -0.3, min(2.0, abs(lateral) / 0.3))
        if abs(forward) > 0.02:
            self._pulse(0.3 if forward > 0 else -0.3, 0.0, min(2.0, abs(forward) / 0.3))
        print(f"[walk-skills] relative_move(lat={lateral:+.2f},fwd={forward:+.2f}) → base={self._base_xy()}", flush=True)

    def _align_to(self, oid: str, max_moves: int = 8) -> np.ndarray:
        """対象オクラをスイートスポットへ（前後 vx / 横 vy の1歩パルスで位置合わせ）。"""
        for _ in range(max_moves):
            p = self._okra_now(oid)
            dx = 0.0 if _SWEET_X[0] <= p[0] <= _SWEET_X[1] else (0.3 if p[0] > _SWEET_X[1] else -0.3)
            # オクラが右(y<sweet)にある→ロボットが右へ動く(vy<0)とオクラの相対yは増える
            dy = 0.0 if _SWEET_Y[0] <= p[1] <= _SWEET_Y[1] else (-0.3 if p[1] < _SWEET_Y[0] else 0.3)
            if dx == 0.0 and dy == 0.0:
                print(f"[walk-skills]   位置合わせ完了 okra_now={tuple(round(float(v),3) for v in p)}", flush=True)
                return p
            # 横歩きはデッドバンドが広く短パルスでは始動しない → 横は 0.55s / 前後は 0.3s
            self._pulse(dx, dy, 0.55 if dy != 0.0 else 0.3)
        p = self._okra_now(oid)
        print(f"[walk-skills]   位置合わせ打ち切り okra_now={tuple(round(float(v),3) for v in p)}", flush=True)
        return p

    def grasp_okra(self, okra: Okra, force: float) -> None:  # noqa: ARG002
        idx, _p0 = self._okra[okra.id]
        try:
            with open(self._target_file, "w") as f:
                f.write(str(idx))
        except Exception as e:  # noqa: BLE001
            print(f"[walk-skills] target file write fail: {e}", flush=True)

        # 0) 事前挙手（第11知見: 静止リーチのクリープ回避。机より上・前方）
        pre = np.array([0.40, -0.16, 0.20])
        r0 = self._ik.solve(pre, [0.0] * 29)
        if r0 is not None:
            self._ramp(list(r0.arm14), 2.0, grip_q=0.0, weight_to=1.0)
        # A) 位置合わせ歩行（挙手のまま歩く）
        print(f"[walk-skills] GRASP okra{okra.id}: 位置合わせ歩行", flush=True)
        p = self._align_to(okra.id)
        # B) リーチ（実測式: x+X_TIP cap / z+Z_COMP）
        tgt = np.array([min(p[0] + X_TIP, X_CMD_MAX), p[1] + 0.013, p[2] + Z_COMP])
        r = self._ik.solve(tgt, [0.0] * 29)
        if r is None:
            print(f"[walk-skills] okra{okra.id} reach IK 解けず（skip）", flush=True)
            self._picked.add(okra.id)
            return
        self._ramp(list(r.arm14), 2.0, grip_q=0.0)
        self._blog.mark()
        self._hold(1.8, grip_q=0.0)
        # C) Δサーボ（fresh 観測・ゲイン0.6±4cm・x上限で半歩前進）
        for it in range(4):
            d = self._blog.delta_for(idx)
            if d is None:
                break
            print(f"[walk-skills]   servo{it}: Δ={tuple(round(v,3) for v in d)}", flush=True)
            if abs(d[0]) < 0.02 and abs(d[1]) < 0.02 and abs(d[2]) < 0.03:
                break
            dd = np.array(d)
            tgt[0] += 0.6 * float(np.clip(dd[0], -0.04, 0.04))
            tgt[1] += 0.6 * float(np.clip(dd[1], -0.04, 0.04))
            tgt[2] += 1.0 * float(np.clip(dd[2], -0.06, 0.06))  # z は倒す前に一気に合わせる
            if tgt[0] > X_CMD_MAX:
                tgt[0] = X_CMD_MAX
                self._pulse(0.3, 0.0, 0.25)  # 半歩前進で x 不足を詰める
            tgt[1] = float(np.clip(tgt[1], -0.30, 0.05))
            tgt[2] = float(np.clip(tgt[2], 0.00, 0.16))  # 下限0: 莢先端でなく中央を掴む（0.05だと先端でノックダウン）
            r = self._ik.solve(tgt, [0.0] * 29)
            if r is None:
                break
            self._ramp(list(r.arm14), 1.2, grip_q=0.0)
            self._blog.mark()
            self._hold(1.8, grip_q=0.0)
        # D) close → lift → 検証。close の瞬間だけ脚凍結（FixStand 相当）＝立位の揺れ（±1-2cm）を
        # 消して莢(φ2cm)×ジョー(5cm)の余裕を守る。凍結は ~2.5s（静的保持の準安定限界内）。
        self._blog.mark()
        open("/tmp/sim_walk_freeze", "w").close()
        try:
            self._ramp(self._cur_arm, 1.5, grip_q=_Q_CLOSE)
            self._hold(0.8, grip_q=_Q_CLOSE)
        finally:
            try:
                os.remove("/tmp/sim_walk_freeze")
            except OSError:
                pass
        lift = np.array([tgt[0] - 0.05, tgt[1], tgt[2] + 0.18])
        r_l = self._ik.solve(lift, [0.0] * 29)
        if r_l is not None:
            self._ramp(list(r_l.arm14), 2.0, grip_q=_Q_CLOSE)
        self._hold(0.8, grip_q=_Q_CLOSE)
        okz = self._blog.okra_z_max()
        self._last_grasp_ok = okz > 0.82
        print(f"[walk-skills]   lift 後 okra_z(max)={okz:.3f} → {'✅把持' if self._last_grasp_ok else '❌未把持'}", flush=True)
        # E) 籠投入（既存 F-07 place を流用）
        ok = self._place()
        print(f"[walk-skills] F-07 籠収納 {'OK' if ok else 'FAIL'}", flush=True)
        self._picked.add(okra.id)

    def verify_harvest(self) -> bool:
        return bool(getattr(self, "_last_grasp_ok", True))
