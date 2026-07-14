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


def _sweet_env(name: str, default: tuple[float, float]) -> tuple[float, float]:
    """スイートスポット窓を env `name`="min,max" で上書き（無ければ default）。"""
    v = os.getenv(name)
    if v:
        a, b = (float(x) for x in v.split(","))
        return (a, b)
    return default


# 把持スイートスポット（okra live torso 座標）[m]。窓が狭いと「十分掴める位置でも把持が
# 発火せず移動を反復」してしまう。腕は IK ワークスペース(ws_y=-0.75..0.20)的に横へ広く届く
# ので、横(y)窓を ~60cm に広げて余計な横歩き（=ヨードリフト源）を減らす。残差は reach＋
# Δサーボが吸収。x は前方リーチ上限(X_CMD_MAX=0.52)で ~24cm が限界。現場調整は env で:
#   SIM_WALK_SWEET_X="min,max" / SIM_WALK_SWEET_Y="min,max"
_SWEET_X = _sweet_env("SIM_WALK_SWEET_X", (0.24, 0.44))
# 横 y は成功帯に絞る（実測: 把持成立は y≈-0.06〜-0.11 に集中。右に寄ると失敗）。中心-0.08・幅12cm。
# 骨盤ピン留め（GRASP_HOLD）で side-step のヨードリフトを吸収できるので、絞って寄せ直しても倒れない。
_SWEET_Y = _sweet_env("SIM_WALK_SWEET_Y", (-0.14, -0.02))  # 幅 12cm（成功帯中心 -0.08）

# 把持中の骨盤ピン留めモード（bridge の SIM_WALK_GRASP_HOLD と対で使う）。
# ON: 位置合わせ後〜lift まで脚凍結を張り続ける → bridge が骨盤をキネマティックにピン留めするので
#     「倒れない土台」で精密把持できる（骨盤ピン中は歩けないので半歩前進はスキップ＝遠め x スタンス前提）。
# OFF: 従来どおり reach/servo は policy に踏ん張らせ、close の一瞬だけ凍結。
_GRASP_HOLD = os.getenv("SIM_WALK_GRASP_HOLD", "0") == "1"

# 実測 base_x の目標スタンス[m]: okra_now.x 推定は spawn 依存でズレる（届かず/近づきすぎ両方を起こす）
# ため、前後の停止は「壊れた推定」ではなく bridge 実測 base_x を**この値まで詰める**で決める。
# 単発把持成功は base_x≈0.08（オクラ torso-x≈0.42 に腕が届く）・机の前板(world 0.35)にも当たらない
# ので既定 0.08。前進(_align_to)も grasp 中の半歩前進も、この値を越えては進めない（机ジャム防止）。
# 机が近い/遠い現場では SIM_WALK_BASE_X_MAX で調整。
_BASE_X_MAX = float(os.getenv("SIM_WALK_BASE_X_MAX", "0.08"))

# 倒れ検知の z 閾値[m]: 立位オクラ中心は z≈0.77、把持成功は >0.82。莢が倒される（なぎ倒し/
# 上から小突き）と z が大きく下がる（実測 0.725）。これを下回る対象は「掴めない」とみなし、
# 物理接近せずスキップする（倒れたオクラへ再把持しに行って散乱を広げるのを防ぐ）。
_FALLEN_Z = float(os.getenv("SIM_WALK_FALLEN_Z", "0.74"))


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

    def _pulse(self, vx: float, vy: float, secs: float, dist: float | None = None) -> None:
        """base 速度パルス。dist[m] 指定時は**変位ベース**: その距離動くまで指令を出し続ける
        （GUI は描画で実時間が遅く、時間ベースだとシム内歩行が1/3以下になり接近が破綻する）。"""
        bx0, by0 = self._base_xy()
        t0 = time.time()
        timeout = max(secs, (dist / 0.25 + 2.0) if dist else secs) + 8.0  # 実時間の安全上限
        while time.time() - t0 < timeout:
            with open(_BASE_MOVE_FILE, "w") as f:
                f.write(f"{vx},{vy}")
            self._hold(0.1, grip_q=self._grip_q)  # 歩行中も腕目標を保持し続ける
            if dist is not None:
                bx, by = self._base_xy()
                if ((bx - bx0) ** 2 + (by - by0) ** 2) ** 0.5 >= dist:
                    break
            elif time.time() - t0 >= secs:
                break
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
        # 収穫順は左端優先（+y=ロボットの左）。左手首に固定のバスケットが未収穫オクラを
        # なぎ倒すのを防ぐため、左端から取り始めてロボットを右へ進める（バスケットは
        # 既収穫側=左へ退くので未収穫オクラに当たらない）。並びは calib torso y の降順。
        # graph の select は熟度同点を検出順で拾う（安定ソート）ため、この順が把持順になる。
        out = []
        for k in sorted(self._okra, key=lambda kk: float(self._okra[kk][1][1]), reverse=True):
            if k in self._picked:
                continue
            p = self._okra_now(k)
            out.append(Okra(id=k, pos_3d={"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                            ripeness=1.0, reachable=True))
        return out

    def relative_move(self, lateral: float, forward: float = 0.0, yaw: float = 0.0) -> None:
        """graph からの相対移動（lateral>0=左）。デッドバンド回避のため v=0.3 の時間パルスで実現。"""
        if abs(lateral) > 0.02:
            self._pulse(0.0, 0.3 if lateral > 0 else -0.3, abs(lateral) / 0.3, dist=abs(lateral))
        if abs(forward) > 0.02:
            self._pulse(0.3 if forward > 0 else -0.3, 0.0, abs(forward) / 0.3, dist=abs(forward))
        print(f"[walk-skills] relative_move(lat={lateral:+.2f},fwd={forward:+.2f}) → base={self._base_xy()}", flush=True)

    def _align_to(self, oid: str, max_moves: int = 12) -> np.ndarray:
        """対象オクラをスイートスポットへ（前後 vx / 横 vy の1歩パルスで位置合わせ）。

        max_moves=12: 骨盤ピン後は歩けない（半歩前進も無効）ので、ピン前の align で
        腕の到達域（x 窓 0.28-0.33 = gap が届く最遠）まで**接近し切る**必要がある。歩数を
        削りすぎると x が届かず「オクラが指の股に入らない＝なぞる」になる（実測）。
        """
        for _ in range(max_moves):
            p = self._okra_now(oid)
            bx = self._base_pose()[0]  # bridge 実測 base_x（ground truth・壊れていない数字）
            # 前後(x): okra_now.x は spawn 依存でズレる（届かず/近づきすぎ両方を起こす）ので使わず、
            #   実測 base_x を実績スタンス _BASE_X_MAX まで詰める。到達で停止・越えたら後退（机ジャム防止）。
            if bx < _BASE_X_MAX - 0.03:
                dx = 0.3
            elif bx > _BASE_X_MAX + 0.03:
                dx = -0.3
            else:
                dx = 0.0
            # 横(y): okra_now.y は正確なのでスイートスポットへ寄せる
            #   オクラが右(y<sweet)にある→ロボットが右へ動く(vy<0)とオクラの相対yは増える
            dy = 0.0 if _SWEET_Y[0] <= p[1] <= _SWEET_Y[1] else (-0.3 if p[1] < _SWEET_Y[0] else 0.3)
            if dx == 0.0 and dy == 0.0:
                print(f"[walk-skills]   位置合わせ完了 okra_now={tuple(round(float(v),3) for v in p)}", flush=True)
                return p
            # 変位ベース1歩（GUI/headless の実時間差に不変）。横はデッドバンド対策で長め上限
            self._pulse(dx, dy, 0.55 if dy != 0.0 else 0.3, dist=0.06)
        p = self._okra_now(oid)
        print(f"[walk-skills]   位置合わせ打ち切り okra_now={tuple(round(float(v),3) for v in p)}", flush=True)
        return p

    # ---- 脚凍結（FixStand 相当）: bridge が /tmp/sim_walk_freeze を見て policy を止め剛PD保持 ----
    def _freeze_legs(self) -> None:
        open("/tmp/sim_walk_freeze", "w").close()

    def _unfreeze_legs(self) -> None:
        try:
            os.remove("/tmp/sim_walk_freeze")
        except OSError:
            pass

    def _okra_z_live(self, idx: int) -> float | None:
        """bridge ログから対象オクラ(idx)の最新 okra_z を読む（倒れ検知用）。nearest=Okra_idx 行のみ。"""
        m = re.findall(rf"nearest=Okra_{idx} okra_z=([0-9.]+)", self._blog._tail())
        return float(m[-1]) if m else None

    def grasp_okra(self, okra: Okra, force: float) -> None:  # noqa: ARG002
        idx, _p0 = self._okra[okra.id]
        # 倒れ検知の分岐: 対象が既に倒れている（z ≤ _FALLEN_Z）なら掴めないので物理接近せずスキップ。
        # 再把持ループ（route_after_verify が同じ target へ GRASP を繰り返す）で倒れたオクラに
        # 何度も突っ込む／さらに散らすのを防ぐ。除外に入れて次のオクラへ進ませる。
        _zl = self._okra_z_live(idx)
        if _zl is not None and _zl <= _FALLEN_Z:
            print(f"[walk-skills] okra{okra.id} 倒れ検知 (z={_zl:.3f}≤{_FALLEN_Z}) → 把持スキップ（取れない）",
                  flush=True)
            self._last_grasp_ok = False
            self._picked.add(okra.id)
            return
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
        # A) 位置合わせ歩行（挙手のまま歩く。ここは歩くので凍結しない）
        print(f"[walk-skills] GRASP okra{okra.id}: 位置合わせ歩行", flush=True)
        p = self._align_to(okra.id)

        # 位置合わせ後の把持工程（reach→servo→close→lift）。
        # held(ON): ここで脚凍結を張り続け、bridge が骨盤をピン留め＝倒れない土台で精密把持。
        #           骨盤ピン中は歩けないので半歩前進はスキップ（遠め x スタンス前提）。
        # held(OFF): reach/servo は policy に踏ん張らせ（凍結すると転倒）、close の一瞬だけ凍結。
        # いずれも早期 return で確実に解除するよう try/finally で保護。
        held = _GRASP_HOLD
        if held:
            self._freeze_legs()
        try:
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
            # C) Δサーボ（fresh 観測・ゲイン0.6±4cm・x上限で半歩前進）。
            # 反復は既定3（単発成功時2反復で収束。x は _align_to が base_x で先に詰めるので z/y 微調整のみ）。
            for it in range(int(os.getenv("SIM_WALK_SERVO_ITERS", "3"))):
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
                    # 半歩前進で x 不足を詰める。ただし実測 base_x が実績スタンスを越えては進めない
                    #（机ジャム防止）。held(骨盤ピン)中は歩けないのでスキップ。
                    if not held and self._base_pose()[0] < _BASE_X_MAX:
                        self._pulse(0.3, 0.0, 0.25, dist=0.04)  # 半歩前進（変位ベース）
                tgt[1] = float(np.clip(tgt[1], -0.45, 0.15))  # 横窓を追随（IK ws_y 内）
                # z 下限: オクラ中心は torso z≈-0.066。ここが高いと骨盤ピンで胴体が高く保持される分
                # ハンドがオクラ上空で空振りする（friction_pick_servo で実証: 下限 -0.08 では隙間が
                # 莢の 5cm 上で頭打ち＝空を握る。-0.10 まで下げて初めて降下→接触→摩擦保持が通った）。
                # 既定 -0.12 で莢中心まで確実に降ろす。SIM_WALK_SERVO_Z_MIN で現場調整可。
                tgt[2] = float(np.clip(tgt[2], float(os.getenv("SIM_WALK_SERVO_Z_MIN", "-0.12")), 0.16))
                r = self._ik.solve(tgt, [0.0] * 29)
                if r is None:
                    break
                self._ramp(list(r.arm14), 1.2, grip_q=0.0)
                self._blog.mark()
                self._hold(1.8, grip_q=0.0)
            # D) close → lift → 検証。held(OFF) は close の一瞬だけ脚凍結（~2.3s＝準安定限界内）で
            # 立位の揺れ（±1-2cm）を消す。held(ON) は既に骨盤ピン中なので追加凍結は不要。
            self._blog.mark()
            if not held:
                self._freeze_legs()
            try:
                self._ramp(self._cur_arm, 2.5, grip_q=_Q_CLOSE)  # close は緩やかに（急閉じで剛体オクラを弾き出さない）
                self._hold(1.0, grip_q=_Q_CLOSE)
            finally:
                if not held:
                    self._unfreeze_legs()
            lift = np.array([tgt[0] - 0.05, tgt[1], tgt[2] + 0.18])
            r_l = self._ik.solve(lift, [0.0] * 29)
            if r_l is not None:
                self._ramp(list(r_l.arm14), 2.0, grip_q=_Q_CLOSE)
            self._hold(0.8, grip_q=_Q_CLOSE)
            okz = self._blog.okra_z_max()
            self._last_grasp_ok = okz > 0.82
            print(f"[walk-skills]   lift 後 okra_z(max)={okz:.3f} → {'✅把持' if self._last_grasp_ok else '❌未把持'}", flush=True)
        finally:
            if held:  # 把持工程終了（成功/失敗/早期return）で骨盤ピンを解除→次のオクラへ歩ける
                self._unfreeze_legs()
        # E) 籠投入（既存 F-07 place を流用）。
        # ※後退モーションは廃止: 歩行 policy が弱く後ろ歩きで転倒するため。代わりに遠めの x
        #   スタンス（SIM_WALK_SWEET_X を届く最遠へ）で最初から机と距離を取り、カゴの引っかかりを避ける。
        ok = self._place()
        print(f"[walk-skills] F-07 籠収納 {'OK' if ok else 'FAIL'}", flush=True)
        self._picked.add(okra.id)

    def verify_harvest(self) -> bool:
        return bool(getattr(self, "_last_grasp_ok", True))
