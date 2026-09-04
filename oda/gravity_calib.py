#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""重力補償の過不足を実測する較正ツール (2026-08-25).

**原理** — 静止して腕を保持しているとき、各関節では次が釣り合っている:

    0 = kp*(q_指令 - q_実測) + tau_ff - tau_重力
    => q_指令 - q_実測 = (tau_重力 - tau_ff) / kp

つまり **残留誤差 x kp = 未補償の重力トルク** がそのまま読める。腕を水平まで
上げた高負荷姿勢で保持し、この残差を測れば「今の重力補償で何 N*m 足りないか」
「先端に何 kg 足せばゼロになるか」が1回の計測で確定する。

**駆動はしない** — arm_home.py と同じ方式で、アプリ(dimos run ...)が**動いている
間に**別ターミナルから `/g1/arm_target` を流すだけ。実際に腕を動かすのはアプリ内の
G1ArmSdkConnection(weight ramp / clip-to-measured / 関節リミット)なので、本スクリプトは
低レベル制御を一切再実装しない。アプリが居なければ motor_states が来ず安全に終了する。

⚠️ アプリを **IK_REACH_LIVE=1** で起動しておくこと。DRY-RUN では publish_cmd=False
   のため腕にトルクが出ず、測定値は無意味になる(その旨を警告する)。
⚠️ 腕が水平まで上がる。可動範囲を空け、e-stop(L2+B)を手元に。
⚠️ 測定中はビューアをクリックしないこと(IK reach が割り込んで姿勢が変わる)。

実行:
  # 端末1: アプリ(クリックしない)
  OKRA_GRAVITY_FF=1 OKRA_TIP_EXTRA_MASS_KG=0.638 IK_REACH_LIVE=1 \
    ROBOT_INTERFACE=enx6c1ff771dc67 .venv/bin/dimos run unitree-g1-okra-ik-diffusion

  # 端末2: 較正
  .venv/bin/python oda/gravity_calib.py

環境変数:
  GC_POSE            : 測定姿勢 "shoulder_pitch,...,wrist_yaw" (rad,7個)
                       既定 "-1.3963,0,0,1.3963,0,0,0" = 肩ピッチ-80度/肘+80度。
                       グリッド探索で **先端質量への感度が最大** になる姿勢を選定
                       (4.52 N*m/kg, 手先 torso[0.414,-0.046,0.205] = 最も前方)。
                       ⚠ 「肩ピッチ-90度・肘0」は直感に反して腕が上を向き
                       (手先 z=0.478)、レバーが短く感度1.92止まりなので使わない。
                       感度が高いほど質量の分解能が上がる: この姿勢なら
                       0.1kg の誤差が 0.32 deg の残差として現れる
  GC_RAMP_S          : 目標姿勢までの移動時間 [s] 既定 5.0 (ゆっくり=安全)
  GC_SETTLE_S        : 到達後、静定を待つ時間 [s] 既定 3.0
  GC_MEASURE_S       : 平均を取る時間 [s] 既定 2.0
  GC_RATE_HZ         : arm_target の発行レート 既定 20
  GC_TIP_EXTRA_MASS_KG : アプリ側に設定した先端ペイロード [kg] (既定 0.638)
                       ※ ここを実際の起動値と一致させないと逆算がずれる
  GC_TIP_EXTRA_COM_XYZ : 同じく重心 既定 "0.113,-0.003,0.0"
  GC_RETURN          : 測定後にホーム姿勢へ戻す 既定 1 (0でその場保持)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pinocchio

from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.act.ik_reach_bridge import (
    _ARM_JOINT_NAMES,
    _LEFT_SLICE,
    _NUM_ARM,
    _RIGHT_SLICE,
)
from dimos.robot.unitree.g1.ik_reach.right_arm_model import DEFAULT_URDF, load_g1_right_arm_ik

# G1ArmSdkConnection の既定ゲイン。右腕7関節ぶん (idx 4,5,6 = 手首は kp_wrist)。
# アプリ側を OKRA_NOACT_KP_ARM 等で変えたなら、ここも合わせること。
KP_ARM = float(os.getenv("GC_KP_ARM", "80.0"))
KP_WRIST = float(os.getenv("GC_KP_WRIST", "40.0"))
KP = np.array([KP_ARM] * 4 + [KP_WRIST] * 3)
JOINT_NAMES = ["sh_pitch", "sh_roll", "sh_yaw", "elbow", "w_roll", "w_pitch", "w_yaw"]

POSE = np.array([float(v) for v in os.getenv("GC_POSE", "-1.3963,0,0,1.3963,0,0,0").split(",")])
RAMP_S = float(os.getenv("GC_RAMP_S", "5.0"))
SETTLE_S = float(os.getenv("GC_SETTLE_S", "3.0"))
MEASURE_S = float(os.getenv("GC_MEASURE_S", "2.0"))
RATE_HZ = float(os.getenv("GC_RATE_HZ", "20"))
TIP_KG = float(os.getenv("GC_TIP_EXTRA_MASS_KG", "0.638"))
TIP_COM = [float(v) for v in os.getenv("GC_TIP_EXTRA_COM_XYZ", "0.113,-0.003,0.0").split(",")]
# アプリが実際に送っている tau は SCALE * g(q)。ここが起動値とズレると、スケール不足を
# 「質量不足」と誤って逆算してしまうので必ず一致させること。
SCALE = float(os.getenv("GC_GRAVITY_TAU_SCALE", "1.0"))
RETURN_HOME = os.getenv("GC_RETURN", "1").strip() == "1"


def _grav_model(extra_kg: float, com: list[float]):
    """アプリと同一条件の重力モデルを組む(先端ペイロードを合成)。"""
    arm = load_g1_right_arm_ik(str(DEFAULT_URDF))
    m = arm.ik.model
    if extra_kg > 0.0:
        last = m.njoints - 1
        m.inertias[last] = m.inertias[last] + pinocchio.Inertia(
            extra_kg, np.asarray(com, dtype=float), np.zeros((3, 3))
        )
    return m, m.createData()


def main() -> int:
    if POSE.shape != (7,):
        raise SystemExit(f"GC_POSE は7個必要: {POSE!r}")
    _model_lo, _model_hi = _grav_model(0.0, TIP_COM)[0].lowerPositionLimit, None
    m0 = load_g1_right_arm_ik(str(DEFAULT_URDF)).ik.model
    if np.any(POSE < m0.lowerPositionLimit) or np.any(POSE > m0.upperPositionLimit):
        raise SystemExit(f"GC_POSE が関節リミット外: {POSE}")

    state: dict = {}
    LCMTransport("/g1/motor_states", JointState).subscribe(
        lambda msg, _=None: state.update(pos=list(msg.position), t=time.time())
    )
    pub = LCMTransport("/g1/arm_target", JointState)

    print("motor_states を待機中 (アプリが動いていないと3秒でタイムアウト)...", flush=True)
    t0 = time.time()
    while "pos" not in state and time.time() - t0 < 3.0:
        time.sleep(0.05)
    if "pos" not in state:
        print(
            "ERROR: motor_states が来ない。アプリ(dimos run ...)が動いているか確認。",
            file=sys.stderr,
        )
        return 1

    def read_arm() -> tuple[np.ndarray, np.ndarray]:
        pos = state["pos"]
        return (
            np.array([float(x) for x in pos[_LEFT_SLICE]]),
            np.array([float(x) for x in pos[_RIGHT_SLICE]]),
        )

    q_left0, q_right0 = read_arm()
    print(f"開始姿勢(右腕) = {np.round(q_right0, 3)}")
    print(f"測定姿勢       = {np.round(POSE, 3)}  ({np.round(np.rad2deg(POSE), 1)} deg)")
    print(f"重力モデル: 先端ペイロード +{TIP_KG:.3f} kg @ {TIP_COM}, tau_scale={SCALE:g}")
    if abs(SCALE - 1.0) > 1e-9:
        print(
            f"  ⚠ tau_scale={SCALE:g} != 1.0。アプリの OKRA_GRAVITY_TAU_SCALE と"
            " 一致しているか必ず確認すること(ズレると逆算が狂う)"
        )
    print(
        f"{RAMP_S:.1f}秒かけて移動 -> {SETTLE_S:.1f}秒静定 -> {MEASURE_S:.1f}秒平均\n", flush=True
    )

    def publish(q_right: np.ndarray) -> None:
        q_left, _ = read_arm()  # 左腕は常に実測を維持(勝手に動かさない)
        pub.publish(
            JointState(
                name=list(_ARM_JOINT_NAMES),
                position=[float(v) for v in np.concatenate([q_left, q_right])],
                velocity=[0.0] * _NUM_ARM,
                effort=[0.0] * _NUM_ARM,
            )
        )

    dt = 1.0 / RATE_HZ
    try:
        # 1) ゆっくり測定姿勢へ
        n = max(1, int(RAMP_S * RATE_HZ))
        for i in range(1, n + 1):
            publish(q_right0 + (POSE - q_right0) * (i / n))
            time.sleep(dt)
        # 2) 静定を待つ(指令は出し続ける = 保持)
        t_end = time.time() + SETTLE_S
        while time.time() < t_end:
            publish(POSE)
            time.sleep(dt)
        # 3) 平均を取る
        samples = []
        t_end = time.time() + MEASURE_S
        while time.time() < t_end:
            publish(POSE)
            _, q_r = read_arm()
            samples.append(q_r)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n中断。腕は最後の指令を保持します(アプリ側でホールド)。", flush=True)
        return 130

    q_meas = np.mean(np.asarray(samples), axis=0)
    q_std = np.std(np.asarray(samples), axis=0)
    dq = POSE - q_meas  # 残留誤差 [rad]
    tau_missing = KP * dq  # 未補償トルク [N*m]

    m, d = _grav_model(TIP_KG, TIP_COM)
    g_model = np.asarray(pinocchio.computeGeneralizedGravity(m, d, q_meas))
    m2, d2 = _grav_model(TIP_KG + 1.0, TIP_COM)
    dtau_dm = np.asarray(pinocchio.computeGeneralizedGravity(m2, d2, q_meas)) - g_model

    print("=" * 72)
    print(f"{'関節':10s} {'指令':>9s} {'実測':>9s} {'残差':>8s} {'未補償':>9s} {'モデルg':>9s}")
    print(f"{'':10s} {'[deg]':>9s} {'[deg]':>9s} {'[deg]':>8s} {'[N*m]':>9s} {'[N*m]':>9s}")
    print("-" * 72)
    for i, n_ in enumerate(JOINT_NAMES):
        print(
            f"{n_:10s} {np.rad2deg(POSE[i]):9.2f} {np.rad2deg(q_meas[i]):9.2f} "
            f"{np.rad2deg(dq[i]):8.2f} {tau_missing[i]:9.2f} {g_model[i]:9.2f}"
        )
    print("=" * 72)
    print(f"測定のばらつき(標準偏差,deg) = {np.round(np.rad2deg(q_std), 3)}")
    if np.max(np.rad2deg(q_std)) > 0.3:
        print("  ⚠ ばらつきが大きい。まだ揺れている可能性 -> GC_SETTLE_S を伸ばす")
    print()

    # 肩ピッチを基準に必要ペイロードを逆算
    i = 0
    if abs(dtau_dm[i]) < 1e-6:
        print("この姿勢では肩ピッチが先端質量にほぼ無感 -> GC_POSE を水平寄りに変更のこと")
        return 0
    # 釣り合い: kp*dq = tau_実負荷 - SCALE*g(q)
    # 質量を need_kg 足して誤差ゼロにしたいので  SCALE*need_kg*dtau_dm = kp*dq
    need_kg = tau_missing[i] / (SCALE * dtau_dm[i])
    print("【肩ピッチ基準の逆算】")
    print(f"  未補償トルク            = {tau_missing[i]:+.2f} N*m")
    print(f"  先端1kgあたりのトルク   = {dtau_dm[i]:+.2f} N*m/kg (x tau_scale {SCALE:g})")
    print(f"  → 追加すべき先端質量   = {need_kg:+.3f} kg")
    print(f"  → 推奨 OKRA_TIP_EXTRA_MASS_KG = {TIP_KG + need_kg:.3f}  (現在 {TIP_KG:.3f})")
    if need_kg < -0.05:
        print("  ⚠ 負 = 補償のかけ過ぎ。腕が自分で持ち上がる方向 -> 値を下げること")
    print()
    print("※ 残差には摩擦・減速機効率も含まれる。重力だけの寄与ではない点に注意。")
    print("※ 姿勢を変えて数点測り、同じ質量が出るかを見ると信頼度が上がる。")

    if RETURN_HOME:
        print("\nホーム姿勢へ戻します...", flush=True)
        try:
            n = max(1, int(RAMP_S * RATE_HZ))
            for i2 in range(1, n + 1):
                publish(POSE + (q_right0 - POSE) * (i2 / n))
                time.sleep(dt)
        except KeyboardInterrupt:
            print("中断。腕はその場で保持。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
