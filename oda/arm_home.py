#!/usr/bin/env python3
"""右腕をホーム姿勢(垂れ下げ)へ滑らかに戻す(2026-07-22, 試行間リセット用).

10回テストで「毎回腕が伸びっぱなしで次がやりづらい」対策。アプリ(コマンドB)が
**動いている間に**別ターミナルから実行する: /g1/motor_states から現在の腕姿勢を
読み、ホーム姿勢まで関節空間で補間した /g1/arm_target 列を流す。実際に腕を駆動
するのはアプリ内の G1ArmSdkConnection (既存経路) なので、本体コードは無変更。

アプリが動いていなければ motor_states が来ずに安全に終了する(何も発行しない)。
グリッパには触れない(閉じたまま/開いたまま維持)。

実行(アプリ稼働中・別ターミナル):
  cd ~/Toyota-auto-body-PoC/DimOS_oda
  LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
  .venv/bin/python oda/arm_home.py

環境変数:
  ARM_HOME_Q_RIGHT    : ホーム姿勢7関節(rad, csv)。既定 "0,0,0,0,0,0,0"(腕垂れ下げ)
  ARM_HOME_DURATION_S : 移動時間(既定 3.0)。ゆっくり=安全
  ARM_HOME_RATE_HZ    : 目標の発行レート(既定 20)
  ARM_HOME_OPEN_Q     : 戻す前にグリッパをこの位置まで開く(既定 3.7)。
                        空文字で無効。挟んだまま腕を引くと支柱ごと持っていく
                        事故(2026-07-22実発生)の再発防止 — 既定でまず開く
"""
from __future__ import annotations

import os
import time

import numpy as np

from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.act.ik_reach_bridge import (
    _ARM_JOINT_NAMES,
    _ARM_START,
    _LEFT_SLICE,
    _NUM_ARM,
    _RIGHT_SLICE,
)

_HOME_RAW = os.getenv("ARM_HOME_Q_RIGHT", "0,0,0,0,0,0,0")
HOME_Q = np.array([float(v) for v in _HOME_RAW.split(",")], dtype=float)
DURATION_S = float(os.getenv("ARM_HOME_DURATION_S", "3.0"))
RATE_HZ = float(os.getenv("ARM_HOME_RATE_HZ", "20"))
_OPEN_RAW = os.getenv("ARM_HOME_OPEN_Q", "3.7").strip()
OPEN_Q = float(_OPEN_RAW) if _OPEN_RAW else None


def main() -> int:
    if HOME_Q.shape != (7,):
        raise SystemExit(f"ARM_HOME_Q_RIGHT must be 7 csv values, got {_HOME_RAW!r}")

    st: dict = {}
    LCMTransport("/g1/motor_states", JointState).subscribe(
        lambda m, _=None: st.update(pos=list(m.position))
    )
    t0 = time.time()
    while "pos" not in st and time.time() - t0 < 3.0:
        time.sleep(0.05)
    pos = st.get("pos")
    if not pos or len(pos) < _ARM_START + _NUM_ARM:
        raise SystemExit(
            "NO motor_states — アプリ(コマンドB)が動いている時だけ使えます。何も発行せず終了。"
        )

    if OPEN_Q is not None:
        # 挟んだまま腕を引かない: まず開き、爪が開き切るのを待ってから動かす
        gpub = LCMTransport("/g1/gripper_target", JointState)
        for _ in range(3):
            gpub.publish(
                JointState(
                    name=["g1/right_gripper"], position=[OPEN_Q],
                    velocity=[0.0], effort=[0.0],
                )
            )
            time.sleep(0.2)
        print(f"gripper open (q={OPEN_Q}) 送信 — 1秒待って腕を戻します")
        time.sleep(1.0)

    q_left = np.array([float(x) for x in pos[_LEFT_SLICE]])
    q_right = np.array([float(x) for x in pos[_RIGHT_SLICE]])
    travel = float(np.max(np.abs(q_right - HOME_Q)))
    if travel < 0.05:
        print(f"already home (max delta {travel:.3f} rad) — nothing to do")
        return 0

    pub = LCMTransport("/g1/arm_target", JointState)
    n = max(int(DURATION_S * RATE_HZ), 2)
    print(
        f"homing right arm: max travel {travel:.2f} rad over {DURATION_S:.1f}s "
        f"({n} steps) — 腕から目を離さないで"
    )
    for i in range(1, n + 1):
        a = i / n
        q_i = (1.0 - a) * q_right + a * HOME_Q
        pub.publish(
            JointState(
                name=list(_ARM_JOINT_NAMES),
                position=[float(x) for x in np.concatenate([q_left, q_i])],
                velocity=[0.0] * _NUM_ARM,
                effort=[0.0] * _NUM_ARM,
            )
        )
        time.sleep(1.0 / RATE_HZ)
    print("done — 右腕ホーム姿勢。次の試行どうぞ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
