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

"""歩行 orchestrator: 仮想ジョイスティック(/tmp/sim_joy.bin)と elastic band(/tmp/sim_band.txt)を
時系列で書き、公式 g1_ctrl FSM を Passive→FixStand→Velocity と遷移させて前進歩行させる。

wireless_remote(40byte) レイアウト（dds_wrapper unitree_joystick.hpp 準拠）:
  [2]=btn下位: bit0=R1,bit1=L1,bit2=Start,bit3=Select,bit4=R2,bit5=L2
  [3]=btn上位: bit0=A,bit1=B,bit2=X,bit3=Y,bit4=up,bit5=right,bit6=down,bit7=left
  [4:8]=lx [8:12]=rx [12:16]=ry [16:20]=L2(analog) [20:24]=ly
  velocity_commands: vx=ly, vy=-lx, yaw=-rx（範囲 vx[-0.5,1.0],vy[-0.3,0.3],yaw[-0.2,0.2]）
  遷移: FixStand=L2+up.on_pressed / Velocity=R1+X.on_pressed / Passive=L2+B.on_pressed
"""

from __future__ import annotations

import os
import struct
import time

JOY = os.getenv("SIM_JOY_FILE", "/tmp/sim_joy.bin")
BAND = os.getenv("SIM_BAND_FILE", "/tmp/sim_band.txt")
BAND_LEN = float(os.getenv("SIM_BAND_LEN0", "0.445"))  # 立位支持 length（重量を支える）
BAND_OFF = float(os.getenv("SIM_BAND_OFF", "2.167"))  # 除荷 length（力≈0）

# btn 下位バイト(byte2)
B2 = {"R1": 0, "L1": 1, "Start": 2, "Select": 3, "R2": 4, "L2": 5}
# btn 上位バイト(byte3)
B3 = {"A": 0, "B": 1, "X": 2, "Y": 3, "up": 4, "right": 5, "down": 6, "left": 7}


def remote(buttons=(), vx=0.0, vy=0.0, yaw=0.0) -> bytes:
    """ボタン集合と速度から 40byte の wireless_remote を作る。"""
    buf = bytearray(40)
    b2 = b3 = 0
    for name in buttons:
        if name in B2:
            b2 |= 1 << B2[name]
        elif name in B3:
            b3 |= 1 << B3[name]
    buf[2] = b2
    buf[3] = b3
    lx = -vy  # vy = -lx
    rx = -yaw  # yaw = -rx
    ly = vx  # vx = ly
    buf[4:8] = struct.pack("f", lx)
    buf[8:12] = struct.pack("f", rx)
    buf[20:24] = struct.pack("f", ly)
    return bytes(buf)


def set_joy(buttons=(), vx=0.0, vy=0.0, yaw=0.0) -> None:
    with open(JOY, "wb") as f:
        f.write(remote(buttons, vx, vy, yaw))


def set_band(length: float, enable: bool) -> None:
    with open(BAND, "w") as f:
        f.write(f"{length} {1 if enable else 0}")


def hold(buttons=(), vx=0.0, vy=0.0, yaw=0.0, secs=0.5, hz=50) -> None:
    """指定状態を secs 秒間ストリーム（on_pressed 検出のため複数フレーム維持）。"""
    n = max(1, int(secs * hz))
    for _ in range(n):
        set_joy(buttons, vx, vy, yaw)
        time.sleep(1.0 / hz)


def main() -> int:
    boot = float(os.getenv("WALK_BOOT_DELAY", "4"))  # g1_ctrl 接続待ち
    walk_secs = float(os.getenv("WALK_SECS", "12"))
    vx = float(os.getenv("WALK_VX", "0.4"))  # 前進速度 [m/s]

    # 0) 中立 + band 係留（脱力落下防止）
    set_band(BAND_LEN, True)
    set_joy()
    print(f"[joy] band 係留(len={BAND_LEN}) + 中立。g1_ctrl 接続待ち {boot}s …", flush=True)
    time.sleep(boot)

    # 1) FixStand: L2 + up（同時押し 0.6s）→ 立位ランプ(約3s)
    print("[joy] → FixStand (L2+up)", flush=True)
    hold(buttons=("L2", "up"), secs=0.6)
    set_joy()
    time.sleep(3.5)  # FixStand ramp 完了待ち

    # 2) Velocity: R1 + X（同時押し 0.6s）→ policy 起動
    print("[joy] → Velocity policy (R1+X)", flush=True)
    hold(buttons=("R1", "X"), secs=0.6)
    set_joy()
    time.sleep(1.2)  # policy が脚制御を引き継ぐ猶予

    # 3) band 滑らか除荷: length 0.445→2.167 で力を 344N→0 に漸減（急な全荷重で転ばないよう）
    #    その後 enable=False。band は水平にも引き戻すため前進前に必須。
    print("[joy] band 滑らか除荷 → 解除", flush=True)
    ramp_t, ramp_hz = 2.0, 20
    for k in range(int(ramp_t * ramp_hz)):
        s = (k + 1) / (ramp_t * ramp_hz)
        set_band(BAND_LEN + (BAND_OFF - BAND_LEN) * s, True)
        set_joy()  # 中立維持
        time.sleep(1.0 / ramp_hz)
    set_band(BAND_OFF, False)
    time.sleep(0.3)

    # 4) その場整定（band解除直後の残留横ずれを policy に吸収させる）
    settle = float(os.getenv("WALK_SETTLE", "3"))
    print(f"[joy] その場整定 {settle}s（vx=0）", flush=True)
    hold(vx=0.0, secs=settle)

    # 5) 前進
    print(f"[joy] 前進 vx={vx} を {walk_secs}s", flush=True)
    hold(vx=vx, secs=walk_secs)

    # 5) 停止
    print("[joy] 停止", flush=True)
    hold(vx=0.0, secs=1.0)
    set_joy()
    print("[joy] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
