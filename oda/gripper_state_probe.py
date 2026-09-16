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

"""Dex1 グリッパの状態受信だけを確認する（何も送信しない・グリッパは動かない）。

harvest_zed 実行で「No rt/dex1/left/state received」が断続的に出る問題の切り分け用。
これ単体で ``rt/dex1/left/state``（既定; ``DEX1_PREFIX`` で ``rt/dex1/right`` に変更可）
を購読するだけで、``cmd`` は一切 publish しない。

実行:
  DEX1_NIC=enx6c1ff771dc67 .venv/bin/python oda/gripper_state_probe.py
"""

from __future__ import annotations

import os
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_

NIC = os.getenv("DEX1_NIC", "enp46s0")
PREFIX = os.getenv("DEX1_PREFIX", "rt/dex1/left")
TIMEOUT_S = float(os.getenv("DEX1_PROBE_TIMEOUT_S", "10"))


def main() -> int:
    ChannelFactoryInitialize(0, NIC)
    st: dict[str, float] = {}
    ChannelSubscriber(f"{PREFIX}/state", MotorStates_).Init(
        lambda m: st.update(q=m.states[0].q, tau=m.states[0].tau_est, t=time.time()), 10
    )
    print(
        f"NIC={NIC!r} prefix={PREFIX!r} で {PREFIX}/state を待ちます"
        f"（何も送信しません・最大{TIMEOUT_S:.0f}秒）..."
    )
    t0 = time.time()
    last_print = 0.0
    while time.time() - t0 < TIMEOUT_S:
        if "q" in st and time.time() - last_print > 0.5:
            print(f"  受信中: q={st['q']:.3f} tau={st['tau']:.3f}")
            last_print = time.time()
        time.sleep(0.05)

    if "q" in st:
        print(f"OK: {PREFIX}/state を受信できました（最後の q={st['q']:.3f}）")
        return 0
    print(
        f"NG: {TIMEOUT_S:.0f}秒待っても {PREFIX}/state を受信できませんでした"
        " — Dex1の電源/ケーブル/prefix（rt/dex1/left か right か）を確認してください"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
