#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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

"""unitree-g1-gravity-hold-test 用の最小オペレータ操作(記録・カメラ購読なし)。

  c -> 右腕を COMPLIANT にする(/g1/reach_done を publish)。
       G1ArmSdkConnection(collection_mode) が kp を 0 までランプダウンしつつ
       重力フィードフォワードトルクを投入する。左腕・腰は STIFF のまま。
  q -> このスクリプトを終了する(腕は再スティッフ化されない — ブループリント側を
       Ctrl+C すると weight->0 へ安全にランプダウンしながら停止する)。

/g1/motor_states を購読して右腕7関節の実測角[deg]を1Hzで表示するので、
手を離した後に角度が動いていないか(その場でキープできているか)を目視確認できる。

実行(ブループリントと同じLCMバス):
  LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
    .venv/bin/python scripts/gravity_hold_toggle.py
"""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
import time
import tty

import numpy as np

from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic

LCM_URL = os.getenv("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=1")
_RIGHT_SLICE = slice(22, 29)  # 29関節ベクトル中の右腕7関節
_ARM_END = 29


def main() -> int:
    lock = threading.Lock()
    latest = {"q": None}

    def on_state(msg, _t):  # type: ignore[no-untyped-def]
        pos = list(msg.position)
        if len(pos) >= _ARM_END:
            with lock:
                latest["q"] = np.array([float(x) for x in pos[_RIGHT_SLICE]], dtype=np.float32)

    lc = LCM(url=LCM_URL)
    lc.start()
    lc.subscribe(Topic("/g1/motor_states", JointState), on_state)
    reach_done_topic = Topic("/g1/reach_done", Bool)

    print(f"[gravity_hold_toggle] LCM {LCM_URL}")
    print("[gravity_hold_toggle] waiting for /g1/motor_states ...")
    t0 = time.time()
    while time.time() - t0 < 5:
        with lock:
            if latest["q"] is not None:
                break
        time.sleep(0.1)
    with lock:
        if latest["q"] is None:
            print("[gravity_hold_toggle] no /g1/motor_states — is the blueprint running?")
            return 2

    print("[gravity_hold_toggle] ready.")
    print("  'c' -> 右腕を支えてから押す(COMPLIANT化: kp->0 + 重力FF)")
    print("  'q' -> このスクリプトのみ終了(腕はブループリント側Ctrl+Cで安全停止)")
    print("  右腕7関節の角度[deg]を1Hzで表示 -> 手を離した後、動いていないか目視確認")

    cmd = {"compliant": False, "quit": False}

    def key_thread() -> None:
        is_tty = sys.stdin.isatty()
        if not is_tty:
            for line in sys.stdin:  # フォールバック: "c"/"q" + Enter
                c = line.strip().lower()
                if c == "q":
                    cmd["quit"] = True
                    return
                if c == "c":
                    cmd["compliant"] = True
            return
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not cmd["quit"]:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    c = sys.stdin.read(1).lower()
                    if c == "q":
                        cmd["quit"] = True
                        return
                    if c == "c":
                        cmd["compliant"] = True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    threading.Thread(target=key_thread, daemon=True).start()

    last_print = 0.0
    try:
        while not cmd["quit"]:
            if cmd["compliant"]:
                cmd["compliant"] = False
                lc.publish(reach_done_topic, Bool(data=True))
                print("[c] -> RIGHT arm COMPLIANT を要求(手で支えてください)。")
            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                with lock:
                    q = None if latest["q"] is None else latest["q"].copy()
                if q is not None:
                    deg = np.degrees(q)
                    print("  q[deg] = " + " ".join(f"{v:7.2f}" for v in deg))
            time.sleep(0.05)
    finally:
        try:
            lc.stop()
        except Exception:
            pass
    print("[gravity_hold_toggle] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
