#!/usr/bin/env python3
"""nav_send_goal.py — dimos nav_stack に目的地を1つ送る（.venv で実行）。

map フレームの (x, y) へ向かわせる。nav_stack（SimplePlanner / MovementManager）が購読する
``/goal#geometry_msgs.PointStamped`` に PointStamped を publish するだけ。

使い方:
  .venv/bin/python docs/sim-setup/nav_send_goal.py 2.0 0.0
（引数省略時は (2.0, 0.0)。座標は chinou 室内の開けた点を指定すること）
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/kota-ueda/Desktop/dimos-hackathon")

from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic

T_GOAL = Topic("/goal", PointStamped)


def main() -> None:
    x = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    bus = LCM()
    goal = PointStamped(x=x, y=y, z=0.0, frame_id="map", ts=time.time())
    # 取りこぼし防止に数回送る
    for _ in range(5):
        bus.publish(T_GOAL, goal)
        time.sleep(0.2)
    print(f"sent /goal = ({x:.2f}, {y:.2f}) in map frame", flush=True)


if __name__ == "__main__":
    main()
