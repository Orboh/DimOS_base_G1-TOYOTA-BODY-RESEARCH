#!/usr/bin/env python3
"""nav_isaac_adapter.py — Isaac(raw LCM) ⇄ dimos nav_stack(LCM) の変換役（.venv で実行）。

Unity を使わずに、我々の Isaac 収穫 sim（chinou+g1bag）を dimos 標準ナビ（nav_stack）へ接続する。
Isaac 側は dimos 非依存（isaac-sim env の dimos は open3d 由来で壊れているため）＝**生 LCM のみ**で
自作の軽量フォーマットを流す。本アダプタ（.venv, dimos が正常動作）がそれを nav_stack が読む
dimos メッセージへ翻訳する。全部 1 本の LCM バス（udpm://239.255.76.67:7667、(a) で lo マルチキャスト設定済み）。

  受信 isaac/lidar (raw: <u32 N><f32 N*3>  world 座標の点群) → publish /registered_scan (PointCloud2)
  受信 isaac/odom  (raw: <f32 x7>  x,y,z,qx,qy,qz,qw)        → publish /odometry (Odometry) + /tf(map→sensor)
  受信 /cmd_vel (Twist, nav_stack の出力)                    → publish isaac/cmd_vel (raw: <f32 x3> vx,vy,wz)

実行: .venv/bin/python docs/sim-setup/nav_isaac_adapter.py
"""
from __future__ import annotations

import struct
import sys
import time

sys.path.insert(0, "/home/kota-ueda/Desktop/dimos-hackathon")

import lcm as lcm_mod
import numpy as np

from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic

# Isaac との生 LCM チャンネル（自作フォーマット）
CH_LIDAR_IN = "isaac/lidar"
CH_ODOM_IN = "isaac/odom"
CH_CMDVEL_OUT = "isaac/cmd_vel"

# nav_stack 側 dimos チャンネル
T_SCAN = Topic("/registered_scan", PointCloud2)
T_ODOM = Topic("/odometry", Odometry)
T_TF = Topic("/tf", Transform)
T_CMD_IN = Topic("/cmd_vel", Twist)

FRAME = "map"
CHILD = "sensor"


def _vec_component(v, idx: int, attr: str) -> float:
    """Vector3 でも list/tuple でも成分を取り出す（dimos の型ゆらぎ対策）。"""
    if v is None:
        return 0.0
    if hasattr(v, attr):
        return float(getattr(v, attr))
    try:
        return float(v[idx])
    except (TypeError, IndexError, KeyError):
        return 0.0


class NavIsaacAdapter:
    def __init__(self) -> None:
        # dimos バス: 型付き publish（PointCloud2/Odometry/TF）＋ /cmd_vel の型付き subscribe。
        self.bus = LCM()
        # 生バス: isaac/* の型なし subscribe＋publish（dimos の encoder は型なしを decode 時に捨てるため別handle）。
        # 既定 URL は dimos と同一（udpm://239.255.76.67:7667）＝同じバスを共有。
        self.raw = lcm_mod.LCM()
        self.n_scan = 0
        self.n_odom = 0
        self.n_cmd = 0
        self.raw.subscribe(CH_LIDAR_IN, self._on_lidar)
        self.raw.subscribe(CH_ODOM_IN, self._on_odom)
        self.bus.subscribe(T_CMD_IN, self._on_cmd)

    # --- Isaac → nav_stack（生 lcm コールバック: (channel, data)）---
    def _on_lidar(self, _channel, data) -> None:
        if not isinstance(data, bytes) or len(data) < 4:
            return
        n = struct.unpack_from("<I", data, 0)[0]
        need = 4 + n * 12
        if n <= 0 or len(data) < need:
            return
        pts = np.frombuffer(data, dtype="<f4", count=n * 3, offset=4).reshape(n, 3).astype(np.float64)
        self.bus.publish(T_SCAN, PointCloud2.from_numpy(pts, frame_id=FRAME, timestamp=time.time()))
        self.n_scan += 1

    def _on_odom(self, _channel, data) -> None:
        if not isinstance(data, bytes) or len(data) < 28:
            return
        x, y, z, qx, qy, qz, qw = struct.unpack_from("<7f", data, 0)
        now = time.time()
        self.bus.publish(
            T_ODOM,
            Odometry(
                ts=now,
                frame_id=FRAME,
                child_frame_id=CHILD,
                pose=Pose(position=[x, y, z], orientation=[qx, qy, qz, qw]),
            ),
        )
        self.bus.publish(
            T_TF,
            Transform(
                translation=Vector3(x, y, z),
                rotation=Quaternion(qx, qy, qz, qw),
                frame_id=FRAME,
                child_frame_id=CHILD,
                ts=now,
            ),
        )
        self.n_odom += 1

    # --- nav_stack → Isaac ---
    def _on_cmd(self, msg, _topic) -> None:
        tw = Twist.lcm_decode(msg) if isinstance(msg, bytes) else msg
        vx = _vec_component(getattr(tw, "linear", None), 0, "x")
        vy = _vec_component(getattr(tw, "linear", None), 1, "y")
        wz = _vec_component(getattr(tw, "angular", None), 2, "z")
        self.raw.publish(CH_CMDVEL_OUT, struct.pack("<3f", vx, vy, wz))
        self.n_cmd += 1

    def spin(self) -> None:
        print("[adapter] up. isaac/lidar,isaac/odom → /registered_scan,/odometry,/tf ; /cmd_vel → isaac/cmd_vel", flush=True)
        last = 0.0
        while True:
            self.bus.l.handle_timeout(100)
            self.raw.handle_timeout(100)
            t = time.time()
            if t - last > 2.0:
                print(f"[adapter] scan={self.n_scan} odom={self.n_odom} cmd={self.n_cmd}", flush=True)
                last = t


if __name__ == "__main__":
    NavIsaacAdapter().spin()
