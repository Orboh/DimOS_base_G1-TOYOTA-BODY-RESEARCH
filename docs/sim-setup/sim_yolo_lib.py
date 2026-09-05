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

"""sim ego_view → 実 YOLO-seg → torso 3D 検出の再利用ライブラリ（.venv）。

`sim_yolo_detect.py`（単発スクリプト）のロジックを関数化し、収穫ループ（HarvestSkills.detect_okra）
からも呼べるようにする。bridge(SIM_PUB_CAMERA=1, near クリップ修正済) の ego_view を1フレーム取得し、
実オクラ weight(okra11n-seg.pt) で YOLO-seg → マスク重心(u,v)+マスク内 depth median →
camera 光学系3D → cam_to_torso で torso 系へ。GT 注入を実検出に置換する中核。
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import os
import time

import cv2
import msgpack
import numpy as np
import zmq

DEFAULT_WEIGHT = os.getenv(
    "OKRA_WEIGHT", "/home/kota-ueda/Desktop/okra-yolo-finetune/hf_upload/okra11n-seg.pt"
)
_MODEL = None  # YOLO モデルはプロセス内でキャッシュ（毎フレーム読み込まない）


def _b(x):
    return base64.b64decode(x) if isinstance(x, str) else bytes(x)


@dataclass
class YoloDet:
    """1 検出: torso 3D・信頼度・画素重心・depth。"""

    torso: tuple[float, float, float]
    conf: float
    uv: tuple[float, float]
    depth_m: float


def grab_frame(
    host: str = "127.0.0.1", port: int = 5555, topic: str = "ego_view", secs: float = 3.0
):
    """ego_view を ~secs 受信し最新の (bgr, depth[m], K, cam_to_torso) を返す。無ければ None。"""
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.SUB)
    s.connect(f"tcp://{host}:{port}")
    s.setsockopt(zmq.SUBSCRIBE, b"")
    s.setsockopt(zmq.RCVTIMEO, 1000)
    bgr = depth = K = c2t = None
    t0 = time.time()
    while time.time() - t0 < secs:
        try:
            payload = s.recv()
        except zmq.Again:
            continue
        msg = msgpack.unpackb(payload, raw=False)
        cb = (msg.get("images") or {}).get(topic)
        if cb is None:
            continue
        img = cv2.imdecode(np.frombuffer(_b(cb), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        bgr = img
        db = (msg.get("depth") or {}).get(topic)
        if db is not None:
            d16 = cv2.imdecode(np.frombuffer(_b(db), np.uint8), cv2.IMREAD_UNCHANGED)
            if d16 is not None:
                depth = d16.astype(np.float32) * float(msg.get("depth_scale", 0.001))
        K = (msg.get("intrinsics") or {}).get(topic) or K
        c2t = msg.get("cam_to_torso") or c2t
    s.close()
    if bgr is None:
        return None
    return bgr, depth, K, c2t


def _quat_rot(p, q):
    """R(q)·p。q=(qx,qy,qz,qw)。"""
    qx, qy, qz, qw = q
    x, y, z = p
    tx = 2 * (qy * z - qz * y)
    ty = 2 * (qz * x - qx * z)
    tz = 2 * (qx * y - qy * x)
    return (
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    )


def detect_okra_torso(
    host: str = "127.0.0.1",
    port: int = 5555,
    topic: str = "ego_view",
    *,
    weight: str = DEFAULT_WEIGHT,
    conf: float = 0.25,
    secs: float = 3.0,
    save_dir: str | None = None,
) -> list[YoloDet]:
    """ego_view を1フレーム取り YOLO-seg → torso 3D の検出リストを返す（信頼度降順）。"""
    global _MODEL
    fr = grab_frame(host, port, topic, secs)
    if fr is None:
        return []
    bgr, depth, K, c2t = fr
    if _MODEL is None:
        from ultralytics import YOLO

        _MODEL = YOLO(weight)
    res = _MODEL.predict(bgr, conf=conf, verbose=False)[0]
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f"{save_dir}/ego_color.png", bgr)
        cv2.imwrite(f"{save_dir}/ego_yolo.png", res.plot())
    nd = 0 if res.boxes is None else len(res.boxes)
    if nd == 0 or not K:
        return []
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]
    t = [0.0, 0.0, 0.0]
    q = [0.0, 0.0, 0.0, 1.0]
    if c2t:
        v = [float(x) for x in c2t.split(",")]
        if len(v) >= 7:
            t, q = v[:3], v[3:7]
    masks = res.masks.data.cpu().numpy() if res.masks is not None else None
    out: list[YoloDet] = []
    for i in range(nd):
        cf = float(res.boxes.conf[i])
        if masks is not None and i < len(masks):
            m = cv2.resize(masks[i], (bgr.shape[1], bgr.shape[0])) > 0.5
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue
            u, v_ = float(xs.mean()), float(ys.mean())
            dd = depth[m] if depth is not None else None
        else:
            x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy()
            u, v_ = (x1 + x2) / 2, (y1 + y2) / 2
            dd = None
        if dd is not None:
            dd = dd[(dd > 0.05) & (dd < 10)]
        d = (
            float(np.median(dd))
            if (dd is not None and dd.size)
            else (float(depth[int(v_), int(u)]) if depth is not None else 0.45)
        )
        p_opt = ((u - cx) / fx * d, (v_ - cy) / fy * d, d)
        rp = _quat_rot(p_opt, q)
        out.append(
            YoloDet(
                torso=(rp[0] + t[0], rp[1] + t[1], rp[2] + t[2]), conf=cf, uv=(u, v_), depth_m=d
            )
        )
    out.sort(key=lambda d: -d.conf)
    return out


__all__ = ["DEFAULT_WEIGHT", "YoloDet", "detect_okra_torso", "grab_frame"]
