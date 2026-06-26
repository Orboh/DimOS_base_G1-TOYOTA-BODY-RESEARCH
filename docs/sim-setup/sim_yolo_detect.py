#!/usr/bin/env python3
"""sim 胸カメラ(ego_view)→ 実 YOLO-seg 検出 → 3D(torso) を実走する（.venv）。

bridge(SIM_PUB_CAMERA=1) が配信する ego_view（color+depth+intrinsics K+cam_to_torso）を受信し、
**実オクラ weight（okra11n-seg.pt）で YOLO-seg**。検出マスク重心(u,v)＋マスク内 depth median から
カメラ光学系3D を作り、`cam_to_torso` で torso 系へ変換して報告する。GT 注入を実検出に置換した版。

⚠️ weight は実農園画像で学習＝sim 画像は OOD。検出は0〜低信頼になり得る（＝精度ではなく
「camera→YOLO→3D の配管が通るか」の実走確認。精度判定は実機, [[12-検証計画-sim]] §10）。

実行:
  # bridge: SIM_PUB_CAMERA=1 SIM_TABLE=1 ... sim_dds_bridge.py
  SIM_CAM_HOST=127.0.0.1 .venv/bin/python docs/sim-setup/sim_yolo_detect.py
"""
import base64
import math
import os
import sys
import time

import cv2
import msgpack
import numpy as np
import zmq

HOST = os.getenv("SIM_CAM_HOST", "127.0.0.1")
PORT = int(os.getenv("SIM_CAM_PORT", "5555"))
TOPIC = os.getenv("SIM_CAM_TOPIC", "ego_view")
WEIGHT = os.getenv("OKRA_WEIGHT", "/home/kota-ueda/Desktop/okra-yolo-finetune/hf_upload/okra11n-seg.pt")
CONF = float(os.getenv("YOLO_CONF", "0.25"))
SECS = float(os.getenv("SUB_SECS", "8"))
OUT = os.getenv("SIM_YOLO_OUT", "/tmp/claude-1000/-home-kota-ueda-Desktop-dimos-hackathon/ff3f8d27-e694-4585-88d3-beb6daaeb929/scratchpad/sim_yolo")


def _b(x):
    return base64.b64decode(x) if isinstance(x, str) else bytes(x)


def quat_rot(p, q):
    """torso<-optical: p_torso = R(q)·p_opt + t は呼び出し側。ここは R(q)·p のみ。q=(qx,qy,qz,qw)。"""
    qx, qy, qz, qw = q
    x, y, z = p
    # v' = v + 2*q_xyz × (q_xyz × v + qw*v)
    tx = 2 * (qy * z - qz * y)
    ty = 2 * (qz * x - qx * z)
    tz = 2 * (qx * y - qy * x)
    rx = x + qw * tx + (qy * tz - qz * ty)
    ry = y + qw * ty + (qz * tx - qx * tz)
    rz = z + qw * tz + (qx * ty - qy * tx)
    return (rx, ry, rz)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    # --- ego_view を数秒受信して最新フレームを得る ---
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.SUB)
    s.connect(f"tcp://{HOST}:{PORT}")
    s.setsockopt(zmq.SUBSCRIBE, b"")
    s.setsockopt(zmq.RCVTIMEO, 1000)
    print(f"[yolo] SUB tcp://{HOST}:{PORT} topic={TOPIC} {SECS}s …", flush=True)
    bgr = depth = K = c2t = None
    n = 0
    t0 = time.time()
    while time.time() - t0 < SECS:
        try:
            payload = s.recv()
        except zmq.Again:
            continue
        msg = msgpack.unpackb(payload, raw=False)
        cb = (msg.get("images") or {}).get(TOPIC)
        if cb is None:
            continue
        img = cv2.imdecode(np.frombuffer(_b(cb), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        bgr = img
        n += 1
        db = (msg.get("depth") or {}).get(TOPIC)
        if db is not None:
            d16 = cv2.imdecode(np.frombuffer(_b(db), np.uint8), cv2.IMREAD_UNCHANGED)
            if d16 is not None:
                depth = d16.astype(np.float32) * float(msg.get("depth_scale", 0.001))
        K = (msg.get("intrinsics") or {}).get(TOPIC) or K
        c2t = msg.get("cam_to_torso") or c2t
    if bgr is None:
        print("[yolo] フレーム受信できず（bridge の SIM_PUB_CAMERA=1 を確認）", flush=True)
        return 2
    cv2.imwrite(f"{OUT}/ego_color.png", bgr)
    print(f"[yolo] received {n} frames, color={bgr.shape}, depth={'yes' if depth is not None else 'no'}, K={K}", flush=True)

    # --- 実 YOLO-seg ---
    from ultralytics import YOLO

    model = YOLO(WEIGHT)
    res = model.predict(bgr, conf=CONF, verbose=False)[0]
    cv2.imwrite(f"{OUT}/ego_yolo.png", res.plot())  # 注釈付き画像
    names = res.names
    nd = 0 if res.boxes is None else len(res.boxes)
    print(f"[yolo] weight={os.path.basename(WEIGHT)} conf>={CONF} → 検出 {nd} 個 names={names}", flush=True)
    if nd == 0:
        print("[yolo] 検出0（sim画像はOOD＝想定内）。配管(camera→YOLO→画像)は通った。注釈画像を保存。", flush=True)
        print(f"[yolo] 画像: {OUT}/ego_color.png / {OUT}/ego_yolo.png", flush=True)
        return 0

    # --- 検出ごとに 3D(torso) 化 ---
    fx, _, cx, _, fy, cy = K[0], K[1], K[2], K[3], K[4], K[5]
    t = [0.0, 0.0, 0.0]
    q = [0.0, 0.0, 0.0, 1.0]
    if c2t:
        v = [float(x) for x in c2t.split(",")]
        if len(v) >= 7:
            t, q = v[:3], v[3:7]
    masks = res.masks.data.cpu().numpy() if res.masks is not None else None
    for i in range(nd):
        conf = float(res.boxes.conf[i])
        cls = int(res.boxes.cls[i])
        # 重心(u,v) と depth median（mask 優先）
        if masks is not None and i < len(masks):
            m = cv2.resize(masks[i], (bgr.shape[1], bgr.shape[0])) > 0.5
            ys, xs = np.where(m)
            u, v_ = float(xs.mean()), float(ys.mean())
            dd = depth[m] if depth is not None else None
        else:
            x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy()
            u, v_ = (x1 + x2) / 2, (y1 + y2) / 2
            dd = None
        if dd is not None:
            dd = dd[(dd > 0.05) & (dd < 10)]
        d = float(np.median(dd)) if (dd is not None and dd.size) else (
            float(depth[int(v_), int(u)]) if depth is not None else 0.45)
        # 光学系3D（X右,Y下,Z前）→ torso
        p_opt = ((u - cx) / fx * d, (v_ - cy) / fy * d, d)
        rp = quat_rot(p_opt, q)
        p_tor = (rp[0] + t[0], rp[1] + t[1], rp[2] + t[2])
        print(f"  det[{i}] {names.get(cls, cls)} conf={conf:.2f} uv=({u:.0f},{v_:.0f}) "
              f"depth={d:.2f}m torso=({p_tor[0]:.3f},{p_tor[1]:.3f},{p_tor[2]:.3f})", flush=True)
    print(f"[yolo] 注釈画像: {OUT}/ego_yolo.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
