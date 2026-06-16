#!/usr/bin/env python3
"""Standalone live-camera check for the okra-ACT service (NO robot motion).

Grabs ONE frame from the head (cam_high) and right-wrist (cam_right_wrist)
teleimager cameras, sends both to act_service.py over ZMQ, and prints the action.
This verifies the full real-camera -> ACT inference path with the two-camera
"tree" model — without moving the arm.

Prereqs:
  # NX:     teleimager-server --rs   (head :55555 AND right_wrist :55557 enabled)
  # laptop: ~/act-okura/.venv_act/bin/python scripts/act_service.py --serve
  # laptop: ~/act-okura/.venv_act/bin/python scripts/verify_act_live_cameras.py --host 192.168.123.164

Note: STATE here is a placeholder (zeros) — this checks the CAMERA + inference
path, not joint state. (The real ActGraspModule fills state from motor_states.)
"""

from __future__ import annotations

import argparse
import time

import cv2
import msgpack
import numpy as np
import zmq

STATE_DIM = 16


def _grab_bgr(getter, name: str, tries: int = 50):
    """Poll a teleimager getter until it returns a decoded BGR frame."""
    for _ in range(tries):
        frame = getter()
        bgr = getattr(frame, "bgr", None) if frame is not None else None
        if bgr is None and isinstance(frame, np.ndarray):
            bgr = frame
        if bgr is not None:
            return bgr
        time.sleep(0.1)
    raise RuntimeError(f"no frame from {name} camera (is it enabled on the NX?)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="192.168.123.164", help="G1/NX teleimager host")
    ap.add_argument("--request-port", type=int, default=60000)
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5701", help="act_service ZMQ")
    ap.add_argument("--steps", type=int, default=3, help="how many frames to send")
    args = ap.parse_args()

    from teleimager.image_client import ImageClient

    print(f"[live] connecting ImageClient host={args.host}")
    client = ImageClient(host=args.host, request_bgr=True)
    cfg = client.get_cam_config()
    print(f"[live] head enabled={cfg['head_camera']['enable_zmq']} "
          f"right_wrist enabled={cfg['right_wrist_camera']['enable_zmq']}")
    if not cfg["right_wrist_camera"]["enable_zmq"]:
        print("[live] ⚠️ right_wrist camera NOT enabled on the NX — the tree model needs it.")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(args.endpoint)

    for i in range(args.steps):
        head = _grab_bgr(client.get_head_frame, "head")
        wrist = _grab_bgr(client.get_right_wrist_frame, "right_wrist")
        print(f"[live] frame {i}: head={head.shape} wrist={wrist.shape}")
        _, head_jpg = cv2.imencode(".jpg", head)
        _, wrist_jpg = cv2.imencode(".jpg", wrist)
        req = {
            "state": [0.0] * STATE_DIM,  # placeholder — camera/inference path check only
            "images": {
                "cam_high": head_jpg.tobytes(),
                "cam_right_wrist": wrist_jpg.tobytes(),
            },
            "reset": i == 0,
        }
        sock.send(msgpack.packb(req, use_bin_type=True))
        resp = msgpack.unpackb(sock.recv(), raw=False)
        if "error" in resp:
            print(f"[live] act_service ERROR: {resp['error']}")
            return 1
        action = np.asarray(resp["action"], dtype=float)
        np.set_printoptions(precision=3, suppress=True)
        print(f"[live] action[{i}] = {action}")
        time.sleep(0.2)

    client.close()
    print("[live] DONE — real cameras -> ACT inference works (no motion).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
