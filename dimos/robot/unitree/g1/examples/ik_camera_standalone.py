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

# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Standalone D435i -> LCM point-cloud publisher for the IK-reach PoC.

Runs ON THE ROBOT'S JETSON. This is a drop-in replacement for
``dimos run unitree-g1-ik-camera`` that publishes BYTE-IDENTICAL LCM output but
**runs no dimos coordinator** — so the laptop's ``dimos run unitree-g1-ik-reach``
is the only Coordinator on the LCM bus and the two no longer collide
(`CoordinatorRPC._ensure_no_existing_service` would otherwise reject the 2nd).

It reuses dimos message + transport code directly (no ModuleCoordinator):
  - ``PointCloud2.from_rgbd`` builds the cloud (optical-frame coords) exactly as
    ``RealSenseCamera`` does.
  - ``LCMTransport`` publishes on the same channels/types/encoding the laptop's
    RerunBridge + IkReachBridge already expect — so the laptop side is UNCHANGED.

Contract (must match RealSenseCamera output; do NOT pre-transform to torso):
  /camera/pointcloud   PointCloud2  frame_id=camera_color_optical_frame  ~3 Hz
  /camera/color_image  Image (RGB)  same frame_id
  /camera/camera_info  CameraInfo (live intrinsics)  ~1 Hz
The clicked point comes back in optical-frame coords; IkReachBridge applies its
own hardcoded optical->torso SE3.

Preconditions (see project_g1_ik_reach_okra memory):
  - D435i free (stop g1-teleimager.service first).
  - LCM route on eth0: ``sudo ip route replace 239.255.76.67/32 dev eth0`` and
    ``sudo sysctl -w net.core.rmem_max=67108864 net.core.rmem_default=67108864``.

Run (in the ik_cam conda env):
  python ik_camera_standalone.py
No PYTEST_VERSION needed (no dimos run = no system configurator).
"""

from __future__ import annotations

import os
import signal
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

# NB: import the low-level LCM directly, NOT dimos.core.transport — the latter
# pulls in the DDS transport (cyclonedds) which is absent on the Jetson ik_cam env.
# LCM(...).publish(Topic(name, MsgType), msg) yields the identical wire format
# (channel "/topic#sensor_msgs.X" + msg.lcm_encode()) that LCMTransport produces.
from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic

SERIAL = os.getenv("IK_CAMERA_SERIAL", "405622072808")
WIDTH = int(os.getenv("IK_CAMERA_WIDTH", "640"))
HEIGHT = int(os.getenv("IK_CAMERA_HEIGHT", "480"))
CAPTURE_FPS = int(os.getenv("IK_CAMERA_CAPTURE_FPS", "15"))
PC_FPS = float(os.getenv("IK_CAMERA_PC_FPS", "3.0"))
INFO_FPS = float(os.getenv("IK_CAMERA_INFO_FPS", "1.0"))
VOXEL = float(
    os.getenv("IK_CAMERA_VOXEL", "0.002")
)  # 2mm voxel: dense enough to read the okra (≈6x the old 5mm). Smaller=denser but heavier.
# Drop points beyond this optical distance [m]. The head camera is pitched down and
# sees the far floor/wall (median ~2m), which swamps the near okra (~0.3-0.6m) and
# makes it un-clickable. Truncating to the near reach-workspace keeps only the okra +
# immediate table so the click lands on the object, not the far background.
DEPTH_TRUNC = float(os.getenv("IK_CAMERA_DEPTH_TRUNC", "0.8"))
LCM_URL = os.getenv("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=1")
# Match RealSenseCamera: cloud is published in the COLOR OPTICAL frame.
OPTICAL_FRAME = "camera_color_optical_frame"

_DISTORTION = {
    rs.distortion.none: "",
    rs.distortion.modified_brown_conrady: "plumb_bob",
    rs.distortion.inverse_brown_conrady: "plumb_bob",
    rs.distortion.ftheta: "equidistant",
    rs.distortion.brown_conrady: "plumb_bob",
    rs.distortion.kannala_brandt4: "equidistant",
}

_running = True


def _stop(*_a: object) -> None:
    global _running
    _running = False


def _camera_info_from_intrinsics(intr: rs.intrinsics) -> CameraInfo:
    """Inline of RealSenseCamera._intrinsics_to_camera_info (camera.py:215-242)."""
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    return CameraInfo(
        height=intr.height,
        width=intr.width,
        distortion_model=_DISTORTION.get(intr.model, ""),
        D=list(intr.coeffs) if intr.coeffs else [],
        K=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        P=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        frame_id=OPTICAL_FRAME,
    )


def main() -> int:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # pyrealsense2 pipeline (inline of RealSenseCamera.start, camera.py:119-160)
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(SERIAL)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, CAPTURE_FPS)
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, CAPTURE_FPS)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_info = _camera_info_from_intrinsics(color_intr)
    print(
        f"[ik-cam] D435i {SERIAL} {WIDTH}x{HEIGHT}@{CAPTURE_FPS} depth_scale={depth_scale:.6f} "
        f"K=[{color_intr.fx:.1f},{color_intr.fy:.1f},{color_intr.ppx:.1f},{color_intr.ppy:.1f}]",
        flush=True,
    )

    lc = LCM(url=LCM_URL)
    lc.start()
    pc_topic = Topic("/camera/pointcloud", PointCloud2)
    color_topic = Topic("/camera/color_image", Image)
    info_topic = Topic("/camera/camera_info", CameraInfo)
    print(f"[ik-cam] publishing on {LCM_URL} (pc {PC_FPS}Hz, info {INFO_FPS}Hz)", flush=True)

    pc_interval = 1.0 / PC_FPS
    info_interval = 1.0 / INFO_FPS
    last_pc = 0.0
    last_info = 0.0
    n = 0
    try:
        while _running:
            try:
                frames = pipeline.wait_for_frames(2000)
            except Exception as e:
                print(f"[ik-cam] wait_for_frames: {e!r}", flush=True)
                continue
            now = time.time()
            if now - last_pc < pc_interval:
                continue
            last_pc = now

            frames = align.process(frames)
            cframe = frames.get_color_frame()
            dframe = frames.get_depth_frame()
            if not cframe or not dframe:
                continue

            ts = now
            rgb = cv2.cvtColor(np.asanyarray(cframe.get_data()), cv2.COLOR_BGR2RGB)
            color_img = Image(data=rgb, format=ImageFormat.RGB, frame_id=OPTICAL_FRAME, ts=ts)
            depth_img = Image(
                data=np.asanyarray(dframe.get_data()),
                format=ImageFormat.DEPTH16,
                frame_id=OPTICAL_FRAME,
                ts=ts,
            )

            cloud = PointCloud2.from_rgbd(
                color_image=color_img,
                depth_image=depth_img,
                camera_info=camera_info,
                depth_scale=depth_scale,
                depth_trunc=DEPTH_TRUNC,
            ).voxel_downsample(VOXEL)

            lc.publish(pc_topic, cloud)
            lc.publish(color_topic, color_img)
            if now - last_info >= info_interval:
                camera_info.ts = ts
                lc.publish(info_topic, camera_info)
                last_info = now

            n += 1
            if n % 30 == 0:
                pts, _ = cloud.as_numpy()
                print(f"[ik-cam] {n} clouds, last={len(pts)} pts", flush=True)
    finally:
        try:
            lc.stop()
        except Exception:
            pass
        pipeline.stop()
        print("[ik-cam] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
