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

"""``docs/sim-setup/sim_dds_bridge.py`` の ego_view ZMQ カメラ配信を受信し、
``perception.DepthCamera`` として ``color_image``/``depth_image``/``camera_info`` を
publish する。

MujocoSimModule の publish パターン（engine を直接持たず、Module が Out port へ配信する
形）と、``docs/sim-setup/dds_cam_sub.py`` のデコードロジック（msgpack + base64 + JPEG/PNG16）
を移植したもの。``ZEDCamera`` と同じ ``perception.DepthCamera`` インターフェースを実装する
ため、``unitree_g1_okra_honban.py`` 等で ``ZEDCamera.blueprint(...)`` の代わりにこの
``IsaacZmqDepthCamera.blueprint(...)`` を挿すだけで ``HarvestModule(use_zed_depth=True)`` の
``color_image``/``depth_image``/``camera_info`` 入力にそのまま接続できる。

wire format（``sim_dds_bridge.py`` の ``SIM_PUB_CAMERA=1`` 送信側、``msgpack``）::

    {
      "images": {topic: base64(jpeg bytes)},
      "timestamps": {topic: float unix time},
      "intrinsics": {topic: [fx, 0, cx, 0, fy, cy, 0, 0, 1]},
      "cam_to_torso": "x,y,z,qx,qy,qz,qw",  # torso<-optical（bridge が実搭載位置から算出）
      "depth": {topic: base64(16bit png, mm)},       # 省略可（SIM_CAM_MODE/深度有効時のみ）
      "depth_scale": 0.001,                          # 省略可（mm -> m）
    }

実行例（sim_dds_bridge.py 側、torsoモード + カメラ配信を有効化）::

    SIM_PUB_CAMERA=1 SIM_CAM_MODE=torso SIM_CAM_PORT=5555 \\
        ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/sim_dds_bridge.py

    # このモジュール側（dimos 実行ホスト）
    IsaacZmqDepthCamera.blueprint(zmq_host="127.0.0.1", zmq_port=5555, zmq_topic="ego_view")
"""

from __future__ import annotations

import base64
import threading
import time
from typing import Any

import cv2
import msgpack
import numpy as np
from pydantic import Field
import reactivex as rx
import zmq

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.hardware.sensors.camera.spec import DepthCameraConfig, DepthCameraHardware
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.spec import perception
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def _default_identity_transform() -> Transform:
    return Transform(translation=Vector3(0.0, 0.0, 0.0), rotation=Quaternion(0.0, 0.0, 0.0, 1.0))


def _parse_cam_to_torso(spec: str) -> Transform | None:
    """``sim_dds_bridge.py`` の ``cam_to_torso`` 文字列 "x,y,z,qx,qy,qz,qw" を Transform に。

    形式は honban.py の ``OKRA_CAM_TO_TORSO``（``HarvestModuleConfig.cam_to_torso_xyzquat``）
    と同じ。参考用（tf publish）に保持するのみで、座標変換自体は HarvestModule 側
    （``_parse_cam_to_torso`` / ``cam_to_torso_xyzquat``）が別途行う。
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    try:
        x, y, z, qx, qy, qz, qw = (float(v) for v in spec.split(","))
    except ValueError:
        logger.warning("IsaacZmqDepthCamera: cam_to_torso 文字列の解析に失敗", spec=spec)
        return None
    return Transform(translation=Vector3(x, y, z), rotation=Quaternion(qx, qy, qz, qw))


class IsaacZmqDepthCameraConfig(ModuleConfig, DepthCameraConfig):
    """``sim_dds_bridge.py`` の ego_view ZMQ 配信を購読する設定。"""

    camera_name: str = "sim_chest_cam"
    width: int = 640
    height: int = 360
    fps: int = 15
    base_frame_id: str = "torso_link"
    base_transform: Transform | None = Field(default_factory=_default_identity_transform)
    align_depth_to_color: bool = True
    enable_depth: bool = True
    enable_pointcloud: bool = False
    pointcloud_fps: float = 5.0
    # Voxel size [m] for downsampling the published cloud. <= 0 disables downsampling.
    pointcloud_voxel: float = 0.005
    camera_info_fps: float = 1.0

    # ZMQ 接続先（sim_dds_bridge.py の SIM_CAM_PORT/SIM_CAM_TOPIC と合わせる）
    zmq_host: str = "127.0.0.1"
    zmq_port: int = 5555
    zmq_topic: str = "ego_view"
    recv_timeout_s: float = 1.0  # [s] SUB recv のタイムアウト（stop() の join を早める）


class IsaacZmqDepthCamera(DepthCameraHardware, Module, perception.DepthCamera):
    """Isaac Sim（sim_dds_bridge.py）の ego_view ZMQ 配信を dimos の DepthCamera Module に変換する。

    ``ZEDCamera`` と同じ ``perception.DepthCamera`` インターフェース
    （color_image/depth_image/camera_info/depth_camera_info）を実装するため、
    ``HarvestModule(use_zed_depth=True)`` の入力にそのまま接続できる。
    """

    config: IsaacZmqDepthCameraConfig
    color_image: Out[Image]
    depth_image: Out[Image]
    pointcloud: Out[PointCloud2]
    camera_info: Out[CameraInfo]
    depth_camera_info: Out[CameraInfo]

    @property
    def _camera_link(self) -> str:
        return f"{self.config.camera_name}_link"

    @property
    def _color_frame(self) -> str:
        return f"{self.config.camera_name}_color_frame"

    @property
    def _color_optical_frame(self) -> str:
        return f"{self.config.camera_name}_color_optical_frame"

    @property
    def _depth_frame(self) -> str:
        return f"{self.config.camera_name}_depth_frame"

    @property
    def _depth_optical_frame(self) -> str:
        return f"{self.config.camera_name}_depth_optical_frame"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ctx: zmq.Context | None = None
        self._sock: zmq.Socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._color_camera_info: CameraInfo | None = None
        self._depth_camera_info: CameraInfo | None = None
        self._cam_to_torso: Transform | None = None
        self._latest_color_img: Image | None = None
        self._latest_depth_img: Image | None = None
        self._pointcloud_lock = threading.Lock()

    @staticmethod
    def _b(x: Any) -> bytes:
        return base64.b64decode(x) if isinstance(x, str) else bytes(x)

    def _build_camera_info(self, K: list[float]) -> None:
        """``dds_cam_sub.py`` と同じ K=[fx,0,cx,0,fy,cy,0,0,1] から CameraInfo を作る。"""
        fx, _, cx, _, fy, cy = K[0], K[1], K[2], K[3], K[4], K[5]
        self._color_camera_info = CameraInfo.from_intrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=self.config.width, height=self.config.height,
            frame_id=self._color_optical_frame,
        )
        depth_frame = (
            self._color_optical_frame
            if self.config.align_depth_to_color
            else self._depth_optical_frame
        )
        self._depth_camera_info = CameraInfo.from_intrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=self.config.width, height=self.config.height,
            frame_id=depth_frame,
        )

    def _publish_camera_info(self) -> None:
        ts = time.time()
        if self._color_camera_info is not None:
            self.camera_info.publish(self._color_camera_info.with_ts(ts))
        if self._depth_camera_info is not None:
            self.depth_camera_info.publish(self._depth_camera_info.with_ts(ts))

    def _publish_tf(self, ts: float) -> None:
        transforms = []
        if self.config.base_transform is not None:
            transforms.append(
                Transform(
                    translation=self.config.base_transform.translation,
                    rotation=self.config.base_transform.rotation,
                    frame_id=self.config.base_frame_id,
                    child_frame_id=self._camera_link,
                    ts=ts,
                )
            )
        # bridge が実搭載位置から算出した cam_to_torso（torso<-optical）。座標変換の正本は
        # HarvestModule 側の cam_to_torso_xyzquat なので、ここは可視化/参考用に publish するのみ。
        if self._cam_to_torso is not None:
            transforms.append(
                Transform(
                    translation=self._cam_to_torso.translation,
                    rotation=self._cam_to_torso.rotation,
                    frame_id=self.config.base_frame_id,
                    child_frame_id=self._color_optical_frame,
                    ts=ts,
                )
            )
        if transforms:
            self.tf.publish(*transforms)

    def _handle_message(self, msg: dict[str, Any]) -> bool:
        """1件の ego_view msgpack メッセージを処理し、Out port へ publish する。

        ``sock.recv()`` から切り離してあるのは、実 ZMQ 接続なしにデコード/publish
        ロジック単体をユニットテストできるようにするため（recv ループ自体は
        ``_recv_loop`` が担う）。処理できたフレームなら True を返す。
        """
        topic = self.config.zmq_topic
        cb = (msg.get("images") or {}).get(topic)
        if cb is None:
            return False
        bgr = cv2.imdecode(np.frombuffer(self._b(cb), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return False
        ts = float((msg.get("timestamps") or {}).get(topic, time.time()))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        color_img = Image(
            data=rgb, format=ImageFormat.RGB, frame_id=self._color_optical_frame, ts=ts
        )
        self.color_image.publish(color_img)

        K = (msg.get("intrinsics") or {}).get(topic)
        if K and self._color_camera_info is None:
            self._build_camera_info(list(K))
            self._publish_camera_info()

        cam_to_torso_str = msg.get("cam_to_torso")
        if cam_to_torso_str and self._cam_to_torso is None:
            self._cam_to_torso = _parse_cam_to_torso(cam_to_torso_str)
            logger.info(
                "IsaacZmqDepthCamera: cam_to_torso 受信（bridge実搭載位置由来・参考値）",
                cam_to_torso=cam_to_torso_str,
            )

        depth_img = None
        if self.config.enable_depth:
            db = (msg.get("depth") or {}).get(topic)
            if db is not None:
                d16 = cv2.imdecode(np.frombuffer(self._b(db), np.uint8), cv2.IMREAD_UNCHANGED)
                if d16 is not None:
                    scale = float(msg.get("depth_scale", 0.001))  # 既定 mm -> m
                    depth_m = d16.astype(np.float32) * scale
                    depth_frame = (
                        self._color_optical_frame
                        if self.config.align_depth_to_color
                        else self._depth_optical_frame
                    )
                    depth_img = Image(
                        data=depth_m, format=ImageFormat.DEPTH, frame_id=depth_frame, ts=ts
                    )
                    self.depth_image.publish(depth_img)

        if self.config.enable_pointcloud and depth_img is not None:
            with self._pointcloud_lock:
                self._latest_color_img = color_img
                self._latest_depth_img = depth_img

        self._publish_tf(ts)
        return True

    def _recv_loop(self) -> None:
        sock = self._sock
        if sock is None:
            return
        first = True
        while self._running:
            try:
                payload = sock.recv()
            except zmq.Again:
                continue
            except Exception as exc:
                if self._running:
                    logger.error("IsaacZmqDepthCamera: recv error", error=str(exc))
                break

            try:
                msg = msgpack.unpackb(payload, raw=False)
            except Exception as exc:
                logger.warning("IsaacZmqDepthCamera: msgpack decode 失敗", error=str(exc))
                continue

            if self._handle_message(msg) and first:
                logger.info("IsaacZmqDepthCamera: first frame received")
                first = False

    def _generate_pointcloud(self) -> None:
        with self._pointcloud_lock:
            color_img = self._latest_color_img
            depth_img = self._latest_depth_img
        if color_img is None or depth_img is None or self._color_camera_info is None:
            return
        try:
            pcd = PointCloud2.from_rgbd(
                color_image=color_img,
                depth_image=depth_img,
                camera_info=self._color_camera_info,
                depth_scale=1.0,  # depth はデコード時点で既にメートル換算済み
            )
            if self.config.pointcloud_voxel > 0.0:
                pcd = pcd.voxel_downsample(self.config.pointcloud_voxel)
            self.pointcloud.publish(pcd)
        except Exception as exc:
            logger.error("IsaacZmqDepthCamera: pointcloud生成エラー", error=str(exc))

    @rpc
    def start(self) -> None:
        self._ctx = zmq.Context.instance()
        sock = self._ctx.socket(zmq.SUB)
        sock.connect(f"tcp://{self.config.zmq_host}:{self.config.zmq_port}")
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVTIMEO, int(self.config.recv_timeout_s * 1000))
        self._sock = sock

        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="IsaacZmqCamRecv"
        )
        self._thread.start()

        interval_sec = 1.0 / self.config.camera_info_fps
        self.register_disposable(
            rx.interval(interval_sec).subscribe(
                on_next=lambda _: self._publish_camera_info(),
                on_error=lambda e: logger.error("CameraInfo publish error", error=str(e)),
            )
        )
        if self.config.enable_pointcloud and self.config.enable_depth:
            pc_interval = 1.0 / self.config.pointcloud_fps
            self.register_disposable(
                rx.interval(pc_interval).subscribe(
                    on_next=lambda _: self._generate_pointcloud(),
                    on_error=lambda e: logger.error("Pointcloud error", error=str(e)),
                )
            )
        logger.info(
            "IsaacZmqDepthCamera started",
            zmq=f"tcp://{self.config.zmq_host}:{self.config.zmq_port}",
            topic=self.config.zmq_topic,
        )

    @rpc
    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:
                pass
            self._sock = None
        self._color_camera_info = None
        self._depth_camera_info = None
        self._cam_to_torso = None
        self._latest_color_img = None
        self._latest_depth_img = None
        super().stop()

    @rpc
    def get_color_camera_info(self) -> CameraInfo | None:
        return self._color_camera_info

    @rpc
    def get_depth_camera_info(self) -> CameraInfo | None:
        return self._depth_camera_info

    @rpc
    def get_depth_scale(self) -> float:
        return 1.0  # depth はデコード時点で既にメートル換算済み


__all__ = ["IsaacZmqDepthCamera", "IsaacZmqDepthCameraConfig"]
