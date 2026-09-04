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

"""Publish a synthetic ``/clicked_point`` to drive an IkReachBridge reach.

Stands in for a human click on the Rerun point cloud, so the IK reach path can be
exercised without a viewer (headless laptop) or a real okra in frame.

FRAME: the okra blueprints set ``click_in_camera_body_frame=True``, so x/y/z are in
the CAMERA BODY frame (URDF convention: +X forward, +Y left, +Z up, origin at the
camera mount). IkReachBridge maps them to torso_link with ``camera_mount_xyzrpy``
alone -- see ik_reach_bridge.py ``_on_click``/``_reach``. ``frame_id`` must equal the
blueprint's ``expected_click_frame`` or the click is rejected (LIVE refuses to move
on a mismatch, by design).

Usage:

    .venv/bin/python oda/g1_click_point.py                 # default probe point
    .venv/bin/python oda/g1_click_point.py 0.34 -0.18 -0.10

ALWAYS dry-run first and read the bridge's reported torso target / hand_tip before
running the same point LIVE.
"""

from __future__ import annotations

import sys
import time

from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.PointStamped import PointStamped

_TOPIC = "/clicked_point"

# Must match IkReachBridgeConfig.expected_click_frame in the blueprint.
_FRAME_ID = "/world/camera/pointcloud"

# Camera-body-frame probe point. Maps to roughly torso [0.45, -0.15, 0.15] with the
# chest ZED mount [0.109, 0.030, 0.248, 0, -0.0209, 0] -- forward, on the right arm's
# side, below shoulder height, comfortably inside IkReachBridge's torso workspace box
# (ws_x [0.05, 0.65], ws_y [-0.75, 0.20], ws_z [-0.35, 0.85]).
_DEFAULT_XYZ = (0.339, -0.180, -0.105)


def main() -> None:
    if len(sys.argv) == 4:
        x, y, z = (float(v) for v in sys.argv[1:4])
    elif len(sys.argv) == 1:
        x, y, z = _DEFAULT_XYZ
    else:
        sys.exit(f"usage: {sys.argv[0]} [x y z]   (camera body frame, metres)")

    transport = LCMTransport(_TOPIC, PointStamped)
    transport.start()

    msg = PointStamped()
    msg.frame_id = _FRAME_ID
    msg.x, msg.y, msg.z = x, y, z
    msg.ts = time.time()

    # One click only: the bridge debounces on reach_min_interval_s, and a repeat
    # would either be swallowed or (with confirm_click=1) read as the confirmation.
    transport.broadcast(None, msg)
    time.sleep(0.3)

    print(f"published {_TOPIC} frame={_FRAME_ID} xyz=({x:.3f}, {y:.3f}, {z:.3f}) [camera body]")


if __name__ == "__main__":
    main()
