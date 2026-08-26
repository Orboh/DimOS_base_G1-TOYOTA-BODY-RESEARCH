#!/usr/bin/env python3
"""Render close-up views of the G1 right-hand UMI mounting envelope.

This is an inspection image, not a motion result.  It makes the relationship between the
G1 visual hand mesh and the collision proxies reviewable without having to infer a 3-D
placement from a deposit video:

* orange: CAD-derived UMI-base outer envelope (91.4 x 172.0 x 58.2 mm);
* black: separately modelled GoPro / Media Mod keepout;
* red: arm, palm and fingertip collision proxies;
* blue: conservative torso keepout.

The supplied Fusion archive is an F3D file, which MuJoCo cannot load directly.  Therefore
the orange object intentionally remains the *measured CAD envelope*, rather than claiming
to be a visual CAD mesh.  It is mounted using the photo-confirmed long-side-across-hand
convention; only the precise physical clamp datum remains to be measured.

Run after regenerating the scene:
    MUJOCO_GL=egl .venv/bin/python oda/mujoco_sim/render_umi_mount_inspection.py
    -> oda/mujoco_sim/output/umi_mount_inspection.png
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dimos.robot.unitree.g1.ik_reach.right_arm_model import load_g1_right_arm_ik
from oda.mujoco_sim.build_g1_scene import _OUT
from oda.mujoco_sim.smoke_sim_arm import TIP_OFFSET
from oda.mujoco_sim.test_basket_deposit import (
    DROP_IK_SEED,
    DROP_TORSO,
    _make_rig,
    _scene_with_free_cargo,
    _solve,
)

_OUT_DIR = Path(__file__).resolve().parent / "output"
_OUT_PNG = _OUT_DIR / "umi_mount_inspection.png"
_RES = (640, 480)


def _lookat_xyaxes(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return MuJoCo camera x/y axes looking from ``eye`` to ``target``."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.concatenate([right, up])


def _scene_with_closeup_cameras(xml_text: str, wrist_pos: np.ndarray, wrist_rot: np.ndarray) -> str:
    """Add three fixed close-up cameras expressed around the current wrist pose."""
    root = ET.fromstring(xml_text)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("scene has no worldbody")

    # The target is at the centre of the UMI plate.  Offsets are wrist-frame vectors;
    # this keeps the views meaningful if the robot base is moved in the scene later.
    target = wrist_pos + wrist_rot @ np.array([0.055, 0.0, 0.040])
    views = [
        ("umi_mount_front_right", np.array([0.31, -0.32, 0.20])),
        # The positive-y side is inside the torso at this right-arm pose.  Keep this
        # second view on the external side but raise it, so the photo-derived mounting
        # envelope is visible rather than being hidden behind the torso keepout.
        ("umi_mount_elevated", np.array([0.36, -0.28, 0.35])),
        ("umi_mount_top", np.array([0.08, -0.18, 0.43])),
    ]
    for name, offset in views:
        eye = wrist_pos + wrist_rot @ offset
        ET.SubElement(
            worldbody,
            "camera",
            name=name,
            pos=" ".join(f"{v:.5f}" for v in eye),
            xyaxes=" ".join(f"{v:.5f}" for v in _lookat_xyaxes(eye, target)),
            fovy="42",
        )
    return ET.tostring(root, encoding="unicode")


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    """Add a compact title strip without affecting the simulated pixels."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 31), (20, 20, 20), -1)
    cv2.putText(frame, text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return frame


def render(scene: Path = _OUT, out_path: Path = _OUT_PNG) -> Path:
    """Write a three-view inspection image at the elbow-forward deposit pose."""
    arm = load_g1_right_arm_ik(gripper_offset_xyz=TIP_OFFSET)
    q_drop = _solve(arm, DROP_TORSO, DROP_IK_SEED)

    probe = _make_rig(_scene_with_free_cargo(scene, held=True))
    probe.data.qpos[22:29] = q_drop
    probe.command_right_arm(q_drop)
    mujoco.mj_forward(probe.model, probe.data)
    wrist_id = mujoco.mj_name2id(probe.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    wrist_pos = np.asarray(probe.data.xpos[wrist_id], dtype=float)
    wrist_rot = np.asarray(probe.data.xmat[wrist_id], dtype=float).reshape(3, 3)

    model = mujoco.MjModel.from_xml_string(
        _scene_with_closeup_cameras(_scene_with_free_cargo(scene, held=True), wrist_pos, wrist_rot)
    )
    data = mujoco.MjData(model)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    data.qpos[22:29] = q_drop
    mujoco.mj_forward(model, data)

    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(model, height=_RES[1], width=_RES[0])
    try:
        for camera, title in [
            ("umi_mount_front_right", "front-right"),
            ("umi_mount_elevated", "elevated"),
            ("umi_mount_top", "top"),
        ]:
            renderer.update_scene(data, camera=camera)
            frames.append(_label(renderer.render().copy(), title))
    finally:
        renderer.close()

    panel = np.concatenate(frames, axis=1)
    cv2.rectangle(panel, (9, 443), (574, 473), (20, 20, 20), -1)
    cv2.putText(
        panel,
        "orange: CAD UMI envelope 91.4 x 172.0 x 58.2 mm; black: GoPro keepout",
        (18, 464),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {out_path}")
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    render()
