#!/usr/bin/env python3
"""Render the basket-deposit sequence (entry -> drop -> release -> fall -> settle) to mp4.

This is a pure visualization companion to ``test_basket_deposit.py`` -- same sequence,
same helper functions (imported, not copied), but every advance-phase also grabs frames
from a camera and writes them to an mp4 so a non-engineer can watch the okra actually land
in the basket instead of reading tracking-error numbers.

``test_basket_deposit.py`` is not edited: this script only *imports* its pure helper
functions (``_scene_with_free_cargo``, ``_make_rig``, ``_solve``, ``_advance``,
``_cargo_torso``) and re-sequences them with rendering interleaved.

A camera is added to the scene *in memory* (not written back to g1_okra_scene.xml) because
the shipped ``spectator`` camera is framed around the far-away pick location
(``OKRA_IN_TORSO``), not the basket near the pelvis. ``deposit_cam`` below reuses
spectator's viewing convention but re-centers on the basket.

Run after regenerating the scene:
    MUJOCO_GL=egl .venv/bin/python oda/mujoco_sim/render_basket_deposit.py
    -> oda/mujoco_sim/output/basket_deposit.mp4
"""

from __future__ import annotations

import argparse
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
from oda.mujoco_sim.build_g1_scene import _BASKET_COLLISION_NAMES, _OUT
from oda.mujoco_sim.smoke_sim_arm import TIP_OFFSET, SimRig
from oda.mujoco_sim.test_basket_deposit import (
    DROP_TORSO,
    ENTRY_TORSO,
    RETREAT_TORSO,
    SETTLE_SECONDS,
    _cargo_torso,
    _make_rig,
    _scene_with_free_cargo,
    _solve,
)

_OUT_DIR = Path(__file__).resolve().parent / "output"

CAM_NAME = "deposit_cam"
# 640x480: the scene has no <visual><global offwidth=.../> override, so this is MuJoCo's
# default offscreen framebuffer size -- matches the existing spectator res in
# smoke_sim_arm.py's _check_cameras().
CAM_RES = (640, 480)
FPS = 15
TARGET_SECONDS = 14.0  # desired wall-clock length of the output video


# The basket is a 5-plate box open upward (+Z): back/front/bottom/left/right plates, no
# lid. A camera above and to the right looks through that physical opening, so the video
# directly verifies the intended top-down insertion instead of relying on translucency.
_DEPOSIT_LOOKAT_TORSO = DROP_TORSO
_DEPOSIT_EYE_OFFSET_TORSO = np.array([0.36, -0.34, 0.42])


def _lookat_xyaxes(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """MuJoCo camera xyaxes (x-axis, y-axis) so the camera's -Z points at ``target``."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.concatenate([right, up])


BASKET_ALPHA = 0.35  # render-only: see the pod through the walls of an otherwise closed box.


def _add_deposit_cam(xml_text: str, torso_world: np.ndarray, torso_rot: np.ndarray) -> str:
    """Add a fixed world-frame camera that looks down through the open (+Z) face.

    Also makes the basket's 5 collision plates translucent *for this render only* (rgba
    alpha is a pure visual property; contype/conaffinity/size are untouched, so physics is
    identical to test_basket_deposit.py). The transparency remains a supplementary view of
    the landing, while the opening itself is visible from the camera's elevated viewpoint.
    """
    target_world = torso_world + torso_rot @ _DEPOSIT_LOOKAT_TORSO
    eye_world = torso_world + torso_rot @ (_DEPOSIT_LOOKAT_TORSO + _DEPOSIT_EYE_OFFSET_TORSO)
    xyaxes = _lookat_xyaxes(eye_world, target_world)
    root = ET.fromstring(xml_text)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("scene has no worldbody")
    ET.SubElement(
        worldbody,
        "camera",
        name=CAM_NAME,
        pos=" ".join(f"{v:.4f}" for v in eye_world),
        xyaxes=" ".join(f"{v:.5f}" for v in xyaxes),
        fovy="48",
    )
    for geom in root.iter("geom"):
        if geom.get("name") in _BASKET_COLLISION_NAMES:
            r, g, b, _a = (float(v) for v in geom.get("rgba", "0.62 0.45 0.28 1").split())
            geom.set("rgba", f"{r} {g} {b} {BASKET_ALPHA}")
    return ET.tostring(root, encoding="unicode")


def _hide_pick_okra(rig: SimRig) -> None:
    """Teleport the far-away pick-location ``okra`` mocap body out of frame.

    It is unrelated to the already-grasped ``cargo_okra`` this script renders (see
    ``build_g1_scene.py``'s ``okra`` body, a separate mocap-teleportable target used by
    the *reach* tests). Left at its default position it is often a long stem crossing
    straight through this close-up basket shot, which reads as a stray line in the video.
    Moving its mocap frame does not touch any qpos index (that is the whole point of
    ``mocap="true"``), so this is a pure visualization tweak.
    """
    okra_body = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_BODY, "okra")
    if okra_body < 0:
        return
    mocap_id = rig.model.body_mocapid[okra_body]
    if mocap_id < 0:
        return
    rig.data.mocap_pos[mocap_id] = np.array([0.0, 0.0, -5.0])
    mujoco.mj_forward(rig.model, rig.data)


class VideoRecorder:
    """A persistent mp4 writer with a swappable renderer.

    The deposit sequence uses two *different* MjModel instances (the held-cargo model,
    then a freshly compiled released-cargo model -- see ``test_basket_deposit.py``'s
    docstring on why). ``mujoco.Renderer`` is tied to one model, but the video must be a
    single continuous file, so the cv2.VideoWriter is opened once and the Renderer is
    rebound via ``rebind()`` when the model changes.
    """

    def __init__(self, out_path: Path, fps: int, res: tuple[int, int]) -> None:
        self.res = res
        self.renderer: mujoco.Renderer | None = None
        self.writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, res
        )
        self.n_frames = 0

    def rebind(self, model: mujoco.MjModel) -> None:
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(model, height=self.res[1], width=self.res[0])

    def grab(self, data: mujoco.MjData) -> None:
        assert self.renderer is not None, "call rebind() before grab()"
        self.renderer.update_scene(data, camera=CAM_NAME)
        frame = self.renderer.render()
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.n_frames += 1

    def hold(self, data: mujoco.MjData, n: int) -> None:
        for _ in range(n):
            self.grab(data)

    def close(self) -> None:
        self.writer.release()
        if self.renderer is not None:
            self.renderer.close()


def _advance_recorded(
    rig: SimRig, q_goal: np.ndarray, seconds: float, rec: VideoRecorder, frame_every: int
) -> None:
    """Same ramp as ``test_basket_deposit._advance``, but grabs a frame every N steps."""
    q_start = rig.q_right()
    steps = max(1, int(seconds / rig.model.opt.timestep))
    for i in range(steps):
        rig.command_right_arm(q_start + (q_goal - q_start) * ((i + 1) / steps))
        mujoco.mj_step(rig.model, rig.data)
        if i % frame_every == 0:
            rec.grab(rig.data)


def render(
    scene: Path,
    out_path: Path,
    *,
    q_start: np.ndarray | None = None,
    tip_world_override: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Run entry->drop->release->retreat/settle, writing an mp4 of ``deposit_cam``.

    Args:
        q_start: right-arm seed for the ENTRY solve. Defaults to the scene's home right
            arm angles (matches ``test_basket_deposit.py``). Pass the arm's post-reach
            configuration to render a deposit that starts from an actual grasp pose.
        tip_world_override: world-frame point the (virtually-held) cargo starts welded
            to. Defaults to the tip position at the scene's home pose. Pass the tip
            position at ``q_start`` to keep the cargo glued to the hand that "grasped" it.
    Returns:
        dict summary (final cargo torso-frame position, settle speed, whether it ended
        inside the basket keepout box) -- reused by the workspace test to report per-clip
        outcomes without re-deriving them.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sizing: total simulated motion time is entry(2s) + drop(2s) + settle(SETTLE_SECONDS)
    # -- render enough frames from that + a hold at start/end so playback lands near
    # TARGET_SECONDS without slowing physics down.
    moving_seconds = 2.0 + 2.0 + SETTLE_SECONDS
    total_frames = int(FPS * TARGET_SECONDS)
    hold_frames = max(1, int(FPS * 1.0))  # 1 s hold at start and end
    moving_frames = max(1, total_frames - 2 * hold_frames)
    frame_every = max(1, int((moving_seconds / rig_timestep_hint()) // moving_frames))

    base_xml = _scene_with_free_cargo(scene, held=True)
    home_rig_probe = _make_rig(base_xml)
    torso_pos, torso_rot = home_rig_probe.torso_pose()
    xml_held = _add_deposit_cam(base_xml, torso_pos, torso_rot)

    rig = _make_rig(xml_held)
    if q_start is not None:
        rig.data.qpos[22:29] = q_start
        rig.command_right_arm(q_start)
        mujoco.mj_forward(rig.model, rig.data)
    if tip_world_override is not None:
        # Snap the (still virtually-welded) cargo onto the actual starting tip pose,
        # matching how _scene_with_free_cargo would have placed it had it been built
        # with this q_start in the first place.
        cargo_body = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_BODY, "cargo_okra")
        cargo_joint = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_JOINT, "cargo_okra_free")
        cqa = rig.model.jnt_qposadr[cargo_joint]
        rig.data.qpos[cqa : cqa + 3] = tip_world_override
        rig.data.qpos[cqa + 3 : cqa + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(rig.model, rig.data)
    _hide_pick_okra(rig)

    cargo_body = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_BODY, "cargo_okra")
    cargo_joint = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_JOINT, "cargo_okra_free")
    cargo_qpos = rig.model.jnt_qposadr[cargo_joint]

    arm = load_g1_right_arm_ik(gripper_offset_xyz=TIP_OFFSET)
    q_entry = _solve(arm, ENTRY_TORSO, rig.q_right())
    q_drop = _solve(arm, DROP_TORSO, q_entry)
    q_retreat = _solve(arm, RETREAT_TORSO, q_drop)
    if verbose:
        print("[render] targets solved: entry/drop/retreat OK")

    rec = VideoRecorder(out_path, FPS, CAM_RES)
    rec.rebind(rig.model)
    rec.hold(rig.data, hold_frames)

    _advance_recorded(rig, q_entry, seconds=2.0, rec=rec, frame_every=frame_every)
    _advance_recorded(rig, q_drop, seconds=2.0, rec=rec, frame_every=frame_every)
    if verbose:
        print(f"[render] carried entry -> drop, cargo(torso)={np.round(_cargo_torso(rig, cargo_body), 4).tolist()}")

    # Release: same recipe as test_basket_deposit.py -- copy qpos across into a freshly
    # compiled, collidable-cargo model at the exact release pose.
    release_qpos = rig.data.qpos.copy()
    cargo_rot_torso = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    cargo_quat = np.empty(4)
    mujoco.mju_mat2Quat(cargo_quat, (torso_rot @ cargo_rot_torso).reshape(-1))
    release_qpos[cargo_qpos + 3 : cargo_qpos + 7] = cargo_quat

    xml_released = _add_deposit_cam(_scene_with_free_cargo(scene, held=False), torso_pos, torso_rot)
    rig = _make_rig(xml_released)
    cargo_body = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_BODY, "cargo_okra")
    cargo_joint = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_JOINT, "cargo_okra_free")
    cargo_qvel = rig.model.jnt_dofadr[cargo_joint]
    rig.data.qpos[:] = release_qpos
    rig.data.qvel[:] = 0.0
    rig.command_right_arm(q_drop)
    mujoco.mj_forward(rig.model, rig.data)
    _hide_pick_okra(rig)

    rec.rebind(rig.model)
    rec.hold(rig.data, hold_frames // 2)
    _advance_recorded(rig, q_retreat, seconds=SETTLE_SECONDS, rec=rec, frame_every=frame_every)
    cargo_t = _cargo_torso(rig, cargo_body)
    cargo_speed = float(np.linalg.norm(rig.data.qvel[cargo_qvel : cargo_qvel + 3]))
    rec.hold(rig.data, hold_frames)
    n_frames = rec.n_frames
    rec.close()

    inside = 0.090 < cargo_t[0] < 0.275 and abs(cargo_t[1]) < 0.040 and -0.165 < cargo_t[2] < -0.080
    if verbose:
        print(f"[render] final cargo(torso)={np.round(cargo_t, 4).tolist()} linear_speed={cargo_speed:.4f} m/s inside={inside}")
        print(f"[render] wrote {out_path} ({n_frames} frames, {n_frames / FPS:.1f} s)")
    return {
        "cargo_torso": cargo_t.tolist(),
        "cargo_speed": cargo_speed,
        "inside_basket": bool(inside),
        "video": str(out_path),
    }


def rig_timestep_hint() -> float:
    # g1_okra_scene.xml always sets timestep=0.002 (see build_g1_scene._add_options);
    # kept as a tiny helper so a future timestep change only needs editing there.
    return 0.002


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=_OUT)
    ap.add_argument("--out", type=Path, default=_OUT_DIR / "basket_deposit.mp4")
    args = ap.parse_args()
    if not args.scene.exists():
        print(f"scene missing: {args.scene}\nrun: .venv/bin/python oda/mujoco_sim/build_g1_scene.py")
        return 2
    render(args.scene, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
