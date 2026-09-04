#!/usr/bin/env python3
"""Generate the MuJoCo scene for the G1 okra IK+Diffusion rig, from ``g1.urdf``.

WHY GENERATE INSTEAD OF SHIPPING AN MJCF: the IK that this sim exists to test is
Pinocchio over ``dimos/robot/unitree/g1/g1.urdf`` (right_arm_model.py). If the sim used a
*different* model (e.g. mujoco_menagerie's unitree_g1) every result would be confounded by
a model mismatch. Deriving the MJCF from the very same URDF makes sim FK and IK FK
identical by construction -- verified: MuJoCo puts ``torso_link`` at [-0.0039635, 0, 0.044],
which is exactly pinocchio's ``torso_in_root`` translation.

Why the repo's existing MuJoCo path is not reused: ``dimos/simulation/mujoco/model.py``
needs ``mujoco_playground`` (not installed in .venv) plus the ``mujoco_sim`` LFS archive
(``git lfs pull`` fails on this checkout), and it builds a *locomotion* scene (ONNX walking
policy, lidar rings, office mesh) which is irrelevant to an arm-only reach test.

Three transforms this file has to get right, because everything downstream depends on them:

1. **Mesh strip.** The repo ships no STLs, so every ``<visual>``/``<collision>`` that
   references a mesh is dropped. Link *inertials* are explicit in the URDF and are kept, so
   the dynamics stay right. Visual bones are then synthesized (see ``_add_bones``).
2. **Base weld.** The URDF root is a floating joint. This rig tests an arm on a standing
   robot, so the free joint is removed and the pelvis is welded at standing height. No
   locomotion policy, no falling over, no confounding base drift in the FK comparison.
3. **Cameras.** MuJoCo cameras look down their own -Z with +X right and +Y up. The chest
   ZED and wrist GoPro are specified in the DimOS/URDF body convention (+X fwd, +Y left,
   +Z up). ``_cam_quat_from_body_rpy`` does that conversion once, here, rather than
   scattering sign flips through the runtime.

Run:
    .venv/bin/python oda/mujoco_sim/build_g1_scene.py
    -> oda/mujoco_sim/g1_okra_scene.xml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_URDF = _REPO / "dimos" / "robot" / "unitree" / "g1" / "g1.urdf"
_OUT = Path(__file__).resolve().parent / "g1_okra_scene.xml"

# ---------------------------------------------------------------------------
# Home posture. 29-DOF canonical order (== make_humanoid_joints("g1")).
# Legs/waist straight; both arms in the natural lowered pose that -- per the hardware
# session on 2026-07-23 -- does NOT occlude the chest camera.
# ---------------------------------------------------------------------------
_LEG_HOME = [0.0] * 12
_WAIST_HOME = [0.0, 0.0, 0.0]
_LEFT_ARM_HOME = [0.20, 0.25, 0.0, 0.40, 0.0, 0.0, 0.0]
_RIGHT_ARM_HOME = [0.20, -0.25, 0.0, 0.40, 0.0, 0.0, 0.0]
HOME_Q = _LEG_HOME + _WAIST_HOME + _LEFT_ARM_HOME + _RIGHT_ARM_HOME

# Actuator gains. The arms use the same kp/kd the real G1ArmSdkConnection defaults to
# (OKRA_NOACT_KP_ARM=80 / KD_ARM=3), so the sim's tracking lag resembles the hardware's --
# the diffusion loop's convergence test is sensitive to exactly that lag.
KP_ARM, KD_ARM = 80.0, 3.0
# Legs/waist only have to hold a posture against gravity; stiff and boring on purpose.
KP_POST, KD_POST = 400.0, 20.0

# Chest ZED Mini mount on torso_link, DimOS body convention [x,y,z,r,p,y] (positive pitch
# = nose DOWN). Translation matches the hardware blueprint's ZED_MOUNT_XYZRPY (measured
# 2026-07-24); the PITCH is deliberately not the hardware's -0.0209 rad. At that near-zero
# pitch a target low enough for the arm to reach sits ~30 deg below the optical axis, i.e.
# clipped at the bottom edge of the frame -- verified by rendering. 0.25 rad (14 deg) of
# nose-down puts the workspace in the middle of the frame with margin.
#
# THIS MUST STAY EQUAL TO the blueprint's mount value, because IkReachBridge uses it as the
# torso<-camera transform for every click. Both read this same env var with this same
# default, so they cannot silently diverge; _check_chest_fov() below asserts the geometry.
SIM_ZED_MOUNT_ENV = "SIM_ZED_MOUNT_XYZRPY"
SIM_ZED_MOUNT_DEFAULT = "0.109,0.030,0.248,0.0,0.25,0.0"
CHEST_MOUNT = [
    float(v) for v in os.getenv(SIM_ZED_MOUNT_ENV, SIM_ZED_MOUNT_DEFAULT).split(",")
]
CHEST_FOVY_DEG = 70.0        # ZED Mini vertical FOV, approximate
CHEST_RES = (640, 360)

# Wrist GoPro mount on right_wrist_yaw_link (the UMI observation camera). The real dex1-1
# rig's GoPro looks forward-and-slightly-inward past the jaw. Pitch is +25 deg (nose down)
# so the gripper occupies the lower part of the frame like the UMI training data.
WRIST_MOUNT = [0.055, -0.02, 0.045, 0.0, 0.436, 0.0]
WRIST_FOVY_DEG = 92.0        # GoPro Wide, de-fisheyed equivalent (sim renders pinhole)
WRIST_RES = (320, 240)

# Spectator camera so the run is watchable in rerun without a MuJoCo GL window.
SPECTATOR_RES = (640, 480)

# Okra target, in torso_link frame. Chosen to satisfy three constraints simultaneously:
#   - inside IkReachBridge's workspace box (ws_x[0.05,0.65] ws_y[-0.75,0.20] ws_z[-0.35,0.85])
#   - >= 0.35 m from the chest camera (the ZED Mini's minimum measurement distance)
#   - reachable by the right arm (neutral tip sits at torso [0.245,-0.152,0.051])
SIM_OKRA_ENV = "SIM_OKRA_IN_TORSO"
OKRA_IN_TORSO = [
    float(v) for v in os.getenv(SIM_OKRA_ENV, "0.45,-0.20,0.10").split(",")
]
OKRA_RADIUS = 0.013
OKRA_HALF_LEN = 0.045
# Keep the target this far inside the frame edge (fraction of the half-FOV). A pod sitting
# on the very edge is technically visible but a click on it lands on background depth.
FOV_MARGIN = 0.75


def _strip_meshes(root: ET.Element) -> int:
    """Drop every visual/collision that references a mesh file. Returns the drop count."""
    dropped = 0
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for el in list(link.findall(tag)):
                if el.find(".//mesh") is not None:
                    link.remove(el)
                    dropped += 1
    return dropped


def _urdf_to_mjcf(urdf_xml: str) -> str:
    """Let MuJoCo do the URDF->MJCF conversion, then hand back the MJCF text.

    Going through MuJoCo (rather than writing MJCF by hand) is what guarantees the body
    tree, joint axes, limits and inertias match the URDF the IK solves against.
    """
    model = mujoco.MjModel.from_xml_string(urdf_xml)
    tmp = Path(_OUT).with_suffix(".raw.xml")
    mujoco.mj_saveLastXML(str(tmp), model)
    text = tmp.read_text()
    tmp.unlink()
    return text


def _joint_names_in_order(root: ET.Element) -> list[str]:
    """Hinge joint names in MJCF document order (== the canonical 29-DOF order)."""
    return [
        j.get("name", "")
        for j in root.iter("joint")
        if j.get("type") in (None, "hinge") and j.get("name")
    ]


def _weld_base(root: ET.Element, pelvis_z: float) -> None:
    """Remove the floating base and pin the pelvis at ``pelvis_z``.

    MuJoCo emits the URDF floating joint as a ``<freejoint>`` (or a 6-DOF joint set) on the
    pelvis body. Deleting it welds the body to its parent -- the world -- at whatever pos
    the body carries, so the pos is set here too.
    """
    for body in root.iter("body"):
        if body.get("name") != "pelvis":
            continue
        for child in list(body):
            if child.tag == "freejoint" or (
                child.tag == "joint" and child.get("type") in ("free", "floating")
            ):
                body.remove(child)
        body.set("pos", f"0 0 {pelvis_z:.6f}")
        return
    raise RuntimeError("pelvis body not found in generated MJCF")


def _cam_quat_from_body_rpy(rpy: list[float]) -> tuple[float, float, float, float]:
    """Body-convention rpy -> MuJoCo camera quaternion (w,x,y,z), MuJoCo frame.

    A DimOS/URDF body frame is +X forward, +Y left, +Z up. A MuJoCo camera looks along its
    own -Z, with +X right and +Y up. The fixed change of basis between them is

        cam_X = -body_Y      (right    = -left)
        cam_Y = +body_Z      (up       =  up)
        cam_Z = -body_X      (backward = -forward)

    so R_cam = R_body(rpy) @ B with B the matrix of those columns. Positive pitch in the
    body convention is nose-DOWN, which this preserves.
    """
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    r_body = rz @ ry @ rx
    # Columns are where cam X, Y, Z point, expressed in the body frame -- written out
    # column-wise on purpose: transposing this matrix aims the camera along body -Y
    # instead of forward, which renders a plausible-looking but useless view.
    basis = np.array(
        [
            [0.0, 0.0, -1.0],   # row x: cam_X has no body-x, cam_Z is -body_X
            [-1.0, 0.0, 0.0],   # row y: cam_X is -body_Y
            [0.0, 1.0, 0.0],    # row z: cam_Y is +body_Z
        ]
    )
    r_cam = r_body @ basis
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, r_cam.flatten())
    return tuple(float(v) for v in quat)  # type: ignore[return-value]


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise RuntimeError(f"body {name!r} not found in generated MJCF")


def _add_camera(
    root: ET.Element, parent_body: str, name: str, mount: list[float], fovy_deg: float
) -> None:
    body = _find_body(root, parent_body)
    qw, qx, qy, qz = _cam_quat_from_body_rpy(mount[3:])
    ET.SubElement(
        body,
        "camera",
        name=name,
        pos=f"{mount[0]:.6f} {mount[1]:.6f} {mount[2]:.6f}",
        quat=f"{qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f}",
        fovy=f"{fovy_deg:.4f}",
    )


def _add_bones(root: ET.Element, model: mujoco.MjModel) -> int:
    """Add one visual capsule per link, so the stripped model is still watchable.

    Each body gets a capsule from its own origin to each child body's origin ("bones"),
    which traces the real kinematic chain. Leaf bodies get a small sphere. These are
    ``contype=0 conaffinity=0`` -- visual only, they must not change the physics that the
    IK/diffusion loop is being measured against.
    """
    children: dict[str, list[str]] = {}
    for bid in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[bid])
        if name and parent:
            children.setdefault(parent, []).append(name)

    added = 0
    for body in root.iter("body"):
        name = body.get("name")
        if not name:
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        kids = children.get(name, [])
        drawn = False
        for kid in kids:
            kid_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, kid)
            offset = np.asarray(model.body_pos[kid_id], dtype=float)
            if float(np.linalg.norm(offset)) < 0.02:
                continue  # too short to be a useful bone
            ET.SubElement(
                body,
                "geom",
                name=f"bone_{name}_{kid}",
                type="capsule",
                fromto=f"0 0 0 {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}",
                size="0.022",
                rgba="0.62 0.66 0.70 1",
                contype="0",
                conaffinity="0",
                group="1",
                mass="0",
            )
            added += 1
            drawn = True
        if not drawn:
            ET.SubElement(
                body,
                "geom",
                name=f"tip_{name}",
                type="sphere",
                size="0.026",
                rgba="0.62 0.66 0.70 1",
                contype="0",
                conaffinity="0",
                group="1",
                mass="0",
            )
            added += 1
    return added


def _add_gravcomp(root: ET.Element) -> int:
    """Turn on MuJoCo per-body gravity compensation for every link.

    WHY: the real G1 arm_sdk applies a gravity feedforward term, so the hardware holds an
    IK pose with only ~1.3 mm of tracking error at kp=80 (measured 2026-07-30, RUN.md
    "track=1.3mm"). A bare position servo at kp=80 in sim droops ~21 mm under gravity,
    which would show up as a fake steady-state offset and dominate the diffusion loop's
    4 mm convergence threshold. gravcomp reproduces the feedforward instead of inflating
    kp, which would have made the sim stiffer than the robot in the transient too.
    """
    count = 0
    for body in root.iter("body"):
        if body.get("name"):
            body.set("gravcomp", "1")
            count += 1
    return count


def _add_actuators(root: ET.Element, joints: list[str]) -> None:
    """One position servo per joint: arms at the hardware's kp/kd, posture joints stiff."""
    actuator = ET.SubElement(root, "actuator")
    for name in joints:
        is_arm = "shoulder" in name or "elbow" in name or "wrist" in name
        kp, kd = (KP_ARM, KD_ARM) if is_arm else (KP_POST, KD_POST)
        ET.SubElement(
            actuator,
            "position",
            name=f"act_{name}",
            joint=name,
            kp=f"{kp:g}",
            kv=f"{kd:g}",
        )


def _add_scene_furniture(root: ET.Element, floor_z: float, okra_world: np.ndarray) -> None:
    """Floor, lights, sky, and the okra (green pod on a thin stem) as a mocap body.

    The okra is ``mocap="true"`` so a test can teleport it without touching qpos indices --
    useful for re-running a reach at several target positions.
    """
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.55 0.66 0.78",
        rgb2="0.10 0.12 0.16",
        width="256",
        height="256",
    )
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1="0.22 0.24 0.26",
        rgb2="0.30 0.32 0.35",
        width="300",
        height="300",
    )
    ET.SubElement(
        asset, "material", name="grid_mat", texture="grid", texrepeat="8 8", reflectance="0.05"
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("generated MJCF has no worldbody")

    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        pos=f"0 0 {floor_z:.6f}",
        size="3 3 0.05",
        material="grid_mat",
    )
    ET.SubElement(worldbody, "light", pos="0.8 0.4 2.4", dir="-0.3 -0.15 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(worldbody, "light", pos="-0.8 -0.8 2.0", dir="0.3 0.3 -1", diffuse="0.35 0.35 0.35")

    # Spectator view: front-right of the robot, looking back at the reach volume.
    ET.SubElement(
        worldbody,
        "camera",
        name="spectator",
        pos=f"{okra_world[0] + 0.85:.4f} {okra_world[1] - 0.75:.4f} {okra_world[2] + 0.42:.4f}",
        xyaxes="0.66 0.75 0 -0.28 0.25 0.93",
        fovy="45",
    )

    okra = ET.SubElement(
        worldbody,
        "body",
        name="okra",
        pos=f"{okra_world[0]:.6f} {okra_world[1]:.6f} {okra_world[2]:.6f}",
        mocap="true",
    )
    # The pod. Capsule along the body's local Z, i.e. hanging vertically.
    ET.SubElement(
        okra,
        "geom",
        name="okra_pod",
        type="capsule",
        size=f"{OKRA_RADIUS:.4f} {OKRA_HALF_LEN:.4f}",
        rgba="0.20 0.62 0.16 1",
        contype="0",
        conaffinity="0",
    )
    # Stem going up out of frame, so the chest camera sees a plant-like object, not a
    # floating pill (the click target is the pod, not the stem).
    ET.SubElement(
        okra,
        "geom",
        name="okra_stem",
        type="capsule",
        fromto=f"0 0 {OKRA_HALF_LEN:.4f} 0 0 {OKRA_HALF_LEN + 0.22:.4f}",
        size="0.005",
        rgba="0.30 0.45 0.18 1",
        contype="0",
        conaffinity="0",
    )


def _add_options(root: ET.Element, joints: list[str]) -> None:
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    map_el = visual.find("map")
    if map_el is None:
        map_el = ET.SubElement(visual, "map")
    # znear matters for the wrist camera: the gripper is centimetres from the lens.
    map_el.set("znear", "0.01")
    map_el.set("zfar", "50")

    keyframe = ET.SubElement(root, "keyframe")
    qpos = " ".join(f"{v:.6f}" for v in HOME_Q)
    ctrl = " ".join(f"{v:.6f}" for v in HOME_Q)
    ET.SubElement(keyframe, "key", name="home", qpos=qpos, ctrl=ctrl)


def _check_chest_fov(res: tuple[int, int], fovy_deg: float) -> tuple[float, float, float]:
    """Assert the okra sits well inside the chest camera's frustum, and >= the ZED min range.

    Done here, at build time, in closed form -- rather than left to be discovered as "the
    click did nothing" at run time. Returns (horizontal deg, vertical deg, distance m) of
    the okra relative to the chest camera's optical axis.
    """
    cam_p = np.asarray(CHEST_MOUNT[:3], dtype=float)
    pitch = float(CHEST_MOUNT[4])
    v = np.asarray(OKRA_IN_TORSO, dtype=float) - cam_p
    dist = float(np.linalg.norm(v))

    # Project onto the pitched camera axes. With positive pitch = nose down, the optical
    # axis in body coords is R_y(pitch) @ x_hat = [cos p, 0, -sin p] and the camera's up is
    # R_y(pitch) @ z_hat = [sin p, 0, cos p]; the components are just those dot products.
    cp, sp = np.cos(pitch), np.sin(pitch)
    fwd = cp * v[0] - sp * v[2]          # along the optical axis
    up = sp * v[0] + cp * v[2]           # + = above the axis
    horiz = np.degrees(np.arctan2(-v[1], fwd))   # + = to the camera's right
    vert = np.degrees(np.arctan2(up, fwd))

    width, height = res
    half_v = fovy_deg / 2.0
    half_h = np.degrees(np.arctan(np.tan(np.radians(half_v)) * width / height))

    problems = []
    if abs(vert) > half_v * FOV_MARGIN:
        problems.append(f"vertical {vert:+.1f} deg exceeds {half_v * FOV_MARGIN:.1f} deg")
    if abs(horiz) > half_h * FOV_MARGIN:
        problems.append(f"horizontal {horiz:+.1f} deg exceeds {half_h * FOV_MARGIN:.1f} deg")
    if dist < 0.35:
        # The real ZED Mini cannot measure closer than ~0.35 m (hardware finding 2026-07-24),
        # so a sim target closer than that would be un-clickable on the actual robot.
        problems.append(f"distance {dist:.3f} m < 0.35 m ZED minimum range")
    if problems:
        raise RuntimeError(
            "okra is not usefully visible to the chest camera: "
            + "; ".join(problems)
            + f"\n  okra(torso)={OKRA_IN_TORSO} cam(torso)={CHEST_MOUNT[:3]} pitch={pitch} rad"
            + f"\n  retune {SIM_OKRA_ENV} and/or the pitch in {SIM_ZED_MOUNT_ENV}"
        )
    return float(horiz), float(vert), dist


def build(out_path: Path = _OUT, verbose: bool = True) -> Path:
    horiz, vert, cam_dist = _check_chest_fov(CHEST_RES, CHEST_FOVY_DEG)

    urdf_root = ET.parse(_URDF).getroot()
    dropped = _strip_meshes(urdf_root)
    mjcf_text = _urdf_to_mjcf(ET.tostring(urdf_root, encoding="unicode"))
    root = ET.fromstring(mjcf_text)
    root.set("model", "g1_okra_sim")

    # A first pass over the floating-base model tells us how far the feet sit below the
    # pelvis at the home posture, which fixes both the weld height and the floor.
    probe_model = mujoco.MjModel.from_xml_string(mjcf_text)
    probe_data = mujoco.MjData(probe_model)
    probe_data.qpos[:3] = [0.0, 0.0, 0.0]
    probe_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    probe_data.qpos[7:] = HOME_Q
    mujoco.mj_forward(probe_model, probe_data)
    ankle_id = mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    foot_drop = float(probe_data.xpos[ankle_id][2])          # negative: below the pelvis
    pelvis_z = -foot_drop + 0.03                             # +3 cm for the sole thickness

    joints = _joint_names_in_order(root)
    if len(joints) != 29:
        raise RuntimeError(f"expected 29 hinge joints, generator found {len(joints)}: {joints}")

    _weld_base(root, pelvis_z)
    _add_bones(root, probe_model)
    gravcomp_n = _add_gravcomp(root)
    _add_actuators(root, joints)
    _add_options(root, joints)

    # torso_link placement is constant once the base is welded, so the okra's world pos is
    # just torso_world + OKRA_IN_TORSO. Recomputed from the welded model below.
    torso_id = mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    torso_world = np.asarray(probe_data.xpos[torso_id], dtype=float) + np.array([0.0, 0.0, pelvis_z])
    okra_world = torso_world + np.asarray(OKRA_IN_TORSO, dtype=float)

    _add_scene_furniture(root, floor_z=0.0, okra_world=okra_world)
    _add_camera(root, "torso_link", "chest_cam", CHEST_MOUNT, CHEST_FOVY_DEG)
    _add_camera(root, "right_wrist_yaw_link", "wrist_cam", WRIST_MOUNT, WRIST_FOVY_DEG)

    ET.indent(root, space="  ")
    out_path.write_text(ET.tostring(root, encoding="unicode"))

    # Fail loudly here rather than at run time if the result does not load.
    check = mujoco.MjModel.from_xml_path(str(out_path))
    if check.nq != 29:
        raise RuntimeError(f"welded model nq={check.nq}, expected 29 (base weld failed?)")
    if check.nu != 29:
        raise RuntimeError(f"welded model nu={check.nu}, expected 29 actuators")
    for cam in ("chest_cam", "wrist_cam", "spectator"):
        if mujoco.mj_name2id(check, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
            raise RuntimeError(f"camera {cam!r} missing from generated scene")

    if verbose:
        print(f"wrote {out_path}")
        print(f"  mesh geoms dropped : {dropped}")
        print(f"  gravcomp bodies    : {gravcomp_n}")
        print(f"  nq/nv/nu           : {check.nq}/{check.nv}/{check.nu}")
        print(f"  pelvis weld z      : {pelvis_z:.4f} m (foot drop {foot_drop:.4f})")
        print(f"  torso_link world   : {np.round(torso_world, 4).tolist()}")
        print(f"  okra world / torso : {np.round(okra_world, 4).tolist()} / {OKRA_IN_TORSO}")
        print(f"  chest mount        : {CHEST_MOUNT}  (env {SIM_ZED_MOUNT_ENV})")
        print(
            f"  okra in chest view : h={horiz:+.1f} deg v={vert:+.1f} deg "
            f"dist={cam_dist:.3f} m  (fovy {CHEST_FOVY_DEG} deg)"
        )
        print(f"  joints             : {joints[:3]} ... {joints[-3:]}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_OUT), help="output MJCF path")
    args = ap.parse_args()
    build(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
