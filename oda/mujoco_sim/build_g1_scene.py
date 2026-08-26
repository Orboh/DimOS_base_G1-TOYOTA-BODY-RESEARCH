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

1. **Visual meshes, simplified collision.** The repository does not version the G1 STL
   assets, but this workstation has a complete Unitree G1 mesh bundle. At build time the
   visual mesh paths are resolved from that bundle (or ``G1_VISUAL_MESH_DIR``); mesh
   *collisions* are still stripped. Link inertials are explicit in the URDF and are kept,
   while the reviewed primitive right-arm/basket collision geometry remains authoritative.
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
_MESH_DIR_ENV = "G1_VISUAL_MESH_DIR"
# These are deliberately outside this repository: STL assets are large third-party files
# and are not checked in here. Set G1_VISUAL_MESH_DIR on a different machine. The first
# entry is the verified bundle on this workstation.
_MESH_DIR_CANDIDATES = (
    Path("/home/techshare/unitree_lerobot/unitree_lerobot/eval_robot/assets/g1"),
    Path("/home/techshare/drl_kit/mujoco_ws/ts_mujoco-main/unitree_robots/g1"),
)

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

# Right-arm collision capsules (20260825 review decision B: minimum viable set -- only the
# parts that can plausibly reach the basket get real collision; left arm and legs keep the
# visual-only STL geometry). This first visual-mesh stage intentionally keeps collision
# simple, using radii already present and reviewed elsewhere in this file/URDF rather than
# inventing new measurements: 0.03 m is the shoulder collision cylinder radius in g1.urdf
# (right_shoulder_roll_link's <collision>), 0.026 m is the wrist/hand "tip" sphere radius
# used by the right-hand stand-in below.
ARM_COLLISION_RADIUS_LIMB = 0.03
ARM_COLLISION_RADIUS_WRIST = 0.026
# The visual mesh was measured directly from the checked-in G1 asset bundle:
# right_rubber_hand.STL spans (x, y, z) = (131.828, 66.558, 106.479) mm in the
# hand-link frame.  The fixed hand-link origin is (41.5, -3, 0) mm forward of
# right_wrist_yaw_link (see g1.urdf).  The old 26 mm wrist sphere therefore did
# not cover the palm or fingers at all.
_HAND_PALM_CENTER = (0.079, 0.009, 0.008)
_HAND_PALM_HALF_EXTENTS = (0.031, 0.034, 0.050)
_HAND_FINGER_CAPSULES = [
    # (centre y/z in wrist frame): four forward capsules cover the finger fan.
    (-0.016, -0.018),
    (0.000, -0.003),
    (0.017, 0.012),
    (0.034, 0.020),
]
_HAND_FINGER_X_FROMTO = (0.100, 0.174)
HAND_FINGER_RADIUS_M = 0.012

# CAD-derived outer envelope of the 3D-printed UMI base mounted on the Dex1-1 hand.
#
# Source: ``uim_base_dex1-1_No_cube.f3d`` supplied on 2026-08-26.  Its Fusion OGS
# display mesh has a bounding span of 172.0 x 91.4 x 58.2 mm.  This model does not
# contain the GoPro body (``No_cube``), so the camera remains a separate keepout below.
# The CAD's long axis is mounted along the G1 hand's +x (wrist -> fingertip) direction;
# the base begins at the wrist origin and sits on top of the hand.  The mount origin is
# photo-derived pending physical measurement, but the collision size is CAD-derived.
UMI_BASE_FULL_EXTENTS = (0.1720, 0.0914, 0.0582)  # x, y, z [m]
UMI_BASE_POS = (0.0860, 0.0000, 0.0291)  # centre in right_wrist_yaw_link [m]

# UMI uses the wrist-mounted GoPro HERO9 + Media Mod rig.  GoPro's specified
# HERO9 body envelope is W x H x D = 71.0 x 55.0 x 33.6 mm.  In the G1 body
# convention these map to y, z, x respectively.  We add 5 mm on every face to
# keep Media Mod frame, cable clearance, and the unmeasured wrist bracket inside
# the collision keepout; it is intentionally conservative until the rig is
# physically measured.  WRIST_MOUNT is the same hardware transform used by the
# simulated UMI camera below.
UMI_GOPRO_BODY_FULL_EXTENTS = (0.0436, 0.0810, 0.0650)  # x, y, z [m]

# The visual torso mesh spans x=[-66.6, 83.6], y=[-107.7, 107.7], and
# z=[-8.9, 311.6] mm in ``torso_link``.  Mesh contacts remain intentionally
# disabled in this phase, so retain a conservative primitive envelope for the
# part of the body that the moving right arm and wrist-mounted UMI can reach.
# A box is used here (rather than an ellipsoid) on purpose: an arm hidden
# inside the visible chest must register as a collision, even near a corner.
TORSO_CORE_CENTER = (0.0085, 0.0000, 0.1510)
TORSO_CORE_HALF_EXTENTS = (0.0755, 0.1080, 0.1605)
# (parent_body, child_body, radius): one capsule per segment, spanning parent origin to
# child origin for this specific chain.
_ARM_COLLISION_SEGMENTS = [
    ("right_shoulder_yaw_link", "right_elbow_link", ARM_COLLISION_RADIUS_LIMB),        # upper arm
    ("right_elbow_link", "right_wrist_roll_link", ARM_COLLISION_RADIUS_LIMB),          # forearm
    ("right_wrist_roll_link", "right_wrist_pitch_link", ARM_COLLISION_RADIUS_WRIST),
    ("right_wrist_pitch_link", "right_wrist_yaw_link", ARM_COLLISION_RADIUS_WRIST),
]

# MuJoCo's default `filterparent` flag drops contacts between a geom's own body and its
# direct parent, which handles most joints in the chain above for free. It does NOT cover
# "skip one" pairs, and the wrist cluster is packed too tightly for that to be free of
# false positives: e.g. the forearm capsule's far end sits exactly at right_wrist_roll_link
# (radius 0.03), 0.038 m short of where the next-but-one wrist_pitch->wrist_yaw capsule
# starts (radius 0.026) -- 0.056 m of combined radius over a 0.038 m gap, so they overlap
# in *every* pose, home included. This is an artifact of approximating a compact multi-DOF
# wrist as a chain of capsules, not a real self-collision; verified via _check_home_contacts
# (2026-08-25) and excluded explicitly rather than shrinking radii, which would just move
# the same problem to a different pair of segments.
_ARM_COLLISION_EXCLUDE_PAIRS = [
    # The upper-arm capsule starts at the shoulder, immediately beside the
    # torso shell.  This neighbouring pair is an intentional mechanical
    # adjacency, not an arm-through-chest event.  All elbow, wrist, hand and
    # UMI bodies remain collidable with the torso core.
    ("torso_link", "right_shoulder_yaw_link"),
    ("right_elbow_link", "right_wrist_pitch_link"),   # forearm vs wrist_pitch->wrist_yaw
    ("right_wrist_roll_link", "right_wrist_yaw_link"),  # wrist_roll->wrist_pitch vs hand
]

# 20260825 decision (Ueda + Yokote): stop chasing the basket's real-world placement error
# with more measurement passes -- accept the ~+/-10mm photo-based z uncertainty documented
# in g1.urdf's basket_joint comment (and the unmeasured x/y mounting slop on top of it), and
# instead make the *simulated* box bigger than the real cardboard box by this margin on
# every face. A too-big keepout volume in sim is a false positive the arm just avoids a
# little early; a too-small one is a real collision the sim never sees. Only the 5 named
# basket collision plates (basket_back/front/bottom/left/right, see g1.urdf) are padded --
# the visual plates keep the true dimensions so renders still look like the actual box.
BASKET_COLLISION_MARGIN_M = 0.015
_BASKET_COLLISION_NAMES = [
    "basket_back", "basket_front", "basket_bottom", "basket_left", "basket_right",
]


def _resolve_visual_mesh_dir(root: ET.Element, requested: Path | None = None) -> Path:
    """Return a directory containing every mesh named by the URDF's visual elements.

    The builder writes absolute paths into its generated, ignored MJCF so a caller does
    not need to copy or symlink third-party STL files into this development repository.
    A missing bundle is an actionable build error, never a silent fallback to a stick
    figure: visual review must show the physical G1 form factor.
    """
    mesh_names = {
        mesh.get("filename", "")
        for visual in root.findall(".//visual")
        for mesh in visual.findall(".//mesh")
    }
    candidates = ([requested] if requested is not None else [])
    env_dir = os.getenv(_MESH_DIR_ENV)
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(_MESH_DIR_CANDIDATES)
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.expanduser().resolve()
        if all((candidate / mesh).is_file() for mesh in mesh_names):
            return candidate
    example = sorted(mesh_names)[0] if mesh_names else "meshes/<link>.STL"
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise RuntimeError(
        "G1 visual mesh bundle is incomplete or unavailable. "
        f"Set {_MESH_DIR_ENV} to the directory containing {example}. Searched: {searched}"
    )


def _prepare_mesh_geometry(root: ET.Element, mesh_dir: Path) -> tuple[int, int]:
    """Keep visual STLs (rewritten as absolute paths) and drop mesh collision geoms.

    Returns ``(visual_meshes_kept, collision_meshes_dropped)``. This intentionally
    separates appearance from contact: Phase 4 keeps its reviewed primitive collision
    model until a measured mesh-collision migration is explicitly validated.
    """
    compiler = root.find("./mujoco/compiler")
    if compiler is None:
        raise RuntimeError("G1 URDF has no <mujoco><compiler> element for mesh resolution")
    # MuJoCo's URDF importer gives <compiler meshdir> precedence over an absolute
    # ``filename`` attribute. Point it at the actual STL directory and retain only the
    # filename, rather than relying on that importer-specific precedence rule.
    compiler.set("meshdir", str((mesh_dir / "meshes").resolve()))

    visual_kept = 0
    collision_dropped = 0
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            for mesh in visual.findall(".//mesh"):
                filename = mesh.get("filename")
                if not filename:
                    raise RuntimeError("visual mesh without a filename")
                path = (mesh_dir / filename).resolve()
                if not path.is_file():
                    raise RuntimeError(f"visual mesh missing: {path}")
                mesh.set("filename", Path(filename).name)
                visual_kept += 1
        for collision in list(link.findall("collision")):
            if collision.find(".//mesh") is not None:
                link.remove(collision)
                collision_dropped += 1
    return visual_kept, collision_dropped


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


def _add_arm_collision(root: ET.Element, model: mujoco.MjModel) -> int:
    """Add real (collidable) capsules along the right arm, per 20260825 decision B.

    The display uses real STL meshes, but their collision geometry is intentionally off in
    this stage. These capsules make the basket enforceable without treating display meshes
    as validated contact geometry.
    ``filterparent`` (MuJoCo's default) suppresses the parent/child false positives this
    would otherwise create against the existing shoulder collision cylinders.
    """
    added = 0
    for parent_name, child_name, radius in _ARM_COLLISION_SEGMENTS:
        body = _find_body(root, parent_name)
        child_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, child_name)
        if child_id < 0:
            raise RuntimeError(f"arm collision segment references unknown body {child_name!r}")
        offset = np.asarray(model.body_pos[child_id], dtype=float)
        ET.SubElement(
            body,
            "geom",
            name=f"armcoll_{parent_name}_{child_name}",
            type="capsule",
            fromto=f"0 0 0 {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}",
            size=f"{radius:.4f}",
            contype="2",
            conaffinity="1",
            rgba="0.85 0.25 0.25 0.5",
        )
        added += 1

    # Keep the wrist sphere for the compact wrist-yaw housing, then cover the hand mesh
    # beyond it with a palm box and individual finger capsules.  These shapes are on the
    # wrist body because right_hand_palm_joint is fixed (see g1.urdf).
    wrist_body = _find_body(root, "right_wrist_yaw_link")
    ET.SubElement(
        wrist_body,
        "geom",
        name="armcoll_right_wrist_housing",
        type="sphere",
        size=f"{ARM_COLLISION_RADIUS_WRIST:.4f}",
        contype="2",
        conaffinity="1",
        rgba="0.85 0.25 0.25 0.5",
    )
    added += 1

    ET.SubElement(
        wrist_body,
        "geom",
        name="armcoll_right_hand_palm",
        type="box",
        pos=" ".join(f"{v:.6f}" for v in _HAND_PALM_CENTER),
        size=" ".join(f"{v:.6f}" for v in _HAND_PALM_HALF_EXTENTS),
        contype="2",
        conaffinity="1",
        rgba="0.85 0.25 0.25 0.5",
    )
    added += 1
    for i, (y, z) in enumerate(_HAND_FINGER_CAPSULES):
        x0, x1 = _HAND_FINGER_X_FROMTO
        ET.SubElement(
            wrist_body,
            "geom",
            name=f"armcoll_right_hand_finger_{i + 1}",
            type="capsule",
            fromto=f"{x0:.6f} {y:.6f} {z:.6f} {x1:.6f} {y:.6f} {z:.6f}",
            size=f"{HAND_FINGER_RADIUS_M:.4f}",
            contype="2",
            conaffinity="1",
            rgba="0.85 0.25 0.25 0.5",
        )
        added += 1

    # The CAD-derived 3D-printed UMI base is a physical keepout.  A single bounding box
    # is deliberate at this stage: it covers the base plate, its upright brackets, and
    # the unmeshed mirror mounting tabs, without treating the visual mesh itself as a
    # validated MuJoCo collision mesh.
    base_x, base_y, base_z = UMI_BASE_FULL_EXTENTS
    ET.SubElement(
        wrist_body,
        "geom",
        name="armcoll_right_umi_base",
        type="box",
        pos=" ".join(f"{v:.6f}" for v in UMI_BASE_POS),
        size=f"{base_x / 2:.6f} {base_y / 2:.6f} {base_z / 2:.6f}",
        contype="2",
        conaffinity="1",
        rgba="0.98 0.50 0.08 0.45",
    )
    added += 1

    # The GoPro/Media Mod body is a separate physical keepout.  Its visual mesh is not
    # known or validated for contact.
    full_x, full_y, full_z = UMI_GOPRO_BODY_FULL_EXTENTS
    ET.SubElement(
        wrist_body,
        "geom",
        name="armcoll_right_umi_gopro",
        type="box",
        pos=" ".join(f"{v:.6f}" for v in WRIST_MOUNT[:3]),
        euler=" ".join(f"{v:.6f}" for v in WRIST_MOUNT[3:]),
        size=f"{full_x / 2:.6f} {full_y / 2:.6f} {full_z / 2:.6f}",
        contype="2",
        conaffinity="1",
        rgba="0.10 0.10 0.10 0.8",
    )
    added += 1
    return added


def _add_torso_collision(root: ET.Element) -> int:
    """Add the simplified torso keepout used for right-arm self-collision.

    It is deliberately an independent primitive, not a switch back to
    unvalidated STL collision meshes.  ``contype=1, conaffinity=2`` makes it
    pair with the right-arm/hand/UMI proxy geoms (2/1) while preserving their
    existing basket contacts.
    """
    torso = _find_body(root, "torso_link")
    ET.SubElement(
        torso,
        "geom",
        name="torso_collision_core",
        type="box",
        pos=" ".join(f"{v:.6f}" for v in TORSO_CORE_CENTER),
        size=" ".join(f"{v:.6f}" for v in TORSO_CORE_HALF_EXTENTS),
        contype="1",
        conaffinity="2",
        rgba="0.30 0.45 0.90 0.18",
    )
    return 1


def _add_contact_excludes(root: ET.Element, pairs: list[tuple[str, str]]) -> int:
    """Write ``<contact><exclude body1=.. body2=..>`` entries, one per (body, body) pair."""
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for body1, body2 in pairs:
        ET.SubElement(
            contact,
            "exclude",
            name=f"exclude_{body1}_{body2}",
            body1=body1,
            body2=body2,
        )
    return len(pairs)


def _pad_basket_collision(root: ET.Element, margin: float) -> int:
    """Grow the basket's collision plates by ``margin`` on every face (see 20260825 note
    on ``BASKET_COLLISION_MARGIN_M`` above). Each plate keeps its center pos and just gets
    ``margin`` added to all three half-extents, so a thin plate becomes a thicker slab that
    also spills outward past its original footprint on every edge -- the basket_link is
    fixed-joint-collapsed into the pelvis body, so this can never create a new same-body
    contact; it only widens what counts as "touching the basket" from other bodies (the
    right arm). Only the named ``<collision>`` geoms are touched -- the paired ``<visual>``
    geoms MuJoCo emits alongside them are untouched, so renders keep the true box size.
    """
    padded = 0
    for geom in root.iter("geom"):
        if geom.get("name") not in _BASKET_COLLISION_NAMES:
            continue
        size = [float(v) + margin for v in geom.get("size", "").split()]
        if len(size) != 3:
            raise RuntimeError(f"basket collision geom {geom.get('name')!r} has size {geom.get('size')!r}, expected 3 values")
        geom.set("size", " ".join(f"{v:.6f}" for v in size))
        padded += 1
    if padded != len(_BASKET_COLLISION_NAMES):
        raise RuntimeError(
            f"expected to pad {len(_BASKET_COLLISION_NAMES)} basket collision geoms, found {padded} "
            f"-- did the names in g1.urdf change?"
        )
    return padded


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


def _check_home_contacts(model: mujoco.MjModel) -> int:
    """Assert the only contact at the home keyframe is feet-on-floor.

    Done here, at build time, for the same reason as ``_check_chest_fov``: with the right
    arm now carrying real collision geoms (``_add_arm_collision``), any overlap baked into
    the geometry itself -- e.g. a capsule radius fat enough to graze the torso visual mesh, or the
    basket sitting closer to the hip than its wall lets on -- shows up as a nonzero-penetration
    contact from frame zero, before any motion is ever commanded. Better to fail the build
    than to discover it as "the diffusion loop diverged" later. Returns the contact count
    excluding floor contacts, i.e. 0 on success.
    """
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        raise RuntimeError("no 'home' keyframe in generated model")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    unexpected = []
    for i in range(data.ncon):
        c = data.contact[i]
        if floor_id in (c.geom1, c.geom2):
            continue  # feet resting on the floor is expected at the home posture
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom#{c.geom1}"
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom#{c.geom2}"
        unexpected.append((n1, n2))
    if unexpected:
        raise RuntimeError(
            "unexpected contact(s) at the home posture (baked-in geometry overlap, not a "
            "real collision event -- shrink a radius, adjust an origin, or add a "
            "<contact><exclude> pair): " + "; ".join(f"{a} vs {b}" for a, b in unexpected)
        )
    return len(unexpected)


def build(
    out_path: Path = _OUT,
    verbose: bool = True,
    mesh_dir: Path | None = None,
) -> Path:
    horiz, vert, cam_dist = _check_chest_fov(CHEST_RES, CHEST_FOVY_DEG)

    urdf_root = ET.parse(_URDF).getroot()
    resolved_mesh_dir = _resolve_visual_mesh_dir(urdf_root, mesh_dir)
    visual_meshes, collision_meshes_dropped = _prepare_mesh_geometry(urdf_root, resolved_mesh_dir)
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
    torso_collision_n = _add_torso_collision(root)
    arm_collision_n = _add_arm_collision(root, probe_model)
    _add_contact_excludes(root, _ARM_COLLISION_EXCLUDE_PAIRS)
    basket_pad_n = _pad_basket_collision(root, BASKET_COLLISION_MARGIN_M)
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
    _check_home_contacts(check)

    if verbose:
        print(f"wrote {out_path}")
        print(f"  visual STL meshes  : {visual_meshes} ({resolved_mesh_dir})")
        print(f"  mesh collisions off: {collision_meshes_dropped} (primitive contacts retained)")
        print(f"  torso collision geoms: {torso_collision_n}")
        print(f"  arm collision geoms: {arm_collision_n}")
        print(f"  basket geoms padded: {basket_pad_n} (+{BASKET_COLLISION_MARGIN_M * 1000:.0f} mm/face)")
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
    ap.add_argument(
        "--mesh-dir",
        type=Path,
        help=f"directory containing meshes/*.STL (defaults to ${_MESH_DIR_ENV} or known local bundles)",
    )
    args = ap.parse_args()
    build(Path(args.out), mesh_dir=args.mesh_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
