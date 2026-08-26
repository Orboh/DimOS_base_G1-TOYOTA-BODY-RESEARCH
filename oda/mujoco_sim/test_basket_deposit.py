#!/usr/bin/env python3
"""Verify a right-arm transfer and release into the fixed abdominal basket.

The G1 URDF ends at the rubber hand: it has no actuated finger joints.  This test
therefore models just the missing *simulation* contract of the gripper with a
MuJoCo weld constraint: while ``cargo_okra`` is held, it is rigidly attached to
the gripper tip; "open gripper" means disabling that weld.  The pod is then a
real free body, so its fall and containment by the five basket collision plates
are evaluated by MuJoCo rather than by a teleport or a visual judgement.

The arm targets are intentionally authored in ``torso_link`` coordinates.  That
makes the deposit pose a constant in the IK convention and keeps it independent
of the pelvis/world placement used by the generated scene.

Run after regenerating the scene:
    MUJOCO_GL=egl .venv/bin/python oda/mujoco_sim/test_basket_deposit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dimos.robot.unitree.g1.ik_reach.right_arm_model import load_g1_right_arm_ik
from oda.mujoco_sim.build_g1_scene import _OUT, BASKET_COLLISION_MARGIN_M
from oda.mujoco_sim.smoke_sim_arm import TIP_OFFSET, SimRig

# The basket is open in +Z. These targets keep the hand/UMI above the rim; the
# virtually held pod is then released and falls vertically into the volume. Values
# are metres in torso_link frame; they deliberately do not depend on world pose.
ENTRY_TORSO = np.array([0.200, -0.025, 0.140])
# The hand/UMI keepout must stay above the basket rim.  The pod is released from
# this point and falls through the open +Z face; the arm never descends into it.
DROP_TORSO = np.array([0.200, -0.025, 0.100])
RETREAT_TORSO = ENTRY_TORSO.copy()

# Position-only IK has a null space: a tip can reach the same point with either
# the elbow/UMI outside the torso or passing through it.  These seeds choose the
# externally-routed, elbow-forward branch: its elbow is about 0.10 m in front of
# the torso origin rather than nearly level with it.  Every interpolated motion
# sample is checked against ``torso_collision_core`` below.  They were obtained
# with the CAD-derived UMI base keepout enabled, not tuned from a visual-only model.
ENTRY_IK_SEED = np.array([-0.518, -0.860, 0.266, -0.353, 0.564, 0.407, 1.477])
DROP_IK_SEED = np.array([-0.581, -1.057, 0.608, -0.280, 0.947, 0.574, 1.159])

CARGO_RADIUS_M = 0.013
CARGO_HALF_LEN_M = 0.045
TRACK_TOL_M = 0.010
# The arm may start the stow from any valid front-side grasp configuration, not only from
# home. Three seconds lets the position actuators settle through that larger joint-space
# displacement while retaining contact checks at every simulation step.
DEPOSIT_MOVE_SECONDS = 3.0
SETTLE_SECONDS = 4.0
# MuJoCo's compliant contact can settle a dense capsule fractionally inside its
# padded contact shell.  Keep the physical 15 mm basket margin intact and allow
# only 1 mm of solver compliance beyond it; this is unrelated to arm/UMI
# self-collision, which remains a strict zero-contact failure.
MAX_BASKET_PENETRATION_M = BASKET_COLLISION_MARGIN_M + 0.001


def _scene_with_free_cargo(scene: Path, *, held: bool) -> str:
    """Add cargo for either the virtual-grasp or released physical state."""
    # Use the scene's home wrist pose for the freejoint initial state.  This
    # avoids a large constraint correction at the first simulation step.
    home = SimRig(scene)
    tip_world = home.tip_world(np.asarray(TIP_OFFSET, dtype=float))

    root = ET.parse(scene).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("scene has no worldbody")
    cargo = ET.SubElement(
        worldbody,
        "body",
        name="cargo_okra",
        pos=" ".join(f"{v:.8f}" for v in tip_world),
    )
    ET.SubElement(cargo, "freejoint", name="cargo_okra_free")
    ET.SubElement(
        cargo,
        "geom",
        name="cargo_okra_pod",
        type="capsule",
        size=f"{CARGO_RADIUS_M} {CARGO_HALF_LEN_M}",
        density="700",
        rgba="0.20 0.62 0.16 1",
        # The pod collides with the basket in *both* the held and released
        # phases. Thus an approach that drags the fruit through a rim or wall
        # is rejected instead of being hidden by the virtual grasp. It does
        # not collide with the hand/UMI proxy: those use contype=2, whereas
        # this pod uses contype=4 and the corresponding masks do not overlap.
        # MuJoCo still needs a newly compiled released model because its
        # free-body state replaces the weld at the exact release pose.
        contype="4",
        conaffinity="1",
    )
    if held:
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        ET.SubElement(
            equality,
            "weld",
            name="virtual_gripper_hold",
            body1="right_wrist_yaw_link",
            body2="cargo_okra",
            relpose=" ".join(f"{v:.8f}" for v in [*TIP_OFFSET, 1.0, 0.0, 0.0, 0.0]),
            solref="0.002 1",
        )
    return ET.tostring(root, encoding="unicode")


def _make_rig(xml: str) -> SimRig:
    """Construct the tiny SimRig wrapper from an in-memory MJCF document."""
    rig = SimRig.__new__(SimRig)
    rig.model = mujoco.MjModel.from_xml_string(xml)
    rig.data = mujoco.MjData(rig.model)
    home_key = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(rig.model, rig.data, home_key)
    rig._torso = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    rig._wrist = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    rig._okra = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_BODY, "okra")
    mujoco.mj_forward(rig.model, rig.data)
    return rig


def _solve(arm, target_torso: np.ndarray, q_seed: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    import pinocchio

    target_root = arm.torso_to_root(target_torso)
    solution, converged, err = arm.ik.solve(
        pinocchio.SE3(np.eye(3), target_root), q_seed
    )
    if not converged or err > 1e-3:
        raise RuntimeError(
            f"IK did not converge for {np.round(target_torso, 4).tolist()}: err={err:.6f}"
        )
    if not arm.clamp_ok(solution):
        raise RuntimeError("IK solution violates a right-arm joint limit")
    return np.asarray(solution, dtype=float)


def _advance(
    rig: SimRig, q_goal: np.ndarray, seconds: float
) -> tuple[float, list[tuple[str, str]], list[tuple[str, str]]]:
    """Ramp the arm command, tracking basket and torso self-collision contacts."""
    q_start = rig.q_right()
    steps = max(1, int(seconds / rig.model.opt.timestep))
    deepest = 0.0
    basket_pairs: list[tuple[str, str]] = []
    torso_pairs: list[tuple[str, str]] = []
    for i in range(steps):
        rig.command_right_arm(q_start + (q_goal - q_start) * ((i + 1) / steps))
        mujoco.mj_step(rig.model, rig.data)
        for j in range(rig.data.ncon):
            contact = rig.data.contact[j]
            a = mujoco.mj_id2name(rig.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or "<unnamed>"
            b = mujoco.mj_id2name(rig.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or "<unnamed>"
            if "basket_" in a or "basket_" in b:
                deepest = min(deepest, float(contact.dist))
                basket_pairs.append((a, b))
            if "torso_collision_core" in (a, b):
                torso_pairs.append((a, b))
    return deepest, basket_pairs, torso_pairs


def _cargo_torso(rig: SimRig, cargo_body: int) -> np.ndarray:
    return rig.to_torso(np.asarray(rig.data.xpos[cargo_body], dtype=float))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=_OUT)
    args = ap.parse_args()
    if not args.scene.exists():
        print(f"scene missing: {args.scene}\nrun: .venv/bin/python oda/mujoco_sim/build_g1_scene.py")
        return 2

    rig = _make_rig(_scene_with_free_cargo(args.scene, held=True))
    cargo_body = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_BODY, "cargo_okra")
    cargo_joint = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_JOINT, "cargo_okra_free")
    cargo_qpos = rig.model.jnt_qposadr[cargo_joint]
    cargo_qvel = rig.model.jnt_dofadr[cargo_joint]

    arm = load_g1_right_arm_ik(gripper_offset_xyz=TIP_OFFSET)
    q_entry = _solve(arm, ENTRY_TORSO, ENTRY_IK_SEED)
    q_drop = _solve(arm, DROP_TORSO, DROP_IK_SEED)
    q_retreat = _solve(arm, RETREAT_TORSO, ENTRY_IK_SEED)

    print("[1] fixed torso-frame targets")
    for label, target, solution in (
        ("entry", ENTRY_TORSO, q_entry),
        ("drop", DROP_TORSO, q_drop),
        ("retreat", RETREAT_TORSO, q_retreat),
    ):
        print(f"    {label:7s} target={np.round(target, 4).tolist()} q={np.round(solution, 4).tolist()}")

    failures: list[str] = []
    print("[2] carry down through the open (+Z) face")
    for label, target, solution in (("entry", ENTRY_TORSO, q_entry), ("drop", DROP_TORSO, q_drop)):
        deepest, pairs, torso_pairs = _advance(rig, solution, seconds=DEPOSIT_MOVE_SECONDS)
        tip_err = float(np.linalg.norm(rig.tip_torso(np.asarray(TIP_OFFSET)) - target))
        arm_basket = [p for p in pairs if not any("cargo_okra" in geom for geom in p)]
        carried_pod_basket = [p for p in pairs if "cargo_okra_pod" in p]
        print(
            f"    {label:7s} tip_err={tip_err * 1000:.2f} mm "
            f"basket_depth={deepest * 1000:.2f} mm"
        )
        if tip_err > TRACK_TOL_M:
            failures.append(f"{label}: tip error {tip_err * 1000:.1f} mm")
        if arm_basket:
            failures.append(f"{label}: arm touched basket ({arm_basket[0]})")
        if carried_pod_basket:
            failures.append(f"{label}: carried okra touched basket ({carried_pod_basket[0]})")
        if torso_pairs:
            failures.append(f"{label}: arm/UMI touched torso ({torso_pairs[0]})")

    print("[3] virtual gripper open -> free fall -> basket containment")
    print(f"    cargo before open (torso)={np.round(_cargo_torso(rig, cargo_body), 4).tolist()}")
    # "Open" is represented by compiling the released state with collision
    # enabled, then copying robot and cargo pose across exactly once.  This is
    # equivalent to removing the virtual weld but preserves MuJoCo's compiled
    # collision-pair table.
    release_qpos = rig.data.qpos.copy()
    # The padded basket has only 80 mm vertical clearance.  An okra therefore
    # enters lengthwise (its capsule axis is torso +X), not upright.  The
    # current right-arm IK is position-only and leaves wrist roll unspecified;
    # represent the final in-hand roll explicitly here.  A hardware gripper
    # model must turn this into an orientation target before Phase 5.
    torso_rot = rig.torso_pose()[1]
    cargo_rot_torso = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    cargo_quat = np.empty(4)
    mujoco.mju_mat2Quat(cargo_quat, (torso_rot @ cargo_rot_torso).reshape(-1))
    release_qpos[cargo_qpos + 3:cargo_qpos + 7] = cargo_quat
    rig = _make_rig(_scene_with_free_cargo(args.scene, held=False))
    cargo_body = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_BODY, "cargo_okra")
    cargo_joint = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_JOINT, "cargo_okra_free")
    cargo_qvel = rig.model.jnt_dofadr[cargo_joint]
    rig.data.qpos[:] = release_qpos
    rig.data.qvel[:] = 0.0
    rig.command_right_arm(q_drop)
    mujoco.mj_forward(rig.model, rig.data)
    release_contacts = [
        (
            mujoco.mj_id2name(rig.model, mujoco.mjtObj.mjOBJ_GEOM, rig.data.contact[i].geom1),
            mujoco.mj_id2name(rig.model, mujoco.mjtObj.mjOBJ_GEOM, rig.data.contact[i].geom2),
            rig.data.contact[i].dist,
        )
        for i in range(rig.data.ncon)
        if "cargo_okra_pod" in {
            mujoco.mj_id2name(rig.model, mujoco.mjtObj.mjOBJ_GEOM, rig.data.contact[i].geom1),
            mujoco.mj_id2name(rig.model, mujoco.mjtObj.mjOBJ_GEOM, rig.data.contact[i].geom2),
        }
    ]
    print(
        f"    cargo pose at open={np.round(release_qpos[cargo_qpos:cargo_qpos + 7], 4).tolist()} "
        f"contacts={release_contacts}"
    )
    deepest, pairs, torso_pairs = _advance(rig, q_retreat, seconds=SETTLE_SECONDS)
    cargo_t = _cargo_torso(rig, cargo_body)
    # The capsule may retain harmless axial spin in MuJoCo after it has come to
    # rest on cardboard.  The placement requirement is translational rest, not
    # zero angular velocity of an idealised low-damping pod.
    cargo_speed = float(np.linalg.norm(rig.data.qvel[cargo_qvel:cargo_qvel + 3]))
    cargo_angular_speed = float(np.linalg.norm(rig.data.qvel[cargo_qvel + 3:cargo_qvel + 6]))
    cargo_basket_contacts = [p for p in pairs if "cargo_okra_pod" in p]
    inside = 0.090 < cargo_t[0] < 0.275 and abs(cargo_t[1]) < 0.040 and -0.165 < cargo_t[2] < -0.080
    print(
        f"    cargo(torso)={np.round(cargo_t, 4).tolist()} linear_speed={cargo_speed:.4f} m/s "
        f"angular_speed={cargo_angular_speed:.4f} rad/s "
        f"basket_contacts={len(cargo_basket_contacts)}"
    )
    if not cargo_basket_contacts:
        failures.append("released cargo never contacted the basket")
    if not inside:
        failures.append(f"released cargo is outside the basket: torso={cargo_t}")
    if cargo_speed > 0.03:
        failures.append(f"released cargo did not settle: speed={cargo_speed:.3f} m/s")
    if torso_pairs:
        failures.append(f"retreat: arm/UMI touched torso ({torso_pairs[0]})")
    if deepest < -MAX_BASKET_PENETRATION_M:
        failures.append(
            f"basket penetration exceeds its {MAX_BASKET_PENETRATION_M * 1000:.0f} mm "
            f"contact tolerance: {deepest * 1000:.1f} mm"
        )

    if failures:
        print("BASKET_DEPOSIT FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("BASKET_DEPOSIT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
