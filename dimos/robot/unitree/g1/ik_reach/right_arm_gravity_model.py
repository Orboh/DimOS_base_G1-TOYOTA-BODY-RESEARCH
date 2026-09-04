"""Reduced right-arm G1 model used only to evaluate gravity torque.

Mirror of :mod:`left_arm_gravity_model` for the RIGHT arm (motors 22..28), added
2026-09-04 so ``IkReachBridge`` reaches can hold a commanded pose against gravity
instead of settling low.

Why a separate model from the IK one: the IK model in
:mod:`dimos.robot.unitree.g1.ik_reach.right_arm_model` carries a ``gripper_tip``
operational frame and is shared with the solver's own data buffers. Evaluating
gravity on it from the control thread would race the solver. This module builds an
independent reduced model with its own data, and callers must opt in explicitly
before any calculated torque can be sent to the robot.

The model locks every non-right-arm joint at the URDF neutral configuration.
"""

from __future__ import annotations

from pathlib import Path

import pinocchio

from dimos.robot.unitree.g1.ik_reach.right_arm_model import DEFAULT_URDF

# This order is both the URDF order after reduction and motors 22..28.
RIGHT_ARM_JOINTS: list[str] = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def load_g1_right_arm_gravity_model(urdf_path: str | Path = DEFAULT_URDF) -> pinocchio.Model:
    """Return the 7-DOF right-arm reduced model, validating its joint order.

    Args:
        urdf_path: URDF to build the gravity model from. For a G1 wearing the
            Dex1-1 gripper pass ``g1_dex1_1_calibrated_550g.urdf`` — the stock
            ``g1.urdf`` models the hand as a single lumped ``right_rubber_hand``
            link, which gets the total mass right but places the centre of mass
            ~3.7 mm short and so under-estimates the shoulder torque by ~11 %.
    """
    path = Path(str(urdf_path))
    if not path.exists():
        raise FileNotFoundError(f"G1 URDF not found: {path}")

    full = pinocchio.buildModelFromUrdf(str(path))
    missing = [joint for joint in RIGHT_ARM_JOINTS if not full.existJointName(joint)]
    if missing:
        raise RuntimeError(f"URDF missing right-arm joints: {missing}")
    keep_ids = {full.getJointId(joint) for joint in RIGHT_ARM_JOINTS}
    lock_ids = [joint_id for joint_id in range(1, full.njoints) if joint_id not in keep_ids]
    reduced = pinocchio.buildReducedModel(full, lock_ids, pinocchio.neutral(full))

    names = [name for name in reduced.names if name in RIGHT_ARM_JOINTS]
    if reduced.nq != len(RIGHT_ARM_JOINTS) or names != RIGHT_ARM_JOINTS:
        raise RuntimeError(
            "unexpected reduced right-arm model: "
            f"nq={reduced.nq}, names={names}, expected={RIGHT_ARM_JOINTS}"
        )
    return reduced
