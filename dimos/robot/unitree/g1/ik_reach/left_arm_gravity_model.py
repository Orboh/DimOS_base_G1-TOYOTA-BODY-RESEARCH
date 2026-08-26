"""Reduced left-arm G1 model used only to evaluate gravity torque.

The model locks every non-left-arm joint at the URDF neutral configuration.  It
is intentionally separate from the IK solver: callers must opt in explicitly
before any calculated torque can be sent to the robot.
"""

from __future__ import annotations

from pathlib import Path

import pinocchio

from dimos.robot.unitree.g1.ik_reach.right_arm_model import DEFAULT_URDF

# This order is both the URDF order after reduction and motors 15..21.
LEFT_ARM_JOINTS: list[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]


def load_g1_left_arm_gravity_model(urdf_path: str | Path = DEFAULT_URDF) -> pinocchio.Model:
    """Return the 7-DOF left-arm reduced model, validating its joint order."""
    path = Path(str(urdf_path))
    if not path.exists():
        raise FileNotFoundError(f"G1 URDF not found: {path}")

    full = pinocchio.buildModelFromUrdf(str(path))
    missing = [joint for joint in LEFT_ARM_JOINTS if not full.existJointName(joint)]
    if missing:
        raise RuntimeError(f"g1.urdf missing left-arm joints: {missing}")
    keep_ids = {full.getJointId(joint) for joint in LEFT_ARM_JOINTS}
    lock_ids = [joint_id for joint_id in range(1, full.njoints) if joint_id not in keep_ids]
    reduced = pinocchio.buildReducedModel(full, lock_ids, pinocchio.neutral(full))

    names = [name for name in reduced.names if name in LEFT_ARM_JOINTS]
    if reduced.nq != len(LEFT_ARM_JOINTS) or names != LEFT_ARM_JOINTS:
        raise RuntimeError(
            "unexpected reduced left-arm model: "
            f"nq={reduced.nq}, names={names}, expected={LEFT_ARM_JOINTS}"
        )
    return reduced
