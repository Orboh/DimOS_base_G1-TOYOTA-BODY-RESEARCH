# Copyright 2025-2026 Dimensional Inc.
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

"""G1 right-arm 7-DOF inverse-kinematics model (reduced from the full URDF).

The full ``g1.urdf`` is a 29-DOF humanoid with a floating base. For an
ACT-independent IK reach we only drive the RIGHT arm (the Dex1 side). This module
builds a 7-DOF *reduced* Pinocchio model by locking every joint except the seven
right-arm joints, and wraps it in the existing :class:`PinocchioIK` solver.

Verified empirically against ``g1.urdf`` (pinocchio 4.0.0, R0c on 2026-06-16):
- ``pinocchio.buildReducedModel(full, lock_all_but_right_arm, neutral)`` yields
  ``reduced.nq == 7`` and preserves the canonical joint order
  (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw), matching
  ``make_humanoid_joints("g1")[22:29]``.
- ``ee_joint_id`` for ``right_wrist_yaw_joint`` is 7 in the reduced model.
- Pinocchio FK (``oMi[ee]``) is expressed in the model ROOT (world/universe)
  frame, NOT torso. At the neutral (locked) configuration ``torso_link`` sits at
  a *constant* placement ``oMtorso`` in that root frame, so torso<->root is a
  fixed rigid transform we capture once here. IK targets are authored in
  ``torso_link`` and converted to the solver's root frame via ``oMtorso``.

This module deliberately does NOT assume the joint order — it reads
``reduced.names`` and exposes the actual order so the caller can verify/permute.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pinocchio

from dimos.control.components import make_humanoid_joints
from dimos.manipulation.planning.kinematics.pinocchio_ik import PinocchioIK, PinocchioIKConfig
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = setup_logger()

# Default URDF ships next to the G1 config (same idiom as dimos/robot/unitree/g1/config.py:34).
DEFAULT_URDF = Path(__file__).resolve().parent.parent / "g1.urdf"

# The seven right-arm joints, in canonical dimos 29-DOF order (== make_humanoid_joints("g1")[22:29]).
RIGHT_ARM_JOINTS: list[str] = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
EE_JOINT_NAME = "right_wrist_yaw_joint"
TORSO_FRAME = "torso_link"
# right_hand_palm_joint is a fixed offset from the wrist (g1.urdf:976); optional EE post-offset.
PALM_OFFSET_FROM_WRIST = np.array([0.0415, -0.003, 0.0])


@dataclass
class G1RightArmIK:
    """A right-arm IK handle: the solver plus the frame/limit metadata it needs.

    Attributes:
        ik: PinocchioIK over the reduced 7-DOF model (solve/FK operate in ROOT frame).
        ee_joint_id: EE joint id in the reduced model (right_wrist_yaw_joint).
        joint_names: actual reduced-model joint order (verify == RIGHT_ARM_JOINTS).
        torso_in_root: constant SE3 placement (pinocchio oMtorso) of torso_link in
            the solver's ROOT frame.
        lower / upper: per-joint position limits (radians), in joint_names order.
    """

    ik: PinocchioIK
    ee_joint_id: int
    joint_names: list[str]
    torso_in_root: Any  # pinocchio.SE3 (oMtorso)
    lower: NDArray[np.floating[Any]]
    upper: NDArray[np.floating[Any]]

    @property
    def order_matches_canonical(self) -> bool:
        return self.joint_names == RIGHT_ARM_JOINTS

    def torso_to_root(self, p_torso: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Map a 3D point from torso_link frame into the solver ROOT frame."""
        return np.asarray(self.torso_in_root.act(np.asarray(p_torso, dtype=np.float64)))

    def root_to_torso_pose(self, oMx: Any) -> Any:
        """Map an SE3 from the solver ROOT frame into torso_link frame."""
        return self.torso_in_root.actInv(oMx)

    def fk_root(self, q: NDArray[np.floating[Any]]) -> Any:
        """EE pose (SE3) in the solver ROOT frame for joint config q."""
        return self.ik.forward_kinematics(np.asarray(q, dtype=np.float64))

    def clamp_ok(self, q: NDArray[np.floating[Any]]) -> bool:
        """True iff every joint of q is within the URDF position limits."""
        q = np.asarray(q, dtype=np.float64).flatten()
        return bool(np.all(q >= self.lower - 1e-6) and np.all(q <= self.upper + 1e-6))


def load_g1_right_arm_ik(
    urdf_path: str | Path = DEFAULT_URDF,
    ik_config: PinocchioIKConfig | None = None,
) -> G1RightArmIK:
    """Build the 7-DOF right-arm reduced model and wrap it in PinocchioIK.

    Raises:
        FileNotFoundError: if the URDF is missing.
        RuntimeError: if reduction does not produce exactly the 7 right-arm joints.
    """
    path = Path(str(urdf_path))
    if not path.exists():
        raise FileNotFoundError(f"G1 URDF not found: {path}")

    full = pinocchio.buildModelFromUrdf(str(path))

    missing = [j for j in RIGHT_ARM_JOINTS if not full.existJointName(j)]
    if missing:
        raise RuntimeError(f"g1.urdf missing right-arm joints: {missing}")

    keep_ids = {full.getJointId(j) for j in RIGHT_ARM_JOINTS}
    lock_ids = [jid for jid in range(1, full.njoints) if jid not in keep_ids]
    q_ref = pinocchio.neutral(full)

    reduced = pinocchio.buildReducedModel(full, lock_ids, q_ref)
    if reduced.nq != len(RIGHT_ARM_JOINTS):
        raise RuntimeError(
            f"reduced model nq={reduced.nq}, expected {len(RIGHT_ARM_JOINTS)}; "
            f"names={list(reduced.names)}"
        )

    # Actual joint order pinocchio produced (jid 0 is 'universe').
    joint_names = [n for n in reduced.names if n in RIGHT_ARM_JOINTS]
    if joint_names != RIGHT_ARM_JOINTS:
        # Not fatal: the caller can permute. But warn loudly — every downstream
        # index (warm-start, q_sol, 14-vec) assumes this order.
        logger.warning(
            f"reduced right-arm order {joint_names} != canonical {RIGHT_ARM_JOINTS}; "
            "caller MUST permute warm-start and q_sol to match."
        )

    data = reduced.createData()
    ee_joint_id = reduced.getJointId(EE_JOINT_NAME)

    # Capture the constant torso_link placement in the ROOT frame at neutral.
    qn = pinocchio.neutral(reduced)
    pinocchio.forwardKinematics(reduced, data, qn)
    pinocchio.updateFramePlacements(reduced, data)
    if not reduced.existFrame(TORSO_FRAME):
        raise RuntimeError(f"reduced model has no frame {TORSO_FRAME!r}")
    torso_in_root = data.oMf[reduced.getFrameId(TORSO_FRAME)].copy()

    if ik_config is None:
        # Reach-to-okra is a POSITION-only task: any approach orientation is acceptable.
        # A full 6-DOF solve over-constrains the 7-DOF arm and forces large wrist
        # reconfigurations / non-convergence. Solve position-only by default.
        ik_config = PinocchioIKConfig(position_only=True)
    ik = PinocchioIK(reduced, data, ee_joint_id, ik_config)

    return G1RightArmIK(
        ik=ik,
        ee_joint_id=ee_joint_id,
        joint_names=joint_names,
        torso_in_root=torso_in_root,
        lower=np.asarray(reduced.lowerPositionLimit, dtype=np.float64).copy(),
        upper=np.asarray(reduced.upperPositionLimit, dtype=np.float64).copy(),
    )


def fk_sanity_check(arm: G1RightArmIK | None = None) -> None:
    """Print reduced-model facts for R0c verification (run in the laptop venv).

    Compares the canonical right-arm slice of make_humanoid_joints("g1") against
    the reduced model order, prints FK at neutral in both ROOT and torso frames.
    """
    arm = arm or load_g1_right_arm_ik()
    canonical = make_humanoid_joints("g1")[22:29]

    def _norm(n: str) -> str:
        # make_humanoid_joints uses 'g1/right_shoulder_pitch'; the URDF uses
        # 'right_shoulder_pitch_joint'. Compare the bare joint identity.
        return n.split("/")[-1].removesuffix("_joint")

    order_ok = [_norm(a) for a in arm.joint_names] == [_norm(c) for c in canonical]
    print(f"reduced nq            : {arm.ik.nq}")
    print(f"reduced joint order   : {arm.joint_names}")
    print(f"canonical [22:29]     : {list(canonical)}")
    print(f"order matches (norm)  : {order_ok}")
    print(f"ee_joint_id           : {arm.ee_joint_id}")
    q0 = np.zeros(arm.ik.nq)
    oMee = arm.fk_root(q0)
    print(f"FK(neutral) ROOT  xyz : {np.asarray(oMee.translation)}")
    print(f"FK(neutral) TORSO xyz : {np.asarray(arm.root_to_torso_pose(oMee).translation)}")
    print(f"torso in ROOT     xyz : {np.asarray(arm.torso_in_root.translation)}")
    print(f"lower limits          : {arm.lower}")
    print(f"upper limits          : {arm.upper}")


if __name__ == "__main__":
    fk_sanity_check()
