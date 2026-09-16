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

"""BasketDepositBridge: grasp_done -> right-arm-only IK deposit into the abdominal basket.

F-07 (SS-06 収納), IK-only cut of the "腹部かご" (abdominal basket, ``basket_link``
fixed to ``pelvis`` in ``g1.urdf``) designed by Yokote (see the Obsidian design doc
``Free-space/yokote/20260825/腹部かご搭載_方針書.md`` + 8/25-8/27 work logs). No VLM,
no learned policy: three fixed torso-frame waypoints (entry -> drop -> retreat),
solved with the SAME 7-DOF right-arm IK as :class:`IkReachBridge`
(``right_arm_model.load_g1_right_arm_ik``), then an open-loop gripper release.

Waypoints default to the values verified in MuJoCo (``oda/mujoco_sim/
test_basket_deposit.py``, Phase-4 "top-open basket" revision, 2026-08-26): the
basket opens +Z, so the tip enters/exits through the top rather than sweeping in
from the side. Position-only IK (the reach orientation is held, matching every
other okra blueprint) -- the sim's explicit in-hand okra rotation is a physics-
visualization detail, not an IK target.

⚠️ SAFETY -- this is NOT the exact motion verified on real hardware. Yokote's
2026-08-27 real-robot Phase 5 demo (arm only, no payload) used an ADDITIONAL pair
of safety waypoints not encoded here -- a right-shoulder-roll outward retreat and
a forward elbow flexion BEFORE approaching the basket -- specifically to avoid a
right-leg collision the direct entry/drop path caused on hardware. The sim's
self-collision checking (torso/leg capsules) does not exist on the real robot.
Before a LIVE run with a real payload:
  1. Run DRY-RUN first and read the logged q_sol for every waypoint.
  2. Consider adding the retreat/flex safety stage back in (ask Yokote for the
     exact joint angles) if the direct path looks like it swings toward the leg.
  3. Keep the remote e-stop in hand; watch the first LIVE cycle at reduced speed.

Wiring (mirrors GripperGraspOnReach -> IkReachBridge upstream):
  grasp_done (Bool, from GripperGraspOnReach's settle timer)
    -> BasketDepositBridge solves entry -> drop -> retreat, publishing arm_target
       for each leg and open-loop-waiting for the arm to settle (same timing model
       as IkReachBridge: no motor-state convergence read, no force feedback)
    -> gripper_target (open_q) releases the okra into the basket
    -> deposit_done (Bool)
"""

from __future__ import annotations

import threading
from threading import Thread
import time
from typing import Any

import numpy as np
import pinocchio
from reactivex.disposable import Disposable

from dimos.control.components import make_humanoid_joints
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.manipulation.planning.kinematics.pinocchio_ik import (
    PinocchioIKConfig,
    check_joint_delta,
)
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.robot.unitree.g1.ik_reach.right_arm_model import DEFAULT_URDF, load_g1_right_arm_ik
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_ARM_START = 15
_NUM_ARM = 14
_LEFT_SLICE = slice(15, 22)
_RIGHT_SLICE = slice(22, 29)
_G1_JOINTS = make_humanoid_joints("g1")
_ARM_JOINT_NAMES = _G1_JOINTS[_ARM_START : _ARM_START + _NUM_ARM]
_RIGHT_GRIPPER_JOINT = "g1/right_gripper"  # matches gripper_grasp_on_reach.py

# Torso-frame waypoints [m] for the abdominal (pelvis-mounted) basket, shared with
# the synchronous LangGraph variant (harvest/basket_deposit.py) so both entry
# points agree on where the basket actually is. Source: oda/mujoco_sim/
# test_basket_deposit.py ENTRY_TORSO/DROP_TORSO (top-open basket, 2026-08-26
# revision) -- verified in MuJoCo only, see the module SAFETY note above.
ENTRY_TORSO: list[float] = [0.200, -0.025, 0.140]
DROP_TORSO: list[float] = [0.200, -0.025, 0.100]
RETREAT_TORSO: list[float] = [0.200, -0.025, 0.140]  # == entry
# SS-06 Q_MAX=5.2 is the blade-safe upper clamp; 3.7 matches this repo's proven
# OKRA_OPEN_Q default (gripper_grasp_on_reach.py).
BASKET_OPEN_Q: float = 3.7


class BasketDepositBridgeConfig(ModuleConfig):
    urdf_path: str = str(DEFAULT_URDF)
    # DRY-RUN safe default: log the solved targets, publish nothing.
    log_only: bool = True
    # Gripper-tip offset from the wrist (right_wrist_yaw_joint), WRIST frame [m].
    # Keep this equal to the grasp reach's OKRA_TIP_OFFSET_XYZ -- the IK model must
    # agree on where the tip physically is regardless of which bridge is driving it.
    gripper_offset_xyz: list[float] = [0.1845, -0.003, 0.0]
    # Torso-frame waypoints [m] (torso_link, +X fwd / +Y left / +Z up). See the
    # module-level ENTRY_TORSO/DROP_TORSO/RETREAT_TORSO for provenance -- shared
    # with the synchronous LangGraph variant (harvest/basket_deposit.py).
    entry_xyz: list[float] = ENTRY_TORSO
    drop_xyz: list[float] = DROP_TORSO
    retreat_xyz: list[float] = RETREAT_TORSO
    # Gripper release (raw Dex1 q). See module-level BASKET_OPEN_Q.
    open_q: float = BASKET_OPEN_Q
    # Settle wait after the release before firing deposit_done [s].
    release_settle_s: float = 1.2
    # Safety gates (same shape as IkReachBridgeConfig -- keep both bridges honest
    # about what "reachable" means for this arm/URDF).
    require_converged: bool = True
    max_reach_pos_err_m: float = 0.02  # deposit is tighter than a grasp reach (narrow rim)
    max_joint_delta_deg: float = 90.0
    ws_x: list[float] = [0.05, 0.65]
    ws_y: list[float] = [-0.75, 0.20]
    ws_z: list[float] = [-0.35, 0.85]
    # Open-loop per-leg wait, same model as IkReachBridge (reach_nominal_speed_rad_s +
    # margin, clamped) -- no motor-state convergence read, no force feedback.
    nominal_speed_rad_s: float = 1.0
    margin_s: float = 0.5
    min_wait_s: float = 0.8
    max_wait_s: float = 3.0
    dry_wait_s: float = 0.1
    max_state_age_s: float = 1.0
    # Spoken phase announcements (same mechanism as IkReachBridge/UmiDiffusionBridge).
    voice: bool = False
    voice_nic: str = ""
    voice_volume: int = 100


class BasketDepositBridge(Module):
    """grasp_done -> right-arm-only IK deposit (entry -> drop -> retreat) -> release."""

    config: BasketDepositBridgeConfig

    grasp_done: In[Bool]  # gripper reported closed (settled) -> start the deposit
    motor_states: In[JointState]  # full 29-DOF measured state (IK warm-start)
    arm_target: Out[JointState]  # 14 arm targets -> G1ArmSdkConnection
    gripper_target: Out[JointState]  # right Dex1 target q (open = release)
    deposit_done: Out[Bool]  # fired once the okra has been released into the basket

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._arm = load_g1_right_arm_ik(
            self.config.urdf_path,
            ik_config=PinocchioIKConfig(position_only=True),
            gripper_offset_xyz=self.config.gripper_offset_xyz,
        )
        if not self._arm.order_matches_canonical:
            raise RuntimeError(
                f"reduced right-arm order {self._arm.joint_names} != canonical order; "
                "refusing (see IkReachBridge for the same guard)."
            )
        self._lock = threading.Lock()
        self._latest_state: JointState | None = None
        self._trigger_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Thread | None = None
        self._count = 0
        from dimos.robot.unitree.g1.act.phase_voice import build_phase_voice

        self._voice = build_phase_voice(
            self.config.voice, self.config.voice_nic, volume=self.config.voice_volume
        )

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.grasp_done.subscribe(self._on_grasp_done)))
        self.register_disposable(Disposable(self.motor_states.subscribe(self._on_state)))
        self._stop_event.clear()
        self._thread = Thread(target=self._worker, daemon=True, name="basket-deposit-bridge")
        self._thread.start()
        logger.info(
            "BasketDepositBridge started",
            log_only=self.config.log_only,
            entry=self.config.entry_xyz,
            drop=self.config.drop_xyz,
            retreat=self.config.retreat_xyz,
        )

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        self._trigger_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        super().stop()

    def _on_state(self, state: JointState) -> None:
        with self._lock:
            self._latest_state = state

    def _on_grasp_done(self, msg: Bool) -> None:
        if not msg.data:
            return
        self._trigger_event.set()

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            self._trigger_event.wait()
            if self._stop_event.is_set():
                break
            self._trigger_event.clear()
            try:
                self._run_deposit()
            except Exception as e:  # never let the worker die on one bad cycle
                logger.warning(f"BasketDepositBridge: deposit failed: {e!r}")

    def _solve_leg(
        self, label: str, p_torso: np.ndarray, rot: np.ndarray, q_seed: np.ndarray
    ) -> np.ndarray | None:
        if not (
            self.config.ws_x[0] <= p_torso[0] <= self.config.ws_x[1]
            and self.config.ws_y[0] <= p_torso[1] <= self.config.ws_y[1]
            and self.config.ws_z[0] <= p_torso[2] <= self.config.ws_z[1]
        ):
            logger.warning(
                f"BasketDepositBridge: {label} torso target {np.round(p_torso, 3)} "
                "outside workspace box; aborting deposit."
            )
            return None
        p_root = self._arm.torso_to_root(p_torso)
        target = pinocchio.SE3(rot, np.asarray(p_root, dtype=float))
        q_sol, converged, err = self._arm.ik.solve(target, q_seed)
        q_sol = np.asarray(q_sol, dtype=float).flatten()
        if self.config.require_converged and not converged and err > self.config.max_reach_pos_err_m:
            logger.warning(
                f"BasketDepositBridge: {label} IK err={err:.4f} m exceeds tol "
                f"{self.config.max_reach_pos_err_m} m; aborting deposit."
            )
            return None
        if not check_joint_delta(q_sol, q_seed, self.config.max_joint_delta_deg):
            logger.warning(
                f"BasketDepositBridge: {label} joint delta exceeds "
                f"{self.config.max_joint_delta_deg}°; aborting deposit."
            )
            return None
        if not self._arm.clamp_ok(q_sol):
            logger.warning(
                f"BasketDepositBridge: {label} solution violates joint limits; aborting deposit."
            )
            return None
        return q_sol

    def _run_deposit(self) -> None:
        with self._lock:
            state = self._latest_state
        if state is None:
            logger.warning("BasketDepositBridge: no motor_states yet; aborting deposit.")
            return
        state_ts = float(getattr(state, "ts", 0.0) or 0.0)
        now = time.time()
        if state_ts and (now - state_ts) > self.config.max_state_age_s:
            logger.warning(
                f"BasketDepositBridge: stale motor_states ({now - state_ts:.1f}s old); "
                "aborting deposit."
            )
            return
        pos = list(state.position)
        if len(pos) < _ARM_START + _NUM_ARM:
            logger.warning(f"motor_states has {len(pos)} joints; expected >= 29; aborting.")
            return
        q_left = np.array([float(x) for x in pos[_LEFT_SLICE]])
        q_right = np.array([float(x) for x in pos[_RIGHT_SLICE]])
        if not (np.all(np.isfinite(q_left)) and np.all(np.isfinite(q_right))):
            logger.warning("BasketDepositBridge: measured arm pose has non-finite values; aborting.")
            return

        self._count += 1
        self._voice.say_phase("basket", "オクラを収納します")
        rot = self._arm.fk_root(q_right).rotation  # position-only: hold the grasp orientation
        q_cur = q_right
        legs = [
            ("entry", np.array(self.config.entry_xyz, dtype=float)),
            ("drop", np.array(self.config.drop_xyz, dtype=float)),
            ("retreat", np.array(self.config.retreat_xyz, dtype=float)),
        ]
        for label, p_torso in legs:
            if self._stop_event.is_set():
                return
            q_sol = self._solve_leg(label, p_torso, rot, q_cur)
            if q_sol is None:
                return
            arm14 = np.concatenate([q_left, q_sol])
            tag = "DRY" if self.config.log_only else "LIVE->arm_sdk"
            logger.info(
                f"[{tag}] BasketDepositBridge #{self._count} {label}: "
                f"torso{np.round(p_torso, 3)} q_right={np.round(q_sol, 3)}"
            )
            if not self.config.log_only:
                self.arm_target.publish(
                    JointState(
                        name=list(_ARM_JOINT_NAMES),
                        position=[float(x) for x in arm14],
                        velocity=[0.0] * _NUM_ARM,
                        effort=[0.0] * _NUM_ARM,
                    )
                )
            delta = float(np.max(np.abs(q_sol - q_cur)))
            if self.config.log_only:
                wait_s = self.config.dry_wait_s
            else:
                wait_s = (
                    delta / max(self.config.nominal_speed_rad_s, 1e-3) + self.config.margin_s
                )
                wait_s = min(max(wait_s, self.config.min_wait_s), self.config.max_wait_s)
            if self._stop_event.wait(wait_s):
                return  # shutting down: never release mid-deposit
            q_cur = q_sol

        if self.config.log_only:
            logger.info(f"[DRY] BasketDepositBridge: would open gripper to q={self.config.open_q:.3f}")
        else:
            logger.info(f"BasketDepositBridge: releasing -> gripper q={self.config.open_q:.3f}")
            self.gripper_target.publish(
                JointState(
                    name=[_RIGHT_GRIPPER_JOINT],
                    position=[self.config.open_q],
                    velocity=[0.0],
                    effort=[0.0],
                )
            )
        if self._stop_event.wait(
            self.config.release_settle_s if not self.config.log_only else self.config.dry_wait_s
        ):
            return
        self._voice.say_phase("basket_done", "収納が完了しました")
        logger.info(f"BasketDepositBridge #{self._count}: deposit complete -> firing deposit_done.")
        self.deposit_done.publish(Bool(data=True))


__all__ = [
    "BASKET_OPEN_Q",
    "DROP_TORSO",
    "ENTRY_TORSO",
    "RETREAT_TORSO",
    "BasketDepositBridge",
    "BasketDepositBridgeConfig",
]
