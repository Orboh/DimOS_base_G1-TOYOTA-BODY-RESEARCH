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

# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Stoppable okra-ACT reach as one grasp episode (the real ``grasp_okra``).

ACT brings the right hand to the cut point (closing/cutting is a separate
program). This packages the verified okra-ACT loop (head camera + joints → ACT
inference over ZMQ → arm/Dex1 targets, the SAME wire contract as ``ActBridge``)
as a single **episode** that the SafetyMonitor can **cancel mid-reach**:

    grasp_okra(okra, force) -> ActGraspModule.run_episode()   # arm reaches
    SafetyMonitor.on_pause   -> ActGraspModule.stop()          # halt mid-reach

The episode ends at ``max_steps`` (≈ a few seconds; a reach-convergence/geometry
end is a future refinement) or when ``stop()`` is called. ``act_call`` and
``encode`` are injectable so the loop/cancel logic is unit-testable with a fake
ACT service and no robot (see ``test_act_grasp.py``).

⚠️ Not robot-verified here. On the robot it needs ``act_service.py`` (the lerobot
inference process) on ``act_endpoint`` and the arm/gripper connections wired.
The okra-ACT model itself (right-arm reach) is the verified Stage-B policy; an
8-DoF gripper-camera retrain is the production target (drops in unchanged here).
"""

from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import Any

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_ARM_START = 15
_NUM_ARM = 14
_RIGHT_GRIP_IDX = 15  # action[15] = right Dex1 target (action[14] left = dropped)
_RIGHT_GRIPPER_JOINT = "g1/right_gripper"
_RIGHT_ARM_START = 22  # right arm = canonical motor 22-28
_NUM_RIGHT_ARM = 7


def make_zmq_act_call(
    endpoint: str = "tcp://127.0.0.1:5701", recv_timeout_ms: int = 2000
) -> Callable[[dict[str, Any]], list[float]]:
    """Build an ``act_call(obs) -> action[16]`` over the ActBridge ZMQ/msgpack wire.

    obs = ``{"state":[16], "image_jpeg":<bytes>, "reset":<bool>}``; reply
    ``{"action":[16]}``. Reconnects on timeout. Reuses ``act_service.py`` (REP).
    """
    import msgpack
    import zmq

    ctx = zmq.Context.instance()
    sock = {"s": None}

    def _connect() -> Any:
        s = ctx.socket(zmq.REQ)
        s.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(endpoint)
        return s

    def act_call(obs: dict[str, Any]) -> list[float]:
        if sock["s"] is None:
            sock["s"] = _connect()
        try:
            sock["s"].send(msgpack.packb(obs, use_bin_type=True))
            resp = msgpack.unpackb(sock["s"].recv(), raw=False)
            return list(resp["action"])
        except zmq.error.Again:
            sock["s"].close()
            sock["s"] = _connect()
            raise

    return act_call


class ActGraspModule:
    """Runs one stoppable okra-ACT reach episode and drives the arm + right Dex1."""

    def __init__(
        self,
        image_getter: Callable[[], Any],
        state_getter: Callable[[], Any],
        gripper_getter: Callable[[], float],
        publish_arm: Callable[[Any], None],
        publish_gripper: Callable[[Any], None],
        *,
        wrist_getter: Callable[[], Any] | None = None,
        act_call: Callable[[dict[str, Any]], list[float]] | None = None,
        act_endpoint: str = "tcp://127.0.0.1:5701",
        rate_hz: float = 30.0,
        max_steps: int = 120,
        grasp_force: float = 0.3,
        reached_fn: Callable[[list[float], int], bool] | None = None,
        right_arm_only_7d: bool = False,
    ) -> None:
        self._image_getter = image_getter
        self._wrist_getter = wrist_getter  # right-wrist frame (2-camera tree model)
        self._state_getter = state_getter
        self._gripper_getter = gripper_getter
        self._publish_arm = publish_arm
        self._publish_gripper = publish_gripper
        self._act_call = act_call
        self._act_endpoint = act_endpoint
        self._rate_hz = rate_hz
        self._max_steps = max_steps
        self._grasp_force = grasp_force
        self._reached_fn = reached_fn
        # 7-DOF wrist-only model (sotata/act-okura-kinesthetic-wrist-7d): state/action
        # are the 7 RIGHT-arm joints only (motor 22-28), wrist camera as the sole image,
        # NO gripper dim (close/cut is OUT of ACT — handled by GraspSequence). When False
        # (default): legacy 8-dim right-only / 16-dim two-arm with a gripper action.
        self._right_arm_only_7d = bool(right_arm_only_7d)
        self._stop = threading.Event()
        # (okra_id, steps, cancelled) per episode — for trace / assertions.
        self.episodes: list[tuple[str, int, bool]] = []
        # Arm joint names (left 15-21, right 22-28), lazily to keep import light.
        from dimos.control.components import make_humanoid_joints

        self._arm_names = list(make_humanoid_joints("g1"))[_ARM_START : _ARM_START + _NUM_ARM]

    def stop(self) -> None:
        """Cancel the in-flight reach (SafetyMonitor.on_pause calls this)."""
        self._stop.set()

    def _encode(self, image: Any) -> bytes:
        import cv2

        ok, jpeg = cv2.imencode(".jpg", image.to_opencv())
        if not ok:
            raise ValueError("failed to JPEG-encode frame")
        return jpeg.tobytes()

    def _build_state(self, state: Any, right_grip: float) -> list[float] | None:
        pos = list(state.position)
        if len(pos) < _ARM_START + _NUM_ARM:
            logger.warning(f"[act-grasp] motor_states has {len(pos)} joints; expected >= 29")
            return None
        if self._right_arm_only_7d:
            # 7-dim: right arm only (motor 22-28), no gripper dim.
            right = pos[_RIGHT_ARM_START : _RIGHT_ARM_START + _NUM_RIGHT_ARM]
            return [float(x) for x in right]
        arms = pos[_ARM_START : _ARM_START + _NUM_ARM]
        return [float(x) for x in arms] + [0.0, float(right_grip)]  # left grip const 0

    def _publish(self, action: Any, force: float) -> None:
        from dimos.msgs.sensor_msgs.JointState import JointState

        if self._right_arm_only_7d:
            # action = 7 right-arm joints. Publish them into the right half of the
            # 14-joint arm_target (left held at the measured pose), and do NOT touch
            # the gripper (cut/close is GraspSequence's job, OUT of ACT).
            st = self._state_getter()
            left = [0.0] * 7
            if st is not None:
                pos = list(st.position)
                if len(pos) >= _ARM_START + _NUM_ARM:
                    left = [float(x) for x in pos[_ARM_START : _ARM_START + 7]]
            right = [float(x) for x in action[:_NUM_RIGHT_ARM]]
            self._publish_arm(
                JointState(
                    name=list(self._arm_names),
                    position=left + right,
                    velocity=[0.0] * _NUM_ARM,
                    effort=[0.0] * _NUM_ARM,
                )
            )
            return
        arms = [float(x) for x in action[:_NUM_ARM]]
        self._publish_arm(
            JointState(
                name=list(self._arm_names),
                position=arms,
                velocity=[0.0] * _NUM_ARM,
                effort=[0.0] * _NUM_ARM,
            )
        )
        self._publish_gripper(
            JointState(
                name=[_RIGHT_GRIPPER_JOINT],
                position=[float(action[_RIGHT_GRIP_IDX])],
                velocity=[0.0],
                effort=[0.0],
            )
        )

    def run_episode(self, okra: Any = None, force: float | None = None) -> bool:
        """Run one ACT reach. Returns True if it reached/completed, False if stopped."""
        self._stop.clear()
        f = self._grasp_force if force is None else float(force)
        if self._act_call is None:
            self._act_call = make_zmq_act_call(self._act_endpoint)
        okra_id = getattr(okra, "id", "?")
        logger.info(
            f"[act-grasp] reach START okra={okra_id} force={f} (max_steps={self._max_steps})"
        )

        period = 1.0 / max(1e-3, self._rate_hz)
        reset, steps, iters = True, 0, 0
        max_iters = self._max_steps * 4  # allow a few no-observation cycles
        next_t = time.perf_counter()
        reached = False
        while steps < self._max_steps and not self._stop.is_set() and iters < max_iters:
            iters += 1
            image, state, grip = self._image_getter(), self._state_getter(), self._gripper_getter()
            wrist = self._wrist_getter() if self._wrist_getter is not None else None
            if self._right_arm_only_7d:
                # Wrist-only model: the wrist frame is the SOLE image; head not needed.
                cams_ready = wrist is not None
            else:
                # 2-camera tree model needs head + wrist; single-cam needs head.
                cams_ready = image is not None and (self._wrist_getter is None or wrist is not None)
            if cams_ready and state is not None:
                obs_state = self._build_state(state, grip)
                if obs_state is not None:
                    try:
                        if self._right_arm_only_7d:
                            # Sole image = wrist. act_service derives head_img_key =
                            # img_keys[0] (the wrist key for this model) and reads it
                            # from image_jpeg, so send the wrist there.
                            wrist_jpeg = self._encode(wrist)
                            images = {"cam_right_wrist": wrist_jpeg}
                            req = {
                                "state": obs_state,
                                "images": images,
                                "image_jpeg": wrist_jpeg,
                                "reset": reset,
                            }
                        else:
                            images = {"cam_high": self._encode(image)}
                            if wrist is not None:
                                images["cam_right_wrist"] = self._encode(wrist)
                            req = {
                                "state": obs_state,
                                "images": images,
                                "image_jpeg": images["cam_high"],
                                "reset": reset,
                            }
                        action = self._act_call(req)
                        reset = False
                        steps += 1
                        self._publish(action, f)
                        if self._reached_fn and self._reached_fn(list(action), steps):
                            reached = True
                            break
                    except Exception as exc:
                        logger.warning(f"[act-grasp] act_call failed: {exc}")
            next_t += period
            wait = next_t - time.perf_counter()
            if wait > 0:
                self._stop.wait(wait)
            else:
                next_t = time.perf_counter()

        if self._stop.is_set():
            logger.warning(f"[act-grasp] STOPPED mid-reach at step {steps}")
            self.episodes.append((okra_id, steps, True))
            return False
        logger.info(f"[act-grasp] reach DONE ({steps} steps, reached={reached})")
        self.episodes.append((okra_id, steps, False))
        return steps > 0


__all__ = ["ActGraspModule", "make_zmq_act_call"]
