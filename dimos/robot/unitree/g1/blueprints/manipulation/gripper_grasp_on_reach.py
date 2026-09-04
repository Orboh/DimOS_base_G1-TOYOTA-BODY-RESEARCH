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

"""GripperGraspOnReach: script a single gripper close on IK reach_done (no ACT).

This is the ACT replacement for the `oda` IK-only okra experiment: where
``unitree_g1_okra_harvest.py`` hands ``reach_done`` to ``ActBridge`` (a learned
policy that drives the arm+gripper for ``grasp_duration_s``), this module just
publishes ONE closed-gripper target and lets ``G1GripperConnection``'s own soft
kp/kd control loop hold it there indefinitely. No arm motion, no policy, no
learned behavior of any kind -- purely "IK reached -> close the hand".

Debounced like ``ActBridge._on_reach_done`` (act_bridge.py:187-200): a
``reach_done`` arriving during ``debounce_s`` after the last one is ignored, so
duplicate/late-redelivered triggers don't double-fire. Unlike ActBridge there is
no fixed-duration "window" to close -- the point of this module is "grasp and
hold", so once the cooldown elapses it simply re-arms for the next click/attempt
(no restart needed to retry with a different close_q while tuning on hardware).

SAFETY: ``close_q`` is a raw Dex1 motor position with NO known-good value in
this repo -- unlike the ACT policy (which learned the right gripper motion from
demonstrations), this is an open-loop scripted command. Tune it on hardware
(watch ``right_gripper_state`` while varying it) before trusting a LIVE grasp.
``dry_run=True`` (default) logs the would-be action and publishes nothing.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.robot.unitree.g1.act.two_click_confirm import TwoClickConfirm
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_RIGHT_GRIPPER_JOINT = "g1/right_gripper"  # matches g1_gripper_connection.py


class GripperGraspOnReachConfig(ModuleConfig):
    # PLACEHOLDER -- untuned. Dex1 q_min=0.0/q_max=9.0 (g1_gripper_connection.py),
    # rest ~3-5, "open" empirically ~2.5-4.5 (larger q = MORE open, per
    # unitree_g1_okra_harvest.py's comments), so closing means a SMALLER q than
    # that. 0.0 is the mechanical closed limit -- likely too aggressive for a
    # soft vegetable. Set OKRA_NOACT_CLOSE_Q from on-hardware tuning before LIVE.
    close_q: float = 0.0
    # Ignore a second reach_done arriving within this many seconds of the last
    # one (duplicate/late-redelivered trigger guard), then re-arm automatically
    # so a fresh click can retry without restarting the process.
    debounce_s: float = 3.0
    # DRY-RUN (default): log the would-be gripper_target, publish nothing.
    dry_run: bool = True
    # Standard pre-grasp OPEN position (raw Dex1 q). If set, every accepted click
    # checks the measured gripper q and, when it is more closed than
    # (open_q - open_tolerance), publishes open_q BEFORE the arm arrives -- so
    # each grasp cycle starts from the same opening regardless of what the
    # previous cycle (basket release / crash / manual fiddling) left behind.
    # None (default) = feature off, preserving the original manual-open workflow.
    # NOTE: an accepted click while HOLDING an okra will open and drop it -- by
    # design (a click means "start a new cycle"); the basket-place motion (SS-06
    # F-07, developed separately) is responsible for releases between cycles.
    open_q: float | None = None
    open_tolerance: float = 0.3
    # Two-click confirm gate (2026-07-22), mirroring IkReachBridgeConfig: when True,
    # the pre-grasp opening fires only on the CONFIRMING (second) click — so a lone
    # ARMED click / phantom viewer-drag click no longer visibly moves the jaw
    # ("勝手に動く"). Keep the three parameters identical to IkReachBridge (wired
    # from the same OKRA_CONFIRM_* envs) so arm and jaw agree on which click fired.
    confirm_click: bool = False
    confirm_radius_m: float = 0.03
    confirm_window_s: float = 2.5
    confirm_min_gap_s: float = 0.35
    # Only feed the confirm gate clicks whose frame_id matches (same value as
    # IkReachBridgeConfig.expected_click_frame). Without this, a click the bridge
    # REJECTS (e.g. on the '/world/clicked_point' marker entity) still re-arms
    # OUR gate, desyncing arm and jaw: the bridge fires but the jaw never opens
    # (observed 2026-07-22 reach #1: blade arrived closed). Empty = accept all
    # frames (legacy).
    expected_click_frame: str = ""


class GripperGraspOnReach(Module):
    """reach_done -> one scripted gripper_target close (no ACT, no arm motion)."""

    config: GripperGraspOnReachConfig

    reach_done: In[Bool]  # IK settled at pre-grasp -> close now
    clicked_point: In[PointStamped]  # new cycle trigger: ensure standard opening
    right_gripper_state: In[JointState]  # measured Dex1 q (from G1GripperConnection)
    gripper_target: Out[JointState]  # right Dex1 target q (position[0])

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._cooldown_until = 0.0
        self._measured_q: float | None = None
        self._confirm = TwoClickConfirm(
            radius_m=self.config.confirm_radius_m,
            window_s=self.config.confirm_window_s,
            min_gap_s=self.config.confirm_min_gap_s,
        )

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.reach_done.subscribe(self._on_reach_done)))
        if self.config.open_q is not None:
            self.register_disposable(Disposable(self.clicked_point.subscribe(self._on_click)))
            self.register_disposable(
                Disposable(self.right_gripper_state.subscribe(self._on_gripper_state))
            )
        logger.warning(
            f"GripperGraspOnReach started: close_q={self.config.close_q:.3f} is a SCRIPTED, "
            f"UNTUNED value (no learned policy) -- tune via OKRA_NOACT_CLOSE_Q on hardware "
            f"before trusting a LIVE grasp. dry_run={self.config.dry_run}, "
            f"debounce_s={self.config.debounce_s}, "
            f"open_q={'OFF' if self.config.open_q is None else self.config.open_q}."
        )

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_gripper_state(self, msg: JointState) -> None:
        pos = list(msg.position)
        if pos:
            with self._lock:
                self._measured_q = float(pos[0])

    def _on_click(self, _msg: PointStamped) -> None:
        """Standardize the opening at the start of each cycle (open_q set only).

        Fires on the raw click (before IkReachBridge validates it) -- opening on
        a click that later gets rejected is harmless, it just restores the
        standard pre-grasp opening. The gripper opens while the arm is still
        slewing, so this adds no cycle time. In confirm mode only the CONFIRMING
        click opens (a lone/phantom click must not visibly move the jaw); the
        arm's reach takes >1 s, ample time for the jaw to open in parallel.
        """
        open_q = self.config.open_q
        if open_q is None:
            return
        if self.config.expected_click_frame and (
            str(getattr(_msg, "frame_id", "")) != self.config.expected_click_frame
        ):
            return  # click on another entity -- the bridge rejects it too; keep gates in sync
        if self.config.confirm_click:
            with self._lock:
                fire = self._confirm.feed(float(_msg.x), float(_msg.y), float(_msg.z), time.time())
            if not fire:
                return  # armed only -- IkReachBridge logs the ARMED message
        with self._lock:
            q = self._measured_q
        if q is not None and q >= open_q - self.config.open_tolerance:
            return  # already at (or beyond) the standard opening
        if self.config.dry_run:
            logger.info(
                f"GripperGraspOnReach: [DRY-RUN] click -> would open gripper to "
                f"q={open_q:.3f} (measured {q if q is not None else 'unknown'})."
            )
            return
        logger.info(
            f"GripperGraspOnReach: click -> opening gripper to q={open_q:.3f} "
            f"(measured {f'{q:.3f}' if q is not None else 'unknown'}; standard pre-grasp opening)."
        )
        self.gripper_target.publish(
            JointState(
                name=[_RIGHT_GRIPPER_JOINT],
                position=[open_q],
                velocity=[0.0],
                effort=[0.0],
            )
        )

    def _on_reach_done(self, msg: Bool) -> None:
        if not msg.data:
            return
        now = time.time()
        with self._lock:
            if now < self._cooldown_until:
                logger.info("GripperGraspOnReach: reach_done during cooldown; ignoring (debounce).")
                return
            self._cooldown_until = now + self.config.debounce_s
        if self.config.dry_run:
            logger.info(
                f"GripperGraspOnReach: [DRY-RUN] would publish gripper_target "
                f"q={self.config.close_q:.3f} (no ACT, no arm motion)."
            )
            return
        logger.info(
            f"GripperGraspOnReach: reach_done -> closing gripper q={self.config.close_q:.3f} "
            "(scripted, no ACT)."
        )
        self.gripper_target.publish(
            JointState(
                name=[_RIGHT_GRIPPER_JOINT],
                position=[self.config.close_q],
                velocity=[0.0],
                effort=[0.0],
            )
        )


__all__ = ["GripperGraspOnReach", "GripperGraspOnReachConfig"]
