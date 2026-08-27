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

"""UmiDiffusionBridge: replace the post-IK fine adjustment with a UMI diffusion policy.

Where ``GripperGraspOnReach`` scripts a single close on ``reach_done``, this module
runs a CLOSED-LOOP end-effector micro-adjustment driven by a trained UMI diffusion
policy (``~/umi/epoch=0110-train_loss=0.012.ckpt``). It sits in the same slot the
ACT ``ActBridge`` occupied (fed by ``IkReachBridge.reach_done``), but only drives the
RIGHT ARM (``arm_target``) — the gripper open/close is handled by the user's SEPARATE
program, which this module hands off to by firing ``adjust_done`` once the adjustment
converges.

Architecture (see oda/umi_diffusion/): the heavy perception+policy stack (GoPro UVC
capture, UMI fisheye/mask preprocessing, DiffusionUnetTimmPolicy.predict_action) runs
in a co-located process in the training-matched ``umi`` conda env
(``umi_policy_server.py``). This module — in the DimOS venv — is the robot-facing half:
each control tick it computes the current EE pose from the G1 right-arm FK, asks the
policy server for the next EE waypoint(s), solves G1 IK for them (reusing
``right_arm_model`` / ``PinocchioIK``), and publishes the 14-joint ``arm_target`` to
``G1ArmSdkConnection`` (whose 250 Hz clip-to-measured loop tracks the setpoint).

IMPORTANT — training data constraint (verified 2026-07-24 from
``~/umi/okra_20260723_ishimaru``): the ``gripper_width`` channel is a dead constant
(1e-4) across all 40 demos, so the policy CANNOT command the gripper. We feed the
policy that same constant for its gripper obs (kept in-distribution); its action head
has no gripper column at all (``action_include_gripper=False`` — 9-dim pos3 + rot6d,
decoded server-side to 6-dim pos+axis-angle). EE-pose adjustment is what it learned.

SAFETY: ``log_only=True`` (DRY-RUN, default) computes and logs everything but publishes
NO ``arm_target``. Every solved waypoint is gated (IK convergence, per-cycle joint
delta, joint limits, torso-frame workspace box) exactly like ``IkReachBridge`` before
it is published. On stop / server timeout the arm is HELD (we stop publishing; the arm
controller holds the last setpoint) — never a lunge. Keep the remote e-stop in hand
(L2+B damping); Ctrl-C clean-stop on this rig is unreliable.

OBSERVABILITY — "what did the policy actually infer?" is answered on three levels:
  * ``umi-infer`` line per inference: the obs sent to the server (tip pose + measured
    joints), server ``n``/``infer_ms``, and the returned action chunk as per-waypoint
    displacements from the measured tip (torso frame, mm).
  * ``adj`` line per tick (``log_every_n``): the commanded joint vector ``q_sol``, its
    per-joint delta from measured in degrees, worst joint, IK residual, settle state.
  * ``episode END`` summary + a JSONL trace (``trace_path``, default = the per-run log
    dir) holding EVERY waypoint of EVERY chunk untruncated, for offline analysis.
The summary's ``net`` (start->end commanded tip displacement) vs ``path`` (sum of
per-tick steps) is the direct read on whether the policy is converging on something or
just dithering in place.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
from threading import Thread
import time
from typing import Any

import numpy as np
import pinocchio
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.manipulation.planning.kinematics.pinocchio_ik import (
    PinocchioIKConfig,
    check_joint_delta,
    get_worst_joint_delta,
)
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.robot.unitree.g1.ik_reach.right_arm_model import (
    DEFAULT_URDF,
    load_g1_right_arm_ik,
)
from dimos.utils.logging_config import get_run_log_dir, setup_logger

logger = setup_logger()

# Canonical 29-DOF G1 joint vector: arms are 15-28 (left 15-21, right 22-28).
_G1_JOINTS = None  # imported lazily below to avoid a heavy import at module load
_ARM_START = 15
_NUM_ARM = 14
_LEFT_SLICE = slice(15, 22)
_RIGHT_SLICE = slice(22, 29)


def _arm_joint_names() -> list[str]:
    global _G1_JOINTS
    if _G1_JOINTS is None:
        from dimos.control.components import make_humanoid_joints

        _G1_JOINTS = make_humanoid_joints("g1")
    return list(_G1_JOINTS[_ARM_START : _ARM_START + _NUM_ARM])


class _PolicyClient:
    """Thin ZMQ REQ client to the co-located umi_policy_server (msgpack).

    One request per control tick. A REQ socket that times out is left in an
    invalid state, so on any timeout/error we close and lazily reconnect. Returns
    the server's whole reply dict — ``{"actions": [[pos3, aa3], ...] (absolute, arm
    ROOT frame), "n": chunk length, "infer_ms": server-side latency}`` — or ``None``
    if the server did not answer in time / replied ``ok=False``. The rows are 6-dim:
    this ckpt has ``action_include_gripper=False``, so there is NO gripper column.
    """

    def __init__(self, addr: str, timeout_ms: int) -> None:
        self._addr = addr
        self._timeout_ms = int(timeout_ms)
        self.last_error = ""
        self._ctx = None
        self._sock = None

    def _connect(self) -> None:
        import zmq

        if self._ctx is None:
            self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._sock.connect(self._addr)

    def _reset_socket(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close(0)
        except Exception:
            pass
        self._sock = None

    def request(self, payload: dict) -> dict | None:
        import msgpack

        try:
            if self._sock is None:
                self._connect()
            self._sock.send(msgpack.packb(payload, use_bin_type=True))
            raw = self._sock.recv()
        except Exception as e:  # timeout or transport error -> reconnect next call
            logger.warning(f"UmiDiffusionBridge: policy request failed ({e!r}); will reconnect.")
            self._reset_socket()
            return None
        rep = msgpack.unpackb(raw, raw=False)
        if not rep.get("ok", False):
            # Hand the reply back rather than discarding it. `ok: False` is the server
            # ANSWERING -- most often "the wrist camera is unusable", with the diagnosis in
            # `reason` (health) or `err` (predict). Returning None here made every camera
            # fault look like a dead server: the health preflight fell through to
            # "camera_unreachable" ("is umi_policy_server.py running?") while the GoPro was
            # simply switched off, and the camera_dead branch became unreachable.
            self.last_error = str(rep.get("reason") or rep.get("err") or "unspecified")
            logger.warning(f"UmiDiffusionBridge: server replied not-ok: {self.last_error}")
            return rep
        self.last_error = ""
        return rep

    def close(self) -> None:
        self._reset_socket()


# ---------------------------------------------------------------------------- EE frame
# Rotation taking a vector from the UMI/roboharvest TCP frame into the G1 `gripper_tip`
# frame. The policy's action translation is expressed in ITS OWN EE frame, so the two
# conventions must be reconciled or every commanded displacement comes out rotated 90°.
#
#   roboharvest scripts_data_processing/06_generate_dataset_plan.py:106
#       pose_cam_tcp = [0, cam_to_center_height, cam_to_tip_offset, 0, 0, 0]
#   The rotation is ZERO, i.e. the TCP frame IS the GoPro optical frame (OpenCV
#   convention): +x right, +y DOWN, +z FORWARD (the approach direction). The 0.086 m
#   "camera above gripper centre" sits on +y and the 0.2196 m "camera to fingertip" on
#   +z, which only holds with y-down / z-forward.
#
#   G1 `gripper_tip` (right_arm_model.fk_tip): at q=0 the frame coincides exactly with
#   torso -- +x FORWARD (approach), +y left, +z UP. Verified with pinocchio.
#
# So UMI +z ("go forward, toward the fruit") == G1 +x. Feeding the pose through
# unconverted made that approach command execute as G1 +z == STRAIGHT UP: on 2026-08-26
# every inference of the run returned a chunk whose EE-frame direction was
# [~0.03, ~-0.10, +0.99] no matter what the camera saw.
#
# Columns = the UMI axes written in gripper_tip coordinates:
#   UMI +x (right) -> G1 -y    UMI +y (down) -> G1 -z    UMI +z (fwd) -> G1 +x
_R_TIP_TO_UMI = np.array(
    [[0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


class UmiDiffusionBridgeConfig(ModuleConfig):
    # ---- policy server (co-located, umi conda env) ----
    server_addr: str = "tcp://127.0.0.1:5599"
    predict_timeout_ms: int = 300      # per-tick request budget; timeout -> hold this tick
    max_consecutive_timeouts: int = 10  # abort the adjustment (hold + fire nothing) after this many

    # ---- control loop ----
    control_hz: float = 10.0           # UMI eval default frequency
    n_exec_per_infer: int = 1          # execute this many returned waypoints per inference
                                       # (1 = receding-horizon, re-infer every tick; needs
                                       #  inference < 1/control_hz. Raise to amortize latency.)
    max_duration_s: float = 30.0       # safety ceiling on one adjustment episode

    # ---- convergence -> handoff (adjust_done) ----
    converge_pos_eps_m: float = 0.004  # commanded EE position step below which we count "settled"
    converge_hold_ticks: int = 8       # consecutive settled ticks -> fire adjust_done

    # ---- IK / arm model ----
    urdf_path: str = str(DEFAULT_URDF)
    # Gripper-tip offset from the wrist [m], WRIST frame == the point IK drives / FK reports
    # as the EE. MUST match the TCP point UMI tracked during data collection (Step 6:
    # ~/umi/okra_20260723_ishimaru dataset_plan grippers[0].tcp_pose). Default = bare Dex1.
    gripper_offset_xyz: list[float] = [0.1845, -0.003, 0.0]
    # v1 fallback: solve POSITION-only and let orientation float (safest; avoids 6-DOF
    # non-convergence). v2: False = full 6-DOF (follow the policy's commanded orientation).
    position_only: bool = False

    # ---- EE frame handed to / received from the policy (see _R_TIP_TO_UMI) ----
    # "camera": send the UMI TCP frame (GoPro optical axes) and convert the returned
    #           waypoints back to gripper_tip. This is the CORRECT setting.
    # "tip":    send the raw G1 gripper_tip frame (pre-2026-08-26 behaviour). Kept only
    #           so the two can be A/B'd on hardware; it makes "approach" come out as "up".
    ee_frame: str = "camera"
    # Position of the UMI TCP point in gripper_tip coordinates [m]. Zero means the G1
    # hand tip and the point UMI tracked during data collection are the same physical
    # point (RUN.md Step 6). It cancels exactly while position_only=True, so leave it at
    # zero until the 6-DOF mode is used.
    tip_to_tcp_xyz: list[float] = [0.0, 0.0, 0.0]

    # ---- safety gates (mirror IkReachBridge) ----
    max_joint_delta_deg: float = 20.0  # per-cycle cap (tighter than IkReach's one-shot 90°)
    max_reach_pos_err_m: float = 0.05
    require_converged: bool = True
    ws_x: list[float] = [0.05, 0.65]
    ws_y: list[float] = [-0.75, 0.20]
    ws_z: list[float] = [-0.35, 0.85]
    max_state_age_s: float = 1.0

    # ---- wrist-camera liveness (this side cannot see the camera; the server reports it) ----
    # The camera dies SILENTLY in two ways, both observed on hardware:
    #  * the capture device vanishes (a USB replug renumbers /dev/videoN) and the server's
    #    ring buffer then serves the SAME frame forever — on 2026-08-25 an entire LIVE run
    #    was driven from a 16-HOUR-old frame, with the policy emitting all-zero actions;
    #  * the card keeps delivering frames while the GoPro outputs nothing → ~100% black
    #    image (training frames are ~21% black from the mask).
    # Neither raises anywhere, and from this side both look exactly like "the policy just
    # isn't moving". So ask the server BEFORE the arm is driven, and refuse to start.
    require_camera_ok: bool = True
    camera_check_timeout_ms: int = 2000   # health round-trip budget (no inference involved)

    # ---- spoken phase announcements (Japanese, via the G1 speaker) ----
    # From outside the robot an IK coarse reach and a diffusion fine-adjustment look the
    # same, and a refused adjustment looks like nothing at all. Off by default; degrades
    # to log-only if the speaker cannot be built, and never raises into the control loop.
    voice: bool = False
    voice_nic: str = ""
    voice_volume: int = 100

    # ---- DRY-RUN (default): compute + log, publish NOTHING ----
    log_only: bool = True

    # ---- observability (see the module docstring) ----
    log_every_n: int = 1        # print the per-tick `adj` line every N ticks (1 = every tick)
    log_joints: bool = True     # include the q_meas / q_sol / delta-q block in that line
    log_chunk_max: int = 4      # waypoints of each chunk to print (-1 = all, 0 = none)
    trace_path: str = "auto"    # JSONL trace: "auto" = <run log dir>/umi_diffusion_trace.jsonl,
                                # explicit path, or "" to disable


class UmiDiffusionBridge(Module):
    """reach_done -> closed-loop EE micro-adjustment via UMI diffusion -> arm_target."""

    config: UmiDiffusionBridgeConfig

    reach_done: In[Bool]             # IK settled at pre-grasp -> start adjustment
    motor_states: In[JointState]     # full 29-DOF measured state (FK + IK warm-start)
    arm_target: Out[JointState]      # 14 arm joint targets (left 7 held, right 7 from IK)
    adjust_done: Out[Bool]           # fired once adjustment converges -> user's gripper program

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._arm = load_g1_right_arm_ik(
            self.config.urdf_path,
            ik_config=PinocchioIKConfig(position_only=self.config.position_only),
            gripper_offset_xyz=self.config.gripper_offset_xyz,
        )
        if not self._arm.order_matches_canonical:
            raise RuntimeError(
                f"reduced right-arm order {self._arm.joint_names} != canonical; refusing "
                "to construct (index mapping would be silently wrong)."
            )
        # gripper_tip -> UMI TCP. "tip" reproduces the pre-fix (wrong) behaviour.
        frame = str(self.config.ee_frame).strip().lower()
        if frame not in ("camera", "tip"):
            raise ValueError(f"ee_frame must be 'camera' or 'tip', got {self.config.ee_frame!r}")
        self._T_tip_umi = pinocchio.SE3(
            _R_TIP_TO_UMI if frame == "camera" else np.eye(3),
            np.asarray(self.config.tip_to_tcp_xyz, dtype=np.float64),
        )
        self._lock = threading.Lock()
        self._latest_state: JointState | None = None
        self._state_recv_t: float = 0.0
        self._client = _PolicyClient(self.config.server_addr, self.config.predict_timeout_ms)
        self._busy = threading.Event()   # an adjustment episode is running
        self._stop_event = threading.Event()
        self._worker: Thread | None = None
        self._count = 0
        # Resolved in start(): __init__ runs before the per-run log dir is published.
        self._trace_path: Path | None = None
        from dimos.robot.unitree.g1.act.phase_voice import build_phase_voice

        self._voice = build_phase_voice(
            self.config.voice, self.config.voice_nic, volume=self.config.voice_volume
        )

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.reach_done.subscribe(self._on_reach_done)))
        self.register_disposable(Disposable(self.motor_states.subscribe(self._on_state)))
        self._trace_path = self._resolve_trace_path()
        c = self.config
        logger.warning(
            "UmiDiffusionBridge started: server=%s control=%.1fHz n_exec=%d position_only=%s "
            "ee_frame=%s tip_to_tcp=%s "
            "log_only=%s tip_offset=%s converge=%.4fm x %d ticks max_duration=%.1fs "
            "log_every_n=%d log_joints=%s log_chunk_max=%d trace=%s "
            "(gripper is the USER's separate program; we fire adjust_done on convergence).",
            c.server_addr, c.control_hz, c.n_exec_per_infer, c.position_only,
            c.ee_frame, c.tip_to_tcp_xyz, c.log_only,
            c.gripper_offset_xyz, c.converge_pos_eps_m, c.converge_hold_ticks, c.max_duration_s,
            c.log_every_n, c.log_joints, c.log_chunk_max, self._trace_path or "OFF",
        )

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        w = self._worker
        if w is not None:
            w.join(timeout=2.0)
            self._worker = None
        self._client.close()
        super().stop()

    def _on_state(self, state: JointState) -> None:
        with self._lock:
            self._latest_state = state
            self._state_recv_t = time.time()

    def _read_arm_q(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (q_left7, q_right7) from the latest fresh motor_states, or None."""
        with self._lock:
            st = self._latest_state
            age = time.time() - self._state_recv_t
        if st is None or age > self.config.max_state_age_s:
            return None
        pos = list(st.position)
        if len(pos) < _ARM_START + _NUM_ARM:
            return None
        q_left = np.array([float(x) for x in pos[_LEFT_SLICE]], dtype=np.float64)
        q_right = np.array([float(x) for x in pos[_RIGHT_SLICE]], dtype=np.float64)
        if not (np.all(np.isfinite(q_left)) and np.all(np.isfinite(q_right))):
            return None
        return q_left, q_right

    # ----------------------------------------------------------- diagnostics
    @staticmethod
    def _j(x: Any) -> Any:
        """numpy -> JSON-safe. np.float64 / ndarray are NOT json.dumps-able."""
        if isinstance(x, np.ndarray):
            return [round(float(v), 6) for v in x.flatten()]
        if isinstance(x, (np.floating, np.integer)):
            return round(float(x), 6)
        if isinstance(x, (list, tuple)):
            return [UmiDiffusionBridge._j(v) for v in x]
        if isinstance(x, float):
            return round(x, 6) if np.isfinite(x) else None
        return x

    def _resolve_trace_path(self) -> Path | None:
        spec = (self.config.trace_path or "").strip()
        if not spec:
            return None
        if spec == "auto":
            d = get_run_log_dir()
            if d is None:
                env_d = os.environ.get("DIMOS_RUN_LOG_DIR")
                d = Path(env_d) if env_d else Path(tempfile.gettempdir())
            p = Path(d) / "umi_diffusion_trace.jsonl"
        else:
            p = Path(spec).expanduser()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"UmiDiffusionBridge: cannot create trace dir {p.parent}: {e!r}; trace OFF.")
            return None
        return p

    def _trace(self, rec: dict) -> None:
        """Append one JSONL record. Diagnostic only — must never disturb the control path."""
        if self._trace_path is None:
            return
        try:
            with open(self._trace_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # disable rather than raise once per tick
            logger.warning(f"UmiDiffusionBridge: trace write failed ({e!r}); disabling trace.")
            self._trace_path = None

    def _camera_health(self) -> dict | None:
        """Ask the server about the wrist camera. None = server unreachable.

        Uses its own (short) timeout: no inference runs, so the predict budget — which is
        sized for a ~135 ms forward pass — would be needlessly tight here.
        """
        client = _PolicyClient(self.config.server_addr, self.config.camera_check_timeout_ms)
        try:
            return client.request({"cmd": "health"})
        finally:
            client.close()

    def _to_torso(self, p_root: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._arm.torso_in_root.actInv(np.asarray(p_root, dtype=np.float64)), dtype=np.float64
        )

    def _umi_wp_to_tip(self, wp: Any, oMtip: Any) -> tuple[np.ndarray, np.ndarray]:
        """One policy waypoint (UMI TCP frame, ROOT) -> gripper_tip pose in ROOT.

        Returns (position, axis-angle). With ``position_only`` the wrist orientation is
        held, so the rotation paired with the commanded point is the MEASURED one --
        using the policy's commanded rotation here would leak it into the position
        through the gripper_tip <- TCP lever arm.
        """
        p_umi = np.asarray(wp[:3], dtype=np.float64)
        aa_umi = np.asarray(wp[3:6], dtype=np.float64)
        rot_umi = (
            oMtip.rotation @ self._T_tip_umi.rotation
            if self.config.position_only
            else pinocchio.exp3(aa_umi)
        )
        oMcmd = pinocchio.SE3(rot_umi, p_umi) * self._T_tip_umi.inverse()
        return (
            np.asarray(oMcmd.translation, dtype=np.float64),
            np.asarray(pinocchio.log3(oMcmd.rotation), dtype=np.float64),
        )

    def _fmt_chunk(self, chunk: list[list[float]], tip_torso: np.ndarray, oMtip: Any) -> str:
        """Chunk as per-waypoint displacement from the measured tip (torso frame, mm)."""
        n = len(chunk)
        if not n or self.config.log_chunk_max == 0:
            return f"(n={n}, not shown)"
        n_show = n if self.config.log_chunk_max < 0 else min(self.config.log_chunk_max, n)

        def _d(i: int) -> str:
            p_tip, _ = self._umi_wp_to_tip(chunk[i], oMtip)
            d = (self._to_torso(p_tip) - tip_torso) * 1000.0
            return f"#{i}[{d[0]:+.1f} {d[1]:+.1f} {d[2]:+.1f}]={float(np.linalg.norm(d)):.1f}"

        parts = [_d(i) for i in range(n_show)]
        if n_show < n:
            parts.append("… " + _d(n - 1))
        span = float(np.linalg.norm(self._to_torso(chunk[-1][:3]) - self._to_torso(chunk[0][:3])))
        return " ".join(parts) + f" span={span*1000:.1f}"

    def _note_skip(self, stats: dict, key: str, msg: str, rec: dict) -> None:
        stats[key] += 1
        logger.warning(msg)
        self._trace({**rec, "kind": "skip", "reason": key})

    def _on_reach_done(self, msg: Bool) -> None:
        if not getattr(msg, "data", False):
            return
        if self._busy.is_set():
            logger.info("UmiDiffusionBridge: reach_done while an adjustment is running; ignoring.")
            return
        self._busy.set()
        self._worker = Thread(target=self._run_adjustment, daemon=True, name="umi-diffusion-adjust")
        self._worker.start()

    # ------------------------------------------------------------------ loop
    def _run_adjustment(self) -> None:
        try:
            self._adjust_loop()
        except Exception as e:
            # traceback, not just repr: the loop is a worker thread, so this is the only
            # place the failure surfaces at all.
            logger.warning(f"UmiDiffusionBridge: adjustment loop failed: {e!r}", exc_info=True)
        finally:
            self._busy.clear()

    def _adjust_loop(self) -> None:
        self._count += 1
        ep = self._count
        period = 1.0 / max(self.config.control_hz, 1e-3)
        tag = "DRY" if self.config.log_only else "LIVE->arm_sdk"
        logger.info(f"UmiDiffusionBridge[{ep}]: reach_done -> starting EE adjustment [{tag}].")
        self._voice.say_phase("diffusion", "ディフュージョンで微調整します")

        t0 = time.time()
        stats: dict[str, Any] = {
            "ended": False, "ticks": 0, "infers": 0, "timeouts": 0,
            "skip_ws": 0, "skip_ik": 0, "skip_delta": 0, "skip_lim": 0,
            "infer_ms": [], "path_m": 0.0, "tip_first": None, "tip_last": None,
        }

        def _end(reason: str) -> None:
            """One summary line + trace record per episode, whichever way it exits."""
            if stats["ended"]:
                return
            stats["ended"] = True
            dur = time.time() - t0
            ims: list[float] = stats["infer_ms"]
            first_tip, last_tip = stats["tip_first"], stats["tip_last"]
            # net = did the policy actually walk the tip somewhere; path = how far it travelled
            # doing it. net ~ 0 with a large path means it is dithering, not adjusting.
            net = 0.0 if first_tip is None or last_tip is None else float(
                np.linalg.norm(last_tip - first_tip)
            )
            avg_ms = float(np.mean(ims)) if ims else 0.0
            max_ms = float(max(ims)) if ims else 0.0
            logger.info(
                f"UmiDiffusionBridge[{ep}] episode END reason={reason} dur={dur:.2f}s "
                f"ticks={stats['ticks']} infers={stats['infers']}\n"
                f"    infer_ms avg={avg_ms:.1f} max={max_ms:.1f} timeouts={stats['timeouts']} "
                f"skips{{ws={stats['skip_ws']} ik={stats['skip_ik']} delta={stats['skip_delta']} "
                f"lim={stats['skip_lim']}}}\n"
                f"    tip(torso) start{np.round(first_tip, 3) if first_tip is not None else '-'} "
                f"end{np.round(last_tip, 3) if last_tip is not None else '-'} "
                f"net={net*1000:.1f}mm path={stats['path_m']*1000:.1f}mm"
            )
            # Speak the outcome. The operator otherwise cannot tell "finished adjusting"
            # from "gave up" — both just leave the arm sitting still.
            self._voice.say_phase("end:" + reason, {
                "converged": "微調整が完了しました",
                "max_duration": "時間切れで微調整を終了します",
                "server_misses": "推論サーバーが応答しません。中止します",
                "camera_dead": "手首カメラの映像が使えません。中止します",
                "camera_unreachable": "推論サーバーに接続できません。中止します",
                "state_stale": "ロボットの状態が取得できません。中止します",
                "no_state": "ロボットの状態が取得できません。中止します",
                "stopped": "停止しました",
                "exception": "エラーが発生しました。中止します",
            }.get(reason, "微調整を終了します"))
            self._voice.reset()  # next reach_done starts a fresh phase sequence
            self._trace({
                "kind": "end", "t": time.time(), "ep": ep, "reason": reason,
                "dur_s": round(dur, 3), "ticks": stats["ticks"], "infers": stats["infers"],
                "timeouts": stats["timeouts"],
                "infer_ms_avg": round(avg_ms, 2), "infer_ms_max": round(max_ms, 2),
                "skips": {k: stats[k] for k in ("skip_ws", "skip_ik", "skip_delta", "skip_lim")},
                "net_m": round(net, 6), "path_m": round(float(stats["path_m"]), 6),
                "tip_first_torso": self._j(first_tip), "tip_last_torso": self._j(last_tip),
            })

        # ---- wrist-camera preflight, BEFORE anything drives the arm ----------------
        # A dead camera is indistinguishable from "the policy is idle" once the loop is
        # running, so check up front and refuse rather than move on a stale/black frame.
        if self.config.require_camera_ok:
            cam = self._camera_health()
            if cam is None:
                logger.error(
                    f"UmiDiffusionBridge[{ep}]: policy server did not answer the camera "
                    "health check; refusing to start (arm HELD, no adjust_done). Is "
                    "umi_policy_server.py running?"
                )
                _end("camera_unreachable")
                return
            age = cam.get("cam_age_ms")
            blk = cam.get("black_frac")
            age_s = "?" if age is None else f"{float(age):.0f}ms"
            blk_s = "?" if blk is None else f"{float(blk) * 100:.1f}%"
            if not cam.get("ok", False):
                logger.error(
                    f"UmiDiffusionBridge[{ep}]: WRIST CAMERA NOT USABLE — refusing to "
                    f"start (arm HELD, no adjust_done).\n    cam_age={age_s} "
                    f"black={blk_s}\n    {cam.get('reason', '')}"
                )
                _end("camera_dead")
                return
            logger.info(
                f"UmiDiffusionBridge[{ep}]: wrist camera OK (age={age_s} black={blk_s}, "
                "training frames are ~21% black)."
            )

        arm = self._read_arm_q()
        if arm is None:
            logger.warning(f"UmiDiffusionBridge[{ep}]: no fresh motor_states; aborting adjustment.")
            _end("no_state")
            return
        q_warm = arm[1].copy()  # right-arm warm start = measured

        settled = 0
        timeouts = 0
        tick = 0  # per-tick counter for log_every_n (NOT self._count, which is the episode)
        last_tip_cmd: np.ndarray | None = None  # previous COMMANDED tip, for the settle test
        first_req = True
        pending: list[list[float]] = []  # buffered waypoints from the last inference
        chunk_n = 0  # length of the chunk `pending` was sliced from (for the wp label)
        n_slice = 0  # how many of those chunk_n waypoints we actually execute
        wp_i = 0     # index of the next waypoint inside that slice
        n_exec = max(1, self.config.n_exec_per_infer)
        next_tick = time.perf_counter()

        try:
            while not self._stop_event.is_set():
                # Safety ceiling FIRST, once per iteration: every other exit below sits
                # behind a `continue`, so a waypoint that every tick fails a gate (or a
                # workspace box that rejects the whole chunk) used to spin here forever
                # with the arm held and no handoff. Verified offline 2026-07-30.
                if time.time() - t0 > self.config.max_duration_s:
                    logger.warning(
                        f"UmiDiffusionBridge[{ep}]: max_duration {self.config.max_duration_s}s "
                        "reached; firing adjust_done (best effort)."
                    )
                    self.adjust_done.publish(Bool(data=True))
                    _end("max_duration")
                    return

                arm = self._read_arm_q()
                if arm is None:
                    logger.warning(
                        f"UmiDiffusionBridge[{ep}]: motor_states went stale; holding + aborting."
                    )
                    _end("state_stale")
                    break
                q_left, q_right = arm

                # Measured tip: BOTH the obs we send to the policy and this tick's tracking
                # reference. One FK per tick (same thread as ik.solve, so no data race).
                oMtip = self._arm.fk_tip(q_right)
                tip_meas = np.asarray(oMtip.translation, dtype=np.float64)
                tip_meas_torso = self._to_torso(tip_meas)

                # (re)infer when the buffered chunk is exhausted
                if not pending:
                    # The policy speaks the UMI TCP frame (GoPro optical axes), not the
                    # G1 gripper_tip frame. Send its pose so `action_pose_repr=relative`
                    # rotates the returned displacement into the axes it was trained on.
                    oMumi = oMtip * self._T_tip_umi
                    eef_pos = np.asarray(oMumi.translation, dtype=np.float64)
                    eef_aa = np.asarray(pinocchio.log3(oMumi.rotation), dtype=np.float64)
                    was_reset = first_req
                    rep = self._client.request({
                        "cmd": "predict",
                        "t": time.time(),
                        "eef_pos": eef_pos.tolist(),
                        "eef_rot_aa": eef_aa.tolist(),
                        "reset": first_req,
                    })
                    first_req = False
                    actions = (rep.get("actions") if rep else None) or []
                    if not actions:
                        timeouts += 1
                        stats["timeouts"] = timeouts
                        if timeouts >= self.config.max_consecutive_timeouts:
                            why = self._client.last_error or "no reply (server down or too slow)"
                            logger.warning(
                                f"UmiDiffusionBridge[{ep}]: {timeouts} consecutive server misses; "
                                f"aborting (arm HELD, no adjust_done).\n    last reason: {why}"
                            )
                            _end("server_misses")
                            return
                        self._sleep_tick(period, next_tick)
                        next_tick += period
                        continue
                    timeouts = 0
                    infer_ms = float(rep.get("infer_ms") or 0.0)
                    chunk_n = int(rep.get("n") or len(actions))
                    stats["infers"] += 1
                    stats["infer_ms"].append(infer_ms)
                    pending = list(actions[:n_exec])
                    n_slice = len(pending)
                    wp_i = 0
                    # Camera health per inference: a camera that dies MID-episode is
                    # otherwise invisible from this side (the preflight only covers start).
                    cam_age = rep.get("cam_age_ms")
                    cam_blk = rep.get("black_frac")
                    cam_s = (
                        ""
                        if cam_age is None
                        else f" cam(age={float(cam_age):.0f}ms black={float(cam_blk or 0) * 100:.0f}%)"
                    )
                    logger.info(
                        f"[{tag}] umi-infer[ep{ep} tick{tick}] "
                        f"obs tip(torso){np.round(tip_meas_torso, 3)} "
                        f"aa_{self.config.ee_frame}{np.round(eef_aa, 3)}\n"
                        f"    q_meas={np.round(q_right, 3)}\n"
                        f"    server n={chunk_n} infer={infer_ms:.1f}ms exec={len(pending)} "
                        f"reset={was_reset}{cam_s}\n"
                        f"    chunk Δtip(torso,mm) "
                        f"{self._fmt_chunk(actions, tip_meas_torso, oMtip)}"
                    )
                    self._trace({
                        "kind": "infer", "t": time.time(), "ep": ep, "tick": tick,
                        "reset": bool(was_reset), "infer_ms": round(infer_ms, 2),
                        "n": chunk_n, "n_exec": len(pending),
                        # eef_* = the gripper_tip pose; obs_* = what was actually SENT
                        # to the policy (UMI TCP frame unless ee_frame="tip").
                        "eef_pos_root": self._j(tip_meas),
                        "eef_aa_root": self._j(np.asarray(pinocchio.log3(oMtip.rotation))),
                        "obs_pos_root": self._j(eef_pos), "obs_aa_root": self._j(eef_aa),
                        "ee_frame": self.config.ee_frame,
                        "eef_pos_torso": self._j(tip_meas_torso), "q_meas": self._j(q_right),
                        # every waypoint, untruncated — the reason the trace file exists
                        "chunk_root": [self._j(np.asarray(a, dtype=np.float64)) for a in actions],
                    })

                wp = pending.pop(0)
                i_wp = wp_i
                wp_i += 1
                # The waypoint is a pose of the UMI TCP frame; bring it back to the
                # gripper_tip frame the IK / gates / logs all speak.
                p_root, aa_root = self._umi_wp_to_tip(wp, oMtip)
                # No gripper column: this ckpt is action_include_gripper=False (and the
                # training gripper_width channel was a dead constant anyway).
                p_torso = self._to_torso(p_root)

                # context shared by this tick's skip / exec trace records
                base = {
                    "t": time.time(), "ep": ep, "tick": tick, "wp": i_wp, "n_chunk": chunk_n,
                    "target_root": self._j(p_root), "target_torso": self._j(p_torso),
                    "target_aa_root": self._j(aa_root),
                    "q_meas": self._j(q_right), "tip_meas_torso": self._j(tip_meas_torso),
                }
                where = (
                    f"[wp{i_wp}/{n_slice}of{chunk_n} tip(torso){np.round(tip_meas_torso, 3)} "
                    f"q_meas={np.round(q_right, 3)}]"
                )

                # workspace gate in torso frame
                if not (
                    self.config.ws_x[0] <= p_torso[0] <= self.config.ws_x[1]
                    and self.config.ws_y[0] <= p_torso[1] <= self.config.ws_y[1]
                    and self.config.ws_z[0] <= p_torso[2] <= self.config.ws_z[1]
                ):
                    self._note_skip(
                        stats, "skip_ws",
                        f"UmiDiffusionBridge[{ep}]: waypoint torso{np.round(p_torso,3)} outside "
                        f"workspace box x{self.config.ws_x} y{self.config.ws_y} "
                        f"z{self.config.ws_z}; skipping (arm holds). {where}",
                        base,
                    )
                    self._sleep_tick(period, next_tick)
                    next_tick += period
                    continue

                if self.config.position_only:
                    rot = self._arm.fk_root(q_right).rotation  # hold current orientation
                else:
                    rot = pinocchio.exp3(aa_root)
                target = pinocchio.SE3(rot, p_root)

                q_sol, converged, err = self._arm.ik.solve(target, q_warm)
                q_sol = np.asarray(q_sol, dtype=np.float64).flatten()
                base["q_sol"] = self._j(q_sol)
                base["err"] = self._j(float(err))
                base["converged"] = bool(converged)

                # gates (mirror IkReachBridge)
                if self.config.require_converged and not converged and err > self.config.max_reach_pos_err_m:
                    self._note_skip(
                        stats, "skip_ik",
                        f"UmiDiffusionBridge[{ep}]: IK err={err:.4f}m > tol; skipping waypoint. "
                        f"tgt(torso){np.round(p_torso,3)} q_sol={np.round(q_sol,3)} {where}",
                        base,
                    )
                    self._sleep_tick(period, next_tick)
                    next_tick += period
                    continue
                if not check_joint_delta(q_sol, q_right, self.config.max_joint_delta_deg):
                    wi, wd = get_worst_joint_delta(q_sol, q_right)
                    self._note_skip(
                        stats, "skip_delta",
                        f"UmiDiffusionBridge[{ep}]: joint {self._arm.joint_names[wi]} delta {wd:.1f}° "
                        f"> {self.config.max_joint_delta_deg}°; skipping waypoint (arm holds). "
                        f"tgt(torso){np.round(p_torso,3)} q_sol={np.round(q_sol,3)} {where}",
                        {**base, "worst_joint": self._arm.joint_names[wi], "worst_deg": float(wd)},
                    )
                    self._sleep_tick(period, next_tick)
                    next_tick += period
                    continue
                if not self._arm.clamp_ok(q_sol):
                    self._note_skip(
                        stats, "skip_lim",
                        f"UmiDiffusionBridge[{ep}]: q_sol violates joint limits; skipping. "
                        f"q_sol={np.round(q_sol,3)} tgt(torso){np.round(p_torso,3)} {where}",
                        base,
                    )
                    self._sleep_tick(period, next_tick)
                    next_tick += period
                    continue

                # Convergence on the commanded EE-position STEP: the distance between
                # SUCCESSIVE commands. Must NOT be measured against the measured tip — the
                # arm carries a steady-state PD droop under load (~46 mm at the default
                # kp_arm=80, measured on hw 2026-07-30), so a command-vs-measured distance
                # never drops below converge_pos_eps_m and the episode could only ever end
                # on the max_duration timeout.
                tip_cmd = np.asarray(self._arm.fk_tip(q_sol).translation, dtype=np.float64)
                step = float("inf") if last_tip_cmd is None else float(
                    np.linalg.norm(tip_cmd - last_tip_cmd)
                )
                if np.isfinite(step):
                    stats["path_m"] += step
                last_tip_cmd = tip_cmd
                tip_cmd_torso = self._to_torso(tip_cmd)
                if stats["tip_first"] is None:
                    stats["tip_first"] = tip_cmd_torso.copy()
                stats["tip_last"] = tip_cmd_torso.copy()
                settled = settled + 1 if step < self.config.converge_pos_eps_m else 0

                tick += 1
                stats["ticks"] = tick
                # `track` is diagnostic only (never gates): commanded vs measured tip, i.e.
                # how far the arm lags its target — see the droop note above.
                track = float(np.linalg.norm(tip_cmd - tip_meas))
                dq_deg = np.degrees(q_sol - q_right)
                wi, wd = get_worst_joint_delta(q_sol, q_right)

                if self.config.log_every_n and tick % self.config.log_every_n == 0:
                    msg = (
                        f"[{tag}] adj[{ep}] tick={tick} wp{i_wp}/{n_slice}of{chunk_n} "
                        f"t={time.time()-t0:.2f}s tgt(torso){np.round(p_torso,3)} "
                        f"tip(torso){np.round(tip_cmd_torso,3)}"
                    )
                    if self.config.log_joints:
                        msg += (
                            f"\n    q_meas ={np.round(q_right,3)}"
                            f"\n    q_sol  ={np.round(q_sol,3)}"
                            f"\n    Δq(deg)={np.round(dq_deg,2)} "
                            f"worst={self._arm.joint_names[wi]} {wd:.1f}°"
                        )
                    step_s = "-" if not np.isfinite(step) else f"{step*1000:.1f}mm"
                    msg += (
                        f"\n    conv={converged} err={err:.4f} step={step_s} "
                        f"track={track*1000:.1f}mm "
                        f"settled={settled}/{self.config.converge_hold_ticks}"
                    )
                    logger.info(msg)

                self._trace({
                    **base, "kind": "exec",
                    "dq_deg": self._j(dq_deg),
                    "worst_joint": self._arm.joint_names[wi], "worst_deg": round(float(wd), 4),
                    "tip_cmd_root": self._j(tip_cmd), "tip_cmd_torso": self._j(tip_cmd_torso),
                    "tip_meas_root": self._j(tip_meas),
                    "step_m": self._j(step), "track_m": self._j(track),
                    "settled": settled, "published": not self.config.log_only,
                })

                if not self.config.log_only:
                    self.arm_target.publish(
                        JointState(
                            name=_arm_joint_names(),
                            position=[float(x) for x in np.concatenate([q_left, q_sol])],
                            velocity=[0.0] * _NUM_ARM,
                            effort=[0.0] * _NUM_ARM,
                        )
                    )
                q_warm = q_sol

                # ---- termination ----
                if settled >= self.config.converge_hold_ticks:
                    logger.info(
                        f"UmiDiffusionBridge[{ep}]: converged ({settled} settled ticks, "
                        f"step<{self.config.converge_pos_eps_m*1000:.0f}mm); firing adjust_done."
                    )
                    self.adjust_done.publish(Bool(data=True))
                    _end("converged")
                    return
                # max_duration is checked at the TOP of the loop so gate-skips count too

                self._sleep_tick(period, next_tick)
                next_tick += period
        except Exception:
            _end("exception")  # summary first, then _run_adjustment logs the traceback
            raise
        _end("stopped")

    def _sleep_tick(self, period: float, next_tick: float) -> None:
        sleep_for = (next_tick + period) - time.perf_counter()
        if sleep_for > 0:
            self._stop_event.wait(sleep_for)


__all__ = ["UmiDiffusionBridge", "UmiDiffusionBridgeConfig"]
