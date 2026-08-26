#!/usr/bin/env python3
"""Sweep okra pick positions across the IK workspace and, for each reachable one, chain a
reach-to-standoff IK solve straight into the entry->drop->release->retreat basket deposit.

``test_basket_deposit.py`` always starts the deposit sequence from the scene's *home*
right-arm pose -- it never actually connects "the arm just grasped an okra somewhere in
the workspace" to "now stow it in the basket". This script closes that gap: for every
sampled torso-frame point it

  1. teleports the ``okra`` mocap body there (no qpos touched -- see build_g1_scene.py's
     "mocap so a test can teleport it" comment),
  2. solves IK for the pre-grasp standoff and drives+settles the real actuators onto it
     (same recipe as smoke_sim_arm.py's ``_check_tracking``),
  3. if that converged, within joint limits, and free of arm/basket/self contact: takes
     the *actual settled* right-arm configuration (not home) as the seed for
     entry->drop->retreat, and runs the same release-and-settle physics
     ``test_basket_deposit.py`` uses,
  4. records success/failure and, on failure, which stage and why.

Every helper that touches the deposit physics (``_scene_with_free_cargo``, ``_make_rig``,
``_solve``, ``_advance``, ``_cargo_torso``) is imported from ``test_basket_deposit.py``,
not copied -- this script only supplies the missing "reach, then deposit" glue and the
sweep/report/video harness around it. Neither ``test_basket_deposit.py`` nor
``build_g1_scene.py`` is modified.

Run after regenerating the scene:
    MUJOCO_GL=egl .venv/bin/python oda/mujoco_sim/test_basket_deposit_workspace.py
    -> oda/mujoco_sim/output/workspace_test_results.json
    -> oda/mujoco_sim/output/workspace_test_plot.png
    -> oda/mujoco_sim/output/deposit_<x>_<y>_<z>.mp4  (a few representative points)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)
import mujoco
import numpy as np
import pinocchio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dimos.robot.unitree.g1.ik_reach.right_arm_model import load_g1_right_arm_ik
from oda.mujoco_sim.build_g1_scene import _OUT
from oda.mujoco_sim.render_basket_deposit import render as render_deposit
from oda.mujoco_sim.smoke_sim_arm import STANDOFF_M, TIP_OFFSET, TRACK_TOL_M
from oda.mujoco_sim.test_basket_deposit import (
    DEPOSIT_MOVE_SECONDS,
    DROP_IK_SEED,
    DROP_TORSO,
    ENTRY_IK_SEED,
    ENTRY_TORSO,
    MAX_BASKET_PENETRATION_M,
    RETREAT_TORSO,
    SETTLE_SECONDS,
    TRACK_TOL_M as DEPOSIT_TRACK_TOL_M,
    _advance,
    _cargo_torso,
    _make_rig,
    _solve,
)

_OUT_DIR = Path(__file__).resolve().parent / "output"
_RIGHT_SLICE = slice(22, 29)
REACH_SETTLE_S = 2.0

# ---------------------------------------------------------------------------------------
# Sample points, torso_link frame [m]. IkReachBridge's workspace box is
# x in [0.05,0.65] y in [-0.75,0.20] z in [-0.35,0.85] -- but that is the box the *bridge*
# clamps a click into, not "the right arm can reach every point in it" (a 7-DOF arm
# mounted at the shoulder plainly cannot reach 0.05 m in front of the torso, or 0.85 m
# above it, while the base is welded upright). The grid below is centered on the known-
# good neutral tip [0.245,-0.152,0.051] (build_g1_scene.py's OKRA_IN_TORSO comment) and
# covers a box around it likely to mostly succeed; the "stress" points deliberately probe
# the edges of the workspace box (near centerline, very close, very far, very high/low) to
# surface real reach-limit failures for the report, per Yokote's request to see where it
# does NOT work, not just where it does.
# ---------------------------------------------------------------------------------------
_GRID_X = [0.30, 0.40, 0.50]
_GRID_Y = [-0.45, -0.30, -0.15, 0.00]
_GRID_Z = [-0.05, 0.15]
_STRESS_POINTS = [
    (0.15, -0.30, 0.10),   # very close to the torso
    (0.60, -0.30, 0.10),   # far reach
    (0.45, 0.15, 0.10),    # near/past the body centerline (left of the right arm's side)
    (0.45, -0.30, 0.45),   # high
    (0.45, -0.30, -0.30),  # low, near basket height
    (0.45, -0.60, 0.10),   # far to the right
    (0.20, -0.60, 0.35),   # corner
    (0.55, 0.10, -0.20),   # corner
]

# Operational front-side harvesting envelope, torso_link frame [m]. This is intentionally
# narrower than the click-clamping box and excludes the torso centreline. It is the area
# that must pass *every* sample before we claim a reliable front-to-basket handoff:
# 30--45 cm ahead, 25--45 cm to the right of torso centre, and 5--25 cm above torso origin.
# A 5 cm grid makes the acceptance set explicit and reproducible (4 x 3 x 3 = 36 points).
FRONT_WORKSPACE_BOX_TORSO = {
    "x": [0.30, 0.45],
    "y": [-0.45, -0.25],
    "z": [0.05, 0.25],
}
_FRONT_GRID_X = [0.30, 0.35, 0.40, 0.45]
_FRONT_GRID_Y = [-0.45, -0.35, -0.25]
_FRONT_GRID_Z = [0.05, 0.15, 0.25]

# Photo-aligned, front-centre-side acceptance profile. The plant appears near the image
# centre in IMG_7961, but a right-arm-only transfer needs 10--20 cm rightward clearance
# from torso centre to keep the UMI and carried pod out of the body/basket. Ten forward
# samples, five lateral samples and two fruit heights provide a 100-case all-pass gate.
CENTER_WORKSPACE_BOX_TORSO = {
    "x": [0.45, 0.54],
    "y": [-0.20, -0.10],
    "z": [0.15, 0.25],
}
_CENTER_GRID_X = [0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54]
_CENTER_GRID_Y = [-0.20, -0.175, -0.15, -0.125, -0.10]
_CENTER_GRID_Z = [0.15, 0.25]

# Route central grasps outward, forward and above the basket rim before carrying them
# inward to the abdominal drop point. Solve this waypoint from the actual grasp posture
# so the IK preserves the local elbow branch instead of switching through the torso.
TRANSFER_STAGE_TORSO = np.array([0.50, -0.25, 0.30])


def sample_points() -> list[tuple[float, float, float]]:
    pts = [(x, y, z) for x in _GRID_X for y in _GRID_Y for z in _GRID_Z]
    pts.extend(_STRESS_POINTS)
    return pts


def front_sample_points() -> list[tuple[float, float, float]]:
    """Return the 36-point, all-pass-required front harvesting acceptance grid."""
    return [(x, y, z) for x in _FRONT_GRID_X for y in _FRONT_GRID_Y for z in _FRONT_GRID_Z]


def center_sample_points() -> list[tuple[float, float, float]]:
    """Return the photo-aligned 100-point, all-pass-required centre-side grid."""
    return [(x, y, z) for x in _CENTER_GRID_X for y in _CENTER_GRID_Y for z in _CENTER_GRID_Z]


def _contact_report(model: mujoco.MjModel, data: mujoco.MjData, ignore: set[str]) -> list[tuple[str, str]]:
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        if floor_id in (c.geom1, c.geom2):
            continue
        a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom#{c.geom1}"
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom#{c.geom2}"
        if a in ignore or b in ignore:
            continue
        out.append((a, b))
    return out


def _reach_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm,  # G1RightArmIK
    point_torso: tuple[float, float, float],
    home_key: int,
    torso_id: int,
    wrist_id: int,
    okra_mocap: int,
) -> dict:
    """Reach phase: teleport the okra, IK to the standoff, settle, check tracking.

    Mirrors smoke_sim_arm.py's ``_check_tracking``, but against an arbitrary torso-frame
    point instead of the scene's fixed ``OKRA_IN_TORSO``, and via mocap teleport instead
    of a scene rebuild (fast enough to run ~30 points/run; also exactly the mechanism
    ``build_g1_scene.py``'s ``okra`` mocap body was designed for).
    """
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    mujoco.mj_forward(model, data)
    torso_pos = np.asarray(data.xpos[torso_id], dtype=float)
    torso_rot = np.asarray(data.xmat[torso_id], dtype=float).reshape(3, 3)

    point = np.asarray(point_torso, dtype=float)
    data.mocap_pos[okra_mocap] = torso_pos + torso_rot @ point
    mujoco.mj_forward(model, data)

    target_t = point - np.array([STANDOFF_M, 0.0, 0.0])
    target_root = arm.torso_to_root(target_t)
    q_seed = data.qpos[_RIGHT_SLICE].copy()
    q_sol, converged, err = arm.ik.solve(pinocchio.SE3(np.eye(3), target_root), q_seed)

    if not converged:
        return {"ok": False, "stage": "reach_ik", "reason": f"IK did not converge (err={err:.4f} m)"}
    if not arm.clamp_ok(q_sol):
        return {"ok": False, "stage": "reach_ik", "reason": "IK solution violates joint limits"}

    data.ctrl[_RIGHT_SLICE] = q_sol
    for _ in range(int(REACH_SETTLE_S / model.opt.timestep)):
        mujoco.mj_step(model, data)

    tip = np.asarray(data.xpos[wrist_id], dtype=float) + np.asarray(
        data.xmat[wrist_id], dtype=float
    ).reshape(3, 3) @ np.asarray(TIP_OFFSET, dtype=float)
    tip_t = torso_rot.T @ (tip - torso_pos)
    track_err = float(np.linalg.norm(tip_t - target_t))
    if track_err > TRACK_TOL_M:
        return {
            "ok": False,
            "stage": "reach_tracking",
            "reason": f"tip did not track standoff: {track_err * 1000:.1f} mm > {TRACK_TOL_M * 1000:.0f} mm",
        }

    contacts = _contact_report(model, data, ignore=set())
    if contacts:
        return {
            "ok": False,
            "stage": "reach_collision",
            "reason": f"arm contact during reach/settle: {contacts[:3]}",
        }

    q_reach = data.qpos[_RIGHT_SLICE].copy()
    wrist_world = np.asarray(data.xpos[wrist_id], dtype=float) + np.asarray(
        data.xmat[wrist_id], dtype=float
    ).reshape(3, 3) @ np.asarray(TIP_OFFSET, dtype=float)
    return {
        "ok": True,
        "q_reach": q_reach.tolist(),
        "tip_world": wrist_world.tolist(),
        "track_err_mm": track_err * 1000.0,
    }


def _deposit_from_pose(scene: Path, q_reach: np.ndarray, tip_world: np.ndarray, arm) -> dict:
    """entry -> drop -> release -> retreat/settle, seeded from an actual reach pose.

    Reuses test_basket_deposit.py's helpers verbatim; the only difference from that
    script's main() is the starting right-arm configuration (``q_reach`` instead of the
    scene's home pose) and the cargo's initial attach point (``tip_world`` instead of the
    home tip).
    """
    import xml.etree.ElementTree as ET

    from oda.mujoco_sim.test_basket_deposit import CARGO_HALF_LEN_M, CARGO_RADIUS_M

    def scene_with_cargo_at(held: bool) -> str:
        root = ET.parse(scene).getroot()
        worldbody = root.find("worldbody")
        cargo = ET.SubElement(
            worldbody, "body", name="cargo_okra", pos=" ".join(f"{v:.8f}" for v in tip_world)
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
            # Match test_basket_deposit: the pod contacts the basket even while
            # virtually grasped, so the transfer cannot pass through the rim.
            # Its category is disjoint from the non-actuated hand/UMI proxy.
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

    rig = _make_rig(scene_with_cargo_at(held=True))
    rig.data.qpos[_RIGHT_SLICE] = q_reach
    rig.command_right_arm(q_reach)
    mujoco.mj_forward(rig.model, rig.data)
    cargo_body = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_BODY, "cargo_okra")
    cargo_joint = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_JOINT, "cargo_okra_free")
    cargo_qpos_adr = rig.model.jnt_qposadr[cargo_joint]

    try:
        q_transfer = _solve(arm, TRANSFER_STAGE_TORSO, q_reach)
        q_entry = _solve(arm, ENTRY_TORSO, ENTRY_IK_SEED)
        q_drop = _solve(arm, DROP_TORSO, DROP_IK_SEED)
        q_retreat = _solve(arm, RETREAT_TORSO, ENTRY_IK_SEED)
    except RuntimeError as exc:
        return {"ok": False, "stage": "deposit_ik", "reason": str(exc)}

    failures: list[str] = []
    for label, target, solution in (
        ("transfer", TRANSFER_STAGE_TORSO, q_transfer),
        ("entry", ENTRY_TORSO, q_entry),
        ("drop", DROP_TORSO, q_drop),
    ):
        deepest, pairs, torso_pairs = _advance(rig, solution, seconds=DEPOSIT_MOVE_SECONDS)
        tip_err = float(np.linalg.norm(rig.tip_torso(np.asarray(TIP_OFFSET)) - target))
        arm_basket = [p for p in pairs if not any("cargo_okra" in geom for geom in p)]
        carried_pod_basket = [p for p in pairs if "cargo_okra_pod" in p]
        if tip_err > DEPOSIT_TRACK_TOL_M:
            failures.append(f"{label}: tip error {tip_err * 1000:.1f} mm")
        if arm_basket:
            failures.append(f"{label}: arm touched basket ({arm_basket[0]})")
        if carried_pod_basket:
            failures.append(f"{label}: carried okra touched basket ({carried_pod_basket[0]})")
        if torso_pairs:
            failures.append(f"{label}: arm/UMI touched torso ({torso_pairs[0]})")
    if failures:
        return {"ok": False, "stage": "deposit_carry", "reason": "; ".join(failures)}

    release_qpos = rig.data.qpos.copy()
    torso_rot = rig.torso_pose()[1]
    cargo_rot_torso = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    cargo_quat = np.empty(4)
    mujoco.mju_mat2Quat(cargo_quat, (torso_rot @ cargo_rot_torso).reshape(-1))
    release_qpos[cargo_qpos_adr + 3 : cargo_qpos_adr + 7] = cargo_quat

    rig = _make_rig(scene_with_cargo_at(held=False))
    cargo_body = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_BODY, "cargo_okra")
    cargo_joint = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_JOINT, "cargo_okra_free")
    cargo_qvel = rig.model.jnt_dofadr[cargo_joint]
    rig.data.qpos[:] = release_qpos
    rig.data.qvel[:] = 0.0
    rig.command_right_arm(q_drop)
    mujoco.mj_forward(rig.model, rig.data)

    deepest, pairs, torso_pairs = _advance(rig, q_retreat, seconds=SETTLE_SECONDS)
    cargo_t = _cargo_torso(rig, cargo_body)
    cargo_speed = float(np.linalg.norm(rig.data.qvel[cargo_qvel : cargo_qvel + 3]))
    cargo_basket_contacts = [p for p in pairs if "cargo_okra_pod" in p]
    inside = 0.090 < cargo_t[0] < 0.275 and abs(cargo_t[1]) < 0.040 and -0.165 < cargo_t[2] < -0.080

    if not cargo_basket_contacts:
        return {"ok": False, "stage": "deposit_release", "reason": "released cargo never contacted the basket"}
    if not inside:
        return {
            "ok": False,
            "stage": "deposit_release",
            "reason": f"released cargo landed outside the basket: torso={np.round(cargo_t, 4).tolist()}",
        }
    if cargo_speed > 0.03:
        return {"ok": False, "stage": "deposit_release", "reason": f"cargo did not settle: speed={cargo_speed:.3f} m/s"}
    if torso_pairs:
        return {"ok": False, "stage": "deposit_release", "reason": f"retreat arm/UMI touched torso: {torso_pairs[0]}"}
    if deepest < -MAX_BASKET_PENETRATION_M:
        return {
            "ok": False,
            "stage": "deposit_release",
            "reason": (
                f"basket penetration exceeds {MAX_BASKET_PENETRATION_M * 1000:.0f} mm "
                f"contact tolerance: {deepest * 1000:.1f} mm"
            ),
        }

    return {"ok": True, "cargo_torso": cargo_t.tolist(), "cargo_speed": cargo_speed}


def run(scene: Path, points: list[tuple[float, float, float]], verbose: bool = True) -> list[dict]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    okra_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "okra")
    okra_mocap = model.body_mocapid[okra_body]

    arm = load_g1_right_arm_ik(gripper_offset_xyz=TIP_OFFSET)

    results = []
    for i, pt in enumerate(points):
        t0 = time.time()
        reach = _reach_point(model, data, arm, pt, home_key, torso_id, wrist_id, okra_mocap)
        record = {"point_torso": list(pt)}
        if not reach["ok"]:
            record.update({"success": False, "stage": reach["stage"], "reason": reach["reason"]})
        else:
            deposit = _deposit_from_pose(
                scene, np.asarray(reach["q_reach"]), np.asarray(reach["tip_world"]), arm
            )
            if not deposit["ok"]:
                record.update({"success": False, "stage": deposit["stage"], "reason": deposit["reason"]})
            else:
                record.update(
                    {
                        "success": True,
                        "cargo_torso": deposit["cargo_torso"],
                        "cargo_speed": deposit["cargo_speed"],
                        "track_err_mm": reach["track_err_mm"],
                    }
                )
        dt = time.time() - t0
        if verbose:
            status = "OK  " if record["success"] else f"FAIL[{record['stage']}]"
            extra = "" if record["success"] else f" -- {record['reason']}"
            print(f"[{i + 1:2d}/{len(points)}] pt={pt} {status} ({dt:.1f}s){extra}")
        results.append(record)
    return results


def summarize(results: list[dict]) -> str:
    n = len(results)
    n_ok = sum(r["success"] for r in results)
    lines = [f"SUCCESS {n_ok}/{n}"]
    fails = [r for r in results if not r["success"]]
    if fails:
        lines.append("failures:")
        for r in fails:
            lines.append(f"  {tuple(r['point_torso'])}: [{r['stage']}] {r['reason']}")
    return "\n".join(lines)


def plot(results: list[dict], out_path: Path) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ok = np.array([r["point_torso"] for r in results if r["success"]])
    bad = np.array([r["point_torso"] for r in results if not r["success"]])
    if len(ok):
        ax.scatter(ok[:, 0], ok[:, 1], ok[:, 2], c="#2ca02c", marker="o", s=60, label=f"success ({len(ok)})")
    if len(bad):
        ax.scatter(bad[:, 0], bad[:, 1], bad[:, 2], c="#d62728", marker="x", s=70, label=f"failure ({len(bad)})")
    ax.set_xlabel("x (torso, m)")
    ax.set_ylabel("y (torso, m)")
    ax.set_zlabel("z (torso, m)")
    ax.set_title("Basket-deposit workspace sweep: okra pick position -> reach+deposit outcome")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def pick_representatives(results: list[dict], n_ok: int = 3, n_fail: int = 2) -> list[dict]:
    oks = [r for r in results if r["success"]]
    fails = [r for r in results if not r["success"]]
    reps = []
    if oks:
        idxs = np.linspace(0, len(oks) - 1, num=min(n_ok, len(oks))).astype(int)
        reps.extend(oks[i] for i in sorted(set(idxs.tolist())))
    reps.extend(fails[: min(n_fail, len(fails))])
    return reps


def render_representatives(scene: Path, reps: list[dict], out_dir: Path) -> list[str]:
    """Re-run the reach+deposit for a handful of points with render_basket_deposit's
    camera/recording, so a person can watch specific successes and failures."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    okra_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "okra")
    okra_mocap = model.body_mocapid[okra_body]
    arm = load_g1_right_arm_ik(gripper_offset_xyz=TIP_OFFSET)

    written = []
    for r in reps:
        pt = tuple(r["point_torso"])
        tag = "_".join(f"{v:+.2f}" for v in pt)
        out_path = out_dir / f"deposit_{tag}.mp4"
        reach = _reach_point(model, data, arm, pt, home_key, torso_id, wrist_id, okra_mocap)
        if not reach["ok"]:
            print(f"  [video] {pt} failed at reach ({reach['reason']}) -- skipping video (nothing to show past that point)")
            continue
        try:
            q_transfer = _solve(arm, TRANSFER_STAGE_TORSO, np.asarray(reach["q_reach"]))
            render_deposit(
                scene,
                out_path,
                q_start=np.asarray(reach["q_reach"]),
                tip_world_override=np.asarray(reach["tip_world"]),
                q_transfer=q_transfer,
                verbose=False,
            )
            label = "success" if r["success"] else f"failure[{r.get('stage')}]"
            print(f"  [video] wrote {out_path} for point {pt} ({label})")
            written.append(str(out_path))
        except RuntimeError as exc:
            print(f"  [video] {pt}: deposit IK failed while rendering ({exc}) -- skipping")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=_OUT)
    ap.add_argument(
        "--profile",
        choices=("diagnostic", "front", "center"),
        default="diagnostic",
        help=(
            "diagnostic: broad/stress sweep; front: 36-point all-pass harvesting envelope; "
            "center: photo-aligned 100-point all-pass centre-side envelope"
        ),
    )
    ap.add_argument("--skip-videos", action="store_true", help="skip representative mp4 rendering")
    args = ap.parse_args()
    if not args.scene.exists():
        print(f"scene missing: {args.scene}\nrun: .venv/bin/python oda/mujoco_sim/build_g1_scene.py")
        return 2

    points = {
        "diagnostic": sample_points,
        "front": front_sample_points,
        "center": center_sample_points,
    }[args.profile]()
    print(f"sweeping {len(points)} points ({args.profile} profile)")
    results = run(args.scene, points)

    print()
    print(summarize(results))

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = _OUT_DIR / f"workspace_{args.profile}_test_results.json"
    workspace_box = {
        "front": FRONT_WORKSPACE_BOX_TORSO,
        "center": CENTER_WORKSPACE_BOX_TORSO,
    }.get(args.profile, {"x": [0.05, 0.65], "y": [-0.75, 0.20], "z": [-0.35, 0.85]})
    results_path.write_text(json.dumps({
        "profile": args.profile,
        "points": results,
        "workspace_box_torso": workspace_box,
        "all_points_required": args.profile in ("front", "center"),
    }, indent=2))
    print(f"wrote {results_path}")

    plot_path = _OUT_DIR / f"workspace_{args.profile}_test_plot.png"
    plot(results, plot_path)
    print(f"wrote {plot_path}")

    if not args.skip_videos:
        reps = pick_representatives(results)
        print(f"rendering {len(reps)} representative videos")
        render_representatives(args.scene, reps, _OUT_DIR)

    if args.profile in ("front", "center") and any(not r["success"] for r in results):
        print(f"{args.profile.upper()}_WORKSPACE FAILED: every profile point must succeed")
        return 1
    if args.profile in ("front", "center"):
        print(f"{args.profile.upper()}_WORKSPACE OK: every profile point succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
