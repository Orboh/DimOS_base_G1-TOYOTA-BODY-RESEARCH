#!/usr/bin/env python3
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

"""摩擦把持の単体プローブ（isaac-sim env, headless）: 位置ずれを完全排除して「今の摩擦設定で
そもそも掴めるのか」だけを確かめる。

歩行 policy / IK / DDS / 位置合わせを一切通さず、対象オブジェクトを **指の隙間のど真ん中**へ
キネマティック固定した状態でグリッパを握らせ、その後 dynamic へ解放して重力下で保持できるかを測る。
→ 完璧に置いて握っても落ちるなら「摩擦/接触設定の問題」、保持できるなら「位置合わせの問題」と切り分く。

物理設定は sim_dds_bridge.py の摩擦把持モード（SIM_GRASP_FRICTION=1）と同一:
  μ_s=SIM_GRIP_FRICTION(1.6) / μ_d=×0.85 / restitution=0 / contactOffset=0.01 / restOffset=0 /
  maxDepenVel=3(obj) / 指 prismatic drive kp=800 kd=40 / close=全閉(grip_sign=+1で upper limit)

対象は掴みやすい円柱（軸=鉛直Z＝重力方向。指は側面を水平に握る＝支持は純摩擦のみ）。
径 D とμを env でスイープし、(D, μ)→保持できたか の表を出す。オクラ質量 ~20g を既定に使う。

実行:
  ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/friction_grasp_probe.py
  # スイープ既定: 径∈{16,20,26,32}mm × μ∈{0.5,1.6,3.0}
  PROBE_DIAMS_MM=20 PROBE_MUS=1.6 PROBE_OBJ_MASS=0.02 \
    ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/friction_grasp_probe.py
"""

import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTHONNOUSERSITE", "1")  # user-site 混入回避（isaac-sim env を汚さない）

import numpy as np

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/docs/sim-setup")

# 物理設定（bridge の摩擦把持モードと同値の既定）
G1_USD = os.getenv("SIM_G1_USD", f"{REPO}/usd_file/g1-29dof-dex1-base-fix-usd/g1bag.usd")
MU_DEFAULTS = [float(x) for x in os.getenv("PROBE_MUS", "0.5,1.6,3.0").split(",")]
DIAMS_MM = [float(x) for x in os.getenv("PROBE_DIAMS_MM", "16,20,26,32").split(",")]
OBJ_MASS = float(os.getenv("PROBE_OBJ_MASS", "0.02"))  # [kg] オクラ ~20g
OBJ_H = float(os.getenv("PROBE_OBJ_H", "0.08"))  # [m] 円柱高さ（莢長 ~8cm）
CONTACT_OFFSET = float(os.getenv("SIM_CONTACT_OFFSET", "0.01"))
REST_OFFSET = float(os.getenv("SIM_REST_OFFSET", "0.0"))
MAX_DEPEN_VEL = float(os.getenv("SIM_MAX_DEPEN_VEL", "3.0"))
GRIP_KP = float(os.getenv("SIM_GRIP_KP", "800.0"))
GRIP_KD = float(os.getenv("SIM_GRIP_KD", "40.0"))
GRIP_CLOSE_FULL = float(os.getenv("SIM_GRIP_CLOSE_FULL", "4.4"))
GRIP_SIGN = float(os.getenv("SIM_GRIP_SIGN", "1.0"))
HEADLESS = os.getenv("PROBE_HEADLESS", "1") == "1"

# 右腕7関節の固定リーチ姿勢（IK target torso=(0.40,-0.16,0.05) を .venv で1回解いた値）。
# 手を体前方・右へ出し、ジョーの握り軸をなるべく水平に保つ（重力を純摩擦で支える最難条件）。
RIGHT7_REACH = [
    float(x)
    for x in os.getenv("PROBE_RIGHT7", "0.1926,0.0107,0.0219,-0.1247,-0.0029,-0.0889,0.0064").split(
        ","
    )
]

CANON_G1_29 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = CANON_G1_29[22:29]  # right_shoulder_pitch .. right_wrist_yaw


def main() -> int:
    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": HEADLESS})

    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    import omni.usd
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()

    add_reference_to_stage(usd_path=G1_USD, prim_path="/G1")
    # ArticulationRoot を探す
    art_root = "/G1"
    for p in Usd.PrimRange(stage.GetPrimAtPath("/G1")):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_root = p.GetPath().pathString
            break
    robot = SingleArticulation(prim_path=art_root, name="g1")
    world.reset()
    robot.initialize()

    dof_names = list(robot.dof_names)
    name_to_idx = {n: i for i, n in enumerate(dof_names)}
    gripper_idx = [i for i, n in enumerate(dof_names) if n not in set(CANON_G1_29)]
    print(
        f"[probe] num_dof={robot.num_dof} gripper(非canon)={[(i, dof_names[i]) for i in gripper_idx]}",
        flush=True,
    )

    # DOF limits（指 prismatic の開閉端）
    try:
        lim = np.asarray(robot._articulation_view.get_dof_limits()).reshape(-1, 2)
        dof_lo, dof_hi = lim[:, 0], lim[:, 1]
    except Exception:
        dp = robot.dof_properties
        dof_lo = np.asarray(dp["lower"], float).reshape(-1)
        dof_hi = np.asarray(dp["upper"], float).reshape(-1)
    for gi in gripper_idx:
        print(
            f"[probe] gripper dof[{gi}] {dof_names[gi]} limit=({dof_lo[gi]:.4f},{dof_hi[gi]:.4f})",
            flush=True,
        )

    # 指 prismatic に drive gains（bridge と同じ）
    ctrl = robot.get_articulation_controller()
    g = ctrl.get_gains()
    kps = np.asarray(g[0], float).reshape(-1)
    kds = np.asarray(g[1], float).reshape(-1)
    for gi in gripper_idx:
        kps[gi] = GRIP_KP
        kds[gi] = GRIP_KD
    ctrl.set_gains(kps=kps, kds=kds)

    # 右腕を固定リーチ姿勢へ（左腕/脚は 0）。数十ステップで定常させる。
    q_arm = np.asarray(robot.get_joint_positions(), float).reshape(-1).copy()
    for off, jn in enumerate(RIGHT_ARM_JOINTS):
        if jn in name_to_idx:
            q_arm[name_to_idx[jn]] = RIGHT7_REACH[off]

    def grip_target(frac: float) -> np.ndarray:
        """grip_q/close_full=frac に対応する全DOF目標（腕は固定リーチ、指は開閉補間）。"""
        q = q_arm.copy()
        for gi in gripper_idx:
            lo, hi = float(dof_lo[gi]), float(dof_hi[gi])
            closed = hi if GRIP_SIGN > 0 else lo
            opened = lo if GRIP_SIGN > 0 else hi
            q[gi] = opened + frac * (closed - opened)
        return q

    def step(n: int, frac: float) -> None:
        act = ArticulationAction(joint_positions=grip_target(frac))
        for _ in range(n):
            robot.apply_action(act)
            world.step(render=not HEADLESS)

    # 開いた状態で腕を寄せて定常化
    step(120, frac=0.0)

    def right_hand_bodies():
        out = []
        for p in Usd.PrimRange(stage.GetPrimAtPath("/G1")):
            if p.HasAPI(UsdPhysics.RigidBodyAPI) and "right_hand" in p.GetName().lower():
                out.append(p.GetPath().pathString)
        return out

    def wpos(path):
        xc2 = UsdGeom.XformCache(Usd.TimeCode.Default())
        t = xc2.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()
        return np.array([float(t[0]), float(t[1]), float(t[2])])

    # 検査モード: Dex1 の全 right_hand リンクの位置を「開」「閉」で測り、実際に動く指パッドと
    # 閉じ幅を特定して終了する。PROBE_INSPECT=1 で起動。
    if os.getenv("PROBE_INSPECT", "0") == "1":
        rhb = right_hand_bodies()
        popen = {p: wpos(p) for p in rhb}
        step(120, frac=1.0)  # 全閉へ
        pclose = {p: wpos(p) for p in rhb}
        print(f"[inspect] right_hand rigidbody {len(rhb)}個:", flush=True)
        for p in rhb:
            nm = p.split("/")[-1]
            d = pclose[p] - popen[p]
            print(
                f"[inspect]  {nm:<28} open={np.round(popen[p], 4).tolist()} "
                f"→close={np.round(pclose[p], 4).tolist()} 移動={np.linalg.norm(d) * 1000:.1f}mm",
                flush=True,
            )
        # 最も動く2リンク＝可動指。閉時の距離＝実際の閉じ幅。
        moved = sorted(rhb, key=lambda p: np.linalg.norm(pclose[p] - popen[p]), reverse=True)
        if len(moved) >= 2:
            a, b = moved[0], moved[1]
            sep_open = np.linalg.norm(popen[a] - popen[b])
            sep_close = np.linalg.norm(pclose[a] - pclose[b])
            print(
                f"[inspect] 可動2指={a.split('/')[-1]},{b.split('/')[-1]} "
                f"開隙間={sep_open * 1000:.1f}mm → 閉隙間={sep_close * 1000:.1f}mm "
                f"(この間の径の物体しか挟めない)",
                flush=True,
            )
        print("INSPECT_DONE", flush=True)
        sim_app.close()
        return 0

    # 握り隙間の中心と軸: Dex1 は2本指×各3リンク（Link1_1..1_3 / Link2_1..2_3）。
    # 対向する指先ペア＝各指の base から最遠のリンク（Link1_3 / Link2_3）。その中点が握り中心。
    def world_pos(path):
        xc2 = UsdGeom.XformCache(Usd.TimeCode.Default())
        t = xc2.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()
        return np.array([float(t[0]), float(t[1]), float(t[2])])

    hand_base = None
    finger1, finger2 = [], []  # (path, pos)
    for p in Usd.PrimRange(stage.GetPrimAtPath("/G1")):
        nm = p.GetName()
        low = nm.lower()
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and "right_hand" in low:
            if "base" in low:
                hand_base = p.GetPath().pathString
            elif "link1" in low:
                finger1.append(p.GetPath().pathString)
            elif "link2" in low:
                finger2.append(p.GetPath().pathString)
    if hand_base is None or not finger1 or not finger2:
        print(
            f"[probe] ERROR: 指リンク検出失敗 base={hand_base} f1={finger1} f2={finger2}",
            flush=True,
        )
        sim_app.close()
        return 1
    hb = world_pos(hand_base)
    tip1 = max(finger1, key=lambda p: np.linalg.norm(world_pos(p) - hb))  # 各指の base 最遠＝指先
    tip2 = max(finger2, key=lambda p: np.linalg.norm(world_pos(p) - hb))
    t1, t2 = world_pos(tip1), world_pos(tip2)
    gap_center = (t1 + t2) / 2.0
    squeeze_axis = t2 - t1
    saxis = squeeze_axis / (np.linalg.norm(squeeze_axis) + 1e-9)
    print(
        f"[probe] 指先ペア tip1={tip1.split('/')[-1]}@{np.round(t1, 4).tolist()} "
        f"tip2={tip2.split('/')[-1]}@{np.round(t2, 4).tolist()}",
        flush=True,
    )
    print(
        f"[probe] gap_center={np.round(gap_center, 4).tolist()} "
        f"squeeze_axis={np.round(saxis, 3).tolist()} 開隙間={np.linalg.norm(squeeze_axis) * 1000:.1f}mm "
        f"(水平成分={np.hypot(saxis[0], saxis[1]):.2f} / 鉛直成分={abs(saxis[2]):.2f})",
        flush=True,
    )

    def make_material(mu):
        muS = float(mu)
        muD = round(muS * 0.85, 3)
        mp = f"/World/PM/mu_{str(mu).replace('.', '_')}"
        gm = UsdShade.Material.Define(stage, mp)
        ga = UsdPhysics.MaterialAPI.Apply(gm.GetPrim())
        ga.CreateStaticFrictionAttr(muS)
        ga.CreateDynamicFrictionAttr(muD)
        ga.CreateRestitutionAttr(0.0)
        return gm

    # グリッパ側 collider にも μ を bind（接触μは両面平均）＋ contact/rest offset
    def bind_gripper_material(gm):
        for p in Usd.PrimRange(stage.GetPrimAtPath("/G1")):
            if p.HasAPI(UsdPhysics.CollisionAPI) and "right_hand" in p.GetName().lower():
                UsdShade.MaterialBindingAPI.Apply(p).Bind(
                    gm, UsdShade.Tokens.weakerThanDescendants, "physics"
                )
                cx = PhysxSchema.PhysxCollisionAPI.Apply(p)
                cx.CreateContactOffsetAttr(CONTACT_OFFSET)
                cx.CreateRestOffsetAttr(REST_OFFSET)

    def spawn_object(diam_m, gm):
        r = diam_m / 2.0
        path = "/World/TestObj"
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
        cyl = UsdGeom.Cylinder.Define(stage, path)
        cyl.CreateRadiusAttr(r)
        cyl.CreateHeightAttr(OBJ_H)
        cyl.CreateAxisAttr("Z")  # 軸=鉛直（重力方向）＝側面を水平に握る
        prim = cyl.GetPrim()
        # 隙間のど真ん中へ（完璧配置）
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in gap_center]))
        UsdPhysics.CollisionAPI.Apply(prim)
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        ma = UsdPhysics.MassAPI.Apply(prim)
        ma.CreateMassAttr(OBJ_MASS)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            gm, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
        cx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cx.CreateContactOffsetAttr(CONTACT_OFFSET)
        cx.CreateRestOffsetAttr(REST_OFFSET)
        pr = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        pr.CreateMaxDepenetrationVelocityAttr(MAX_DEPEN_VEL)
        pr.CreateStabilizationThresholdAttr(0.001)
        pr.CreateSleepThresholdAttr(0.0)
        # 接触力を報告させる（法線/摩擦の集計に必須）
        cr = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        cr.CreateThresholdAttr(0.0)
        return path, rb

    def set_kinematic(rb, flag):
        a = rb.GetKinematicEnabledAttr()
        if not a:
            a = rb.CreateKinematicEnabledAttr(False)
        a.Set(bool(flag))

    # 接触力の計測（法線＝クランプ力 / 接線＝摩擦力）
    # PhysX contact report を購読し、test object が絡む接触の impulse を法線・接線に分解して集計。
    # 力 = Σimpulse / (計測窓の sim 時間)。窓開始で reset → 窓終了で平均力を読む。
    OBJ_PATH = "/World/TestObj"
    _dt = float(world.get_physics_dt())
    _accum = {"n_imp": 0.0, "f_imp": 0.0, "npair": 0}

    def _on_contact(headers, data):
        for h in headers:
            try:
                a = PhysicsSchemaTools.intToSdfPath(h.actor0).pathString
                b = PhysicsSchemaTools.intToSdfPath(h.actor1).pathString
            except Exception:
                continue
            if OBJ_PATH not in a and OBJ_PATH not in b:
                continue
            off, cnt = h.contact_data_offset, h.num_contact_data
            for i in range(off, off + cnt):
                cd = data[i]
                iv = np.array([cd.impulse[0], cd.impulse[1], cd.impulse[2]], float)
                nv = np.array([cd.normal[0], cd.normal[1], cd.normal[2]], float)
                n_along = float(np.dot(iv, nv))
                _accum["n_imp"] += abs(n_along)
                _accum["f_imp"] += float(np.linalg.norm(iv - n_along * nv))
                _accum["npair"] += 1

    _contact_ok = True
    try:
        from omni.physx import get_physx_simulation_interface

        _csub = get_physx_simulation_interface().subscribe_contact_report_events(_on_contact)
    except Exception as _e:
        _contact_ok = False
        print(
            f"[probe] WARN: contact report 購読不可 → 接触力は測れず z のみで判定: {_e}", flush=True
        )

    def measure_forces(rb_path, nsteps, frac):
        """nsteps ステップ回して、その窓の平均 法線力/摩擦力[N] を返す。"""
        _accum["n_imp"] = _accum["f_imp"] = 0.0
        _accum["npair"] = 0
        act = ArticulationAction(joint_positions=grip_target(frac))
        for _ in range(nsteps):
            robot.apply_action(act)
            world.step(render=not HEADLESS)
        win = max(nsteps * _dt, 1e-6)
        return _accum["n_imp"] / win, _accum["f_imp"] / win

    weight = OBJ_MASS * 9.81  # [N] 支えるべき重量
    results = []
    for mu in MU_DEFAULTS:
        gm = make_material(mu)
        bind_gripper_material(gm)
        for dmm in DIAMS_MM:
            diam = dmm / 1000.0
            # 開く → オブジェクトをど真ん中へ kinematic 固定
            step(40, frac=0.0)
            path, rb = spawn_object(diam, gm)
            set_kinematic(rb, True)
            for _ in range(10):
                world.step(render=not HEADLESS)
            # ① 握る: kinematic 固定のまま指を全閉方向へ押し込む＝クランプ力(法線力)を溜める。
            #    最後の窓で法線力 N_clamp を計測（μ×N_clamp が支えられる摩擦上限）。
            for _k in range(50):
                robot.apply_action(ArticulationAction(joint_positions=grip_target(1.0)))
                world.step(render=not HEADLESS)
            n_clamp, _ = measure_forces(path, 30, frac=1.0)
            world_pos(path)[2]
            # ② dynamic 解放（重力 ON）→ 純摩擦で保持できるか。摩擦力 F_fric（接線）を計測。
            set_kinematic(rb, False)
            for _ in range(5):
                world.step(render=not HEADLESS)
            z0 = world_pos(path)[2]
            f_normal, f_fric = measure_forces(path, 150, frac=1.0)  # ~2.5s の平均
            for _k in range(30):  # 追加で沈静化を見る
                robot.apply_action(ArticulationAction(joint_positions=grip_target(1.0)))
                world.step(render=not HEADLESS)
            zf = world_pos(path)[2]
            dz = zf - z0
            # 成功指標（複合）:
            #   (a) z 保持: |Δz| < 2cm  (b) 摩擦余裕: μ×N_clamp ≥ 重量  (c) 接触継続: 解放後も法線力>0
            fric_capacity = mu * n_clamp  # [N] 使える最大摩擦
            margin = fric_capacity / weight if weight > 0 else 0.0
            z_ok = abs(dz) < 0.02
            contact_ok = f_normal > 0.05  # 解放後も握れている（接触が続いている）
            held = z_ok and contact_ok
            qcur = np.asarray(robot.get_joint_positions(), float).reshape(-1)
            fpos = [round(float(qcur[gi]), 4) for gi in gripper_idx]
            results.append((mu, dmm, held, dz, n_clamp, f_fric, margin, f_normal))
            print(
                f"[probe] μ={mu} D={dmm}mm → {'✅HELD' if held else '❌SLIP'} "
                f"Δz={dz * 1000:+.1f}mm | N_clamp={n_clamp:.2f}N μN={fric_capacity:.2f}N "
                f"vs 重量={weight:.2f}N (余裕×{margin:.1f}) | 解放後 法線={f_normal:.2f}N 摩擦={f_fric:.2f}N "
                f"finger={fpos}",
                flush=True,
            )
            stage.RemovePrim(path)

    print("\n[probe] ===== 摩擦把持プローブ結果表 =====", flush=True)
    print(
        f"[probe] 支えるべき重量 = {weight:.2f} N (質量 {OBJ_MASS * 1000:.0f}g)"
        + ("" if _contact_ok else "  ※contact report 不可＝力は0表示、z のみ有効"),
        flush=True,
    )
    print("[probe]  μ    D(mm) 結果  Δz(mm)  N_clamp  μN(=摩擦上限)  余裕   解放後摩擦", flush=True)
    for mu, dmm, held, dz, n_clamp, f_fric, margin, f_normal in results:
        print(
            f"[probe]  {mu:<4} {dmm:<4} {'HELD' if held else 'SLIP'}  {dz * 1000:+6.1f}  "
            f"{n_clamp:6.2f}N  {mu * n_clamp:7.2f}N     ×{margin:<4.1f} {f_fric:6.2f}N",
            flush=True,
        )
    n_held = sum(1 for r in results if r[2])
    print(
        f"[probe] 保持成立 {n_held}/{len(results)}  "
        f"（μ×N_clamp ≥ 重量 が摩擦で掴める条件。z 保持と接触継続の両立で HELD 判定）",
        flush=True,
    )
    print("PROBE_DONE", flush=True)
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
