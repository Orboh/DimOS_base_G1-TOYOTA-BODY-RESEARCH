#!/usr/bin/env python3
"""歩行ポリシー単体検証（A方式・段階1）: base-free g1bag を unitree_rl_lab の velocity policy で
「立つ→cmd_velで歩く」を確認する最小 Isaac スクリプト（収穫は無し）。

設計（[[12-検証計画-sim]] / LocoClient 相当ラッパーの中身）:
  - policy = unitree_rl_lab/deploy g1_29dof velocity/v0/policy.onnx（学習済, obs[480]→action[29]）
  - ★obs[480] = **per-term 履歴連結** [ang_vel×0.2 の履歴5 | proj_grav×5 | cmd×5 | jpos_rel×5 |
      jvel_rel×0.05×5 | last_action×5]（term 内は最古→最新）。deploy.yaml に use_gym_history が
      無い＝C++ 既定 false のこのレイアウトが正（per-step [96×5] だと発散→転倒）。
  - action[29] → q_target[motor=joint_ids_map[i]] = action[i]*0.25 + default_joint_pos[i]
    （default/scale/offset は policy slot 順。**stiffness/damping はモーターSDK順**＝jmap不要）
  - 50Hz: physics_dt=1/200 × 4substep = 20ms。armature=0.01 全関節（訓練/mujoco と一致）。
  - g1bag は base-fix なので root_joint を jointEnabled=False＋RemoveAPI(ArticulationRoot)、
    root API は **pelvis（実剛体）へ**。physxArticulation の solver 32 反復も pelvis へ移植。
  - 実績: 立位 20s 転倒0 / 前進 vx=0.3 で 20s 6.9m（実測 0.35m/s, 横ズレ 0.3m）転倒0。

実行:
  PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
    WALK_VX=0.3 WALK_SECS=20 WALK_BASE_Z=0.80 WALK_SETTLE_STEPS=5 \
    ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/sim_walk_policy.py [--gui]
  WALK_ROOM=0 で部屋なし（地面のみ）。VX=0 で立位のみ。
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
ROOM = f"{REPO}/usd_file/chinou_center.usd"
G1 = f"{REPO}/usd_file/g1-29dof-dex1-base-fix-usd/g1bag.usd"
POLICY = f"{REPO}/usd_file/walk_policy/policy.onnx"

import numpy as np
import yaml

# deploy.yaml（policy 仕様）。リポ同梱 params をローカルに置いた物を読む（無ければ既知値）。
DEPLOY_YAML = f"{REPO}/usd_file/walk_policy/deploy.yaml"

# 正準 G1 29 関節名（motor order = CANON_G1_29, lowstate/motor_cmd 順）
CANON_G1_29 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    vx = float(os.getenv("WALK_VX", "0.0"))
    vy = float(os.getenv("WALK_VY", "0.0"))
    wz = float(os.getenv("WALK_WZ", "0.0"))
    secs = float(os.getenv("WALK_SECS", "8"))

    dp = yaml.safe_load(open(DEPLOY_YAML))
    jmap = list(dp["joint_ids_map"])                  # policy slot i -> motor index
    default_q = np.array(dp["default_joint_pos"], dtype=np.float32)  # policy order
    stiffness = np.array(dp["stiffness"], dtype=np.float32)
    damping = np.array(dp["damping"], dtype=np.float32)
    act_scale = np.array(dp["actions"]["JointPositionAction"]["scale"], dtype=np.float32)
    act_offset = np.array(dp["actions"]["JointPositionAction"]["offset"], dtype=np.float32)
    obs_ang_scale = np.array(dp["observations"]["base_ang_vel"]["scale"], dtype=np.float32)
    obs_jvel_scale = np.array(dp["observations"]["joint_vel_rel"]["scale"], dtype=np.float32)
    HIST = int(dp["observations"]["base_ang_vel"]["history_length"])
    step_dt = float(dp["step_dt"])
    print(f"[walk] policy obs hist={HIST} step_dt={step_dt} cmd=({vx},{vy},{wz})", flush=True)

    import onnxruntime as ort
    sess = ort.InferenceSession(POLICY, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": not args.gui})
    from pxr import Usd, UsdGeom, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage, open_stage
    import omni.usd
    try:
        from isaacsim.core.prims import SingleArticulation as ArtCls
    except Exception:  # noqa: BLE001
        from isaacsim.core.api.articulations import Articulation as ArtCls  # type: ignore
    from isaacsim.core.utils.types import ArticulationAction

    if os.getenv("WALK_ROOM", "1") == "1":
        open_stage(ROOM)
    else:
        print("[walk] WALK_ROOM=0: 部屋なし（ground plane のみ）で切り分け", flush=True)
    # ★physics_dt=1/200(5ms): policy は step_dt=0.02(50Hz) 前提。既定 1/60 のままだと 4step=66ms≈15Hz
    #   になり balance が崩れる。5ms×4step=20ms で policy の 50Hz と一致させる。
    PHYS_DT = 1.0 / 200.0
    world = World(stage_units_in_meters=1.0, physics_dt=PHYS_DT, rendering_dt=1.0 / 50.0)
    stage = omni.usd.get_context().get_stage()
    # ★床コライダー保証: floating base が確実に立てる足場を z=0 に1枚追加（chinou のコライダー頼みに
    # せず、すり抜け/埋もれを防ぐ）。摩擦は既定（足裏は物理材で別途）。
    try:
        world.scene.add_default_ground_plane(z_position=0.0)
        print("[walk] ground plane @z=0 追加（足場保証）", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[walk] ground plane warn: {_e}", flush=True)
    add_reference_to_stage(usd_path=G1, prim_path="/G1")
    g1 = stage.GetPrimAtPath("/G1")

    # ★base-free 化（参照レイヤ対応）: root_joint = world(body0=[])→pelvis(body1) の FixedJoint で、
    #   ここに ArticulationRootAPI が乗っている＝これが world 固定＆fixed-base articulation の正体。
    #   参照 prim は RemovePrim/SetActive が効かない（サイレント失敗）ので、属性/APIの override で直す:
    #     (1) jointEnabled=False で FixedJoint の world 固定制約を無効化（base を解放）
    #     (2) root_joint から ArticulationRootAPI を除去（RemoveAPI は削除 listOp を authored＝参照にも効く）
    #     (3) floating articulation の root API を **base link=pelvis（実剛体）** に付与
    #   注意: (2) を省くと root 二重（nested）で全リンク Invalid transform→NaN。
    #        (3) を top Xform /G1（純 Xform・非剛体）にすると root link が曖昧→無効慣性→やはり NaN。
    #        → root API は必ず実剛体 pelvis に置く。imu/hand の内部 FixedJoint は温存。
    pelvis_path = next((p.GetPath() for p in Usd.PrimRange(g1) if p.GetName() == "pelvis"), None)
    rj_paths = [p.GetPath() for p in Usd.PrimRange(g1) if p.GetName() == "root_joint"]
    for rp in rj_paths:
        jp = stage.GetPrimAtPath(rp)
        UsdPhysics.Joint(jp).CreateJointEnabledAttr(False)
        jp.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    g1.RemoveAPI(UsdPhysics.ArticulationRootAPI)  # 念のため /G1 側にも付いていたら外す
    pelvis0 = stage.GetPrimAtPath(pelvis_path) if pelvis_path else None
    if pelvis0 and not pelvis0.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(pelvis0)
    # ★solver 設定を pelvis へ移植: authored の physxArticulation 設定（solver 32 反復・self-collision
    #   無効）は root_joint に付いており、root を pelvis に移すと **PhysX 既定の反復4に落ちる**。
    #   反復4では剛PD+33kg の接触が解けず関節が実効的に柔らかくなり「像ごと崩れる」。必ず移植する。
    if pelvis0:
        from pxr import PhysxSchema
        pa = PhysxSchema.PhysxArticulationAPI.Apply(pelvis0)
        pa.CreateSolverPositionIterationCountAttr(32)
        pa.CreateSolverVelocityIterationCountAttr(4)
        pa.CreateEnabledSelfCollisionsAttr(False)
        print("[walk] pelvis に physxArticulation 設定移植: solverPosIter=32 velIter=4 selfColl=False",
              flush=True)
        # ★armature 注入: 訓練(IsaacLab unitree.py)も sim2sim(mujoco XML)も **全関節 armature=0.01**
        #   前提。USD は armature 無し(0)のため、軽い足首リンクが接地でチャタリング→policy が増幅→
        #   発振転倒（実測: ankle 2.4→30 rad/s に発散）。訓練と同じ 0.01 を全29関節へ。
        _arm = float(os.getenv("WALK_ARMATURE", "0.01"))  # [kg·m^2]
        _n_arm = 0
        for p in Usd.PrimRange(g1):
            if p.GetName() in CANON_G1_29:
                PhysxSchema.PhysxJointAPI.Apply(p).CreateArmatureAttr(_arm)
                _n_arm += 1
        print(f"[walk] armature={_arm} を {_n_arm}/29 関節に注入", flush=True)
    _rj_hasapi = [bool(stage.GetPrimAtPath(rp).HasAPI(UsdPhysics.ArticulationRootAPI)) for rp in rj_paths]
    _pv_hasapi = bool(pelvis0.HasAPI(UsdPhysics.ArticulationRootAPI)) if pelvis0 else False
    print(f"[walk] base-free 化: jointEnabled=False / root_joint.artAPI={_rj_hasapi}（False望む）"
          f" / pelvis.artAPI={_pv_hasapi}（True望む） pelvis={pelvis_path}", flush=True)

    # 接地までの持ち上げ量（shift）を bbox から算出。★USD の xform は触らない（xformOpOrder 競合回避）。
    #   持ち上げは reset 後に articulation API（set_world_pose）で行う。
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    lift = -float(bbc.ComputeWorldBound(g1).ComputeAlignedRange().GetMin()[2]) + 0.02
    print(f"[walk] 接地 shift={lift:.3f}（reset 後に set_world_pose で適用）", flush=True)

    art_root = pelvis_path.pathString if pelvis_path else "/G1"
    robot = ArtCls(prim_path=art_root, name="g1")
    world.scene.add(robot)
    world.reset()
    # ★重要: reset 後は **一度も step しない** まま先に配置する。authored 姿勢は pelvis@z=0 で
    #   足が world z=-0.757（地面 z=0 を踏み抜き）＝この状態で step すると巨大反力で射出され 29°傾く。
    #   → 配置（直立・高さ 0.80）を済ませてから初めて step する。

    def pelvis_world():
        # ★PhysX 直読み: UsdGeom.XformCache は物理更新を USD に反映しない（authored 値=静止）ため
        #   base の位置/姿勢は必ず articulation API（get_world_pose）から取る。quat=(w,x,y,z)。
        pos, quat = robot.get_world_pose()
        return (np.asarray(pos, dtype=np.float32).reshape(-1)[:3],
                np.asarray(quat, dtype=np.float32).reshape(-1)[:4])

    # authored pelvis は identity 向き（オフライン確認済）＝直立。配置にはこれを使う。
    q_upright = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    dof_names = list(robot.dof_names)
    # motor(CANON) index -> Isaac dof index
    canon_to_isaac = {ci: dof_names.index(nm) for ci, nm in enumerate(CANON_G1_29) if nm in dof_names}
    isaac_idx_for_motor = [canon_to_isaac.get(m) for m in range(29)]
    print(f"[walk] num_dof={robot.num_dof} mapped {len(canon_to_isaac)}/29 clean_quat={np.round(q_upright,3).tolist()}", flush=True)
    print(f"[diag] isaac dof_names={dof_names}", flush=True)
    print(f"[diag] isaac_idx_for_motor(CANON順)={isaac_idx_for_motor}", flush=True)

    # ★PD ゲインを **先に** 設定（剛性ゼロのまま gravity で step すると脚が即崩れて転倒姿勢→NaN）。
    #   WALK_KP_SCALE: ゲイン倍率ノブ（単位解釈ズレの切り分け用。USD drive は度単位・API は rad 単位で
    #   混同すると実効剛性が 1/57 になり「ぐしゃ崩れ(自由落下)」になる）。
    kp_scale = float(os.getenv("WALK_KP_SCALE", "1.0"))
    nd = robot.num_dof
    kp = np.zeros(nd, dtype=np.float32)
    kd = np.zeros(nd, dtype=np.float32)
    # ★★deploy.yaml の stiffness/damping は **モーター(SDK)順**（公式C++ State_RLBase.h:
    #   `motor_cmd()[i].kp() = joint_stiffness[i]`＝jmap を通さない。articulation.h に "// sdk order"）。
    #   default/scale/offset は policy slot 順（jmap 経由）という非対称仕様に注意。
    #   gains に jmap を通すと右膝40(本来150)/右股roll40(本来100)/腰100(本来200)とスクランブルされ、
    #   右脚から崩壊する（実際にこのバグで転倒していた）。
    for m in range(29):
        ii = isaac_idx_for_motor[m]
        if ii is not None:
            kp[ii] = stiffness[m] * kp_scale
            kd[ii] = damping[m] * kp_scale
    try:
        robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
        # ★読み返し検証: 実際に articulation に入った値を確認（set が単位変換/黙殺されていないか）。
        try:
            _rk, _rd = robot.get_articulation_controller().get_gains()
            _lk = isaac_idx_for_motor[3]  # left_knee
            print(f"[walk] PD gains 設定 kp_scale={kp_scale} | set knee kp={kp[_lk]:.1f} kd={kd[_lk]:.2f} "
                  f"→ 読返し kp={float(np.asarray(_rk).reshape(-1)[_lk]):.3f} "
                  f"kd={float(np.asarray(_rd).reshape(-1)[_lk]):.3f}", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[walk] PD gains 設定（読返し不可: {_e}）", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[walk] set_gains warn: {_e}", flush=True)

    # default 関節姿勢（policy slot i は motor jmap[i]）を motor→Isaac dof へ。
    q_init = np.asarray(robot.get_joint_positions(), dtype=np.float32).reshape(-1).copy()
    for i in range(29):
        ii = isaac_idx_for_motor[jmap[i]]
        if ii is not None:
            q_init[ii] = default_q[i]
    robot.set_joint_positions(q_init)

    # ★接地配置: clean な直立 quat のまま pelvis を絶対高さ base_z0 へ（feet が床直上）。
    #   ここまで gravity で step していないので倒れていない＝直立配置が成立。
    base_z0 = float(os.getenv("WALK_BASE_Z", "0.80"))
    _p, _ = robot.get_world_pose()
    _p = np.asarray(_p, dtype=np.float32).reshape(-1)
    robot.set_world_pose(position=np.array([_p[0], _p[1], base_z0], dtype=np.float32),
                         orientation=q_upright)
    robot.set_joint_positions(q_init)  # 再度 default（取りこぼし防止）
    print(f"[walk] 接地配置: base_z0={base_z0:.3f}", flush=True)

    # ★settle: default 姿勢を PD 保持して足を接地させるだけ（既定25step）。**倒れ始める前**に policy へ
    #   渡すのが要点（静的保持のみでは ~s=40 から倒れ始める＝実測）。25step で z≈0.79 の接地立位。
    #   角速度判定は使わない（get_angular_velocity が不正値のため。obs は quat 有限差分で別途取得）。
    settle_n = int(os.getenv("WALK_SETTLE_STEPS", "25"))
    for _s in range(settle_n):
        robot.apply_action(ArticulationAction(joint_positions=q_init))
        world.step(render=args.gui)
        if _s % 50 == 49:  # 長い settle の観測用（静的保持がどこまで持つか）
            _pz = float(pelvis_world()[0][2])
            print(f"[walk]   settle s={_s+1} z={_pz:.3f}", flush=True)
    try:  # 残留速度を零化（policy に持ち込まない）
        robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=np.float32))
    except Exception:  # noqa: BLE001
        pass
    _p0, _q0 = pelvis_world()
    print(f"[walk] settle 後({settle_n}step) pelvis_z={float(np.asarray(_p0).reshape(-1)[2]):.3f} "
          f"quat={np.round(np.asarray(_q0).reshape(-1),3).tolist()}（直立=w≈1）", flush=True)
    # --- 診断: get_world_pose の生構造・base 高さの別ソース・脚関節が動いているか ---
    _wp = robot.get_world_pose()
    print(f"[diag] world_pose: e0={np.round(np.asarray(_wp[0]).reshape(-1),4).tolist()} "
          f"e1={np.round(np.asarray(_wp[1]).reshape(-1),4).tolist()}", flush=True)
    print(f"[diag] art_root={art_root} num_dof={robot.num_dof} num_bodies="
          f"{getattr(robot, 'num_bodies', '?')}", flush=True)
    try:  # 各リンクのワールド座標（base=最上位リンクの z が真の高さ）
        _lp = np.asarray(robot.get_link_positions() if hasattr(robot, 'get_link_positions')
                         else robot.get_world_poses()[0]).reshape(-1, 3)
        print(f"[diag] link_z min/max={float(_lp[:,2].min()):.3f}/{float(_lp[:,2].max()):.3f} "
              f"(root link_z={float(_lp[0,2]):.3f})", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[diag] link pos API 無し: {_e}", flush=True)
    _knee = isaac_idx_for_motor[3]  # left_knee（motor idx3）
    _q0 = float(np.asarray(robot.get_joint_positions()).reshape(-1)[_knee]) if _knee is not None else 0.0
    print(f"[diag] left_knee isaac_idx={_knee} q={_q0:.3f}（default={float(default_q[jmap.index(3)]):.3f}）", flush=True)

    # 観測履歴バッファ（最古→最新, HIST 個）。各 step 96 次元。
    last_action = np.zeros(29, dtype=np.float32)
    cmd = np.array([vx, vy, wz], dtype=np.float32)

    def motor_q_dq():
        q = np.asarray(robot.get_joint_positions(), dtype=np.float32).reshape(-1)
        dq = np.asarray(robot.get_joint_velocities(), dtype=np.float32).reshape(-1)
        mq = np.zeros(29, dtype=np.float32); mdq = np.zeros(29, dtype=np.float32)
        for m in range(29):
            ii = isaac_idx_for_motor[m]
            if ii is not None:
                mq[m] = q[ii]; mdq[m] = dq[ii]
        return mq, mdq

    # ★角速度は get_angular_velocity() を使わない: この articulation では静止時でも ~6.5rad/s の
    #   不正値を返し（z 一定でも |angvel|=6.5）、obs を汚染して policy を暴走させる。代わりに base
    #   quaternion の有限差分で body 系角速度を推定する（dt=step_dt）。prev を保持。
    _prev_q = [None]

    def base_obs():
        # base 角速度（body 系）と projected_gravity（body 系の重力方向）
        pos, quat = pelvis_world()  # quat (w,x,y,z) ＝ PhysX 実姿勢
        q = np.asarray(quat, dtype=np.float32).reshape(-1)[:4]
        qw, qx, qy, qz = [float(v) for v in q]
        # world→body 回転（R は body→world なので body 系へは R^T を掛ける）
        R = np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
        ], dtype=np.float32)
        g_body = R.T @ np.array([0, 0, -1], dtype=np.float32)
        # 角速度 = 前回 quat からの相対回転を angle-axis 化 / dt → world 系 → body 系
        if _prev_q[0] is None:
            w_body = np.zeros(3, dtype=np.float32)
        else:
            qp = _prev_q[0]
            # Δq = q ⊗ conj(qp)（world 系の増分回転）
            pw, px, py, pz = -qp[0], qp[1], qp[2], qp[3]  # conj は虚部反転だが下で符号を合わせる
            cw, cx, cy, cz = qp[0], -qp[1], -qp[2], -qp[3]  # conj(qp)
            dw = qw*cw - qx*cx - qy*cy - qz*cz
            dx = qw*cx + qx*cw + qy*cz - qz*cy
            dy = qw*cy - qx*cz + qy*cw + qz*cx
            dz = qw*cz + qx*cy - qy*cx + qz*cw
            vnorm = float(np.sqrt(dx*dx + dy*dy + dz*dz))
            angle = 2.0 * float(np.arctan2(vnorm, abs(dw)))
            if angle > np.pi:
                angle -= 2.0 * np.pi
            if vnorm > 1e-8:
                sgn = 1.0 if dw >= 0 else -1.0
                axis = np.array([dx, dy, dz], dtype=np.float32) * (sgn / vnorm)
                w_world = axis * (angle / step_dt)
            else:
                w_world = np.zeros(3, dtype=np.float32)
            w_body = (R.T @ w_world).astype(np.float32)
        _prev_q[0] = q.copy()
        return w_body, g_body

    # ★★obs レイアウト＝ **per-term 履歴連結**（[ang×5 | grav×5 | cmd×5 | jpos×5 | jvel×5 | act×5]）。
    #   deploy.yaml に use_gym_history キーが無い＝C++ 既定 false → observation_manager.h は
    #   per-step フレーム連結ではなく「各 term の履歴5個を term ごとに平坦化して連結」する
    #   （manager_term_cfg.h get()）。per-step [96×5] だと policy 入力が全て位置ズレし、静止でも
    #   action が発散して転倒する（実測: 0.7→8.4 に単調発散）。各 term は deque で履歴保持、
    #   初回は C++ reset() と同じく同値で埋める。
    from collections import deque as _deque
    _TERM_ORDER = ["ang", "grav", "cmd", "jpos", "jvel", "act"]
    _t_hist: dict[str, "_deque[np.ndarray]"] = {k: _deque(maxlen=HIST) for k in _TERM_ORDER}

    def build_obs():
        mq, mdq = motor_q_dq()
        # policy 順へ: slot i = motor jmap[i]
        jpos_rel = np.array([mq[jmap[i]] - default_q[i] for i in range(29)], dtype=np.float32)
        jvel_rel = np.array([mdq[jmap[i]] for i in range(29)], dtype=np.float32) * obs_jvel_scale
        w, g_body = base_obs()
        terms = {"ang": (w * obs_ang_scale).astype(np.float32), "grav": g_body.astype(np.float32),
                 "cmd": cmd.astype(np.float32), "jpos": jpos_rel, "jvel": jvel_rel,
                 "act": last_action.astype(np.float32)}
        for k in _TERM_ORDER:
            if not _t_hist[k]:  # 初回: 履歴を同値で埋める（C++ reset() と同じ）
                for _ in range(HIST):
                    _t_hist[k].append(terms[k].copy())
            else:
                _t_hist[k].append(terms[k].copy())  # maxlen=HIST が最古を自動排出
        obs = np.concatenate([np.concatenate(list(_t_hist[k])) for k in _TERM_ORDER])
        return obs.astype(np.float32)[None, :]  # [1,480] per-term（term 内は最古→最新）

    # === 診断A: 「真の直立 obs」(grav=[0,0,-1], 他=0) を policy に通した理想 action。
    #   これが ~0 なら onnx 配線・スケールは正常＝直立で default を保つ。大きいなら onnx 側の解釈ズレ。
    _nom = np.concatenate([np.zeros(3 * HIST), np.tile([0, 0, -1], HIST), np.zeros(3 * HIST),
                           np.zeros(29 * HIST), np.zeros(29 * HIST), np.zeros(29 * HIST)]
                          ).astype(np.float32)[None, :]  # per-term レイアウト
    _a_nom = sess.run(None, {in_name: _nom})[0].reshape(-1)[:29].astype(np.float32)
    print(f"[diag] 理想action(直立obs) |max|={np.abs(_a_nom).max():.3f} "
          f"legs(slot0-11)={np.round(_a_nom[:12], 2).tolist()}", flush=True)

    hold_ideal = os.getenv("WALK_HOLD_IDEAL", "0") == "1"  # 開ループ検証: 理想立位actionを固定保持
    if hold_ideal:
        print("[walk] ※WALK_HOLD_IDEAL=1: policy出力を使わず理想立位actionを固定保持（マッピング検証）", flush=True)

    print(f"[walk] loop start（{secs}s, GUI={args.gui}）", flush=True)
    import time as _t
    t0 = _t.time()
    step = 0
    while _t.time() - t0 < secs and sim_app.is_running():
        obs = build_obs()
        action = sess.run(None, {in_name: obs})[0].reshape(-1)[:29].astype(np.float32)
        if hold_ideal:
            action = _a_nom.copy()  # 理想立位action（直立obsに対する出力）を固定適用
        # === 診断B: 最初の10step の obs 実値と action を dump ＋ 発振源特定（|action|/|jvel| 上位3関節）。
        if step < 10:
            _w, _g = base_obs()
            _mq, _mdq = motor_q_dq()
            _jpr = np.array([_mq[jmap[i]] - default_q[i] for i in range(29)], dtype=np.float32)
            _jvrw = np.array([_mdq[jmap[i]] for i in range(29)], dtype=np.float32)  # raw rad/s
            _sn = [CANON_G1_29[jmap[i]].replace("_joint", "") for i in range(29)]  # slot i の関節名
            _ta = np.argsort(-np.abs(action))[:3]
            _tv = np.argsort(-np.abs(_jvrw))[:3]
            _pz = float(pelvis_world()[0][2])
            print(f"[diag] step{step}: z={_pz:.3f} grav={np.round(_g,2).tolist()} "
                  f"| act_top={[f'{_sn[k]}:{action[k]:+.2f}' for k in _ta]} "
                  f"| vel_top={[f'{_sn[k]}:{_jvrw[k]:+.1f}' for k in _tv]} rad/s", flush=True)
        last_action = action.copy()
        q_tgt_motor = action * act_scale + act_offset      # policy 順
        # Isaac dof 目標へ
        full = np.asarray(robot.get_joint_positions(), dtype=np.float32).reshape(-1).copy()
        for i in range(29):
            ii = isaac_idx_for_motor[jmap[i]]
            if ii is not None:
                full[ii] = q_tgt_motor[i]
        robot.apply_action(ArticulationAction(joint_positions=full))
        # ★1 action = step_dt(20ms) だけ物理を進める。GUI では step(render=True) が
        #   rendering_dt(1/50)=4×physics_dt ぶん（=4サブステップ）を1回で進めるため **1回だけ** 呼ぶ。
        #   headless の step(render=False) は 1 サブステップ(5ms)ずつ → 4回。
        #   （GUI で4回呼ぶと 80ms/action=12.5Hz になり policy が破綻して転倒する）
        if args.gui:
            world.step(render=True)
        else:
            for _ in range(max(1, int(step_dt / PHYS_DT))):
                world.step(render=False)
        step += 1
        if step % 25 == 0:
            pos, _ = pelvis_world()
            z = float(np.asarray(pos).reshape(-1)[2])
            try:
                _lp = np.asarray(robot.get_link_positions() if hasattr(robot, 'get_link_positions')
                                 else robot.get_world_poses()[0]).reshape(-1, 3)
                zmax = float(_lp[:, 2].max())
            except Exception:  # noqa: BLE001
                zmax = -1.0
            _kn = float(np.asarray(robot.get_joint_positions()).reshape(-1)[_knee]) if _knee is not None else 0.0
            _px, _py = float(np.asarray(pos).reshape(-1)[0]), float(np.asarray(pos).reshape(-1)[1])
            print(f"[walk] t={_t.time()-t0:.1f}s pos=({_px:+.2f},{_py:+.2f},{z:.3f}) "
                  f"knee={_kn:.3f}（立=z~0.7±, 転=低z）", flush=True)

    print("[walk] done", flush=True)
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
