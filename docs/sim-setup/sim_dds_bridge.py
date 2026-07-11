#!/usr/bin/env python3
"""Isaac Sim ⇄ dimos DDS ブリッジ（sim-in-the-loop）。

dimos(Jetson) が出す Unitree DDS を Isaac Sim の仮想G1に橋渡しする:
  - 購読 ``rt/arm_sdk`` (unitree_hg LowCmd_): motor_cmd[15..28].q を G1 Articulation の
    腕14関節へ適用（weight=motor_cmd[29]>0 のときのみ）。正準G1_29順→Isaac dof名でマップ。
  - 発行 ``rt/lowstate`` (unitree_hg LowState_): mode_machine + 29関節の q/dq を sim から。
    （dimos の G1ArmSdkConnection は lowstate の mode_machine が来ないと最大10s待って起動しない）

transport は raw cyclonedds で制御（unitree_sdk2py の ChannelFactoryInitialize は使わない＝
CYCLONEDDS_URI 無視・config 固定のため）。env で interface/peers/multicast を切替:
  SIM_DDS_IFACE  : NIC名（既定 "lo"=loopback検証。same-LAN は "wlp0s20f3"、remote は "tailscale0"）
  SIM_DDS_PEERS  : unicast peer IP のカンマ区切り（tailscale 用。空=マルチキャスト）
  SIM_DDS_DOMAIN : DDS domain id（既定 0 = unitree 既定）
  SIM_LOAD_ROOM  : "1" で部屋(ROOM_USD=既定 chinou_center.usd)も読む（既定 0 = G1 のみ・loopback高速）
  SIM_ROOM_USD   : 部屋USD（既定 chinou_center.usd。旧 room.usd に差替可）
  SIM_G1_USD     : ロボットUSD（既定 g1bag.usd=収穫構成。旧 base-fix に差替可）
  SIM_GRAVITY    : "1" で重力ON（動力学検証）。既定OFF=弱PDのg1bag腕を指令角で保持（運動学確認）
  SIM_HEADLESS   : "0" で GUI（既定 1 = headless）
  SIM_WALK_POLICY: "1" で本物歩行モード（既定 0 = 従来）。base 移動を set_world_pose の
                   キネマ移動ではなく unitree_rl_lab velocity policy の脚歩行で行う。
                   floating base 化＋重力ON＋50Hz policy ループ。腕は arm_sdk が来ている間
                   だけ policy 出力を上書き（rad repo の arm_override と同型）。指令は
                   SIM_BASE_MOVE_FILE の "vx,vy[,wz]"（watchdog 途切れで cmd=0＝立位保持）。

実行:
  PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
  PYTHONPATH=/home/kota-ueda/Desktop/unitree_sdk2_python \
  SIM_DDS_IFACE=lo ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/sim_dds_bridge.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
sys.path.insert(0, "/home/kota-ueda/Desktop/unitree_sdk2_python")

REPO = os.getenv("SIM_REPO", "/home/kota-ueda/Desktop/dimos-hackathon")
# 既定＝正しい収穫シーン: chinou 実測室 ＋ g1bag 収穫構成（左手首バスケット＋右手 Dex1）。
# 旧ループバック検証用に room.usd / base-fix G1 を使う場合は env で差し替える。
# ※ g1bag の可動関節は CANON_G1_29 と 29/29 一致（＋Dex1 prismatic 2本）を確認済み。
ROOM_USD = os.getenv("SIM_ROOM_USD", f"{REPO}/usd_file/chinou_center.usd")
G1_USD = os.getenv("SIM_G1_USD", f"{REPO}/usd_file/g1-29dof-dex1-base-fix-usd/g1bag.usd")

# 正準 G1 29-DOF 関節名（Unitree G1_29_JointIndex 順）
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
ARM_CANON_IDX = list(range(15, 29))  # 腕14関節（左7 + 右7）
WEIGHT_IDX = 29


def build_cyclonedds_config() -> str:
    iface = os.getenv("SIM_DDS_IFACE", "lo")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "").split(",") if p.strip()]
    mc = "false" if peers else "true"  # unicast peers 指定時はマルチキャスト切る
    peers_xml = ""
    if peers:
        peers_xml = "<Peers>" + "".join(f'<Peer address="{p}"/>' for p in peers) + "</Peers>"
    allow_mc = "<AllowMulticast>false</AllowMulticast>" if peers else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <Interfaces><NetworkInterface name="{iface}" priority="default" multicast="{mc}"/></Interfaces>
      {allow_mc}
    </General>
    <Discovery>{peers_xml}</Discovery>
  </Domain>
</CycloneDDS>"""


def main() -> None:
    headless = os.getenv("SIM_HEADLESS", "1") != "0"
    load_room = os.getenv("SIM_LOAD_ROOM", "0") == "1"
    # 本物歩行モード: velocity policy で base 移動（キネマ set_world_pose の置換）。
    # 実装の正本は sim_walk_lib.py（floating base 化・SDK順 gains・per-term obs の6大要点）。
    walk_mode = os.getenv("SIM_WALK_POLICY", "0") == "1"
    domain_id = int(os.getenv("SIM_DDS_DOMAIN", "0"))
    cfg = build_cyclonedds_config()
    print(f"[bridge] cyclonedds config:\n{cfg}", flush=True)

    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": headless})

    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage, open_stage
    import omni.usd

    try:
        from isaacsim.core.prims import SingleArticulation as ArtCls
    except Exception:  # noqa: BLE001
        from isaacsim.core.api.articulations import Articulation as ArtCls  # type: ignore
    from isaacsim.core.utils.types import ArticulationAction

    # --- シーン構築 ---
    if load_room:
        open_stage(ROOM_USD)
    if walk_mode:
        # 歩行モード: physics 1/200s × 4substep = policy 50Hz（1/60 のままだと ≈15Hz で転倒）
        import sim_walk_lib as wl  # 同ディレクトリ（実装の正本）
        world = World(stage_units_in_meters=1.0, physics_dt=wl.PHYS_DT, rendering_dt=4 * wl.PHYS_DT)
    else:
        world = World(stage_units_in_meters=1.0)  # fix_base G1 は地面不要（cloud asset 取得を避ける）
    stage = omni.usd.get_context().get_stage()

    # 重力: 既定OFF。g1bag は関節 PD が弱く重力下で腕が垂れて指令角を保持できないため、
    # 運動学の追従確認（S0/IK）では重力OFFで腕を指令角に固定する。SIM_GRAVITY=1 で
    # 通常重力（把持/籠/切断の動力学を物理検証する時）。view_chinou.py と同じ方針。
    # 歩行モードは重力ON必須（policy が全身バランスを取る）＝OFF指定は無視する。
    if walk_mode:
        print("[bridge] walk_mode: 重力ON固定（policy が全身バランス。SIM_GRAVITY 指定は無視）", flush=True)
        if not load_room:  # 部屋なし検証の足場。部屋ありは chinou の床コライダー(z=0)で立つ
            try:
                world.scene.add_default_ground_plane(z_position=0.0)
                print("[bridge] walk_mode: ground plane @z=0 追加（部屋なし）", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[bridge] ground plane warn: {_e}", flush=True)
    elif os.getenv("SIM_GRAVITY", "0") != "1":
        try:
            world.get_physics_context().set_gravity(0.0)  # [m/s^2]
            print("[bridge] gravity OFF（弱PDのg1bag腕を指令角で保持。SIM_GRAVITY=1で重力ON）", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[bridge] set_gravity failed: {_e}", flush=True)

    add_reference_to_stage(usd_path=G1_USD, prim_path="/G1")
    g1 = stage.GetPrimAtPath("/G1")
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    lift = -float(bbc.ComputeWorldBound(g1).ComputeAlignedRange().GetMin()[2])
    if walk_mode:
        # 歩行モード: base-fix → floating base 化（root=pelvis, solver 32/4, armature, selfColl OFF）。
        # /G1 の xform は触らない（xformOpOrder 競合で NaN の罠）→配置は reset 後 place_upright で。
        art_root = wl.make_floating_base(stage, g1)
        if art_root is None:
            print("[bridge] ERROR: pelvis が見つからない（walk_mode 不成立）", flush=True)
            sim_app.close()
            return
    else:
        UsdGeom.XformCommonAPI(g1).SetTranslate(Gf.Vec3d(0.0, 0.0, lift))
        art_root = None
        for p in Usd.PrimRange(g1):
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                art_root = p.GetPath().pathString
                break
    robot = ArtCls(prim_path=art_root or "/G1", name="g1")
    world.scene.add(robot)

    # self-collision: 実機同様にロボット自身のリンク同士（右グリッパ ⇄ 左手首の籠 など）を衝突させる。
    # 「実機で暴れる前に sim で自己干渉を捕まえる」ため既定ON。隣接(親子)リンクは PhysX が自動除外する
    # ので通常は非隣接リンクのめり込みだけ検出する。SIM_SELF_COLLISION=0 で従来(OFF)に戻せる。
    # ※ 静止姿勢で非隣接リンクが既にめり込んでいると起動時に震える/弾かれることがある（その場合は要 collision filter）。
    if walk_mode:
        # 歩行 policy の実証条件は selfColl=False（sim_walk_lib が設定済み）。ON だと未検証。
        print("[bridge] walk_mode: self-collision OFF 固定（policy 実証条件）", flush=True)
    elif os.getenv("SIM_SELF_COLLISION", "1") == "1" and art_root is not None:
        try:
            from pxr import PhysxSchema

            PhysxSchema.PhysxArticulationAPI.Apply(
                stage.GetPrimAtPath(art_root)
            ).CreateEnabledSelfCollisionsAttr(True)
            print(f"[bridge] self-collision ON（{art_root}）— 実機同様に自己干渉を検出", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[bridge] self-collision 設定 warn: {_e}", flush=True)
    else:
        print("[bridge] self-collision OFF（SIM_SELF_COLLISION=1 で実機同様に有効化）", flush=True)

    # 摩擦把持モード（SIM_GRASP_FRICTION=1）: 磁石(kinematic/FixedJoint)を使わず、Dex1 の指
    # prismatic 関節を grip_q で実際に閉じ、指↔オクラの摩擦で掴む（B / Phase2・§3/§4-3）。
    # 薄物の接触は不安定になりやすいので、ソルバ反復を上げて貫通/滑りを抑える。
    grasp_friction = os.getenv("SIM_GRASP_FRICTION", "0") == "1"
    if grasp_friction and art_root is not None:
        try:
            from pxr import PhysxSchema

            _pa = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(art_root))
            _pa.CreateSolverPositionIterationCountAttr(int(os.getenv("SIM_SOLVER_POS_ITERS", "32")))
            _pa.CreateSolverVelocityIterationCountAttr(int(os.getenv("SIM_SOLVER_VEL_ITERS", "4")))
            print(f"[bridge] 摩擦把持モード ON（solver pos/vel iters="
                  f"{os.getenv('SIM_SOLVER_POS_ITERS','32')}/{os.getenv('SIM_SOLVER_VEL_ITERS','4')}）", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[bridge] friction solver 設定 warn: {_e}", flush=True)

    # 重力ON時は per-body 制御: 「ロボット各リンク=重力OFF（弱PDでも指令角を保持して腕が垂れない）／
    # オクラ=重力ON（籠コライダーへ物理落下）」。world 重力は既定(-9.81)のまま、ロボットだけ無効化する。
    # これで F-07 を「本番の腕モーションで運ぶ＋物理で着籠」させつつ弱PD腕の垂れを回避できる。
    if os.getenv("SIM_GRAVITY", "0") == "1" and not walk_mode:
        # ※歩行モードでは無効: policy が全身重力下でバランスするため robot の disableGravity は禁止。
        try:
            from pxr import PhysxSchema

            _ng = 0
            for _p in Usd.PrimRange(g1):
                if _p.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxRigidBodyAPI.Apply(_p).CreateDisableGravityAttr(True)
                    _ng += 1
            print(f"[bridge] gravity ON: robot {_ng} links disableGravity=True"
                  "（腕は指令角保持／オクラのみ落下）", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[bridge] per-body gravity warn: {_e}", flush=True)

    # 机上オクラ（A配置）: SIM_TABLE=1 で view_chinou と同一配置を載せる（M2/M3 用）。
    # world.reset() の前に stage へ追加する。配置の正本は sim_scene.build_table_okra。
    okra_paths: list[str] = []
    hand_path = None    # 右手リンク（机上ピックでオクラを固定する先）
    basket_path = None  # 左手首の籠（F-07 籠収納の投入先）
    if os.getenv("SIM_TABLE", "0") == "1":
        import sim_scene  # 同ディレクトリ（docs/sim-setup）

        n_okra = int(os.getenv("SIM_OKRA", "10"))
        table_h = float(os.getenv("SIM_TABLE_H", "0.72"))
        okra_paths = sim_scene.build_table_okra(stage, table_h=table_h, n_okra=n_okra)
        for _p in Usd.PrimRange(g1):
            nm = _p.GetName()
            if nm == "right_hand_base_link" and hand_path is None:
                hand_path = _p.GetPath().pathString
            elif "basket" in nm.lower() and basket_path is None:
                basket_path = _p.GetPath().pathString
        print(f"[bridge] 机+オクラ {len(okra_paths)}本 配置（A配置, 天板{table_h}m）hand={hand_path} basket={basket_path}", flush=True)

    world.reset()
    walker = None
    if walk_mode:
        # ★reset 後は一度も step せず先に直立配置（authored 姿勢は足が床を 0.757m 踏み抜いて
        #   おり、先に step すると射出される罠）。gains は SDK モーター順（policy slot 順ではない）。
        wdp = wl.load_deploy(f"{REPO}/usd_file/walk_policy/deploy.yaml")
        w_imap = wl.motor_to_isaac_map(list(robot.dof_names))
        wl.apply_sdk_gains(robot, wdp, w_imap)
        walker = wl.PolicyWalker(f"{REPO}/usd_file/walk_policy/policy.onnx", wdp, w_imap)
        # スポーン位置（歩行接近の検証用）: 机から離れて生まれ、歩いて接近する＝本番 F-08 と同型。
        _spawn_xy = (float(os.getenv("SIM_WALK_SPAWN_X", "0.0")),
                     float(os.getenv("SIM_WALK_SPAWN_Y", "0.0")))
        walker.place_upright(robot, world, render=not headless, xy=_spawn_xy,
                             base_z=float(os.getenv("SIM_WALK_BASE_Z", str(wl.BASE_Z0))))
        # 腕は policy の管轄から常時除外（obs マスク）: 腕を IK/arm_sdk で default から大きく
        # 動かすと policy が訓練分布外の腕状態を見て転倒するため。腕の実目標は
        # arm_sdk（無ければ deploy の default 姿勢）へのレート制限ブレンドのみ。
        w_arm_mask = list(range(15, 29))  # SDK motor 15..28 = 腕14関節
        w_arm_default: dict[int, float] = {}
        for _slot in range(29):
            _m = wdp["jmap"][_slot]
            if 15 <= _m <= 28 and w_imap[_m] is not None:
                w_arm_default[w_imap[_m]] = float(wdp["default_q"][_slot])
    else:
        try:
            robot.set_world_pose(position=np.array([0.0, 0.0, lift]))
        except Exception:  # noqa: BLE001
            pass
        for _ in range(20):
            world.step(render=not headless)

    # 籠の世界位置（F-07: 離したオクラをここへ world アンカーで固定＝投入）。左腕は rest 保持なので ~固定。
    basket_pos = None
    if basket_path is not None:
        try:
            _bm = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
                stage.GetPrimAtPath(basket_path))
            _bt = _bm.ExtractTranslation()
            basket_pos = (float(_bt[0]), float(_bt[1]), float(_bt[2]))
            print(f"[bridge] basket world pos = {tuple(round(v,3) for v in basket_pos)}", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[bridge] basket pos warn: {_e}", flush=True)

    dof_names = list(robot.dof_names)
    # 正準index -> Isaac dof index のマップ（名前一致）
    canon_to_isaac = {}
    for ci, nm in enumerate(CANON_G1_29):
        if nm in dof_names:
            canon_to_isaac[ci] = dof_names.index(nm)
    arm_isaac_idx = [canon_to_isaac[ci] for ci in ARM_CANON_IDX if ci in canon_to_isaac]
    print(f"[bridge] num_dof={robot.num_dof} mapped {len(canon_to_isaac)}/29 canon joints; "
          f"arm mapped={len(arm_isaac_idx)}", flush=True)

    # 摩擦把持: 非canon DOF = Dex1 の指 prismatic（num_dof-29）。grip_q で駆動するため
    # index / limits / drive gains を取得し診断出力。close 方向は env(SIM_GRIP_SIGN)で反転可。
    gripper_isaac_idx = [i for i, nm in enumerate(dof_names) if nm not in set(CANON_G1_29)]
    dof_lower = dof_upper = None
    grip_finger_kp = float(os.getenv("SIM_GRIP_KP", "800.0"))     # 指 prismatic drive stiffness
    grip_finger_kd = float(os.getenv("SIM_GRIP_KD", "40.0"))
    grip_close_full = float(os.getenv("SIM_GRIP_CLOSE_FULL", "4.4"))  # grip_q がこの値で全閉
    grip_sign = float(os.getenv("SIM_GRIP_SIGN", "1.0"))         # +1: upper側が閉じ / -1: lower側が閉じ
    if grasp_friction:
        # DOF limits 取得（Isaac の版差を吸収）: dof_properties → articulation_view → USD prim の順。
        for _how, _fn in (
            ("dof_properties",
             lambda: (np.asarray(robot.dof_properties["lower"], dtype=float).reshape(-1),
                      np.asarray(robot.dof_properties["upper"], dtype=float).reshape(-1))),
            ("articulation_view",
             lambda: (lambda L: (np.asarray(L).reshape(-1, 2)[:, 0],
                                 np.asarray(L).reshape(-1, 2)[:, 1]))(robot._articulation_view.get_dof_limits())),
        ):
            try:
                dof_lower, dof_upper = _fn()
                print(f"[bridge] dof limits via {_how}", flush=True)
                break
            except Exception as _e:  # noqa: BLE001
                print(f"[bridge] dof limits {_how} 不可: {_e}", flush=True)
        if dof_lower is None:
            # USD フォールバック: prismatic joint prim の lower/upper を dof 名で引く（stage単位=m）。
            dof_lower = np.full(robot.num_dof, -0.05)
            dof_upper = np.full(robot.num_dof, 0.05)
            for gi in gripper_isaac_idx:
                for _pp in Usd.PrimRange(g1):
                    if _pp.GetName() == dof_names[gi]:
                        _lo, _hi = _pp.GetAttribute("physics:lowerLimit"), _pp.GetAttribute("physics:upperLimit")
                        if _lo and _lo.HasValue():
                            dof_lower[gi] = float(_lo.Get())
                        if _hi and _hi.HasValue():
                            dof_upper[gi] = float(_hi.Get())
                        break
            print("[bridge] dof limits = USD フォールバック", flush=True)
        print(f"[bridge] gripper DOF(非canon)={[(i, dof_names[i]) for i in gripper_isaac_idx]}", flush=True)
        if dof_lower is not None:
            for gi in gripper_isaac_idx:
                print(f"[bridge]   dof[{gi}] {dof_names[gi]} limit=({dof_lower[gi]:.4f},{dof_upper[gi]:.4f})", flush=True)
        # 指 prismatic に drive gains を設定（位置目標で押し込み＝把持力を出すため）
        try:
            _ctrl = robot.get_articulation_controller()
            _g = _ctrl.get_gains()
            _kps = np.asarray(_g[0], dtype=float).reshape(-1)
            _kds = np.asarray(_g[1], dtype=float).reshape(-1)
            for gi in gripper_isaac_idx:
                _kps[gi] = grip_finger_kp
                _kds[gi] = grip_finger_kd
            _ctrl.set_gains(kps=_kps, kds=_kds)
            print(f"[bridge]   指 drive gains set kp={grip_finger_kp} kd={grip_finger_kd}", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[bridge] gripper gains warn: {_e}", flush=True)

    # 一発キャリブレーション（摩擦把持の reach 補正用・SIM_GRASP_FRICTION=1 時）:
    #   各オクラの実 torso 座標 / 右手リンク↔ジョーリンクのオフセット を出す。
    #   これで「IK を実オクラ座標へ当てる」補正値が分かる。jaw_link_paths は loop の距離診断にも使う。
    jaw_link_paths: list[str] = []
    torso_path: str | None = None  # loop の jaw↔okra(torso) 診断で使う
    if grasp_friction:
        try:
            _xc = UsdGeom.XformCache(Usd.TimeCode.Default())
            _torso_pp = None
            for _tp in Usd.PrimRange(g1):
                if _tp.GetName() == "torso_link":
                    _torso_pp = _tp.GetPath().pathString
                    break
            torso_path = _torso_pp
            _Tinv = (_xc.GetLocalToWorldTransform(stage.GetPrimAtPath(_torso_pp)).GetInverse()
                     if _torso_pp else None)

            def _tx(_p):
                _w = _xc.GetLocalToWorldTransform(stage.GetPrimAtPath(_p)).ExtractTranslation()
                if _Tinv is None:
                    return (round(float(_w[0]), 3), round(float(_w[1]), 3), round(float(_w[2]), 3))
                _t = _Tinv.Transform(Gf.Vec3d(float(_w[0]), float(_w[1]), float(_w[2])))
                return (round(float(_t[0]), 3), round(float(_t[1]), 3), round(float(_t[2]), 3))

            print(f"[calib] torso_link={_torso_pp}", flush=True)
            for _i, _op in enumerate(okra_paths):
                print(f"[calib] Okra_{_i} torso={_tx(_op)}", flush=True)
            _hb = (_xc.GetLocalToWorldTransform(stage.GetPrimAtPath(hand_path)).ExtractTranslation()
                   if hand_path else None)
            print(f"[calib] right_hand_base_link torso={_tx(hand_path) if hand_path else None}", flush=True)
            for _p in Usd.PrimRange(g1):
                if (_p.HasAPI(UsdPhysics.RigidBodyAPI) and "right_hand" in _p.GetName().lower()
                        and _p.GetPath().pathString != hand_path):
                    _lp = _p.GetPath().pathString
                    jaw_link_paths.append(_lp)
                    _lw = _xc.GetLocalToWorldTransform(stage.GetPrimAtPath(_lp)).ExtractTranslation()
                    _off = ((round(float(_lw[0]-_hb[0]), 3), round(float(_lw[1]-_hb[1]), 3),
                             round(float(_lw[2]-_hb[2]), 3)) if _hb is not None else None)
                    print(f"[calib] jaw link {_p.GetName()} torso={_tx(_lp)} offset_from_base={_off}", flush=True)
            print(f"[calib] jaw_link_paths={len(jaw_link_paths)}個", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[calib] warn: {_e}", flush=True)

    # 籠の torso 座標（F-07 IK 投入のターゲット GT）。SIM_DUMP_BASKET=1 なら左腕を「提示姿勢(L字)」へ
    # 置いて実測し終了する（その値を SIM_BASKET_TORSO="X,Y,Z" として収穫ランナへ渡す）。
    if basket_path is not None:
        _dump = os.getenv("SIM_DUMP_BASKET", "0") == "1"
        _torso_p = None
        for _tp in Usd.PrimRange(g1):
            if _tp.GetName() == "torso_link":
                _torso_p = _tp.GetPath().pathString
                break
        if _dump:
            # place_basket.LEFT_PRESENT_BASKET と同値（bridge は isaac-sim env で dimos 非依存のため複製）
            LEFT_PRESENT = [-0.1170, -0.0167, -0.3997, 1.1330, 0.0834, -1.0673, -0.2355]
            _tgt = np.asarray(robot.get_joint_positions(), dtype=float).copy()
            for _off, _val in enumerate(LEFT_PRESENT):
                _ii = canon_to_isaac.get(15 + _off)
                if _ii is not None:
                    _tgt[_ii] = _val
            for _ in range(150):
                robot.apply_action(ArticulationAction(joint_positions=_tgt))
                world.step(render=not headless)
        if _torso_p is not None:
            _xc = UsdGeom.XformCache(Usd.TimeCode.Default())
            _Tt = _xc.GetLocalToWorldTransform(stage.GetPrimAtPath(_torso_p))
            _bw = _xc.GetLocalToWorldTransform(stage.GetPrimAtPath(basket_path)).ExtractTranslation()
            _btt = _Tt.GetInverse().Transform(Gf.Vec3d(float(_bw[0]), float(_bw[1]), float(_bw[2])))
            _lbl = "左腕提示後" if _dump else "現姿勢"
            print(f"[bridge] basket torso ({_lbl}) = "
                  f"({_btt[0]:.3f},{_btt[1]:.3f},{_btt[2]:.3f}) → SIM_BASKET_TORSO に設定", flush=True)
        if _dump:
            print("[bridge] DUMP 完了 -> exit", flush=True)
            sim_app.close()
            return

    # --- DDS セットアップ（raw cyclonedds） ---
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.topic import Topic
    from cyclonedds.sub import DataReader
    from cyclonedds.pub import DataWriter
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as make_lowstate

    _dom = Domain(domain_id, cfg)  # ★参照を保持（GCされると domain ごと破棄され discovery 不成立）
    dp = DomainParticipant(domain_id)
    assert _dom is not None
    t_cmd = Topic(dp, "rt/arm_sdk", LowCmd_)
    t_state = Topic(dp, "rt/lowstate", LowState_)
    reader = DataReader(dp, t_cmd)
    writer = DataWriter(dp, t_state)
    # M3 机上ピック: グリッパ指令 rt/dex1/right/cmd（MotorCmds_, cmds[0].q）を購読
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

    t_grip = Topic(dp, "rt/dex1/right/cmd", MotorCmds_)
    grip_reader = DataReader(dp, t_grip)
    print(f"[bridge] DDS up: domain={domain_id} sub=rt/arm_sdk,rt/dex1/right/cmd pub=rt/lowstate", flush=True)

    # シーン照明（room が暗くカメラ像が見えない対策。SIM_ADD_LIGHT=0 で無効）
    # 照明はシーン(stage)単位＝同じ stage の全カメラで共有。view_chinou と見た目を揃えるため、
    # 部屋ロード時は共有関数 sim_scene.add_ceiling_lights で「同じ天井 SphereLight」を置く（単一ソース）。
    if os.getenv("SIM_ADD_LIGHT", "1") == "1":
        try:
            from pxr import UsdLux

            if load_room:
                # 部屋あり: chinou_center.usd に焼き込んだ天井ライトを使用（単一ソース）。
                # add_ceiling_lights は冪等＝焼き込み済みなら skip(0)、無い部屋なら補填。
                import sim_scene  # 同ディレクトリ
                ncl = sim_scene.add_ceiling_lights(
                    stage, intensity=float(os.getenv("SIM_LIGHT_INTENSITY", "6000"))
                )
                print(f"[bridge] ceiling lights: {'焼き込み使用(skip)' if ncl == 0 else f'補填 x{ncl}'}",
                      flush=True)
            else:
                # 部屋なし(G1のみ)はドーム+太陽のフィル（天井が無いので SphereLight 不可）
                UsdLux.DomeLight.Define(stage, "/World/sim_fill_dome").CreateIntensityAttr(1500.0)
                UsdLux.DistantLight.Define(stage, "/World/sim_fill_sun").CreateIntensityAttr(3000.0)
                print("[bridge] added fill light (dome+sun)", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[bridge] light warn: {_e}", flush=True)

    # --- カメラ配信（ego_view ZMQ, dimos ZmqCamera 互換: msgpack{images:{topic:b64-jpeg}}）---
    cam = None
    cam_pub = None
    cam_enabled = os.getenv("SIM_PUB_CAMERA", "0") == "1"
    cam_topic = os.getenv("SIM_CAM_TOPIC", "ego_view")
    cam_period = 1.0 / max(1.0, float(os.getenv("SIM_CAM_FPS", "15")))
    last_cam_pub = 0.0
    cam_K = None          # 公開する intrinsics [fx,0,cx,0,fy,cy,0,0,1]
    cam_to_torso_str = ""  # torso<-optical の取付（dimos の OKRA_CAM_TO_TORSO 形式 x,y,z,qx,qy,qz,qw）
    if cam_enabled:
        import base64
        import cv2
        import math
        import msgpack
        import zmq
        from isaacsim.sensors.camera import Camera

        def _vec(name: str, default: list) -> "np.ndarray":  # noqa: F821
            v = os.getenv(name)
            return np.array([float(x) for x in v.split(",")], dtype=float) if v else np.array(default, dtype=float)

        cam_w = int(os.getenv("SIM_CAM_W", "640"))
        cam_h = int(os.getenv("SIM_CAM_H", "360"))
        cam_port = int(os.getenv("SIM_CAM_PORT", "5555"))
        hfov = math.radians(float(os.getenv("SIM_CAM_HFOV", "90")))  # 水平画角
        cam_mode = os.getenv("SIM_CAM_MODE", "torso")  # torso=リンク追従 / fixed=固定look-at

        # torso_link prim を探索（実 ZED 取付基準）
        torso_path = None
        for _p in Usd.PrimRange(g1):
            if _p.GetName() == "torso_link":
                torso_path = _p.GetPath().pathString
                break
        if torso_path is None:
            for _p in Usd.PrimRange(g1):
                if "torso" in _p.GetName().lower():
                    torso_path = _p.GetPath().pathString
                    break

        if cam_mode == "torso" and torso_path is not None:
            # torso フレームでの取付: 位置 + 前方やや下向き（実 ZED 相当）
            lpos = _vec("SIM_CAM_LOCAL_POS", [0.08, 0.0, 0.20])
            fwd = _vec("SIM_CAM_LOCAL_FWD", [1.0, 0.0, -0.35])
            # SIM_CAM_LOOK_WORLD="x,y,z" 指定時: torso の向きに関係なく「ワールドのその点」を
            # 見るよう、胸カメラの torso-local 前方ベクトルを逆算する（机のオクラ確実捕捉用）。
            _lookw = os.getenv("SIM_CAM_LOOK_WORLD", "")
            if _lookw:
                try:
                    _tw = [float(x) for x in _lookw.split(",")]
                    _Mt = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
                        stage.GetPrimAtPath(torso_path))
                    _camw = _Mt.Transform(Gf.Vec3d(float(lpos[0]), float(lpos[1]), float(lpos[2])))
                    _dirw = Gf.Vec3d(_tw[0] - _camw[0], _tw[1] - _camw[1], _tw[2] - _camw[2])
                    _dl = _Mt.GetInverse().TransformDir(_dirw)  # world dir → torso-local dir
                    fwd = np.array([_dl[0], _dl[1], _dl[2]], dtype=float)
                    print(f"[bridge] chest cam look-at world {_tw} → torso-local fwd {np.round(fwd, 3)}", flush=True)
                except Exception as _e:  # noqa: BLE001
                    print(f"[bridge] look-world warn: {_e}", flush=True)
            fwd = fwd / np.linalg.norm(fwd)
            up = np.array([0.0, 0.0, 1.0])
            camZ = -fwd                                   # USDカメラは -Z が視線
            camX = np.cross(up, camZ); camX /= np.linalg.norm(camX)
            camY = np.cross(camZ, camX)
            # ローカル変換（torso<-usd_cam, Gf は row-vector: 行=基底）
            def _mat2quat(cols):
                R = np.column_stack(cols)
                t = R[0, 0] + R[1, 1] + R[2, 2]
                if t > 0:
                    s = math.sqrt(t + 1.0) * 2
                    w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
                elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                    s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                    w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
                elif R[1, 1] > R[2, 2]:
                    s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                    w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
                else:
                    s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                    w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
                return w, x, y, z

            # USD カメラ姿勢（基底= camX,camY,camZ）を translate+orient op で設定（hydra 互換）
            _cw, _cx, _cy, _cz = _mat2quat([camX, camY, camZ])
            cam = Camera(prim_path=f"{torso_path}/chest_cam", resolution=(cam_w, cam_h))
            cam.initialize()
            # Camera.initialize() が作る orient op は quatd（倍精度）。precision を合わせる（不一致は Tf例外→crash）
            try:
                _xf = UsdGeom.Xformable(stage.GetPrimAtPath(f"{torso_path}/chest_cam"))
                _xf.ClearXformOpOrder()
                _xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                    Gf.Vec3d(float(lpos[0]), float(lpos[1]), float(lpos[2]))
                )
                _xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                    Gf.Quatd(float(_cw), float(_cx), float(_cy), float(_cz))
                )
            except Exception as _e:  # noqa: BLE001
                print(f"[bridge] cam xform warn: {_e}", flush=True)
            # cam_to_torso (torso<-optical): optical 基底= X=camX, Y=-camY, Z=fwd
            _ow, _ox, _oy, _oz = _mat2quat([camX, -camY, fwd])
            cam_to_torso_str = f"{lpos[0]:.4f},{lpos[1]:.4f},{lpos[2]:.4f},{_ox:.5f},{_oy:.5f},{_oz:.5f},{_ow:.5f}"
            print(f"[bridge] camera attached to {torso_path} (link-following). cam_to_torso={cam_to_torso_str}", flush=True)
        else:
            # フォールバック: 固定 look-at
            eye = _vec("SIM_CAM_EYE", [1.6, -0.5, 1.3])
            tgt = _vec("SIM_CAM_TARGET", [-0.2, -0.5, 0.6])
            _m = Gf.Matrix4d()
            _m.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1))
            _q = _m.GetInverse().ExtractRotationQuat()
            _ori = np.array([_q.GetReal(), _q.GetImaginary()[0], _q.GetImaginary()[1], _q.GetImaginary()[2]])
            cam = Camera(prim_path="/sim_head_cam", resolution=(cam_w, cam_h))
            cam.initialize()
            cam.set_world_pose(eye, _ori, camera_axes="usd")
            print("[bridge] camera fixed look-at (torso_link 未検出 or mode=fixed)", flush=True)

        # GUI のメインビューポートをこのカメラ視点へ切替＝画面＝カメラ映像（YOLO入力）を常時表示。
        # SIM_VIEWPORT_CAM=0 で無効（俯瞰のまま）。GUI 時のみ。
        if (not headless) and os.getenv("SIM_VIEWPORT_CAM", "1") == "1":
            _campath = (f"{torso_path}/chest_cam"
                        if (cam_mode == "torso" and torso_path is not None) else "/sim_head_cam")
            try:
                import omni.kit.viewport.utility as _vpu

                _vp = _vpu.get_active_viewport()
                try:
                    _vp.set_active_camera(_campath)
                except Exception:  # noqa: BLE001
                    _vp.camera_path = _campath
                print(f"[bridge] GUI ビューポート → {_campath}（カメラ映像を常時表示）", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[bridge] viewport cam set warn: {_e}", flush=True)

        # intrinsics（HFOV から焦点距離を設定。比 focal/aperture が画角を決める＝単位は相殺）
        _A = 20.955
        _F = (_A / 2.0) / math.tan(hfov / 2.0)
        try:
            cam.set_horizontal_aperture(_A)
            cam.set_vertical_aperture(_A * cam_h / cam_w)
            cam.set_focal_length(_F)
        except Exception as _e:  # noqa: BLE001
            print(f"[bridge] set intrinsics warn: {_e}", flush=True)
        # ★near クリップ: 既定が遠いと近接の机/オクラ(~0.4m)が切られ「透視」して奥の部屋が写る。
        # near を小さく(既定0.03m)して近接物を描画する。SIM_CAM_NEAR / SIM_CAM_FAR で調整。
        try:
            _near = float(os.getenv("SIM_CAM_NEAR", "0.03"))
            _far = float(os.getenv("SIM_CAM_FAR", "1000000.0"))
            cam.set_clipping_range(_near, _far)
            print(f"[bridge] camera clipping range = ({_near}, {_far})", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[bridge] set clipping warn: {_e}", flush=True)
        try:
            cam.add_distance_to_image_plane_to_frame()  # depth 有効化
        except Exception as _e:  # noqa: BLE001
            print(f"[bridge] depth enable warn: {_e}", flush=True)
        for _ in range(25):
            world.step(render=True)
        _fx = (cam_w / 2.0) / math.tan(hfov / 2.0)
        _cx, _cy = cam_w / 2.0, cam_h / 2.0
        cam_K = [_fx, 0.0, _cx, 0.0, _fx, _cy, 0.0, 0.0, 1.0]

        _ctx = zmq.Context.instance()
        cam_pub = _ctx.socket(zmq.PUB)
        cam_pub.bind(f"tcp://0.0.0.0:{cam_port}")
        print(f"[bridge] camera pub: ZMQ tcp://0.0.0.0:{cam_port} topic={cam_topic} {cam_w}x{cam_h} "
              f"hfov={math.degrees(hfov):.0f}deg fx={_fx:.1f} depth=on", flush=True)

    render_on = (not headless) or cam_enabled  # カメラ配信時は headless でも render 必須

    ls = make_lowstate()
    ls.mode_machine = 1  # dimos の arm controller が起動するために非ゼロ必須

    last_q_target = None
    pub_every = 5  # ~publishは数stepごと
    step = 0
    last_log = time.time()
    cmd_count = 0
    t0 = time.time()
    run_secs = float(os.getenv("SIM_RUN_SECS", "0"))  # >0 で打ち切り（検証用）。0=無限
    stop_file = "/tmp/sim_bridge_stop"
    if os.path.exists(stop_file):  # 起動時の残骸を除去（前回の停止フラグで即終了するのを防ぐ）
        try:
            os.remove(stop_file)
        except Exception:  # noqa: BLE001
            pass
    ii22 = canon_to_isaac.get(22)  # right_shoulder_pitch（測定確認用）
    ii25 = canon_to_isaac.get(25)  # right_elbow

    # M3/M4 机上ピック状態（複数把持対応: 把持対象 index はファイル or env から）
    grip_q = 0.0
    grasped_set: set[int] = set()
    # 把持方式: SIM_GRASP_KINEMATIC=1(既定) はオクラを kinematic 化し毎ステップ手に追従させる
    # （手に反力ゼロ＝手が引っ張られない）。=0 で従来の FixedJoint 接着（双方向＝手が引かれる）。
    grasp_kinematic = os.getenv("SIM_GRASP_KINEMATIC", "1") == "1"
    grasped_kin: dict[int, "Gf.Matrix4d"] = {}  # {okra idx: rel = okraW * handW^-1}（追従用）
    basket_count = 0   # F-07: 籠に投入済みの本数（積み重ねオフセット用）
    placed_at: dict[int, int] = {}   # F-07 物理落下: {okra idx: 投入した step}（着籠判定用）
    grasp_target = int(os.getenv("SIM_GRASP_OKRA", "1"))   # 既定 /Okra_1（単発時）
    grasp_close = float(os.getenv("SIM_GRASP_CLOSE", "2.0"))  # cmds[0].q がこれ以上で閉じ=把持
    grip_open_q = float(os.getenv("SIM_GRIP_OPEN", "5.0"))    # これ以上=開き(リリース)。close=4.4/open=5.2 を区別
    grasp_target_file = os.getenv("SIM_GRASP_TARGET_FILE", "/tmp/sim_grasp_target.txt")  # graph がここに次の対象を書く
    _go = [float(x) for x in os.getenv("SIM_GRASP_OFFSET", "0,0,0").split(",")]  # 把持位置オフセット

    # base 移動（cmd_vel 歩行/reposition）: SIM_BASE_MOVE=1（既定OFF）でオプトイン。
    # 設計書 SS-07 準拠＝LocoClient は「速度指令で連続移動する（足は選べない＝ルンバ）」。これを
    # 再現: skills が base_move_file に **world 速度 "vx,vy" を 10Hz でストリーム**、bridge は毎ステップ
    # `base_xy += v*dt` で積分し set_world_pose（g1bag は base-fix なのでキネマティック平行移動＝脚は
    # 動かさない。脚の二足歩行物理は LocoClient/実機の責務で sim 対象外）。watchdog: 指令が
    # SIM_BASE_CMD_TIMEOUT 秒来なければ停止（nav_bridge と同方式）。既定OFFで並行 nav/従来に無影響。
    base_move = os.getenv("SIM_BASE_MOVE", "0") == "1"
    base_move_file = os.getenv("SIM_BASE_MOVE_FILE", "/tmp/sim_base_move.txt")
    base_cmd_timeout = float(os.getenv("SIM_BASE_CMD_TIMEOUT", "0.3"))  # [s] 指令途切れで停止
    base_xy = [0.0, 0.0]   # world 累積 base 位置 [m]
    base_last_t = time.time()  # 積分用の前回時刻
    if walk_mode:
        # 歩行モードは同じファイル/プロトコルを policy の velocity 指令として使う（キネマ積分は無効）。
        base_move = False
        try:
            os.remove(base_move_file)
        except OSError:
            pass
        print(f"[bridge] walk_mode: {base_move_file} の \"vx,vy[,wz]\" を policy 指令に"
              f"（watchdog {base_cmd_timeout}s 途切れで cmd=0＝立位保持）", flush=True)
    elif base_move:
        try:  # 起動時に古いファイルを消す（前回の残骸でいきなり動かないように）
            os.remove(base_move_file)
        except OSError:
            pass
        print(f"[bridge] base-move ON（SIM_BASE_MOVE=1）: {base_move_file} の world 速度 vx,vy を "
              f"10Hz ストリーム→積分移動（LocoClient 風・watchdog {base_cmd_timeout}s）", flush=True)
    # 歩行モードの腕上書き: arm_sdk の最新目標（isaac dof idx → q）。weight<=0.01 で解除＝policy に返す。
    # walk_arm_cur = 実際に適用中の値（レート制限付きブレンド）。腕14関節を一気にスナップさせると
    # policy バランスが崩れて転倒する（実測）ため、目標へ SIM_WALK_ARM_RATE [rad/s] で漸近する
    # （rad repo の arm_override も entry 姿勢ブレンドで同じ対策）。解除時も policy 出力へ滑らかに戻す。
    walk_arm_tgt: dict[int, float] = {}
    walk_arm_cur: dict[int, float] = {}
    walk_arm_rate = float(os.getenv("SIM_WALK_ARM_RATE", "3.0"))  # [rad/s] 腕上書きの最大変化率

    def _world_xyz(path):
        """prim の現在のワールド座標 (x,y,z) を実測（籠は左腕で動くので毎回読む）。"""
        if not path:
            return None
        try:
            _t = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
                stage.GetPrimAtPath(path)).ExtractTranslation()
            return (round(float(_t[0]), 3), round(float(_t[1]), 3), round(float(_t[2]), 3))
        except Exception:  # noqa: BLE001
            return None

    print("[bridge] loop start（Ctrl-C / touch /tmp/sim_bridge_stop で終了）", flush=True)
    while True:
        if run_secs > 0 and (time.time() - t0) > run_secs:
            print("[bridge] run_secs reached -> stop", flush=True)
            break
        if os.path.exists(stop_file):
            print("[bridge] stop file -> stop", flush=True)
            break
        # 0) base 移動（cmd_vel 歩行/reposition）: file の world 速度 vx,vy を読み、鮮度(mtime)が
        #    watchdog 内なら base_xy += v*dt を積分→set_world_pose（LocoClient 風の連続移動）。
        if base_move:
            _now = time.time()
            _dt = max(0.0, min(0.1, _now - base_last_t))  # 実時間 dt（暴れ防止に上限0.1s）
            base_last_t = _now
            _vx = _vy = 0.0
            try:
                if _now - os.path.getmtime(base_move_file) <= base_cmd_timeout:  # 鮮度内のみ採用
                    with open(base_move_file) as _bf:
                        _vx, _vy = (float(v) for v in _bf.read().strip().split(",")[:2])
            except (OSError, ValueError):
                _vx = _vy = 0.0  # ファイル無し/古い=停止（watchdog）
            if _vx != 0.0 or _vy != 0.0:
                base_xy[0] += _vx * _dt
                base_xy[1] += _vy * _dt
                try:
                    robot.set_world_pose(position=np.array([base_xy[0], base_xy[1], lift]),
                                         orientation=np.array([1.0, 0.0, 0.0, 0.0]))
                except Exception:  # noqa: BLE001
                    pass
        # 1) arm_sdk 受信 → 腕適用（InvalidSample 等 motor_cmd を持たない物は除外）
        try:
            samples = [s for s in reader.take(N=20) if hasattr(s, "motor_cmd")]
        except Exception:  # noqa: BLE001
            samples = []
        if samples:
            lc = samples[-1]
            cmd_count += 1
            try:
                weight = float(lc.motor_cmd[WEIGHT_IDX].q)
            except Exception:  # noqa: BLE001
                weight = 1.0
            if weight > 0.01 and arm_isaac_idx:
                if walk_mode:
                    # 歩行モード: 直接 apply せず「最新の腕目標」を記憶 → walk tick で policy 出力に上書き
                    # （rad repo の arm_override と同型: 脚腰=policy バランス / 腕=外部指令）。
                    for ci in ARM_CANON_IDX:
                        ii = canon_to_isaac.get(ci)
                        if ii is not None:
                            walk_arm_tgt[ii] = float(lc.motor_cmd[ci].q)
                else:
                    q_full = np.asarray(robot.get_joint_positions(), dtype=float)
                    tgt = q_full.copy()
                    for ci in ARM_CANON_IDX:
                        ii = canon_to_isaac.get(ci)
                        if ii is not None:
                            tgt[ii] = float(lc.motor_cmd[ci].q)
                    robot.apply_action(ArticulationAction(joint_positions=tgt))
                    last_q_target = tgt
            elif walk_mode and weight <= 0.01 and walk_arm_tgt:
                walk_arm_tgt.clear()  # weight=0 ＝腕を policy に返す（default 姿勢へ）

        # 1b) グリッパ受信 → 机上ピック（閉じ: world joint 外し→手リンクへ FixedJoint / 開き: 解放）
        try:
            gsm = [s for s in grip_reader.take(N=10) if hasattr(s, "cmds") and len(s.cmds) > 0]
        except Exception:  # noqa: BLE001
            gsm = []
        if gsm:
            try:
                grip_q = float(gsm[-1].cmds[0].q)
            except Exception:  # noqa: BLE001
                pass
        # 1c) 摩擦把持: grip_q を Dex1 指 prismatic の位置目標へ写像し毎ステップ駆動（磁石なし）。
        #     frac=0(開)..1(閉)。閉じ側 limit は SIM_GRIP_SIGN で選ぶ。okra は指↔莢の摩擦だけで保持。
        #     歩行モードでは直接 apply せず walk_grip_tgt に記憶（walk tick で policy 出力に上書き）。
        walk_grip_tgt: dict[int, float] = {}
        if grasp_friction and gripper_isaac_idx and dof_lower is not None:
            frac = max(0.0, min(1.0, (grip_q / grip_close_full) if grip_close_full else 0.0))
            base = (last_q_target.copy() if last_q_target is not None
                    else np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1).copy())
            for gi in gripper_isaac_idx:
                lo, hi = float(dof_lower[gi]), float(dof_upper[gi])
                closed = hi if grip_sign > 0 else lo
                opened = lo if grip_sign > 0 else hi
                base[gi] = opened + frac * (closed - opened)
                walk_grip_tgt[gi] = float(base[gi])
            if not walk_mode:
                robot.apply_action(ArticulationAction(joint_positions=base))
        # 把持対象 index の決定:
        #  (a) SIM_GRASP_NEAREST=1: 閉じる瞬間、右手リンク world 位置に最も近い未把持オクラ prim を
        #      自動選択（YOLO 検出は prim index を知らないため＝実検出ループ用）。
        #  (b) 既定: graph が書くファイル優先、無ければ env（GT index ループ用）。
        gt_idx = grasp_target
        if os.getenv("SIM_GRASP_NEAREST", "0") == "1" and grip_q >= grasp_close and hand_path:
            try:
                _hw = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
                    stage.GetPrimAtPath(hand_path)).ExtractTranslation()
                _best, _bd = None, 1e9
                for _i, _op in enumerate(okra_paths):
                    if _i in grasped_set:
                        continue
                    _ow = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
                        stage.GetPrimAtPath(_op)).ExtractTranslation()
                    _dd = (_hw[0]-_ow[0])**2 + (_hw[1]-_ow[1])**2 + (_hw[2]-_ow[2])**2
                    if _dd < _bd:
                        _best, _bd = _i, _dd
                if _best is not None and _bd <= float(os.getenv("SIM_GRASP_NEAREST_MAX", "0.20"))**2:
                    gt_idx = _best
                else:
                    gt_idx = -1  # 近傍に未把持オクラ無し→把持しない
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                with open(grasp_target_file) as _f:
                    gt_idx = int(_f.read().strip())
            except Exception:  # noqa: BLE001
                pass
        # 摩擦把持モード: 磁石（アンカー削除/kinematic/FixedJoint）は一切使わない。指の摩擦だけで
        # 掴み、持ち上げの引張でオクラの破断 FixedJoint(SIM_OKRA_BREAK_N) が切れて収穫＝物理のみ。
        if grasp_friction:
            gt_idx = -1  # 以降の磁石ブロックを不活性化（grasped_set は空のまま）
        # 閉じ かつ 未把持の対象 → world アンカー除去＋手リンクへ FixedJoint（複数可, ユニーク joint）
        if grasp_close <= grip_q < grip_open_q and hand_path and gt_idx not in grasped_set and 0 <= gt_idx < len(okra_paths):
            okp = okra_paths[gt_idx]
            wj = f"/World/OkraJoints/joint_{gt_idx}"
            if stage.GetPrimAtPath(wj):
                stage.RemovePrim(wj)
            if grasp_kinematic:
                # kinematic 追従: オクラを kinematic 化（外力で動かない＝手に反力ゼロ）し、相対変換
                # rel = okraW * handW^-1 を記録。毎ステップ okraW = rel * handW で手に追従させる。
                _xc = UsdGeom.XformCache(Usd.TimeCode.Default())
                _hw = _xc.GetLocalToWorldTransform(stage.GetPrimAtPath(hand_path))
                _ow = _xc.GetLocalToWorldTransform(stage.GetPrimAtPath(okp))
                grasped_kin[gt_idx] = _ow * _hw.GetInverse()
                try:
                    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(okp)).CreateKinematicEnabledAttr(True)
                except Exception as _e:  # noqa: BLE001
                    print(f"[bridge] okra kinematic 設定 warn: {_e}", flush=True)
            else:
                # 従来: FixedJoint 接着（双方向＝オクラの質量/慣性が手を引っ張る）
                gj = UsdPhysics.FixedJoint.Define(stage, f"/World/GraspJoint_{gt_idx}")
                gj.CreateBody0Rel().SetTargets([hand_path])
                gj.CreateBody1Rel().SetTargets([okp])
                gj.CreateLocalPos0Attr(Gf.Vec3f(_go[0], _go[1], _go[2]))
                gj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            grasped_set.add(gt_idx)
            print(f"[bridge] GRASP {okp} → {hand_path}（idx={gt_idx}, grip_q={grip_q:.2f}, "
                  f"{'kinematic追従' if grasp_kinematic else 'FixedJoint'}, 計{len(grasped_set)}本）", flush=True)
        # 開き かつ 把持中 → リリース（手リンクの GraspJoint を外す）＝F-07。
        #   重力ON: 手を離すだけ→重力ONのオクラが籠コライダーへ物理落下（本番の腕モーションで運んだ上で着籠）。
        #   重力OFF: 従来どおり world アンカーで籠位置へ収める（弱PD回避の運動学デモ）。
        elif (grip_q >= grip_open_q or grip_q < grasp_close) and grasped_set:
            _grav_on = os.getenv("SIM_GRAVITY", "0") == "1"
            for _idx in sorted(grasped_set):
                gjp = f"/World/GraspJoint_{_idx}"
                if stage.GetPrimAtPath(gjp):
                    stage.RemovePrim(gjp)
                if _idx in grasped_kin:
                    # kinematic を解除し dynamic に戻す（重力ON なら手を離した位置から物理落下）
                    grasped_kin.pop(_idx, None)
                    try:
                        UsdPhysics.RigidBodyAPI.Apply(
                            stage.GetPrimAtPath(okra_paths[_idx])).CreateKinematicEnabledAttr(False)
                    except Exception:  # noqa: BLE001
                        pass
                if _grav_on:
                    placed_at[_idx] = step
                    print(f"[bridge] PLACE okra idx={_idx} → 手を離す（重力で物理落下）｜"
                          f"basket={_world_xyz(basket_path)} hand={_world_xyz(hand_path)} "
                          f"okra={_world_xyz(okra_paths[_idx])}", flush=True)
                elif basket_pos is not None:
                    # 籠内で少しずつ位置をずらして積む（world アンカー）
                    bx = basket_pos[0] + 0.02 * (basket_count % 3 - 1)
                    by = basket_pos[1] + 0.02 * ((basket_count // 3) % 3 - 1)
                    bz = basket_pos[2] + 0.05 + 0.015 * basket_count
                    bj = UsdPhysics.FixedJoint.Define(stage, f"/World/BasketAnchor_{_idx}")
                    bj.CreateBody1Rel().SetTargets([okra_paths[_idx]])
                    bj.CreateLocalPos0Attr(Gf.Vec3f(bx, by, bz))   # world アンカー＝籠位置へ
                    bj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                    basket_count += 1
                    print(f"[bridge] PLACE okra idx={_idx} → 籠（world アンカー, 投入{basket_count}本目）", flush=True)
            grasped_set.clear()

        # 2) sim 1step（歩行モード: policy 1tick=20ms=4substep / 従来: 1step）
        if walk_mode:
            # velocity 指令: base_move_file の "vx,vy[,wz]"（body系 [m/s, m/s, rad/s]）。
            # watchdog 途切れ → cmd=0（policy が立位保持）＝把持フェーズ。
            _cmd = np.zeros(3, dtype=np.float32)
            try:
                if time.time() - os.path.getmtime(base_move_file) <= base_cmd_timeout:
                    with open(base_move_file) as _bf:
                        _v = [float(x) for x in _bf.read().strip().split(",")[:3]]
                    _cmd[:len(_v)] = _v
            except (OSError, ValueError):
                pass
            _tgt = walker.tick(robot, _cmd, mask_motors=w_arm_mask)
            # 腕: policy 出力は使わず、arm_sdk 目標（無ければ default 姿勢）へレート制限ブレンド
            #（スナップで balance が崩れるのを防ぐ。obs 側は w_arm_mask で常時マスク済み）。
            _dq_max = walk_arm_rate * 0.02  # [rad/tick] (=rate × 20ms)
            for _wi in arm_isaac_idx:
                _des = walk_arm_tgt.get(_wi, w_arm_default.get(_wi, float(_tgt[_wi])))
                _cur = walk_arm_cur.get(_wi)
                if _cur is None:
                    _cur = w_arm_default.get(_wi, float(_tgt[_wi]))  # 初回は default 姿勢から
                _cur += float(np.clip(_des - _cur, -_dq_max, _dq_max))
                walk_arm_cur[_wi] = _cur
                _tgt[_wi] = _cur
            for _gi, _gq in walk_grip_tgt.items():  # 指: 摩擦把持の目標を上書き
                _tgt[_gi] = _gq
            robot.apply_action(ArticulationAction(joint_positions=_tgt))
            last_q_target = _tgt
            # 20ms 進める。render 時は rendering_dt=4×physics_dt を1回で（GUI 4倍進む罠）
            if render_on:
                world.step(render=True)
            else:
                for _ in range(4):
                    world.step(render=False)
            if step % int(os.getenv("SIM_WALK_LOG_TICKS", "100")) == 0:  # base 位置ログ（既定~2s。接近の閉ループ監視は 25=0.5s に）
                _bp, _bq = robot.get_world_pose()
                _bp = np.asarray(_bp, dtype=float).reshape(-1)
                print(f"[bridge][walk] step={step} cmd=({_cmd[0]:+.2f},{_cmd[1]:+.2f},{_cmd[2]:+.2f}) "
                      f"base=({_bp[0]:+.2f},{_bp[1]:+.2f},{_bp[2]:.3f}) arm_ovr={len(walk_arm_tgt)}",
                      flush=True)
        else:
            world.step(render=render_on)
        step += 1

        # 2b) kinematic 把持の追従: 把持中オクラの world 姿勢 = rel * hand_world（手に反力ゼロで追従）
        if grasped_kin:
            _xc = UsdGeom.XformCache(Usd.TimeCode.Default())
            _hw = _xc.GetLocalToWorldTransform(stage.GetPrimAtPath(hand_path))
            for _idx, _rel in grasped_kin.items():
                _m = _rel * _hw  # 目標 world 変換
                _t = _m.ExtractTranslation()
                _q = _m.ExtractRotationQuat()
                _xf = UsdGeom.Xformable(stage.GetPrimAtPath(okra_paths[_idx]))
                _ops = {op.GetOpName(): op for op in _xf.GetOrderedXformOps()}
                _to = _ops.get("xformOp:translate")
                _oo = _ops.get("xformOp:orient")
                if _to is not None:
                    _to.Set(Gf.Vec3d(_t[0], _t[1], _t[2]))
                if _oo is not None:
                    _oo.Set(Gf.Quatd(_q.GetReal(), _q.GetImaginary()))

        # F-07 物理落下の着籠判定（投入から ~1.5s 後に1回, 重力ON時のみ）。籠は左腕で動くので
        # 籠の現在ワールド座標を実測し、それを中心に±半径/高さで内外を判定する。
        if placed_at:
            _bc = _world_xyz(basket_path) or basket_pos
            _done = []
            for _idx, _s0 in placed_at.items():
                if step - _s0 < 90:
                    continue
                try:
                    _o = _world_xyz(okra_paths[_idx])
                    if _o is None or _bc is None:
                        raise RuntimeError("座標取得不可")
                    _dx, _dy, _dz = _o[0] - _bc[0], _o[1] - _bc[1], _o[2] - _bc[2]
                    _rad = (_dx * _dx + _dy * _dy) ** 0.5
                    _inb = (_rad < 0.12) and (-0.05 <= _dz <= 0.25)
                    _verdict = "受けた(籠内)" if _inb else ("床へ落下" if _o[2] < 0.15 else "籠外/縁")
                    _msg = (f"[bridge] F-07 着籠判定 okra{_idx}: okra={_o} 籠={_bc} "
                            f"水平ズレr={_rad:.3f} 高さ差dz={_dz:.3f} -> {_verdict}")
                    print(_msg, flush=True)
                    with open("/tmp/f07_place.txt", "a") as _f:
                        _f.write(_msg + "\n")
                except Exception as _e:  # noqa: BLE001
                    print(f"[bridge] 着籠判定 warn idx={_idx}: {_e}", flush=True)
                _done.append(_idx)
            for _idx in _done:
                placed_at.pop(_idx, None)

        # 3) lowstate 発行
        if step % pub_every == 0:
            q = np.atleast_1d(np.asarray(robot.get_joint_positions(), dtype=float).squeeze())
            dq = np.atleast_1d(np.asarray(robot.get_joint_velocities(), dtype=float).squeeze())
            # アーティキュレーション破綻時 get_joint_positions が 0次元/空を返し crash した実績あり。
            # 形が想定外（必要 dof 未満）ならこのフレームの lowstate 発行をスキップ（落とさない）。
            _need = (max(canon_to_isaac.values()) + 1) if canon_to_isaac else 0
            if q.ndim != 1 or q.size < _need or dq.size < _need:
                if step % 250 == 0:
                    print(f"[bridge] lowstate skip: joint array shape={q.shape}（articulation 不安定?）", flush=True)
                continue
            for ci, nm in enumerate(CANON_G1_29):
                ii = canon_to_isaac.get(ci)
                if ii is not None:
                    ls.motor_state[ci].q = float(q[ii])
                    ls.motor_state[ci].dq = float(dq[ii])
            try:
                writer.write(ls)
            except Exception as e:  # noqa: BLE001
                if step % 250 == 0:
                    print(f"[bridge] lowstate write err: {e}", flush=True)

        # 3b) カメラ配信（ego_view ZMQ: color + depth + intrinsics + cam_to_torso）
        if cam is not None and (time.time() - last_cam_pub) >= cam_period:
            try:
                rgba = cam.get_rgba()
                if rgba is not None and getattr(rgba, "size", 0) > 0:
                    bgr = cv2.cvtColor(np.asarray(rgba)[:, :, :3].astype("uint8"), cv2.COLOR_RGB2BGR)
                    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        msg = {
                            "images": {cam_topic: base64.b64encode(jpg.tobytes()).decode("ascii")},
                            "timestamps": {cam_topic: time.time()},
                            "intrinsics": {cam_topic: cam_K},   # [fx,0,cx,0,fy,cy,0,0,1]
                            "cam_to_torso": cam_to_torso_str,    # torso<-optical (x,y,z,qx,qy,qz,qw)
                        }
                        try:
                            depth = cam.get_depth()
                            if depth is not None and getattr(depth, "size", 0) > 0:
                                d = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                                d16 = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)  # mm
                                okd, dpng = cv2.imencode(".png", d16)
                                if okd:
                                    msg["depth"] = {cam_topic: base64.b64encode(dpng.tobytes()).decode("ascii")}
                                    msg["depth_scale"] = 0.001  # mm -> m
                        except Exception:  # noqa: BLE001
                            pass
                        cam_pub.send(msgpack.packb(msg))
                        last_cam_pub = time.time()
            except Exception as e:  # noqa: BLE001
                if step % 250 == 0:
                    print(f"[bridge] cam pub err: {e}", flush=True)

        # 4) ログ（測定値: 右肩pitch/右肘が指令に追従しているか）。間隔は SIM_LOG_EVERY[s] で調整。
        if time.time() - last_log > float(os.getenv("SIM_LOG_EVERY", "2.0")):
            qm = np.asarray(robot.get_joint_positions(), dtype=float)
            mq22 = qm[ii22] if ii22 is not None else float("nan")
            mq25 = qm[ii25] if ii25 is not None else float("nan")
            _fr = ""
            if grasp_friction and gripper_isaac_idx and hand_path and okra_paths:
                # 摩擦把持の客観診断: 手z・最近傍オクラz・指DOF実測。lift で okra z が上がれば把持成立。
                try:
                    _hz = float(_world_xyz(hand_path)[2])
                    _hw = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
                        stage.GetPrimAtPath(hand_path)).ExtractTranslation()
                    _bi, _bz, _bd = -1, float("nan"), 1e9
                    for _i, _op in enumerate(okra_paths):
                        _ow = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
                            stage.GetPrimAtPath(_op)).ExtractTranslation()
                        _dd = (_hw[0]-_ow[0])**2 + (_hw[1]-_ow[1])**2 + (_hw[2]-_ow[2])**2
                        if _dd < _bd:
                            _bi, _bz, _bd = _i, float(_ow[2]), _dd
                    _fg = [round(float(qm[gi]), 4) for gi in gripper_isaac_idx]
                    _fr = f" | hand_z={_hz:.3f} nearest=Okra_{_bi} okra_z={_bz:.3f} dist={_bd**0.5:.3f} finger={_fg}"
                    # ジョー隙間の中心 vs オクラ（torso 系）。delta = okra - gap = IK 目標に足す補正量。
                    if jaw_link_paths and torso_path and _bi >= 0:
                        _xc2 = UsdGeom.XformCache(Usd.TimeCode.Default())
                        _gc = np.mean([np.asarray(_xc2.GetLocalToWorldTransform(
                            stage.GetPrimAtPath(_jp)).ExtractTranslation(), dtype=float)
                            for _jp in jaw_link_paths], axis=0)
                        _ok = np.asarray(_xc2.GetLocalToWorldTransform(
                            stage.GetPrimAtPath(okra_paths[_bi])).ExtractTranslation(), dtype=float)
                        _Tti = _xc2.GetLocalToWorldTransform(stage.GetPrimAtPath(torso_path)).GetInverse()
                        _gct = _Tti.Transform(Gf.Vec3d(*_gc.tolist()))
                        _okt = _Tti.Transform(Gf.Vec3d(*_ok.tolist()))
                        _delta = (round(_okt[0]-_gct[0], 3), round(_okt[1]-_gct[1], 3), round(_okt[2]-_gct[2], 3))
                        _fr += f" gap_torso=({_gct[0]:.3f},{_gct[1]:.3f},{_gct[2]:.3f}) okra_torso=({_okt[0]:.3f},{_okt[1]:.3f},{_okt[2]:.3f}) Δ(okra-gap)={_delta}"
                except Exception:  # noqa: BLE001
                    pass
            print(f"[bridge] step={step} cmds_rx={cmd_count} measured r_shoulder_pitch={mq22:.3f} r_elbow={mq25:.3f} grip_q={grip_q:.2f} grasped={sorted(grasped_set)}{_fr}", flush=True)
            last_log = time.time()

    sim_app.close()


if __name__ == "__main__":
    main()
