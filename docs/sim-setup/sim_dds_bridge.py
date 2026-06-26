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
    world = World(stage_units_in_meters=1.0)  # fix_base G1 は地面不要（cloud asset 取得を避ける）
    stage = omni.usd.get_context().get_stage()

    # 重力: 既定OFF。g1bag は関節 PD が弱く重力下で腕が垂れて指令角を保持できないため、
    # 運動学の追従確認（S0/IK）では重力OFFで腕を指令角に固定する。SIM_GRAVITY=1 で
    # 通常重力（把持/籠/切断の動力学を物理検証する時）。view_chinou.py と同じ方針。
    if os.getenv("SIM_GRAVITY", "0") != "1":
        try:
            world.get_physics_context().set_gravity(0.0)  # [m/s^2]
            print("[bridge] gravity OFF（弱PDのg1bag腕を指令角で保持。SIM_GRAVITY=1で重力ON）", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[bridge] set_gravity failed: {_e}", flush=True)

    add_reference_to_stage(usd_path=G1_USD, prim_path="/G1")
    g1 = stage.GetPrimAtPath("/G1")
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    lift = -float(bbc.ComputeWorldBound(g1).ComputeAlignedRange().GetMin()[2])
    UsdGeom.XformCommonAPI(g1).SetTranslate(Gf.Vec3d(0.0, 0.0, lift))

    art_root = None
    for p in Usd.PrimRange(g1):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_root = p.GetPath().pathString
            break
    robot = ArtCls(prim_path=art_root or "/G1", name="g1")
    world.scene.add(robot)

    # 机上オクラ（A配置）: SIM_TABLE=1 で view_chinou と同一配置を載せる（M2/M3 用）。
    # world.reset() の前に stage へ追加する。配置の正本は sim_scene.build_table_okra。
    okra_paths: list[str] = []
    hand_path = None  # 右手リンク（M3 机上ピックでオクラを固定する先）
    if os.getenv("SIM_TABLE", "0") == "1":
        import sim_scene  # 同ディレクトリ（docs/sim-setup）

        n_okra = int(os.getenv("SIM_OKRA", "10"))
        table_h = float(os.getenv("SIM_TABLE_H", "0.72"))
        okra_paths = sim_scene.build_table_okra(stage, table_h=table_h, n_okra=n_okra)
        for _p in Usd.PrimRange(g1):
            if _p.GetName() == "right_hand_base_link":
                hand_path = _p.GetPath().pathString
                break
        print(f"[bridge] 机+オクラ {len(okra_paths)}本 配置（A配置, 天板{table_h}m）hand={hand_path}", flush=True)

    world.reset()
    try:
        robot.set_world_pose(position=np.array([0.0, 0.0, lift]))
    except Exception:  # noqa: BLE001
        pass
    for _ in range(20):
        world.step(render=not headless)

    dof_names = list(robot.dof_names)
    # 正準index -> Isaac dof index のマップ（名前一致）
    canon_to_isaac = {}
    for ci, nm in enumerate(CANON_G1_29):
        if nm in dof_names:
            canon_to_isaac[ci] = dof_names.index(nm)
    arm_isaac_idx = [canon_to_isaac[ci] for ci in ARM_CANON_IDX if ci in canon_to_isaac]
    print(f"[bridge] num_dof={robot.num_dof} mapped {len(canon_to_isaac)}/29 canon joints; "
          f"arm mapped={len(arm_isaac_idx)}", flush=True)

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
    if os.getenv("SIM_ADD_LIGHT", "1") == "1":
        try:
            from pxr import UsdLux

            UsdLux.DomeLight.Define(stage, "/World/sim_fill_dome").CreateIntensityAttr(
                float(os.getenv("SIM_LIGHT_INTENSITY", "1500"))
            )
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

        # intrinsics（HFOV から焦点距離を設定。比 focal/aperture が画角を決める＝単位は相殺）
        _A = 20.955
        _F = (_A / 2.0) / math.tan(hfov / 2.0)
        try:
            cam.set_horizontal_aperture(_A)
            cam.set_vertical_aperture(_A * cam_h / cam_w)
            cam.set_focal_length(_F)
        except Exception as _e:  # noqa: BLE001
            print(f"[bridge] set intrinsics warn: {_e}", flush=True)
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
    grasp_target = int(os.getenv("SIM_GRASP_OKRA", "1"))   # 既定 /Okra_1（単発時）
    grasp_close = float(os.getenv("SIM_GRASP_CLOSE", "2.0"))  # cmds[0].q がこれ以上で閉じ=把持
    grasp_target_file = os.getenv("SIM_GRASP_TARGET_FILE", "/tmp/sim_grasp_target.txt")  # graph がここに次の対象を書く
    _go = [float(x) for x in os.getenv("SIM_GRASP_OFFSET", "0,0,0").split(",")]  # 把持位置オフセット

    print("[bridge] loop start（Ctrl-C / touch /tmp/sim_bridge_stop で終了）", flush=True)
    while True:
        if run_secs > 0 and (time.time() - t0) > run_secs:
            print("[bridge] run_secs reached -> stop", flush=True)
            break
        if os.path.exists(stop_file):
            print("[bridge] stop file -> stop", flush=True)
            break
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
                q_full = np.asarray(robot.get_joint_positions(), dtype=float)
                tgt = q_full.copy()
                for ci in ARM_CANON_IDX:
                    ii = canon_to_isaac.get(ci)
                    if ii is not None:
                        tgt[ii] = float(lc.motor_cmd[ci].q)
                robot.apply_action(ArticulationAction(joint_positions=tgt))
                last_q_target = tgt

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
        # 把持対象 index: graph が書くファイル優先、無ければ env 既定（単発）
        gt_idx = grasp_target
        try:
            with open(grasp_target_file) as _f:
                gt_idx = int(_f.read().strip())
        except Exception:  # noqa: BLE001
            pass
        # 閉じ かつ 未把持の対象 → world アンカー除去＋手リンクへ FixedJoint（複数可, ユニーク joint）
        if grip_q >= grasp_close and hand_path and gt_idx not in grasped_set and 0 <= gt_idx < len(okra_paths):
            okp = okra_paths[gt_idx]
            wj = f"/World/OkraJoints/joint_{gt_idx}"
            if stage.GetPrimAtPath(wj):
                stage.RemovePrim(wj)
            gj = UsdPhysics.FixedJoint.Define(stage, f"/World/GraspJoint_{gt_idx}")
            gj.CreateBody0Rel().SetTargets([hand_path])
            gj.CreateBody1Rel().SetTargets([okp])
            gj.CreateLocalPos0Attr(Gf.Vec3f(_go[0], _go[1], _go[2]))
            gj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            grasped_set.add(gt_idx)
            print(f"[bridge] GRASP {okp} → {hand_path}（idx={gt_idx}, grip_q={grip_q:.2f}, 計{len(grasped_set)}本）", flush=True)

        # 2) sim 1step
        world.step(render=render_on)
        step += 1

        # 3) lowstate 発行
        if step % pub_every == 0:
            q = np.asarray(robot.get_joint_positions(), dtype=float)
            dq = np.asarray(robot.get_joint_velocities(), dtype=float)
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

        # 4) ログ（測定値: 右肩pitch/右肘が指令に追従しているか）
        if time.time() - last_log > 2.0:
            qm = np.asarray(robot.get_joint_positions(), dtype=float)
            mq22 = qm[ii22] if ii22 is not None else float("nan")
            mq25 = qm[ii25] if ii25 is not None else float("nan")
            print(f"[bridge] step={step} cmds_rx={cmd_count} measured r_shoulder_pitch={mq22:.3f} r_elbow={mq25:.3f} grip_q={grip_q:.2f} grasped={sorted(grasped_set)}", flush=True)
            last_log = time.time()

    sim_app.close()


if __name__ == "__main__":
    main()
