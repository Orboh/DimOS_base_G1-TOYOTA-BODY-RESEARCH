"""歩行ポリシー共有ライブラリ — Isaac Sim 上で unitree_rl_lab velocity policy を動かす部品。

sim_walk_policy.py（単体実証: 立位20s転倒0 / 前進6.9m/20s・GUI 5.5m 転倒0, 2026-07-11）で
確立した実装の正本。sim_dds_bridge.py の歩行モード（SIM_WALK_POLICY=1）もこれを使う。

## 実装が守るべき6大要点（全部必要。欠くと転倒/NaN — 詳細はメモリ g1-isaac-policy-walk-floating-base）
1. 参照レイヤの root_joint は RemovePrim/SetActive 不可 → jointEnabled=False override
   ＋ RemoveAPI(ArticulationRootAPI)。ArticulationRoot は実剛体 pelvis へ付け直す。
2. authored の physxArticulation 設定（solver 32反復）は root_joint 側に付いており、root を
   pelvis に移すと PhysX 既定の反復4に落ちる → pelvis に solver 32/4 を再設定。
3. get_angular_velocity() はこの articulation で不正値（静止時も ~6.5rad/s）→ base quat の
   有限差分で角速度を推定する。
4. deploy.yaml の stiffness/damping は **SDKモーター順**（公式C++ State_RLBase.h 準拠・jmap 不要）。
   default/scale/offset だけが policy slot 順（jmap 経由）。
5. armature=0.01 [kg·m²] を全29関節へ（訓練 IsaacLab cfg / mujoco XML と同一前提）。
6. obs[480] は **per-term 履歴連結** [ang×5 | grav×5 | cmd×5 | jpos×5 | jvel×5 | act×5]
   （deploy.yaml に use_gym_history キー無し＝C++既定 false のレイアウト）。

物理は physics_dt=1/200 [s]、policy は step_dt=0.02 [s]（=4サブステップ/tick, 50Hz）。
GUI の world.step(render=True) は rendering_dt(1/50) ぶん＝4サブステップ進むため 1回/tick、
headless は step(render=False)×4（呼び出し側の責務）。
"""
from __future__ import annotations

from collections import deque

import numpy as np
import yaml

# 正準 G1 29 関節名（SDK モーター順 = rt/lowstate・motor_cmd の並び）
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

PHYS_DT = 1.0 / 200.0   # [s] 物理サブステップ（policy 50Hz の 1/4）
BASE_Z0 = 0.80          # [m] 直立配置の pelvis 高さ（default 立位 ≈0.79 の少し上→数stepで接地）


def load_deploy(path: str) -> dict:
    """deploy.yaml を読み、policy 駆動に必要な配列/定数を dict で返す。

    注意: stiffness/damping は SDK モーター順、default/scale/offset は policy slot 順。
    """
    dp = yaml.safe_load(open(path))
    return {
        "jmap": list(dp["joint_ids_map"]),  # policy slot i -> SDK motor index
        "default_q": np.array(dp["default_joint_pos"], dtype=np.float32),      # slot順
        "stiffness": np.array(dp["stiffness"], dtype=np.float32),              # SDK順
        "damping": np.array(dp["damping"], dtype=np.float32),                  # SDK順
        "act_scale": np.array(dp["actions"]["JointPositionAction"]["scale"], dtype=np.float32),
        "act_offset": np.array(dp["actions"]["JointPositionAction"]["offset"], dtype=np.float32),
        "ang_scale": np.array(dp["observations"]["base_ang_vel"]["scale"], dtype=np.float32),
        "jvel_scale": np.array(dp["observations"]["joint_vel_rel"]["scale"], dtype=np.float32),
        "hist": int(dp["observations"]["base_ang_vel"]["history_length"]),
        "step_dt": float(dp["step_dt"]),  # [s]
    }


def make_floating_base(stage, g1_prim, armature: float = 0.01,
                       solver_pos: int = 32, solver_vel: int = 4) -> str | None:
    """base-fix の g1bag（参照読み込み）を floating base 化し、pelvis prim path を返す。

    要点1/2/5 を一括適用。self-collision は False（歩行 policy の検証条件）。
    ArtCls(prim_path=返り値) で articulation を作ること。
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    pelvis_path = None
    for p in Usd.PrimRange(g1_prim):
        if p.GetName() == "pelvis":
            pelvis_path = p.GetPath()
            break
    # (1) world 固定の root_joint を無効化（属性/API override は参照 prim にも効く）
    for p in Usd.PrimRange(g1_prim):
        if p.GetName() == "root_joint":
            UsdPhysics.Joint(p).CreateJointEnabledAttr(False)
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    g1_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)  # 誤って top Xform に付いていたら外す
    if pelvis_path is None:
        return None
    pelvis = stage.GetPrimAtPath(pelvis_path)
    if not pelvis.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(pelvis)
    # (2) solver 設定を pelvis へ（root 移植で既定反復4に落ちるのを防ぐ）
    pa = PhysxSchema.PhysxArticulationAPI.Apply(pelvis)
    pa.CreateSolverPositionIterationCountAttr(solver_pos)
    pa.CreateSolverVelocityIterationCountAttr(solver_vel)
    pa.CreateEnabledSelfCollisionsAttr(False)
    # (5) armature を全29関節へ
    n_arm = 0
    for p in Usd.PrimRange(g1_prim):
        if p.GetName() in CANON_G1_29:
            PhysxSchema.PhysxJointAPI.Apply(p).CreateArmatureAttr(armature)
            n_arm += 1
    print(f"[walklib] floating base 化: root={pelvis_path} solver={solver_pos}/{solver_vel} "
          f"selfColl=False armature={armature}×{n_arm}関節", flush=True)
    return pelvis_path.pathString


def motor_to_isaac_map(dof_names: list[str]) -> list[int | None]:
    """SDK モーター index → Isaac dof index の対応（名前一致）。"""
    return [dof_names.index(nm) if nm in dof_names else None for nm in CANON_G1_29]


def apply_sdk_gains(robot, dp: dict, isaac_idx_for_motor: list[int | None],
                    kp_scale: float = 1.0) -> None:
    """deploy.yaml の PD gains を **SDKモーター順のまま** Isaac dof に適用（要点4）。

    非 canon dof（Dex1 指など）のゲインは触らない（現状値を維持）。
    """
    ctrl = robot.get_articulation_controller()
    cur = ctrl.get_gains()
    kp = np.asarray(cur[0], dtype=np.float32).reshape(-1).copy()
    kd = np.asarray(cur[1], dtype=np.float32).reshape(-1).copy()
    for m in range(29):
        ii = isaac_idx_for_motor[m]
        if ii is not None:
            kp[ii] = dp["stiffness"][m] * kp_scale
            kd[ii] = dp["damping"][m] * kp_scale
    ctrl.set_gains(kps=kp, kds=kd)
    knee = isaac_idx_for_motor[3]
    print(f"[walklib] SDK順 PD gains 適用 kp_scale={kp_scale}"
          f"（knee kp={kp[knee]:.0f} kd={kd[knee]:.1f}）", flush=True)


class PolicyWalker:
    """velocity policy の 1tick 実行器（obs 構築→onnx→ full-dof 位置目標ベクトル）。

    呼び出し側の責務:
      - tick() が返す目標に腕/指の上書きを乗せて apply_action し、20ms（=4×PHYS_DT）進める
      - GUI: world.step(render=True)×1 / headless: world.step(render=False)×4（要点: GUI4倍進む罠）
    """

    _TERMS = ("ang", "grav", "cmd", "jpos", "jvel", "act")

    def __init__(self, onnx_path: str, dp: dict, isaac_idx_for_motor: list[int | None]):
        import onnxruntime as ort

        self.dp = dp
        self.map = isaac_idx_for_motor
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.reset()

    def reset(self) -> None:
        """履歴/last_action を初期化（policy 開始・停止から再開時に呼ぶ）。"""
        self._hist: dict[str, deque] = {k: deque(maxlen=self.dp["hist"]) for k in self._TERMS}
        self.last_action = np.zeros(29, dtype=np.float32)
        self._prev_q: np.ndarray | None = None
        self.dbg: dict = {}

    def default_full(self, robot) -> np.ndarray:
        """現姿勢に default_joint_pos（29関節）を上書きした full-dof ベクトル（初期配置用）。"""
        q = np.asarray(robot.get_joint_positions(), dtype=np.float32).reshape(-1).copy()
        jmap, default_q = self.dp["jmap"], self.dp["default_q"]
        for i in range(29):
            ii = self.map[jmap[i]]
            if ii is not None:
                q[ii] = default_q[i]
        return q

    def place_upright(self, robot, world, render: bool = False, settle: int = 5,
                      base_z: float = BASE_Z0, xy: tuple[float, float] = (0.0, 0.0)) -> None:
        """reset 直後（★一度も step する前）に直立配置し、短い settle で接地させる。

        authored 姿勢は足が地面を 0.757m 踏み抜くため、先に step すると射出される（要点の罠）。
        settle は既定5tick（静的保持は metastable ですぐ policy に渡すのが正解）。
        """
        from isaacsim.core.utils.types import ArticulationAction

        q_init = self.default_full(robot)
        robot.set_world_pose(position=np.array([xy[0], xy[1], base_z], dtype=np.float32),
                             orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        robot.set_joint_positions(q_init)
        # ★速度の完全零化（配置直後・settle 前）: world.reset() 時に authored 姿勢の脚が床の
        #   体積コライダー（例: chinou の床ボックス厚0.2m）を貫通していると、PhysX の貫通解消で
        #   articulation に ~4m/s 級の初速が付く。set_world_pose は姿勢しか直さないため、
        #   零化しないと初速のまま吹っ飛ぶ（部屋モードで実測）。
        try:
            robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=np.float32))
        except Exception:  # noqa: BLE001
            pass
        try:
            robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        except Exception as _e:  # noqa: BLE001
            print(f"[walklib] base velocity 零化 warn: {_e}", flush=True)
        for _ in range(settle):
            robot.apply_action(ArticulationAction(joint_positions=q_init))
            if render:
                world.step(render=True)
            else:
                for _s in range(4):
                    world.step(render=False)
        try:
            robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=np.float32))
        except Exception:  # noqa: BLE001
            pass
        p, q = robot.get_world_pose()
        print(f"[walklib] 直立配置+settle{settle}: pelvis_z={float(np.asarray(p).reshape(-1)[2]):.3f} "
              f"quat_w={float(np.asarray(q).reshape(-1)[0]):.3f}", flush=True)

    def _base_obs(self, robot) -> tuple[np.ndarray, np.ndarray]:
        """base 角速度（body系, quat 有限差分＝要点3）と projected_gravity（body系）。"""
        pos, quat = robot.get_world_pose()
        q = np.asarray(quat, dtype=np.float32).reshape(-1)[:4]
        qw, qx, qy, qz = (float(v) for v in q)
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ], dtype=np.float32)
        g_body = R.T @ np.array([0, 0, -1], dtype=np.float32)
        if self._prev_q is None:
            w_body = np.zeros(3, dtype=np.float32)
        else:
            qp = self._prev_q
            cw, cx, cy, cz = qp[0], -qp[1], -qp[2], -qp[3]  # conj(prev)
            dw = qw * cw - qx * cx - qy * cy - qz * cz
            dx = qw * cx + qx * cw + qy * cz - qz * cy
            dy = qw * cy - qx * cz + qy * cw + qz * cx
            dz = qw * cz + qx * cy - qy * cx + qz * cw
            vnorm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            angle = 2.0 * float(np.arctan2(vnorm, abs(dw)))  # [rad]
            if angle > np.pi:
                angle -= 2.0 * np.pi
            if vnorm > 1e-8:
                sgn = 1.0 if dw >= 0 else -1.0
                axis = np.array([dx, dy, dz], dtype=np.float32) * (sgn / vnorm)
                w_world = axis * (angle / self.dp["step_dt"])  # [rad/s]
                w_body = (R.T @ w_world).astype(np.float32)
            else:
                w_body = np.zeros(3, dtype=np.float32)
        self._prev_q = q.copy()
        return w_body, g_body

    def tick(self, robot, cmd, mask_motors: list[int] | None = None) -> np.ndarray:
        """policy 1step: obs[480](per-term) → onnx → full-dof 位置目標ベクトルを返す。

        cmd = (vx, vy, wz) body系 [m/s, m/s, rad/s]。返り値の腕/指 index を上書きしてから
        apply_action してよい（policy へは自身の raw action が last_action として入る）。

        mask_motors: SDK モーター index のリスト。指定した関節を obs 上で「常に default・
        静止・action=0」に見せる（=policy の管轄から外す）。腕を外部制御（arm_sdk/IK）で
        default から大きく動かすと policy が訓練分布外の腕状態を見て転倒するため、
        収穫 bridge では腕14関節（15..28）を常時マスクする。腕の質量移動は外乱として
        policy の頑健性で吸収させる。
        """
        dp, jmap = self.dp, self.dp["jmap"]
        q = np.asarray(robot.get_joint_positions(), dtype=np.float32).reshape(-1)
        dq = np.asarray(robot.get_joint_velocities(), dtype=np.float32).reshape(-1)
        mq = np.zeros(29, dtype=np.float32)
        mdq = np.zeros(29, dtype=np.float32)
        for m in range(29):
            ii = self.map[m]
            if ii is not None:
                mq[m] = q[ii]
                mdq[m] = dq[ii]
        jpos_rel = np.array([mq[jmap[i]] - dp["default_q"][i] for i in range(29)], dtype=np.float32)
        jvel_rel = np.array([mdq[jmap[i]] for i in range(29)], dtype=np.float32) * dp["jvel_scale"]
        act_obs = self.last_action.astype(np.float32)
        if mask_motors:
            _mm = set(mask_motors)
            for i in range(29):
                if jmap[i] in _mm:
                    jpos_rel[i] = 0.0
                    jvel_rel[i] = 0.0
                    act_obs = act_obs.copy()
                    act_obs[i] = 0.0
        w, g = self._base_obs(robot)
        terms = {"ang": (w * dp["ang_scale"]).astype(np.float32), "grav": g.astype(np.float32),
                 "cmd": np.asarray(cmd, dtype=np.float32).reshape(3),
                 "jpos": jpos_rel, "jvel": jvel_rel, "act": act_obs}
        for k in self._TERMS:
            if not self._hist[k]:  # 初回: 履歴を同値で埋める（C++ reset() と同じ）
                for _ in range(dp["hist"]):
                    self._hist[k].append(terms[k].copy())
            else:
                self._hist[k].append(terms[k].copy())
        obs = np.concatenate([np.concatenate(list(self._hist[k])) for k in self._TERMS]
                             ).astype(np.float32)[None, :]
        action = self.sess.run(None, {self.in_name: obs})[0].reshape(-1)[:29].astype(np.float32)
        self.last_action = action.copy()
        self.dbg = {"grav": g, "angvel": w, "action": action}
        # 目標: q_tgt[slot i] = action[i]*scale + offset を motor jmap[i] → Isaac dof へ
        tgt = q.copy()
        q_tgt_motor = action * dp["act_scale"] + dp["act_offset"]
        for i in range(29):
            ii = self.map[jmap[i]]
            if ii is not None:
                tgt[ii] = q_tgt_motor[i]
        return tgt
