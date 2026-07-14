#!/usr/bin/env python3
"""公式 sim2sim 歩行コントローラ（A方式）: unitree_mujoco の G1 を unitree_rl_lab の velocity
policy で歩かせる。rt/lowstate 購読→obs[480]→policy.onnx→rt/lowcmd 発行（PD）。LocoClient 相当。

構成（実機と同じ DDS）:
  [端末1] unitree_mujoco（python, ROBOT="g1", DOMAIN_ID=1, INTERFACE="lo"）
      cd ~/Desktop/unitree_mujoco/simulate_python && python unitree_mujoco.py
  [端末2] 本コントローラ
      .venv/bin/python docs/sim-setup/sim_walk_mujoco_ctrl.py
      WALK_VX=0.0（立つ）→ 立てたら WALK_VX=0.3（前進）

policy 仕様（unitree_rl_lab g1_29dof velocity/v0, deploy.yaml）:
  obs[480] = for h 0..4(最古→最新): [ang_vel×0.2(3), proj_grav(3), cmd(3),
              jpos_rel(29), jvel_rel(29), last_action(29)]
  action[29] → motor[jmap[i]].q = action[i]*0.25 + default[i]、PD=stiffness/damping、50Hz。
  起動時 FixStand（default 姿勢へ ~2s ランプ）→ Velocity（policy）。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import yaml

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, "/home/kota-ueda/Desktop/unitree_sdk2_python")
POLICY = f"{REPO}/usd_file/walk_policy/policy.onnx"
DEPLOY_YAML = f"{REPO}/usd_file/walk_policy/deploy.yaml"

DOMAIN_ID = int(os.getenv("WALK_DOMAIN", "1"))   # unitree_mujoco config.py と一致（既定1）
IFACE = os.getenv("WALK_IFACE", "lo")


def main() -> int:
    vx = float(os.getenv("WALK_VX", "0.0"))
    vy = float(os.getenv("WALK_VY", "0.0"))
    wz = float(os.getenv("WALK_WZ", "0.0"))
    secs = float(os.getenv("WALK_SECS", "20"))

    dp = yaml.safe_load(open(DEPLOY_YAML))
    jmap = list(dp["joint_ids_map"])                # policy slot i -> motor index
    default_q = np.array(dp["default_joint_pos"], dtype=np.float32)
    stiffness = np.array(dp["stiffness"], dtype=np.float32)
    damping = np.array(dp["damping"], dtype=np.float32)
    act_scale = np.array(dp["actions"]["JointPositionAction"]["scale"], dtype=np.float32)
    act_offset = np.array(dp["actions"]["JointPositionAction"]["offset"], dtype=np.float32)
    ang_scale = np.array(dp["observations"]["base_ang_vel"]["scale"], dtype=np.float32)
    jvel_scale = np.array(dp["observations"]["joint_vel_rel"]["scale"], dtype=np.float32)
    HIST = int(dp["observations"]["base_ang_vel"]["history_length"])
    step_dt = float(dp["step_dt"])

    import onnxruntime as ort
    sess = ort.InferenceSession(POLICY, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as make_lowcmd
    from unitree_sdk2py.utils.crc import CRC

    ChannelFactoryInitialize(DOMAIN_ID, IFACE)
    crc = CRC()
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    state = {"q": np.zeros(29, np.float32), "dq": np.zeros(29, np.float32),
             "gyro": np.zeros(3, np.float32), "quat": np.array([1, 0, 0, 0], np.float32),
             "mode_machine": 0, "got": False}

    def on_state(msg: "LowState_") -> None:
        for i in range(29):
            state["q"][i] = msg.motor_state[i].q
            state["dq"][i] = msg.motor_state[i].dq
        state["gyro"] = np.array(msg.imu_state.gyroscope, np.float32)
        state["quat"] = np.array(msg.imu_state.quaternion, np.float32)  # (w,x,y,z)
        state["mode_machine"] = msg.mode_machine
        state["got"] = True

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    print(f"[walk-mj] DDS up domain={DOMAIN_ID} iface={IFACE}; rt/lowstate待ち…", flush=True)
    t0 = time.time()
    while not state["got"] and time.time() - t0 < 10:
        time.sleep(0.05)
    if not state["got"]:
        print("[walk-mj] ❌ rt/lowstate が来ない。unitree_mujoco(ROBOT=g1,DOMAIN_ID=1) 起動中か確認", flush=True)
        return 1
    print(f"[walk-mj] lowstate 受信 OK（mode_machine={state['mode_machine']}）", flush=True)

    lc = make_lowcmd()
    for i in range(29):
        lc.motor_cmd[i].mode = 1  # PR mode

    def send(q_motor, kp, kd):
        lc.mode_machine = state["mode_machine"]
        for i in range(29):
            lc.motor_cmd[i].q = float(q_motor[i])
            lc.motor_cmd[i].dq = 0.0
            lc.motor_cmd[i].tau = 0.0
            lc.motor_cmd[i].kp = float(kp[i])
            lc.motor_cmd[i].kd = float(kd[i])
        lc.crc = crc.Crc(lc)
        pub.Write(lc)

    # PD（motor 順）: policy slot i -> motor jmap[i]
    kp = np.zeros(29, np.float32); kd = np.zeros(29, np.float32)
    default_motor = np.zeros(29, np.float32)
    for i in range(29):
        kp[jmap[i]] = stiffness[i]; kd[jmap[i]] = damping[i]
        default_motor[jmap[i]] = default_q[i]

    # ① FixStand: 現在姿勢 → default へ 2s ランプ（PD で立たせる）
    q_start = state["q"].copy()
    T = 2.0
    n = int(T / step_dt)
    print("[walk-mj] FixStand: default 姿勢へランプ", flush=True)
    for k in range(n):
        s = (k + 1) / n
        send(q_start * (1 - s) + default_motor * s, kp, kd)
        time.sleep(step_dt)

    # ② Velocity: policy ループ（50Hz）
    hist: list[np.ndarray] = []
    last_action = np.zeros(29, np.float32)
    cmd = np.array([vx, vy, wz], np.float32)

    def proj_gravity(quat):
        qw, qx, qy, qz = [float(v) for v in quat]
        R = np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
        ], np.float32)
        return R.T @ np.array([0, 0, -1], np.float32)

    print(f"[walk-mj] Velocity policy 開始 cmd=({vx},{vy},{wz}) {secs}s", flush=True)
    t0 = time.time()
    step = 0
    while time.time() - t0 < secs:
        q = state["q"]; dq = state["dq"]
        jpos_rel = np.array([q[jmap[i]] - default_q[i] for i in range(29)], np.float32)
        jvel_rel = np.array([dq[jmap[i]] for i in range(29)], np.float32) * jvel_scale
        step_vec = np.concatenate([
            state["gyro"] * ang_scale, proj_gravity(state["quat"]), cmd,
            jpos_rel, jvel_rel, last_action,
        ]).astype(np.float32)  # 96
        hist.append(step_vec)
        while len(hist) < HIST:
            hist.append(step_vec.copy())
        if len(hist) > HIST:
            hist.pop(0)
        obs = np.concatenate(hist).astype(np.float32)[None, :]  # [1,480] 最古→最新
        action = sess.run(None, {in_name: obs})[0].reshape(-1)[:29].astype(np.float32)
        last_action = action.copy()
        q_tgt = np.zeros(29, np.float32)
        for i in range(29):
            q_tgt[jmap[i]] = action[i] * act_scale[i] + act_offset[i]
        send(q_tgt, kp, kd)
        step += 1
        if step % 50 == 0:
            print(f"[walk-mj] t={time.time()-t0:.1f}s mode={state['mode_machine']} "
                  f"gz={state['gyro'][2]:+.2f}", flush=True)
        time.sleep(step_dt)

    print("[walk-mj] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
