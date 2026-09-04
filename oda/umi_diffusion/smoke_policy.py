#!/usr/bin/env python
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

"""
Step 1 de-risk (NO ROBOT): load the trained UMI diffusion ckpt in the `umi` conda env,
run predict_action on a dummy obs, and verify:
  - checkpoint loads (deps match training),
  - action conversion returns (N, 7) [pos3 + axis-angle3 + gripper1],
  - inference latency fits the 10 Hz control budget (<100 ms) on this PC's GPU,
  - dump cfg.task (shape_meta, pose_repr, mirror/fisheye-relevant fields) for Step 2/3.

Run:
  conda run -n umi python oda/umi_diffusion/smoke_policy.py
"""

import os
import sys
import time
import traceback

import numpy as np

UMI_ROOT = os.path.expanduser("~/umi/universal_manipulation_interface")
CKPT = os.path.expanduser("~/umi/epoch=0110-train_loss=0.012.ckpt")

sys.path.append(UMI_ROOT)
os.chdir(UMI_ROOT)  # match eval_real_umi.py (some configs use cwd-relative paths)

from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
import dill
import hydra
from omegaconf import OmegaConf
import torch
from umi.common.pose_util import mat_to_pose, pose10d_to_mat, pose_to_mat
from umi.real_world.real_inference_util import get_real_umi_obs_dict

OmegaConf.register_new_resolver("eval", eval, replace=True)


def decode_umi_action(raw_action, env_obs, action_pose_repr):
    """Decode a UMI diffusion action to absolute EE poses in the arm ROOT frame.

    This checkpoint was trained with action_include_gripper=False, so the action is
    9-dim (pos3 + rot6d) — NOT the stock 10-dim (which stock get_real_umi_action
    assumes via //10). We therefore decode the 9-dim pose10d directly. Returns (N, 6)
    = pos3 + axis-angle3 (no gripper: the gripper is the user's separate program).
    """
    pose10d = raw_action[..., :9]  # tolerate a trailing gripper col if ever present
    action_pose_mat = pose10d_to_mat(pose10d)
    cur = pose_to_mat(
        np.concatenate(
            [
                env_obs["robot0_eef_pos"][-1],
                env_obs["robot0_eef_rot_axis_angle"][-1],
            ],
            axis=-1,
        )
    )
    action_mat = convert_pose_mat_rep(
        action_pose_mat, base_pose_mat=cur, pose_rep=action_pose_repr, backward=True
    )
    return mat_to_pose(action_mat)  # (N, 6) pos + axis-angle


def main():
    assert os.path.exists(CKPT), f"ckpt not found: {CKPT}"
    print(
        f"torch {torch.__version__} | cuda avail {torch.cuda.is_available()} | "
        f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    payload = torch.load(open(CKPT, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    print("\n=== ckpt cfg summary ===")
    print("workspace _target_:", cfg._target_)
    print("obs_encoder model_name:", cfg.policy.obs_encoder.model_name)
    print(
        "obs_pose_repr:",
        cfg.task.pose_repr.obs_pose_repr,
        "| action_pose_repr:",
        cfg.task.pose_repr.action_pose_repr,
    )
    print("use_ema:", cfg.training.use_ema)
    try:
        print("dataset_path:", cfg.task.dataset.dataset_path)
    except Exception:
        pass

    shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    print("obs keys:", list(shape_meta["obs"].keys()))
    for k, v in shape_meta["obs"].items():
        print(f"   {k}: {v}")
    print("action:", shape_meta["action"])

    # Dump the full task cfg — this is where mirror / fisheye / sim_fov / camera settings live (Step 2/3).
    print("\n=== cfg.task (full) ===")
    print(OmegaConf.to_yaml(cfg.task))

    # build policy exactly like eval_real_umi.py
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = 16
    device = torch.device("cuda")
    policy.eval().to(device)
    obs_pose_repr = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr

    # dummy obs (horizon T, gripper_width pinned to the training-constant 1e-4)
    co, ho, wo = shape_meta["obs"]["camera0_rgb"]["shape"]
    T = shape_meta["obs"]["camera0_rgb"]["horizon"]
    # a plausible pre-grasp EE pose in the arm-root frame (pos + axis-angle)
    eef_pos = np.array([0.02, -0.22, 0.29], dtype=np.float32)
    eef_rot = np.array([-2.0, 0.03, 0.03], dtype=np.float32)
    env_obs = {
        "camera0_rgb": np.zeros((T, ho, wo, 3), dtype=np.uint8),
        "robot0_eef_pos": np.tile(eef_pos, (T, 1)),
        "robot0_eef_rot_axis_angle": np.tile(eef_rot, (T, 1)),
        "robot0_gripper_width": np.full((T, 1), 1e-4, dtype=np.float32),
        "timestamp": np.arange(T, dtype=np.float64),
    }
    episode_start_pose = [np.concatenate([eef_pos, eef_rot]).astype(np.float64)]

    obs_dict_np = get_real_umi_obs_dict(
        env_obs=env_obs,
        shape_meta=shape_meta,
        obs_pose_repr=obs_pose_repr,
        episode_start_pose=episode_start_pose,
    )
    print("\n=== built obs_dict shapes ===")
    for k, v in obs_dict_np.items():
        print(f"   {k}: {v.shape} {v.dtype}")
    obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

    with torch.no_grad():
        policy.reset()
        result = policy.predict_action(obs_dict)
        raw = result["action_pred"][0].detach().cpu().numpy()
        print("\nraw action shape:", raw.shape, "(this ckpt: 9 = pos3+rot6d, NO gripper)")
        assert raw.shape[-1] in (9, 10), f"unexpected raw action dim {raw.shape}"
        act = decode_umi_action(raw, env_obs, action_pose_repr)
        print("decoded EE action shape:", act.shape, "(expect (N,6) pos+axis-angle)")
        assert act.shape[-1] == 6, f"expected decoded action dim 6, got {act.shape}"
        print("first waypoint  pos:", np.round(act[0, :3], 4), " aa:", np.round(act[0, 3:], 4))
        print("last  waypoint  pos:", np.round(act[-1, :3], 4), " aa:", np.round(act[-1, 3:], 4))

        # timed
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        N = 10
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        for _ in range(N):
            _ = policy.predict_action(obs_dict)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt = (time.time() - t0) / N

    print(f"\nmean inference latency: {dt * 1000:.1f} ms  (10 Hz budget = 100 ms)")
    if torch.cuda.is_available():
        print(
            f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB (this PC RTX3070 = 8 GB)"
        )
    verdict = "OK" if dt < 0.1 else "OVER-BUDGET (consider Orin / steps_per_inference)"
    print(f"\nSMOKE {verdict}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
