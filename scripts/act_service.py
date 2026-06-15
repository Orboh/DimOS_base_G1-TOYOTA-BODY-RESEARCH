#!/usr/bin/env python3
"""ACT inference service, bridged to dimos over ZMQ.

Runs in the dedicated lerobot venv (.venv_act, separate process from dimos). It
owns the heavy lerobot/torch dependency; dimos stays clean and talks to it over
a neutral ZMQ wire (msgpack), so neither side needs the other's Python types.

Wire protocol (ZMQ REP on tcp://127.0.0.1:5701) — UNCHANGED:
  request  (msgpack): {"state": [16 floats], "image_jpeg": <bytes>, "reset": <bool optional>}
  response (msgpack): {"action": [16 floats]}

State / action layout (identity-mapped to dimos):
  [0:7]   left arm   (dimos motor index 15-21)
  [7:14]  right arm  (dimos motor index 22-28)
  [14]    left gripper  (Dex1, constant 0)   [15] right gripper (Dex1)

IMPORTANT — normalization (root-cause fix 2026-06-12):
  This lerobot version moved normalization OUT of the policy and INTO a
  preprocessor/postprocessor pipeline built from the *dataset stats*
  (`make_pre_post_processors`). Calling ``policy.select_action`` on RAW values —
  as this file used to — feeds the network un-normalized inputs and returns
  normalized-space outputs, which we were sending to the robot as raw radians:
  the arms moved, but to garbage targets ("strange motion").

  We now run the EXACT verified eval_g1.py path:
      raw obs -> preprocessor(normalize state+image) -> policy -> postprocessor(un-normalize) -> action
  Offline check vs the dataset's recorded first action: max error 0.056 rad
  (the old raw path was off by 2.21 rad).

Run (from the dimos repo root; the venv lives at ~/act-okura/.venv_act):
  service : ~/act-okura/.venv_act/bin/python scripts/act_service.py --serve
  selftest: ~/act-okura/.venv_act/bin/python scripts/act_service.py --selftest
"""

from __future__ import annotations

import argparse
from copy import copy

import cv2
import msgpack
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.utils import get_safe_torch_device

REPO_ID = "sotata/act-okura-pick-06102026"   # policy checkpoint
DATASET_REPO = "Orboh/okura-sub-lerobot"      # source of normalization stats + task
IMG_KEY = "observation.images.cam_left_high"
STATE_KEY = "observation.state"
IMG_H, IMG_W = 480, 640  # model expects [3, 480, 640]
STATE_DIM = 16
ENDPOINT = "tcp://127.0.0.1:5701"


class ActService:
    def __init__(self, repo_id: str = REPO_ID, dataset_repo: str = DATASET_REPO) -> None:
        self.device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")

        # The dataset provides the normalization stats AND the task string the
        # policy was trained with (e.g. "pick up cube.").
        dataset = LeRobotDataset(repo_id=dataset_repo)
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        self.task = dataset[from_idx].get("task", "") if hasattr(dataset[from_idx], "get") else ""

        cfg = PreTrainedConfig.from_pretrained(repo_id)
        cfg.pretrained_path = repo_id
        self.policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=repo_id,
            dataset_stats=rename_stats(dataset.meta.stats, {}),
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        self._reset()
        print(f"[act] loaded {repo_id} on {self.device} | task={self.task!r} "
              f"| normalization via preprocessor/postprocessor (dataset={dataset_repo})")

    def _reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    @torch.no_grad()
    def infer(self, state: np.ndarray, bgr_image: np.ndarray, reset: bool = False) -> np.ndarray:
        """Run one closed-loop step exactly as eval_g1.py's predict_action does.

        ``state`` is the raw 16-dim observation (radians); ``bgr_image`` is the
        decoded camera frame (BGR, as cv2.imdecode returns). Normalization of
        state + image and un-normalization of the action are handled by the
        lerobot preprocessor/postprocessor — NOT done here by hand.
        """
        if reset:
            self._reset()

        # BGR (cv2) -> RGB, HxWxC uint8. predict_action does /255 + CHW + normalize.
        if bgr_image.shape[:2] != (IMG_H, IMG_W):
            bgr_image = cv2.resize(bgr_image, (IMG_W, IMG_H))  # cv2 size = (W, H)
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        observation = {
            STATE_KEY: torch.from_numpy(state.astype(np.float32)),
            IMG_KEY: torch.from_numpy(np.ascontiguousarray(rgb)),  # HWC uint8
        }
        # --- predict_action (verbatim from unitree_lerobot eval path) ---
        observation = copy(observation)
        for name in list(observation):
            if not hasattr(observation[name], "unsqueeze"):
                continue
            if "images" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0).to(self.device)
        observation["task"] = self.task
        observation["robot_type"] = ""
        observation = self.preprocessor(observation)
        action = self.policy.select_action(observation)
        action = self.postprocessor(action)
        return action.squeeze(0).to("cpu").numpy()

    def serve(self, endpoint: str = ENDPOINT) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REP)
        sock.bind(endpoint)
        print(f"[act] serving on {endpoint} (Ctrl-C to stop)", flush=True)
        while True:
            req = msgpack.unpackb(sock.recv(), raw=False)
            state = np.asarray(req["state"], dtype=np.float32)
            jpeg = np.frombuffer(req["image_jpeg"], dtype=np.uint8)
            bgr = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
            action = self.infer(state, bgr, reset=bool(req.get("reset", False)))
            sock.send(msgpack.packb({"action": action.astype(float).tolist()}, use_bin_type=True))


def _selftest() -> int:
    """Verify the service reproduces the dataset's recorded actions (normalization OK)."""
    svc = ActService()
    dataset = LeRobotDataset(repo_id=DATASET_REPO)
    from_idx = dataset.meta.episodes["dataset_from_index"][0]
    frame = dataset[from_idx]
    state = frame[STATE_KEY].float().numpy()
    img_chw = frame[IMG_KEY]
    # dataset image is CHW float[0,1]; convert to the BGR uint8 a camera would give
    rgb_uint8 = (img_chw.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    rec = frame["action"].float().numpy()

    a = svc.infer(state, bgr, reset=True)
    err = float(np.max(np.abs(a - rec)))
    np.set_printoptions(precision=3, suppress=True)
    print(f"\n[selftest] recorded action : {rec}")
    print(f"[selftest] service action  : {a}")
    print(f"[selftest] max|err| = {err:.4f} rad  -> {'OK' if err < 0.2 else 'FAIL (normalization?)'}")
    return 0 if err < 0.2 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true", help="run the ZMQ REP service")
    ap.add_argument("--selftest", action="store_true", help="reproduce recorded action check")
    ap.add_argument("--endpoint", default=ENDPOINT)
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if args.serve:
        ActService().serve(args.endpoint)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
