#!/usr/bin/env python3
"""ACT inference service, bridged to dimos over ZMQ.

Runs in the dedicated lerobot venv (.venv_act, separate process from dimos). It
owns the heavy lerobot/torch dependency; dimos stays clean and talks to it over
a neutral ZMQ wire (msgpack), so neither side needs the other's Python types.

Model (default): ``sotata/act-okura-pick-tree-06152026`` — okra "tree" pick.
  * state 16, action 16 (same layout as before)
  * TWO camera images: ``cam_high`` (head) + ``cam_right_wrist`` (right wrist)
  * normalization stats from dataset ``sotata/okura-pick-tree-20260615``
The image keys, resolutions and state dim are AUTO-DETECTED from the model
config, so ``--repo``/``--dataset`` can also load the older single-camera model.

Wire protocol (ZMQ REP on tcp://127.0.0.1:5701):
  request  (msgpack): {"state": [16 floats],
                       "images": {"cam_high": <jpeg bytes>, "cam_right_wrist": <jpeg bytes>},
                       "reset": <bool optional>}
                      (legacy single-cam: {"state":..., "image_jpeg": <bytes>, ...})
  response (msgpack): {"action": [16 floats]}

State / action layout (identity-mapped to dimos):
  [0:7]   left arm   (dimos motor index 15-21)
  [7:14]  right arm  (dimos motor index 22-28)
  [14]    left gripper  (Dex1, constant 0)   [15] right gripper (Dex1)

IMPORTANT — normalization (root-cause fix 2026-06-12):
  This lerobot version moved normalization OUT of the policy and INTO a
  preprocessor/postprocessor pipeline built from the *dataset stats*
  (`make_pre_post_processors`). We run the EXACT verified eval_g1.py path:
      raw obs -> preprocessor(normalize state+images) -> policy -> postprocessor(un-normalize)
  Calling ``policy.select_action`` on RAW values feeds garbage. The dataset
  must MATCH the model (its stats normalise the obs), hence --dataset tracks --repo.

Run (from the dimos repo root; the venv lives at ~/act-okura/.venv_act):
  service : ~/act-okura/.venv_act/bin/python scripts/act_service.py --serve
  selftest: ~/act-okura/.venv_act/bin/python scripts/act_service.py --selftest
  (older model: ... --repo sotata/act-okura-pick-06102026 --dataset Orboh/okura-sub-lerobot)
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

REPO_ID = "sotata/act-okura-pick-tree-06152026"  # policy checkpoint (okra tree pick)
DATASET_REPO = "sotata/okura-pick-tree-20260615"  # normalization stats + task (MUST match repo)
STATE_KEY = "observation.state"
_IMG_PREFIX = "observation.images."  # model image keys are this + short name (e.g. cam_high)
ENDPOINT = "tcp://127.0.0.1:5701"


def _short(img_key: str) -> str:
    """'observation.images.cam_high' -> 'cam_high' (the wire short name)."""
    return img_key[len(_IMG_PREFIX):] if img_key.startswith(_IMG_PREFIX) else img_key


class ActService:
    def __init__(self, repo_id: str = REPO_ID, dataset_repo: str = DATASET_REPO) -> None:
        self.device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")

        # The dataset provides the normalization stats AND the task string.
        dataset = LeRobotDataset(repo_id=dataset_repo)
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        self.task = dataset[from_idx].get("task", "") if hasattr(dataset[from_idx], "get") else ""

        cfg = PreTrainedConfig.from_pretrained(repo_id)
        cfg.pretrained_path = repo_id

        # Auto-detect the model's expected inputs (works for 1- or 2-camera models).
        feats = cfg.input_features
        self.state_key = next((k for k in feats if "state" in k), STATE_KEY)
        self.image_keys = [k for k in feats if "image" in k]
        # Per-image (H, W) from the feature shape [C, H, W].
        self.image_hw = {}
        for k in self.image_keys:
            shape = list(getattr(feats[k], "shape", [3, 480, 640]))
            self.image_hw[k] = (int(shape[1]), int(shape[2])) if len(shape) == 3 else (480, 640)

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
        cams = ", ".join(_short(k) for k in self.image_keys)
        print(f"[act] loaded {repo_id} on {self.device} | task={self.task!r} | cameras=[{cams}] "
              f"| normalization via preprocessor/postprocessor (dataset={dataset_repo})")

    def _reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    @torch.no_grad()
    def infer(
        self, state: np.ndarray, bgr_images: dict[str, np.ndarray], reset: bool = False
    ) -> np.ndarray:
        """One closed-loop step (eval_g1.py predict_action path), multi-camera.

        ``bgr_images`` maps each model image key to its decoded BGR frame.
        Normalization of state + images and un-normalization of the action are
        done by the lerobot preprocessor/postprocessor — NOT by hand here.
        """
        if reset:
            self._reset()

        observation = {self.state_key: torch.from_numpy(state.astype(np.float32))}
        for k in self.image_keys:
            bgr = bgr_images[k]
            h, w = self.image_hw[k]
            if bgr.shape[:2] != (h, w):
                bgr = cv2.resize(bgr, (w, h))  # cv2 size = (W, H)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            observation[k] = torch.from_numpy(np.ascontiguousarray(rgb))  # HWC uint8

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

    def _images_from_request(self, req: dict) -> dict[str, np.ndarray]:
        """Decode the request's JPEG(s) into a {model_image_key: BGR} dict."""
        def _decode(buf: bytes) -> np.ndarray:
            return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)

        wire = req.get("images") or {}
        out: dict[str, np.ndarray] = {}
        by_short = {_short(k): k for k in self.image_keys}
        for short, buf in wire.items():  # new multi-camera form: {short_name: jpeg}
            key = by_short.get(short, _IMG_PREFIX + short)
            if key in self.image_keys:
                out[key] = _decode(buf)
        # Legacy single-image fallback: if exactly one model image is still missing
        # and a bare image_jpeg was sent, use it (keeps old single-cam clients working).
        still_missing = [k for k in self.image_keys if k not in out]
        if len(still_missing) == 1 and "image_jpeg" in req:
            out[still_missing[0]] = _decode(req["image_jpeg"])
        missing = [_short(k) for k in self.image_keys if k not in out]
        if missing:
            raise ValueError(f"request missing camera image(s): {missing}; model needs "
                             f"{[_short(k) for k in self.image_keys]}")
        return out

    def serve(self, endpoint: str = ENDPOINT) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REP)
        sock.bind(endpoint)
        print(f"[act] serving on {endpoint} (Ctrl-C to stop)", flush=True)
        while True:
            req = msgpack.unpackb(sock.recv(), raw=False)
            try:
                state = np.asarray(req["state"], dtype=np.float32)
                images = self._images_from_request(req)
                action = self.infer(state, images, reset=bool(req.get("reset", False)))
                sock.send(msgpack.packb({"action": action.astype(float).tolist()}, use_bin_type=True))
            except Exception as exc:  # noqa: BLE001 — keep the service alive, report the error
                sock.send(msgpack.packb({"error": str(exc)}, use_bin_type=True))


def _selftest(repo_id: str = REPO_ID, dataset_repo: str = DATASET_REPO) -> int:
    """Verify the service reproduces the dataset's recorded action (normalization OK).

    Standalone model check — uses the dataset's first frame (all cameras), no
    robot / live camera. A small error means the model loads and infers correctly.
    """
    svc = ActService(repo_id, dataset_repo)
    dataset = LeRobotDataset(repo_id=dataset_repo)
    from_idx = dataset.meta.episodes["dataset_from_index"][0]
    frame = dataset[from_idx]
    state = frame[svc.state_key].float().numpy()
    bgr_images = {}
    for k in svc.image_keys:
        img_chw = frame[k]  # CHW float[0,1]
        rgb_uint8 = (img_chw.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
        bgr_images[k] = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    rec = frame["action"].float().numpy()

    a = svc.infer(state, bgr_images, reset=True)
    err = float(np.max(np.abs(a - rec)))
    np.set_printoptions(precision=3, suppress=True)
    print(f"\n[selftest] cameras         : {[_short(k) for k in svc.image_keys]}")
    print(f"[selftest] recorded action : {rec}")
    print(f"[selftest] service action  : {a}")
    print(f"[selftest] max|err| = {err:.4f} rad  -> {'OK' if err < 0.2 else 'FAIL (normalization?)'}")
    return 0 if err < 0.2 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true", help="run the ZMQ REP service")
    ap.add_argument("--selftest", action="store_true", help="reproduce recorded action check")
    ap.add_argument("--repo", default=REPO_ID, help="policy checkpoint repo id")
    ap.add_argument("--dataset", default=DATASET_REPO, help="dataset repo id (normalization stats)")
    ap.add_argument("--endpoint", default=ENDPOINT)
    args = ap.parse_args()
    if args.selftest:
        return _selftest(args.repo, args.dataset)
    if args.serve:
        ActService(args.repo, args.dataset).serve(args.endpoint)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
