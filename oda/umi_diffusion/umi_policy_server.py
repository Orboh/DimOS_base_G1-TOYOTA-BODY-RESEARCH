#!/usr/bin/env python
"""Co-located UMI diffusion policy inference server (runs in the `umi` conda env).

The robot-facing half is the DimOS ``UmiDiffusionBridge`` (DimOS venv). This is the
perception + policy half, kept in the training-matched ``umi`` conda env so the
checkpoint runs in EXACTLY the deps it was trained with (torch2.1 / diffusers0.18.2 /
timm0.9.7). They talk over localhost ZMQ REQ/REP (msgpack).

Per control tick the bridge sends the current EE pose (from G1 right-arm FK, ROOT
frame, pos + axis-angle); this server:
  1. appends it to a pose ring buffer (own GoPro frames are buffered by a camera thread),
  2. assembles the horizon-2 obs (nearest-neighbour camera, interpolated pose,
     gripper_width pinned to the training-constant 1e-4 — the dead channel),
  3. runs DiffusionUnetTimmPolicy.predict_action,
  4. decodes the 9-dim action (pos3 + rot6d; this ckpt has action_include_gripper=False)
     to absolute EE waypoints in the ROOT frame,
  5. replies with a list of [pos3, aa3] waypoints.

Camera preprocessing (resize/fisheye/mask) MUST match the training-data generation;
those knobs (--fisheye/--camera-intrinsics/--sim-fov/--no-mirror) are pinned in Step 2
by overlaying a live GoPro frame on a training frame. Use --dummy-cam to validate the
IPC + decode + policy path with NO GoPro (offline, this PC).

Run (LIVE, once Step 2 pins the preprocessing):
  conda run -n umi python oda/umi_diffusion/umi_policy_server.py   # --cam-device defaults to the by-id Elgato path
Offline IPC check:
  conda run -n umi python oda/umi_diffusion/umi_policy_server.py --dummy-cam
"""
import os
import sys
import threading
import time
from collections import deque

import numpy as np

UMI_ROOT = os.path.expanduser("~/umi/universal_manipulation_interface")

# GoPro HERO9 -> Media Mod micro-HDMI -> Elgato HD60 X.  by-id, NOT /dev/videoN: the numbers get
# reassigned on replug (ZED-M and the Elgato swapped 4<->6 on 2026-07-29, silently feeding ZED-M
# frames into the smoke test).  Keep in sync with smoke_gopro.GOPRO_DEV -- duplicated as a literal
# so this server never imports the smoke script.
GOPRO_DEV = "/dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB3442072PE-video-index0"
sys.path.append(UMI_ROOT)
os.chdir(UMI_ROOT)

import click
import torch
import dill
import hydra
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from umi.real_world.real_inference_util import get_real_umi_obs_dict
from umi.common.pose_util import pose_to_mat, mat_to_pose, pose10d_to_mat

OmegaConf.register_new_resolver("eval", eval, replace=True)

_GRIPPER_CONST = 1e-4  # dead training channel (verified in okra_20260723_ishimaru)


# ------------------------------------------------------------------ preprocessing
def build_preproc(fisheye_converter, no_mirror, out_res=(224, 224)):
    """Return f(bgr_frame)->float32 RGB (H,W,3) in [0,1], matching umi_env.py's tf()."""
    import cv2
    from diffusion_policy.common.cv2_util import get_image_transform
    from umi.common.cv_util import draw_predefined_mask

    def f(bgr):
        if fisheye_converter is None:
            tf = get_image_transform(input_res=(bgr.shape[1], bgr.shape[0]),
                                     output_res=out_res, bgr_to_rgb=True)
            img = np.ascontiguousarray(tf(bgr))
            img = draw_predefined_mask(img, color=(0, 0, 0),
                                       mirror=no_mirror, gripper=True, finger=False, use_aa=True)
        else:
            img = fisheye_converter.forward(bgr)
            img = img[..., ::-1]  # bgr -> rgb
        return img.astype(np.float32) / 255.0

    return f


class CameraThread:
    """Grab BGR frames from a UVC device (or a dummy) and keep a timestamped ring buffer."""

    def __init__(self, device, preproc, dummy=False, cap_res=(1920, 1080), buf=16):
        self._device = device
        self._preproc = preproc
        self._dummy = dummy
        self._cap_res = cap_res
        self._buf = deque(maxlen=buf)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="gopro-cam")
        self._thread.start()

    def _run(self):
        import cv2

        cap = None
        if not self._dummy:
            cap = cv2.VideoCapture(self._device)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cap_res[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cap_res[1])
            if not cap.isOpened():
                raise RuntimeError(f"cannot open camera {self._device}")
        while not self._stop.is_set():
            if self._dummy:
                bgr = np.zeros((self._cap_res[1], self._cap_res[0], 3), dtype=np.uint8)
                time.sleep(1 / 30)
            else:
                ok, bgr = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue
            img = self._preproc(bgr)
            with self._lock:
                self._buf.append((time.time(), img))
        if cap is not None:
            cap.release()

    def snapshot(self):
        with self._lock:
            return list(self._buf)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class PolicyServer:
    def __init__(self, policy, cfg, shape_meta, cam, control_hz, device):
        self._policy = policy
        self._cfg = cfg
        self._shape_meta = shape_meta
        self._cam = cam
        self._dt = 1.0 / control_hz
        self._device = device
        self._obs_pose_repr = cfg.task.pose_repr.obs_pose_repr
        self._action_pose_repr = cfg.task.pose_repr.action_pose_repr
        self._cam_ds = int(shape_meta["obs"]["camera0_rgb"]["down_sample_steps"])
        self._pose_ds = int(shape_meta["obs"]["robot0_eef_pos"]["down_sample_steps"])
        self._cam_h = int(shape_meta["obs"]["camera0_rgb"]["horizon"])
        self._pose_h = int(shape_meta["obs"]["robot0_eef_pos"]["horizon"])
        self._poses = deque(maxlen=256)  # (t, pos3, aa3)
        self._episode_start = None

    # ---- pose buffer / interpolation ----
    def push_pose(self, t, pos, aa):
        self._poses.append((float(t), np.asarray(pos, float), np.asarray(aa, float)))

    def _interp_pose(self, ts):
        """Nearest-neighbour pose at each timestamp ts (list). Returns (N,6) pos+aa."""
        arr_t = np.array([p[0] for p in self._poses])
        out = []
        for t in ts:
            i = int(np.argmin(np.abs(arr_t - t)))
            out.append(np.concatenate([self._poses[i][1], self._poses[i][2]]))
        return np.asarray(out, dtype=np.float64)

    def reset(self, pos, aa):
        self._poses.clear()
        self._policy.reset()
        self._episode_start = np.concatenate([np.asarray(pos, float), np.asarray(aa, float)])

    def predict(self, t_now, pos, aa):
        """-> (actions (N,6) absolute pos+aa in ROOT, diag dict for logging/trace)."""
        self.push_pose(t_now, pos, aa)
        if self._episode_start is None:
            self._episode_start = np.concatenate([np.asarray(pos, float), np.asarray(aa, float)])
        frames = self._cam.snapshot()
        if not frames:
            raise RuntimeError("no camera frames yet")
        last_t = frames[-1][0]
        # freshness of the newest frame AT OBS-ASSEMBLY TIME (not after inference, which
        # would just add the inference latency back in and read as a stalled camera)
        cam_age_ms = (time.time() - last_t) * 1e3
        cam_t = np.array([t[0] for t in frames])

        cam_ts = last_t - np.arange(self._cam_h)[::-1] * self._cam_ds * self._dt
        idxs = [int(np.argmin(np.abs(cam_t - tt))) for tt in cam_ts]
        cam_obs = np.stack([frames[i][1] for i in idxs])  # (h, 224,224,3)

        # pose obs at the same alignment clock (use the camera 'now')
        pose_ts = last_t - np.arange(self._pose_h)[::-1] * self._pose_ds * self._dt
        pose_obs = self._interp_pose(pose_ts)

        env_obs = {
            "camera0_rgb": cam_obs,
            "robot0_eef_pos": pose_obs[:, :3].astype(np.float32),
            "robot0_eef_rot_axis_angle": pose_obs[:, 3:].astype(np.float32),
            "robot0_gripper_width": np.full((self._pose_h, 1), _GRIPPER_CONST, np.float32),
            "timestamp": cam_ts,
        }
        obs_dict_np = get_real_umi_obs_dict(
            env_obs=env_obs, shape_meta=self._shape_meta,
            obs_pose_repr=self._obs_pose_repr,
            episode_start_pose=[self._episode_start],
        )
        obs_dict = dict_apply(obs_dict_np,
                              lambda x: torch.from_numpy(x).unsqueeze(0).to(self._device))
        with torch.no_grad():
            raw = self._policy.predict_action(obs_dict)["action_pred"][0].detach().cpu().numpy()
        # decode 9-dim (pos3 + rot6d, NO gripper) -> absolute EE (N,6) pos+aa in ROOT
        cur = pose_to_mat(np.concatenate([env_obs["robot0_eef_pos"][-1],
                                          env_obs["robot0_eef_rot_axis_angle"][-1]], axis=-1))
        action_mat = convert_pose_mat_rep(pose10d_to_mat(raw[..., :9]),
                                          base_pose_mat=cur,
                                          pose_rep=self._action_pose_repr, backward=True)
        # diag: everything the bridge cannot see from its side of the socket. A stalled
        # GoPro (cam_age_ms climbing) or an obs assembled from one repeated pose
        # (pose_buf == 1) looks identical to "the policy just isn't moving" downstream.
        diag = {
            "cam_buf": len(frames),
            "cam_age_ms": cam_age_ms,
            "pose_buf": len(self._poses),
            "n_pred": int(raw.shape[0]),
            "obs_pos": env_obs["robot0_eef_pos"][-1].astype(np.float64),
            "obs_aa": env_obs["robot0_eef_rot_axis_angle"][-1].astype(np.float64),
            "raw": raw.astype(np.float64),  # (N,9) pre-decode, the actual net output
        }
        return mat_to_pose(action_mat), diag  # (N,6), diag


def load_policy(ckpt, device):
    payload = torch.load(open(ckpt, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.num_inference_steps = 16
    policy.eval().to(device)
    shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    return policy, cfg, shape_meta


@click.command()
@click.option("--ckpt", default=os.path.expanduser("~/umi/epoch=0110-train_loss=0.012.ckpt"))
@click.option("--addr", default="tcp://127.0.0.1:5599")
@click.option("--cam-device", default=GOPRO_DEV,
              help="default = Elgato HD60 X by-id path (/dev/videoN numbers shuffle on replug)")
@click.option("--dummy-cam", is_flag=True, default=False, help="no GoPro: feed zeros (offline IPC test)")
@click.option("--control-hz", default=10.0, type=float)
@click.option("--fisheye/--no-fisheye", default=False, help="apply GoPro fisheye rectification")
@click.option("--camera-intrinsics",
              default=os.path.join(UMI_ROOT, "example/calibration/gopro_intrinsics_2_7k.json"))
@click.option("--sim-fov", default=None, type=float, help="target FOV for fisheye rect")
@click.option("--no-mirror", is_flag=True, default=False)
@click.option("--log-every", default=1, type=int,
              help="print a diagnostic line every N requests (0 = never)")
@click.option("--quiet", is_flag=True, default=False, help="no per-request lines at all")
@click.option("--trace", default=None,
              help="JSONL trace path: per request the FULL raw (N,9) net output + decoded actions")
def main(ckpt, addr, cam_device, dummy_cam, control_hz, fisheye, camera_intrinsics, sim_fov,
         no_mirror, log_every, quiet, trace):
    import zmq
    import msgpack
    import json
    import traceback

    def f3(v):
        return " ".join(f"{float(x):+.3f}" for x in np.asarray(v).flatten())

    device = torch.device("cuda")
    print(f"loading policy from {ckpt} ...")
    policy, cfg, shape_meta = load_policy(ckpt, device)
    out_res = tuple(shape_meta["obs"]["camera0_rgb"]["shape"][1:])  # (224,224)

    fisheye_converter = None
    if fisheye:
        from umi.common.cv_util import parse_fisheye_intrinsics, FisheyeRectConverter
        assert sim_fov is not None, "--fisheye requires --sim-fov"
        intr = parse_fisheye_intrinsics(json.load(open(camera_intrinsics)))
        fisheye_converter = FisheyeRectConverter(**intr, out_size=out_res, out_fov=sim_fov)
        print(f"fisheye rectification ON (fov={sim_fov})")

    preproc = build_preproc(fisheye_converter, no_mirror, out_res=out_res)
    cam = CameraThread(cam_device, preproc, dummy=dummy_cam)
    cam.start()
    server = PolicyServer(policy, cfg, shape_meta, cam, control_hz, device)

    trace_f = open(trace, "a") if trace else None
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(addr)
    print(f"umi_policy_server ready on {addr} (dummy_cam={dummy_cam}, log_every={log_every}, "
          f"trace={trace or 'OFF'}). Ctrl-C to stop.", flush=True)
    n_req = 0
    try:
        while True:
            req = msgpack.unpackb(sock.recv(), raw=False)
            n_req += 1
            try:
                if req.get("reset"):
                    server.reset(req["eef_pos"], req["eef_rot_aa"])
                    if not quiet:
                        print(f"RESET req#{n_req} episode_start pos[{f3(req['eef_pos'])}] "
                              f"aa[{f3(req['eef_rot_aa'])}]", flush=True)
                t0 = time.time()
                acts, diag = server.predict(req["t"], req["eef_pos"], req["eef_rot_aa"])
                infer_ms = (time.time() - t0) * 1e3
                rep = {"ok": True, "actions": [[float(x) for x in row] for row in acts],
                       "n": int(acts.shape[0]), "infer_ms": infer_ms}
                if not quiet and log_every > 0 and n_req % log_every == 0:
                    d0 = (acts[0, :3] - np.asarray(req["eef_pos"], dtype=np.float64)) * 1e3
                    span = float(np.linalg.norm(acts[-1, :3] - acts[0, :3])) * 1e3
                    print(
                        f"req#{n_req} obs pos[{f3(diag['obs_pos'])}] "
                        f"cam_age={diag['cam_age_ms']:.0f}ms buf={diag['cam_buf']} "
                        f"pose_buf={diag['pose_buf']} | infer={infer_ms:.1f}ms n={rep['n']}\n"
                        f"   raw9[0]=[{f3(diag['raw'][0])}] a0(abs)[{f3(acts[0, :3])}] "
                        f"Δ[{d0[0]:+.1f} {d0[1]:+.1f} {d0[2]:+.1f}]mm span={span:.1f}mm",
                        flush=True,
                    )
                if trace_f is not None:
                    trace_f.write(json.dumps({
                        "t": time.time(), "req": n_req, "reset": bool(req.get("reset")),
                        "obs_pos": diag["obs_pos"].round(6).tolist(),
                        "obs_aa": diag["obs_aa"].round(6).tolist(),
                        "cam_age_ms": round(diag["cam_age_ms"], 2),
                        "cam_buf": diag["cam_buf"], "pose_buf": diag["pose_buf"],
                        "infer_ms": round(infer_ms, 2),
                        "raw": diag["raw"].round(6).tolist(),          # (N,9) net output
                        "actions": np.asarray(acts).round(6).tolist(),  # (N,6) decoded absolute
                    }) + "\n")
                    trace_f.flush()
            except Exception as e:
                # The bridge only ever sees repr(e); keep the traceback on this side.
                traceback.print_exc()
                rep = {"ok": False, "err": repr(e)}
            sock.send(msgpack.packb(rep, use_bin_type=True))
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        sock.close(0)
        if trace_f is not None:
            trace_f.close()


if __name__ == "__main__":
    main()
