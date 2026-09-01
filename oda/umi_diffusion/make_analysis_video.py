"""Build a side-by-side analysis video: what the policy saw, next to what it did.

A run produces records that are painful to correlate by hand:

  * ``<record>/cam_%06d.png`` + ``cam_index.jsonl`` -- the capture stream (server --record),
    logged at --record-fps so the arm's motion between decisions is actually visible
  * ``<record>/frame_%06d.png``  -- the single frame fed to each inference
  * ``<record>/server_trace.jsonl`` -- per-request obs, raw output, latency
  * ``<log>/umi_diffusion_trace.jsonl`` -- per-tick joint angles, tip pose (bridge)

This stitches them into one MP4: the camera on the left, and on the right two plots that
share the video's timeline -- where the hand actually went, and what the policy asked for.
Watching the hand drift while every chunk points the same way is far more legible than
reading either log alone.

    conda run -n umi --no-capture-output python make_analysis_video.py \
        --record ~/okra_runs/run1 \
        --bridge-trace ~/Toyota-auto-body-PoC/DimOS_oda/logs/<run>/umi_diffusion_trace.jsonl

Falls back to the per-inference frames when no capture stream was logged.
"""

import json
import os

import click
import cv2
import numpy as np

_PANEL_W, _PANEL_H = 760, 420
_FRAME_PX = 560

# BGR. Green = up/down (the axis in question), amber = forward/back, grey = reference.
_C_Z = (90, 210, 90)
_C_X = (70, 175, 235)
_C_REF = (140, 140, 140)


def _read_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass          # a run killed mid-write leaves one truncated line; skip it
    return out


def _jp(img, text, org, size=17, colour=(235, 235, 235)):
    """Draw Japanese text via PIL. cv2.putText cannot render CJK -- it emits '?' boxes."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, colour, 1, cv2.LINE_AA)
        return img
    import glob as _glob
    cands = (_glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK*Regular*.ttc")
             + _glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-DemiLight.ttc")
             + _glob.glob("/usr/share/fonts/truetype/fonts-japanese-*.ttf")
             + _glob.glob("/usr/share/fonts/**/NotoSansCJK*", recursive=True))
    for path in cands:
        if os.path.exists(path):
            font = ImageFont.truetype(path, size)
            break
    else:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, colour, 1, cv2.LINE_AA)
        return img
    pil = Image.fromarray(img[..., ::-1])
    ImageDraw.Draw(pil).text(org, text, font=font, fill=tuple(colour[::-1]))
    img[:] = np.asarray(pil)[..., ::-1]
    return img


def _plot(series, cursor, title, ylab, note=""):
    """series: list of (values, label, colour). x axis == the video's frame index."""
    img = np.full((_PANEL_H, _PANEL_W, 3), 24, np.uint8)
    pad_l, pad_r, pad_t, pad_b = 74, 16, 46, 40
    w, h = _PANEL_W - pad_l - pad_r, _PANEL_H - pad_t - pad_b
    finite = [v for s, _, _ in series for v in s if np.isfinite(v)]
    if not finite:
        return img
    lo, hi = float(min(finite)), float(max(finite))
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad = (hi - lo) * 0.10
    lo, hi = lo - pad, hi + pad
    n = max(len(s) for s, _, _ in series)

    def xy(i, v):
        return (pad_l + int(w * (i / max(1, n - 1))),
                pad_t + int(h * (1.0 - (v - lo) / (hi - lo))))

    for k in range(5):
        v = lo + (hi - lo) * k / 4
        _, y = xy(0, v)
        cv2.line(img, (pad_l, y), (pad_l + w, y), (54, 54, 54), 1)
        cv2.putText(img, f"{v:6.0f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (155, 155, 155), 1, cv2.LINE_AA)
    if lo < 0 < hi:
        _, y0 = xy(0, 0.0)
        cv2.line(img, (pad_l, y0), (pad_l + w, y0), (105, 105, 105), 1)

    for s, _, c in series:
        pts = [xy(i, v) for i, v in enumerate(s) if np.isfinite(v)]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, c, 2, cv2.LINE_AA)
    if 0 <= cursor < n:
        x, _ = xy(cursor, lo)
        cv2.line(img, (x, pad_t), (x, pad_t + h), (255, 255, 255), 1)
        for s, _, c in series:
            if cursor < len(s) and np.isfinite(s[cursor]):
                cv2.circle(img, xy(cursor, s[cursor]), 5, c, -1, cv2.LINE_AA)

    _jp(img, title, (pad_l, 8), 19)
    if note:
        _jp(img, note, (pad_l, 30), 13, (150, 150, 150))
    _jp(img, ylab, (6, 8), 13, (150, 150, 150))
    lx = pad_l
    for _, lab, c in series:
        _jp(img, "● " + lab, (lx, _PANEL_H - 30), 15, c)
        lx += 30 + 13 * len(lab)
    return img


@click.command()
@click.option("--record", required=True, help="server --record directory")
@click.option("--bridge-trace", default=None, help="umi_diffusion_trace.jsonl from the dimos run")
@click.option("--out", default=None, help="output mp4 (default: <record>/analysis.mp4)")
@click.option("--fps", default=15.0, type=float, help="playback rate; match --record-fps for real time")
def main(record, bridge_trace, out, fps):
    cam_idx = _read_jsonl(os.path.join(record, "cam_index.jsonl"))
    srv = _read_jsonl(os.path.join(record, "server_trace.jsonl"))
    brg = [r for r in _read_jsonl(bridge_trace) if r.get("kind") in ("infer", "exec")]
    out = out or os.path.join(record, "analysis.mp4")

    if cam_idx:
        # cam_index.jsonl is appended to across server restarts and `n` restarts at 1 each
        # time, so older sessions' entries point at files a later session overwrote. Keep
        # only the last monotonic run of n -- that is the session that produced the frames
        # currently on disk.
        if any("tag" in r for r in cam_idx):
            # Newer servers stamp a per-session tag, so frames never collide and the
            # sessions can simply be separated by it.
            last = [r["tag"] for r in cam_idx if "tag" in r][-1]
            dropped = len(cam_idx) - sum(1 for r in cam_idx if r.get("tag") == last)
            if dropped:
                print(f"cam_index: {dropped} entries from earlier sessions ignored")
            cam_idx = [r for r in cam_idx if r.get("tag") == last]
            frames = [(r["t"], os.path.join(record, "cam_%s_%06d.png" % (r["tag"], r["n"])))
                      for r in cam_idx]
        else:
            # Legacy layout: `n` restarted at 1 each session into a shared namespace, so a
            # restart overwrote the previous run's files. Keep the last monotonic run only.
            cut = 0
            for i in range(len(cam_idx) - 1, 0, -1):
                if cam_idx[i]["n"] <= cam_idx[i - 1]["n"]:
                    cut = i
                    break
            if cut:
                print(f"cam_index: dropping {cut} entries from earlier server sessions")
                cam_idx = cam_idx[cut:]
            frames = [(r["t"], os.path.join(record, "cam_%06d.png" % r["n"])) for r in cam_idx]
        frames = [(t, p) for t, p in frames if os.path.exists(p)]
        src = "capture stream"
    else:
        fs = sorted(f for f in os.listdir(record) if f.startswith("frame_") and f.endswith(".png"))
        ts = [r["t"] for r in srv] if len(srv) == len(fs) else list(range(len(fs)))
        frames = [(t, os.path.join(record, f)) for t, f in zip(ts, fs)]
        src = "per-inference frames only (re-run with --record-fps for a real video)"
    if not frames:
        raise SystemExit(f"no frames in {record}")
    # The capture log spans the server's whole lifetime; the episode is the slice where
    # inference actually ran. Keep a little lead-in/out so the IK reach is visible too.
    # Prefer the bridge trace for the window: server_trace.jsonl also accumulates across
    # restarts, so its min/max can span hours of idle recording.
    win = [r["t"] for r in brg] or [r["t"] for r in srv]
    if win:
        t_lo = min(win) - 4.0
        t_hi = max(win) + 3.0
        n_all = len(frames)
        frames = [(t, f) for t, f in frames if t_lo <= t <= t_hi]
        print(f"trimmed to episode window: {n_all} -> {len(frames)} frames")
    print(f"source: {src}\nframes={len(frames)}  server_trace={len(srv)}  bridge={len(brg)}")

    # Everything is aligned on wall-clock, so a value is carried forward until the next
    # record arrives -- that is what the robot was acting on during those video frames.
    def latest(recs, t, key):
        best = None
        for r in recs:
            if r.get("t", 0) <= t and key in r:
                best = r[key]
            elif r.get("t", 0) > t:
                break
        return best

    t0 = frames[0][0]
    tip_z, tip_x, cmd_dz, cmd_dx, lat = [], [], [], [], []
    for t, _ in frames:
        a = latest(srv, t, "actions")
        p = latest(brg, t, "eef_pos_torso") or latest(srv, t, "obs_pos")
        if a:
            arr = np.asarray(a, float)
            d = (arr[-1, :3] - arr[0, :3]) * 1000
            cmd_dx.append(d[0]); cmd_dz.append(d[2])
        else:
            cmd_dx.append(np.nan); cmd_dz.append(np.nan)
        if p:
            q = np.asarray(p, float) * 1000
            tip_x.append(q[0]); tip_z.append(q[2])
        else:
            tip_x.append(np.nan); tip_z.append(np.nan)
        lat.append(latest(srv, t, "infer_ms"))

    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (_FRAME_PX + _PANEL_W, _PANEL_H * 2))
    for i, (t, path) in enumerate(frames):
        im = cv2.imread(path)
        if im is None:
            continue
        col = np.full((_PANEL_H * 2, _FRAME_PX, 3), 24, np.uint8)
        col[:_FRAME_PX] = cv2.resize(im, (_FRAME_PX, _FRAME_PX), interpolation=cv2.INTER_NEAREST)
        _jp(col, "方策が見ている映像 (224x224)", (12, _FRAME_PX + 14), 18)
        _jp(col, f"経過 {t - t0:5.2f} 秒     {i + 1}/{len(frames)} コマ",
            (12, _FRAME_PX + 44), 16, (185, 185, 185))
        if np.isfinite(cmd_dz[i]):
            _jp(col, f"この時の指令   上下 {cmd_dz[i]:+6.1f} mm    前後 {cmd_dx[i]:+6.1f} mm",
                (12, _FRAME_PX + 72), 16, (185, 185, 185))
        if lat[i]:
            _jp(col, f"推論 {lat[i]:.0f} ms", (12, _FRAME_PX + 100), 15, (150, 150, 150))

        top = _plot([(tip_z, "上下 z", _C_Z), (tip_x, "前後 x", _C_X)], i,
                    "手先が実際にいた位置（胴体基準）", "mm",
                    "z が増える＝手が上がった / x が減る＝体に近づいた")
        bot = _plot([(cmd_dz, "上下 z", _C_Z), (cmd_dx, "前後 x", _C_X)], i,
                    "方策が出した移動量（16点先の目標まで）", "mm",
                    "0 より上＝そちらへ動けという指令")
        vw.write(np.concatenate([col, np.concatenate([top, bot], axis=0)], axis=1))
    vw.release()

    z = np.array([v for v in cmd_dz if np.isfinite(v)])
    if z.size:
        print(f"policy dz: mean {z.mean():+.1f} mm   up {(z > 0).sum()}/{z.size}")
    tz = np.array([v for v in tip_z if np.isfinite(v)])
    if tz.size > 1:
        print(f"tip z:     {tz[0]:+.1f} -> {tz[-1]:+.1f} mm  ({tz[-1] - tz[0]:+.1f} mm)")
    print(f"duration:  {frames[-1][0] - t0:.1f} s at {fps:g} fps\n\nwrote {out}")


if __name__ == "__main__":
    main()
