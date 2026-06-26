#!/usr/bin/env python3
"""7-C 検証: VLM 切断可否を「画像送信→何か返れば OK」の liveness で代替（F-02）。

計画書 [[12-検証計画-sim]] §8 7-C / §5 F-02。狙いは moondream の**判定の正しさ**ではなく
**呼び出し配管**（ollama 稼働・画像エンコード・HTTP・応答パース）の確認。sim 画像は OOD で
判定は当てにならないため、sim では「非空応答なら切断可（次へ）」とし、判定の正しさは実機（§10）。

提供物:
  - ``ollama_caption(image_bytes, host, model)`` : 画像を moondream に投げ caption 文字列を返す。
  - ``make_cut_ok_liveness(frame_getter, host, model)`` : GraspSequence の ``cut_ok_fn`` に渡せる
    ``() -> bool``。非空応答で True、frame無し/ollama停止で False（安全側＝切らない）。

実行（検証）:
  .venv/bin/python docs/sim-setup/sim_vlm_liveness.py
  # host/model を変える:
  SIM_VLM_HOST=http://100.113.43.64:11434 SIM_VLM_MODEL=moondream .venv/bin/python docs/sim-setup/sim_vlm_liveness.py
"""
from __future__ import annotations

import base64
import os
import sys
from collections.abc import Callable
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from dimos.robot.unitree.g1.harvest.blackboard import Okra
from dimos.robot.unitree.g1.harvest.grasp_sequence import GraspSequence
from dimos.robot.unitree.g1.harvest.ollama_vlm import CAPTION_PROMPT

DEFAULT_HOST = os.getenv("SIM_VLM_HOST", "http://100.113.43.64:11434")  # Jetson moondream (tailscale)
DEFAULT_MODEL = os.getenv("SIM_VLM_MODEL", "moondream")


def _to_jpeg_b64(frame: Any) -> str:
    """frame を JPEG base64 に。bytes(画像) / dimos frame(to_opencv) / numpy(cv2) を受ける。"""
    if isinstance(frame, (bytes, bytearray)):
        return base64.b64encode(bytes(frame)).decode("ascii")
    img = frame.to_opencv() if hasattr(frame, "to_opencv") else frame
    import cv2  # 実フレーム時のみ必要

    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("failed to JPEG-encode frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def ollama_caption(
    image: Any, *, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL,
    prompt: str = CAPTION_PROMPT, num_predict: int = 24, timeout: float = 60.0,
) -> str:
    """画像を ollama vision モデルに投げ、応答文字列を返す（ollama_vlm の生成相当）。"""
    import requests

    b64 = _to_jpeg_b64(image)
    url = host.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "images": [b64], "stream": False,
               "options": {"num_predict": num_predict}}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "") or ""


def make_cut_ok_liveness(
    frame_getter: Callable[[], Any], *, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL,
    prompt: str = CAPTION_PROMPT,
) -> Callable[[], bool]:
    """GraspSequence の cut_ok_fn 用: 画像送信→非空応答で True（liveness）。

    frame無し/ollama停止/エラーは False（安全側＝切らない）。判定の正しさは問わない。
    """
    def cut_ok() -> bool:
        frame = frame_getter()
        if frame is None:
            return False
        try:
            resp = ollama_caption(frame, host=host, model=model, prompt=prompt)
            return len(resp.strip()) > 0  # ★ 何か単語が返れば OK
        except Exception:
            return False  # ollama 停止 → 切らない
    return cut_ok


# ----------------------------- 検証 -----------------------------
def _sol(wait_s: float = 0.0):
    from types import SimpleNamespace
    return SimpleNamespace(arm14=[0.0] * 14, joint_names=[f"j{i}" for i in range(14)], wait_s=wait_s)


def main() -> int:
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chinou_scale_check.png")
    if not os.path.exists(img_path):
        print(f"[7-C] テスト画像が無い: {img_path}")
        return 1
    image_bytes = open(img_path, "rb").read()
    okra = Okra(id="vlm", pos_3d={"x": 0.30, "y": 0.45, "z": 0.80}, ripeness=1.0, reachable=True)

    print("========== 7-C: VLM 切断可否 liveness 検証 ==========")
    print(f"[7-C] host={DEFAULT_HOST} model={DEFAULT_MODEL} image={os.path.basename(img_path)}")

    # (a) 実 moondream へ画像送信 → 非空 caption → cut_ok True
    try:
        caption = ollama_caption(image_bytes)
        a_ok = len(caption.strip()) > 0
        print(f"  [{'OK ' if a_ok else 'NG '}] (a) moondream 応答: 非空={a_ok}  caption={caption.strip()[:80]!r}")
    except Exception as exc:  # noqa: BLE001
        a_ok = False
        print(f"  [NG ] (a) moondream 呼び出し失敗: {exc}")

    # (b) fail-safe: 停止中ホスト → False（切らない）
    cut_ok_dead = make_cut_ok_liveness(lambda: image_bytes, host="http://127.0.0.1:1")  # 接続不可
    b_ok = cut_ok_dead() is False
    print(f"  [{'OK ' if b_ok else 'NG '}] (b) fail-safe（ollama停止→cut_ok=False）: {b_ok}")

    # (c) GraspSequence 連携: liveness True → 切断到達 / dead → 切断ゲートで失敗
    cut_live = make_cut_ok_liveness(lambda: image_bytes)  # 実 moondream
    cuts_live: list[float] = []
    seq_live = GraspSequence(ik_solve=lambda o: _sol(), publish_gripper=lambda js: cuts_live.append(js.position[0]),
                             cut_ok_fn=cut_live)
    r_live = seq_live.run_episode(okra)
    c1 = r_live is True and len(cuts_live) == 1  # 切断到達
    cuts_dead: list[float] = []
    seq_dead = GraspSequence(ik_solve=lambda o: _sol(), publish_gripper=lambda js: cuts_dead.append(js.position[0]),
                             cut_ok_fn=make_cut_ok_liveness(lambda: image_bytes, host="http://127.0.0.1:1"))
    r_dead = seq_dead.run_episode(okra)
    c2 = r_dead is False and len(cuts_dead) == 0  # ゲートで停止＝切断せず
    c_ok = c1 and c2
    print(f"  [{'OK ' if c_ok else 'NG '}] (c) ゲート連携: liveOK→切断到達={c1}  dead→切断せず={c2}")

    all_ok = a_ok and b_ok and c_ok
    print(f"[7-C] RESULT: {'PASS ✅' if all_ok else 'FAIL ❌'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
