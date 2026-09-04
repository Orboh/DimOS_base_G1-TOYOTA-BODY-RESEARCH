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

# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""``verify_harvest`` backed by a LOCAL Ollama vision model (Moondream / Qwen3-VL).

Ollama is the local runtime/server; it serves whatever model you ``ollama pull``.
This builds the ``verify_harvest`` skill: send the latest head-camera frame to an
Ollama vision model and decide whether the gripper is holding a picked okra.
Fully local (no cloud) — ideal for the backpack Jetson.

Two strategies, chosen automatically by model name (measured on a Jetson AGX
Orin 64GB serving the models over HTTP):

* **moondream** (default) — ``moondream`` ignores direct yes/no questions over
  Ollama (returns empty) but answers "describe..." in ~1 s. So we ask it to
  briefly describe what is grasped, then keyword-match the caption. ~1 s, no
  "thinking" — best for the frequent verify check. (Japanese is weak, but verify
  only needs an English keyword match.)
* **qwen3-vl** — answers yes/no but only with reasoning on (~5 s; reasoning off
  returns empty). Used via chat yes/no. Slower; good when you also want one
  multilingual model for the Phase-2 language brain.

``llm`` / ``generate`` / ``encode`` are injectable so the wiring is unit-testable
with no Ollama server (see ``test_ollama_vlm.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

DEFAULT_HOST = "http://192.168.123.222:11434"  # Jetson Ollama (override per deploy)

# qwen3-vl path: a direct yes/no question (needs reasoning on to answer).
YESNO_PROMPT = (
    "Look at the robot gripper in the image. Is it holding a picked fruit "
    "(okra / the target object), clearly separated from the plant? "
    "Answer with only 'yes' or 'no'."
)
# moondream path: ask for a short description, then keyword-match the caption.
CAPTION_PROMPT = "Briefly describe the object the robot gripper is holding."
# Caption words that indicate a successful grasp (the target is held).
_HOLD_KEYWORDS = ("okra", "fruit", "holding", "held", "grasp", "gripp", "wrapped", "pod", "green")
# Caption words that indicate nothing is grasped (override a stray match).
_EMPTY_KEYWORDS = ("empty", "nothing", "no object", "not holding")

_AFFIRMATIVE = ("yes", "y", "true", "はい", "holding", "ある")

# detection (scene-level "is there okra in view?")
# moondream path: describe the scene, then keyword-match for okra presence.
DETECT_CAPTION_PROMPT = "Briefly describe any okra or green vegetables/pods visible in the image."
# qwen3-vl path: a direct yes/no question.
DETECT_YESNO_PROMPT = (
    "Is there an okra (a green ridged seed pod) visible in the image? "
    "Answer with only 'yes' or 'no'."
)
# Caption words that indicate okra is present in the scene.
_OKRA_PRESENT_KEYWORDS = (
    "okra",
    "okura",
    "pod",
    "ladyfinger",
    "lady finger",
    "green vegetable",
    "green bean",
    "green",
)
# Caption words that override a stray match (no okra in the scene).
_OKRA_ABSENT_KEYWORDS = ("no okra", "no green", "no vegetable", "nothing", "empty")

# Fixed IN-REACH position [m] used for a VLM detection — the default
# HarvestConfig reach-box centre (x=lateral, y=depth, z=height). The okra-ACT
# grasp is visuomotor (camera-driven, ignores pos_3d), so this only routes
# select -> grasp; override via ``position`` to match a tuned reach box.
_DEFAULT_IN_REACH = {"x": 0.30, "y": 0.45, "z": 0.75}


def _is_moondream(model: str) -> bool:
    return "moondream" in model.lower()


def make_ollama_verify(
    frame_getter: Callable[[], Any],
    *,
    model: str = "moondream",
    host: str | None = DEFAULT_HOST,
    prompt: str | None = None,
    num_predict: int = 24,
    llm: Any = None,
    generate: Callable[[str, str], str] | None = None,
    encode: Callable[[Any], str] | None = None,
) -> Callable[[], bool]:
    """Build a ``verify_harvest`` callable backed by a local Ollama vision model.

    Returns ``verify() -> bool``: grabs the latest frame and asks the model
    whether an okra is held. ``moondream`` uses the caption+keyword strategy;
    other models use a chat yes/no. Connection/model errors are logged and
    treated as ``False`` (never claim an unconfirmed pick). ``llm`` (chat path),
    ``generate`` (moondream path) and ``encode`` are injectable for tests.
    """
    moondream = _is_moondream(model)
    state: dict[str, Any] = {"llm": llm}

    def _encode(frame: Any) -> str:
        import base64

        import cv2

        ok, buf = cv2.imencode(".jpg", frame.to_opencv())
        if not ok:
            raise ValueError("failed to JPEG-encode frame")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    enc = encode or _encode

    # moondream: /api/generate caption, then keyword-match
    def _default_generate(b64: str, text: str) -> str:
        import requests

        url = (host or DEFAULT_HOST).rstrip("/") + "/api/generate"
        payload = {
            "model": model,
            "prompt": text,
            "images": [b64],
            "stream": False,
            "options": {"num_predict": num_predict},
        }
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "") or ""

    gen = generate or _default_generate
    caption_prompt = prompt or CAPTION_PROMPT

    def _verify_moondream(frame: Any) -> bool:
        caption = gen(enc(frame), caption_prompt).strip().lower()
        if not caption:
            logger.info("[ollama-verify] moondream: empty caption -> False")
            return False
        if any(k in caption for k in _EMPTY_KEYWORDS):
            result = False
        else:
            result = any(k in caption for k in _HOLD_KEYWORDS)
        logger.info(f"[ollama-verify] moondream caption {caption[:60]!r} -> {result}")
        return result

    # qwen3-vl & others: chat yes/no (reasoning on)
    def _build_llm() -> Any:
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {"model": model, "reasoning": True, "num_predict": 128}
        if host:
            kwargs["base_url"] = host
        return ChatOllama(**kwargs)

    yesno_prompt = prompt or YESNO_PROMPT

    def _verify_chat(frame: Any) -> bool:
        from langchain_core.messages import HumanMessage

        if state["llm"] is None:
            state["llm"] = _build_llm()
        b64 = enc(frame)
        msg = HumanMessage(
            content=[
                {"type": "text", "text": yesno_prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"},
            ]
        )
        answer = (getattr(state["llm"].invoke([msg]), "content", "") or "").strip().lower()
        result = answer.startswith(_AFFIRMATIVE) or answer.startswith("yes")
        logger.info(f"[ollama-verify] {model}: {answer[:40]!r} -> {result}")
        return result

    def verify() -> bool:
        frame = frame_getter()
        if frame is None:
            logger.info("[ollama-verify] no camera frame yet -> False")
            return False
        try:
            return _verify_moondream(frame) if moondream else _verify_chat(frame)
        except Exception as exc:
            logger.warning(
                f"[ollama-verify] failed ({exc}); is Ollama running with '{model}'? -> False"
            )
            return False

    return verify


def make_ollama_detect_okra(
    frame_getter: Callable[[], Any],
    *,
    model: str = "moondream",
    host: str | None = DEFAULT_HOST,
    position: dict[str, float] | None = None,
    ripeness: float = 1.0,
    prompt: str | None = None,
    num_predict: int = 32,
    llm: Any = None,
    generate: Callable[[str, str], str] | None = None,
    encode: Callable[[Any], str] | None = None,
) -> Callable[[], list[Any]]:
    """Build a ``detect_okra`` callable backed by a local Ollama vision model.

    Unlike YOLO, the VLM only reports PRESENCE, not a location. So when okra is
    seen this emits ONE :class:`Okra` at a fixed IN-REACH ``position`` (default
    the HarvestConfig reach-box centre) with a fresh id, which routes
    ``select -> grasp``. The okra-ACT grasp is visuomotor (driven by the camera,
    NOT by ``pos_3d``), so the faked position only gates the decision — the arm
    still reaches whatever it sees. This lets you exercise the post-detection
    flow (grasp / verify / record / sweep) without an okra-trained YOLO weight.

    ``moondream`` uses the caption+keyword strategy (it ignores yes/no); other
    models use a chat yes/no. Fail-safe: no frame / Ollama down -> ``[]`` (never
    fabricate a detection on error). ``llm`` / ``generate`` / ``encode`` are
    injectable for tests.
    """
    from dimos.robot.unitree.g1.harvest.blackboard import Okra

    moondream = _is_moondream(model)
    pos = dict(position) if position else dict(_DEFAULT_IN_REACH)
    state: dict[str, Any] = {"llm": llm, "n": 0}

    def _encode(frame: Any) -> str:
        import base64

        import cv2

        ok, buf = cv2.imencode(".jpg", frame.to_opencv())
        if not ok:
            raise ValueError("failed to JPEG-encode frame")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    enc = encode or _encode

    def _default_generate(b64: str, text: str) -> str:
        import requests

        url = (host or DEFAULT_HOST).rstrip("/") + "/api/generate"
        payload = {
            "model": model,
            "prompt": text,
            "images": [b64],
            "stream": False,
            "options": {"num_predict": num_predict},
        }
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "") or ""

    gen = generate or _default_generate
    caption_prompt = prompt or DETECT_CAPTION_PROMPT
    yesno_prompt = prompt or DETECT_YESNO_PROMPT

    def _present_moondream(frame: Any) -> bool:
        caption = gen(enc(frame), caption_prompt).strip().lower()
        if not caption:
            logger.info("[ollama-detect] moondream: empty caption -> no okra")
            return False
        if any(k in caption for k in _OKRA_ABSENT_KEYWORDS):
            result = False
        else:
            result = any(k in caption for k in _OKRA_PRESENT_KEYWORDS)
        logger.info(f"[ollama-detect] moondream caption {caption[:60]!r} -> okra={result}")
        return result

    def _build_llm() -> Any:
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {"model": model, "reasoning": True, "num_predict": 128}
        if host:
            kwargs["base_url"] = host
        return ChatOllama(**kwargs)

    def _present_chat(frame: Any) -> bool:
        from langchain_core.messages import HumanMessage

        if state["llm"] is None:
            state["llm"] = _build_llm()
        b64 = enc(frame)
        msg = HumanMessage(
            content=[
                {"type": "text", "text": yesno_prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"},
            ]
        )
        answer = (getattr(state["llm"].invoke([msg]), "content", "") or "").strip().lower()
        result = answer.startswith("yes") or answer.startswith(_AFFIRMATIVE)
        logger.info(f"[ollama-detect] {model}: {answer[:40]!r} -> okra={result}")
        return result

    def detect_okra() -> list[Any]:
        frame = frame_getter()
        if frame is None:
            logger.info("[ollama-detect] no camera frame yet -> []")
            return []
        try:
            present = _present_moondream(frame) if moondream else _present_chat(frame)
        except Exception as exc:
            logger.warning(
                f"[ollama-detect] failed ({exc}); is Ollama running with '{model}'? -> []"
            )
            return []
        if not present:
            return []
        state["n"] += 1
        # Fresh id each sighting so the flow keeps engaging (bounded by
        # max_harvest_iterations); in-reach + ripe so select routes to grasp.
        return [
            Okra(id=f"vlm_okra_{state['n']}", pos_3d=dict(pos), ripeness=ripeness, reachable=True)
        ]

    return detect_okra


__all__ = [
    "CAPTION_PROMPT",
    "DEFAULT_HOST",
    "DETECT_CAPTION_PROMPT",
    "DETECT_YESNO_PROMPT",
    "YESNO_PROMPT",
    "make_ollama_detect_okra",
    "make_ollama_verify",
]
