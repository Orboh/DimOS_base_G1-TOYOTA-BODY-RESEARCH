# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""``verify_harvest`` backed by a LOCAL Ollama vision model (Moondream / Qwen2.5-VL).

Ollama is the local runtime/server; it serves whatever model you ``ollama pull``.
This builds the ``verify_harvest`` skill: send the latest head-camera frame + a
yes/no prompt to an Ollama vision model and parse the answer. Fully local (no
cloud) — ideal for the backpack Jetson.

Model choice (Ollama, swap by name): ``moondream`` (tiny, fast — frequent checks)
or ``qwen2.5vl`` (heavier, multilingual — if Moondream's vision is insufficient).
The heavy LLM "brain" / planner (``qwen2.5``) is a separate Ollama model (Phase 2).

The ``llm`` and ``encode`` args are injectable so the wiring is unit-testable
with no Ollama server (see ``test_ollama_vlm.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

DEFAULT_VERIFY_PROMPT = (
    "Look at the robot gripper in the image. Is it holding a picked fruit "
    "(okra / the target object), clearly separated from the plant? "
    "Answer with only 'yes' or 'no'."
)
_AFFIRMATIVE = ("yes", "y", "true", "はい", "holding", "ある")


def make_ollama_verify(
    frame_getter: Callable[[], Any],
    *,
    model: str = "moondream",
    host: str | None = None,
    prompt: str = DEFAULT_VERIFY_PROMPT,
    llm: Any = None,
    encode: Callable[[Any], str] | None = None,
) -> Callable[[], bool]:
    """Build a ``verify_harvest`` callable backed by an Ollama vision model.

    Returns ``verify() -> bool``: grabs the latest frame, asks the vision model
    the yes/no ``prompt``, and returns True for an affirmative answer. Connection
    or model errors are logged and treated as ``False`` (conservative — never
    claim a pick we could not confirm). ``llm``/``encode`` are injectable for tests.
    """
    state: dict[str, Any] = {"llm": llm}

    def _build_llm() -> Any:
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {"model": model}
        if host:
            kwargs["base_url"] = host
        return ChatOllama(**kwargs)

    def _default_encode(frame: Any) -> str:
        import base64

        import cv2

        bgr = frame.to_opencv()
        ok, buf = cv2.imencode(".jpg", bgr)
        if not ok:
            raise ValueError("failed to JPEG-encode frame")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    enc = encode or _default_encode

    def verify() -> bool:
        frame = frame_getter()
        if frame is None:
            logger.info("[ollama-verify] no camera frame yet -> False")
            return False
        try:
            from langchain_core.messages import HumanMessage

            if state["llm"] is None:
                state["llm"] = _build_llm()
            b64 = enc(frame)
            msg = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"},
                ]
            )
            resp = state["llm"].invoke([msg])
            answer = (getattr(resp, "content", "") or "").strip().lower()
            result = answer.startswith(_AFFIRMATIVE)
            logger.info(f"[ollama-verify] {model}: {answer[:40]!r} -> {result}")
            return result
        except Exception as exc:  # noqa: BLE001 — Ollama down / model error => not verified
            logger.warning(f"[ollama-verify] failed ({exc}); is Ollama running with '{model}'? -> False")
            return False

    return verify


__all__ = ["make_ollama_verify", "DEFAULT_VERIFY_PROMPT"]
