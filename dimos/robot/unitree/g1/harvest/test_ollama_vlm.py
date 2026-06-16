# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Offline tests for the Ollama vision-backed verify_harvest (no Ollama server).

Covers both strategies: moondream (caption via an injected ``generate`` →
keyword match) and qwen3-vl (chat yes/no via an injected ``llm``). Stubs replace
the network so nothing touches a model or robot. Also checks the fail-safe
(no frame / error => False).
"""

from __future__ import annotations

from dimos.robot.unitree.g1.harvest.ollama_vlm import make_ollama_verify


# --- moondream caption path (default model) ---------------------------------

def _moondream_verify(caption: str, frame=object(), **kw):
    """make_ollama_verify on moondream with an injected generate() returning caption."""
    return make_ollama_verify(
        lambda: frame, model="moondream", generate=lambda b64, prompt: caption,
        encode=lambda f: "B64", **kw
    )


def test_moondream_caption_with_object_is_true() -> None:
    cap = "A robot gripper holding a green okra, fingers wrapped around it."
    assert _moondream_verify(cap)() is True


def test_moondream_empty_caption_is_false() -> None:
    assert _moondream_verify("")() is False


def test_moondream_empty_keyword_overrides() -> None:
    # "holding" would match, but "empty"/"nothing" forces False.
    assert _moondream_verify("The gripper is empty, holding nothing.")() is False


def test_moondream_unrelated_caption_is_false() -> None:
    assert _moondream_verify("A plain white wall.")() is False


def test_moondream_no_frame_is_false() -> None:
    assert _moondream_verify("okra", frame=None)() is False


def test_moondream_generate_error_is_false() -> None:
    def boom(b64, prompt):
        raise RuntimeError("connection refused")

    v = make_ollama_verify(lambda: object(), model="moondream", generate=boom, encode=lambda f: "B64")
    assert v() is False  # fail-safe


def test_moondream_sends_caption_prompt_and_image() -> None:
    seen = {}

    def gen(b64, prompt):
        seen["b64"] = b64
        seen["prompt"] = prompt
        return "holding an okra"

    make_ollama_verify(lambda: object(), model="moondream", generate=gen,
                       encode=lambda f: "ABC123", prompt="DESCRIBE")()
    assert seen["b64"] == "ABC123"
    assert seen["prompt"] == "DESCRIBE"


# --- qwen3-vl chat yes/no path ----------------------------------------------

class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubLLM:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list = []

    def invoke(self, messages):  # noqa: ANN001
        self.calls.append(messages)
        return _Resp(self._content)


def _chat_verify(content, frame=object(), **kw):
    return make_ollama_verify(
        lambda: frame, model="qwen3-vl:2b", llm=_StubLLM(content),
        encode=lambda f: "B64", **kw
    )


def test_chat_affirmative_is_true() -> None:
    assert _chat_verify("Yes, it is holding an okra.")() is True


def test_chat_negative_is_false() -> None:
    assert _chat_verify("No, the gripper is empty.")() is False


def test_chat_error_is_false() -> None:
    class _BadLLM:
        def invoke(self, messages):  # noqa: ANN001
            raise RuntimeError("connection refused")

    v = make_ollama_verify(lambda: object(), model="qwen3-vl:2b", llm=_BadLLM(), encode=lambda f: "B64")
    assert v() is False


def test_chat_sends_prompt_and_image() -> None:
    stub = _StubLLM("yes")
    make_ollama_verify(lambda: object(), model="qwen3-vl:2b", llm=stub,
                       encode=lambda f: "ABC123", prompt="PICKED?")()
    content = stub.calls[0][0].content  # HumanMessage.content blocks
    assert any(b.get("type") == "text" and b.get("text") == "PICKED?" for b in content)
    assert any(b.get("type") == "image_url" and "ABC123" in b.get("image_url", "") for b in content)
