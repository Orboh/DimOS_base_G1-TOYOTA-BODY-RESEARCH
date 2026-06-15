# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Offline tests for the Ollama vision-backed verify_harvest (no Ollama server).

A stub LLM stands in for ChatOllama and a stub encoder skips real image encoding,
so we verify the yes/no parsing, the message (prompt + image) sent, and the
fail-safe (no frame / Ollama error => False) without any model or robot.
"""

from __future__ import annotations

from dimos.robot.unitree.g1.harvest.ollama_vlm import make_ollama_verify


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


def _verify(content, frame=object(), **kw):
    return make_ollama_verify(
        lambda: frame, llm=_StubLLM(content), encode=lambda f: "B64", **kw
    )


def test_affirmative_is_true() -> None:
    assert _verify("Yes, it is holding an okra.")() is True


def test_negative_is_false() -> None:
    assert _verify("No, the gripper is empty.")() is False


def test_no_frame_is_false() -> None:
    assert _verify("yes", frame=None)() is False


def test_ollama_error_is_false() -> None:
    class _BadLLM:
        def invoke(self, messages):  # noqa: ANN001
            raise RuntimeError("connection refused")

    v = make_ollama_verify(lambda: object(), llm=_BadLLM(), encode=lambda f: "B64")
    assert v() is False  # fail-safe: never claim a pick we couldn't confirm


def test_sends_prompt_and_image() -> None:
    stub = _StubLLM("yes")
    v = make_ollama_verify(
        lambda: object(), llm=stub, encode=lambda f: "ABC123", prompt="PICKED?"
    )
    v()
    content = stub.calls[0][0].content  # HumanMessage.content blocks
    assert any(b.get("type") == "text" and b.get("text") == "PICKED?" for b in content)
    assert any(
        b.get("type") == "image_url" and "ABC123" in b.get("image_url", "") for b in content
    )
