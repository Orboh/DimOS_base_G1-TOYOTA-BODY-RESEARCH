# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Spoken (Japanese) announcements for the harvest workflow.

The handbook (§6 HMI) wants the robot to keep the human informed: when the agent
makes a decision or the world state changes, it should say so out loud. The
phrasing is FIXED Japanese templates (not LLM-generated) — announcements are
routine status, so they must be predictable and cost nothing per cycle. The
LLM/VLM is reserved for judgment, never for narration.

Like :mod:`skills`, the speaker is an injectable interface so the graph logic is
testable with no robot:

* :class:`NullAnnouncer` — say nothing (default).
* :class:`RecordingAnnouncer` — record the spoken text (dry runs / tests):
  verify *what the robot would say* with no audio hardware.
* :class:`CallableAnnouncer` — wrap any ``speak(text)`` callable, for the real
  robot. Failures are swallowed so a TTS hiccup never breaks the harvest loop.

Real wiring (see ``README.md``): route ``speak`` to the G1's onboard speaker via
``unitree_sdk2py.g1.audio.AudioClient`` — either ``TtsMaker(text, speaker_id)``
(onboard TTS) or synthesise Japanese audio off-board (OpenAI/pyttsx) and push it
with ``PlayStream(...)`` if the onboard TTS lacks Japanese. The DimOS
``SpeakSkill`` (OpenAI TTS, Japanese-capable) plays on the host's speaker
instead, if robot-side output is not required.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Lateral/forward base-move direction → the phrase that explains it.
MoveDir = str  # one of: "forward" | "back" | "left" | "right"


@runtime_checkable
class Announcer(Protocol):
    """Speaks a line of (Japanese) text to the human."""

    def say(self, text: str) -> None:
        ...


class NullAnnouncer:
    """Says nothing. The default when no speaker is wired."""

    def say(self, text: str) -> None:  # noqa: D102
        return None


class RecordingAnnouncer:
    """Records spoken lines instead of playing them (dry runs / tests)."""

    def __init__(self) -> None:
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)


class CallableAnnouncer:
    """Speaks via an injected ``speak(text)`` callable (real robot / SpeakSkill).

    The callable should be NON-BLOCKING (e.g. ``SpeakSkill.speak(text,
    blocking=False)`` or ``AudioClient.TtsMaker``) so audio never stalls the
    harvest loop. Any exception is logged and swallowed.
    """

    def __init__(self, speak: Callable[[str], None]) -> None:
        self._speak = speak

    def say(self, text: str) -> None:
        try:
            self._speak(text)
        except Exception as exc:  # noqa: BLE001 — a TTS failure must not stop harvesting
            logger.warning("announce failed", text=text, error=str(exc))


# --- Fixed Japanese phrases (one place to review / tweak the wording) ----------


def start() -> str:
    return "収穫を開始します。"


def grasping() -> str:
    return "オクラを収穫します。"


def skip_height() -> str:
    return "高い位置のオクラは届かないので飛ばします。"


def approaching(direction: MoveDir) -> str:
    return {
        "forward": "オクラが遠いので前に進みます。",
        "back": "近すぎるので少し下がります。",
        "left": "取りやすい位置まで左に移動します。",
        "right": "取りやすい位置まで右に移動します。",
    }.get(direction, "オクラに近づきます。")


def regrasp() -> str:
    return "うまくつかめなかったので、つかみ直します。"


def picked(count: int) -> str:
    return f"{count}個目を収穫しました。"


def searching() -> str:
    return "この場所にオクラはありません。次を探しに移動します。"


def revisiting() -> str:
    return "取り残したオクラを採りに戻ります。"


def next_station() -> str:
    return "この場所は採り終わりました。次の収穫場所に移動します。"


def basket_swap() -> str:
    return "カゴがいっぱいになりました。空のカゴに交換してきます。"


def give_up() -> str:
    return "このオクラは収穫できませんでした。次に進みます。"


def done(count: int) -> str:
    return f"収穫を完了しました。全部で{count}個収穫しました。"


__all__ = [
    "Announcer",
    "NullAnnouncer",
    "RecordingAnnouncer",
    "CallableAnnouncer",
    "start",
    "grasping",
    "skip_height",
    "approaching",
    "regrasp",
    "picked",
    "searching",
    "revisiting",
    "next_station",
    "basket_swap",
    "give_up",
    "done",
]
