# Copyright 2025-2026 Dimensional Inc.
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

"""Spoken phase announcements for the click -> IK -> diffusion grasp pipeline.

The LangGraph harvest app announces every node it enters, so an operator standing next
to the robot always knows what it is doing. This bridge-driven pipeline had no such cue:
the arm just moves, and from outside there is no way to tell an IK coarse reach from the
diffusion fine-adjustment — or to notice that the adjustment refused to start at all.

Reuses the production speaker (``harvest/g1_speaker.py``, hardware-verified): Japanese is
synthesised OFF-BOARD with pyopenjtalk (the G1 onboard TTS cannot speak Japanese) and the
PCM is streamed via ``AudioClient.PlayStream``. Fully local — no network.

DESIGN NOTES
* **Never breaks the robot loop.** Every failure path (no pyopenjtalk, no audio client,
  DDS not up, synth error) degrades to logging the line instead of raising. Announcements
  are an operator convenience; losing them must never stop a grasp or leave the arm in an
  unknown state.
* **Non-blocking.** ``G1SpeakerAnnouncer.say`` only enqueues; a worker thread synthesises
  and streams. Control loops (250 Hz arm_sdk, 10 Hz diffusion) never wait on audio.
* **De-duplicated.** ``say_phase`` drops a repeat of the phase currently being announced,
  so a per-tick call site cannot flood the queue with the same sentence.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol

from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class Announcer(Protocol):
    """Minimal announcer contract (same shape as the harvest package's)."""

    def say(self, text: str) -> None: ...


class LogAnnouncer:
    """Fallback: prints the line instead of speaking it.

    Used whenever the real speaker cannot be built, so the phase trace still shows up in
    the run log and the pipeline behaves identically with and without audio.
    """

    def say(self, text: str) -> None:
        logger.info(f"[VOICE] {text}")


class PhaseVoice:
    """Speaks short Japanese phase lines, de-duplicated and failure-tolerant."""

    def __init__(self, announcer: Announcer | None = None) -> None:
        self._announcer: Announcer = announcer or LogAnnouncer()
        self._lock = threading.Lock()
        self._last: str | None = None

    def say_phase(self, phase: str, text: str) -> None:
        """Announce ``text`` unless ``phase`` is already the one being announced."""
        with self._lock:
            if phase == self._last:
                return
            self._last = phase
        self.say(text)

    def say(self, text: str) -> None:
        """Announce unconditionally. Swallows every audio error by design."""
        try:
            # Log every spoken line too: with the real speaker the run log otherwise
            # carries no record of what the robot said, so a post-mortem cannot tell
            # which phase was announced (or whether the line was swallowed by audio).
            if not isinstance(self._announcer, LogAnnouncer):
                logger.info(f"[VOICE] {text}")
            self._announcer.say(text)
        except Exception as exc:
            logger.warning(f"phase voice failed ({exc!r}); line was: {text}")

    def reset(self) -> None:
        """Forget the last phase so the next call speaks even if it repeats."""
        with self._lock:
            self._last = None


# One speaker per process. Each G1SpeakerAnnouncer calls AudioClient.PlayStop() before
# streaming, so two announcers sharing the single physical speaker cut each other off:
# the IK bridge's "接近しました" and the diffusion bridge's "ディフュージョンで微調整します"
# are issued ~50 ms apart, and the second PlayStop killed the first line mid-stream while
# both interleaved chunks into the same device. Sharing one announcer serialises them
# through its single worker queue, so every line is heard in full and in order.
_shared_announcer: Any = None
_shared_lock = threading.Lock()


def build_phase_voice(
    enabled: bool,
    network_interface: str = "",
    *,
    init_dds: bool = False,
    volume: int | None = 100,
) -> PhaseVoice:
    """Build a :class:`PhaseVoice`, falling back to log-only on any failure.

    Args:
        enabled: False -> log-only (no audio client is built at all).
        network_interface: wired NIC to the G1.
        init_dds: leave False inside a running DimOS deployment — ``G1ArmSdkConnection``
            has already called ``ChannelFactoryInitialize`` via the idempotent
            ``ensure_channel_factory``, and a second init is at best a no-op.
        volume: speaker volume 0-100.
    """
    if not enabled:
        return PhaseVoice(LogAnnouncer())
    global _shared_announcer
    with _shared_lock:
        if _shared_announcer is not None:
            logger.info("phase voice: reusing the shared G1 speaker")
            return PhaseVoice(_shared_announcer)
        try:
            from dimos.robot.unitree.g1.act.dds_init import ensure_channel_factory
            from dimos.robot.unitree.g1.harvest.g1_speaker import make_g1_playstream_announcer

            # Share the process-wide DDS factory rather than initialising our own.
            if network_interface:
                ensure_channel_factory(network_interface)
            ann: Any = make_g1_playstream_announcer(
                network_interface or None, init_dds=init_dds, volume=volume
            )
            logger.info(f"phase voice: G1 speaker ready (volume={volume})")
            _shared_announcer = ann
            return PhaseVoice(ann)
        except Exception as exc:
            logger.warning(
                f"phase voice: G1 speaker unavailable ({exc!r}); falling back to log-only. "
                "Japanese TTS needs `pyopenjtalk` + `scipy` in this venv."
            )
            return PhaseVoice(LogAnnouncer())


__all__ = ["Announcer", "LogAnnouncer", "PhaseVoice", "build_phase_voice"]
