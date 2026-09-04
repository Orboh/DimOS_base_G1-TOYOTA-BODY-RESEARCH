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

"""Offline tests for the G1 speaker announcer (real pyopenjtalk synth, stub audio).

No robot: synthesis runs locally (pyopenjtalk) and a stub stands in for the G1
AudioClient, so we verify the PCM is produced, cached, and streamed (PlayStream +
PlayStop) without any hardware.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyopenjtalk")  # local Japanese TTS; skip the suite if absent
pytest.importorskip("scipy")

from dimos.robot.unitree.g1.harvest.announce import Announcer
from dimos.robot.unitree.g1.harvest.g1_speaker import (
    G1SpeakerAnnouncer,
    synth_pcm_jp,
)


class _StubAudio:
    """Records AudioClient calls instead of touching a robot."""

    def __init__(self) -> None:
        self.streams: list[tuple[str, str, int]] = []  # (app, stream_id, n_bytes)
        self.stops = 0
        self.volume: int | None = None

    def SetVolume(self, v: int) -> int:
        self.volume = v
        return 0

    def PlayStream(self, app: str, stream_id: str, pcm: bytes) -> tuple[int, str]:
        self.streams.append((app, stream_id, len(pcm)))
        return (0, "")

    def PlayStop(self, app: str) -> int:
        self.stops += 1
        return 0


def test_synth_pcm_jp_produces_audio() -> None:
    pcm = synth_pcm_jp("こんにちは。", rate=16000)
    assert isinstance(pcm, bytes)
    assert len(pcm) > 16000 * 2 * 0.2  # at least ~0.2s of 16 kHz s16le audio
    assert len(pcm) % 2 == 0  # 16-bit aligned


def test_announcer_satisfies_protocol_and_sets_volume() -> None:
    stub = _StubAudio()
    ann = G1SpeakerAnnouncer(stub, volume=100, chunk_ms=50)
    try:
        assert isinstance(ann, Announcer)
        assert stub.volume == 100
    finally:
        ann.stop()


def test_play_now_streams_and_stops() -> None:
    stub = _StubAudio()
    ann = G1SpeakerAnnouncer(stub, chunk_ms=50)
    try:
        ann._play_now("オクラを収穫します。")  # synchronous play (no thread/timing)
        assert len(stub.streams) >= 1  # streamed at least one chunk
        assert all(s[0] == "okra-harvest" for s in stub.streams)
        assert len({s[1] for s in stub.streams}) == 1  # one stream_id for the utterance
        assert stub.stops >= 2  # clear-before + stop-after
    finally:
        ann.stop()


def test_play_caches_by_text() -> None:
    stub = _StubAudio()
    ann = G1SpeakerAnnouncer(stub, chunk_ms=50)
    try:
        ann._play_now("テスト")
        first = dict(ann._cache)
        ann._play_now("テスト")  # second time: served from cache, not re-synthesised
        assert "テスト" in ann._cache
        assert ann._cache["テスト"] == first["テスト"]
    finally:
        ann.stop()
