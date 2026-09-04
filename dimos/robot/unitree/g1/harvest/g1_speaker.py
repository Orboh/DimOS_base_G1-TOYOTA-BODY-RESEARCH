# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Japanese announcements through the G1 speaker via off-board synthesis + PlayStream.

The G1 onboard TTS can't speak Japanese (verified — it sounds English), so we
synthesise the audio off-board with **pyopenjtalk** (local, pip-only, no network)
and push the raw PCM to the speaker with ``AudioClient.PlayStream`` (verified on
the real robot at 16 kHz mono s16le). Fully local — ideal for the backpack Jetson.

:class:`G1SpeakerAnnouncer` implements the harvest :class:`Announcer` protocol:
``say(text)`` enqueues; a background worker synthesises (cached per text) and
plays sequentially, so the harvest loop is never blocked. A unique ``stream_id``
per utterance (+ a PlayStop first) avoids the "reused stream is silent" gotcha.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_RATE = 16000  # G1 PlayStream PCM rate [Hz] (verified)
_APP = "okra-harvest"


def synth_pcm_jp(text: str, rate: int = _RATE) -> bytes:
    """Synthesise Japanese ``text`` to raw s16le mono PCM at ``rate`` (pyopenjtalk).

    Local, no network. Returns little-endian 16-bit mono PCM bytes.
    """
    import numpy as np
    import pyopenjtalk
    from scipy.signal import resample_poly

    wav, sr = pyopenjtalk.tts(text)  # float waveform, sr typically 48000
    wav = np.asarray(wav, dtype=np.float64)
    peak = max(1e-9, float(np.max(np.abs(wav))))
    wav = wav / peak * 0.98  # normalise loud (helps the quiet-ish speaker)
    if sr != rate:
        from math import gcd

        g = gcd(int(sr), int(rate))
        wav = resample_poly(wav, rate // g, sr // g)
    return (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class G1SpeakerAnnouncer:
    """Speaks Japanese via the G1 speaker (PlayStream), non-blocking + cached."""

    def __init__(
        self,
        audio_client: Any,
        *,
        rate: int = _RATE,
        app_name: str = _APP,
        chunk_ms: int = 200,
        volume: int | None = 100,
    ) -> None:
        self._audio = audio_client
        self._rate = rate
        self._app = app_name
        self._chunk_ms = chunk_ms
        self._cache: dict[str, bytes] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._stream_no = 0
        if volume is not None:
            try:
                self._audio.SetVolume(volume)
            except Exception as exc:  # noqa: BLE001
                logger.warning("G1 SetVolume failed", error=str(exc))
        self._thread = threading.Thread(target=self._run, daemon=True, name="g1-speaker")
        self._thread.start()

    def say(self, text: str) -> None:
        """Enqueue a line to speak (returns immediately; played in the background)."""
        self._queue.put(text)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._play_now(text)
            except Exception as exc:  # noqa: BLE001 — audio must never break harvesting
                logger.warning("G1 speaker play failed", text=text, error=str(exc))

    def _play_now(self, text: str) -> None:
        """Synthesise (cached) and stream ``text`` to the speaker (blocking, in-worker)."""
        pcm = self._cache.get(text)
        if pcm is None:
            pcm = synth_pcm_jp(text, self._rate)
            self._cache[text] = pcm
        # Clear any prior stream and use a fresh id (a reused id can play silent).
        self._audio.PlayStop(self._app)
        self._stop.wait(0.05)
        self._stream_no += 1
        stream_id = f"okra_jp_{self._stream_no}"
        chunk = max(2, int(self._rate * 2 * self._chunk_ms / 1000))
        chunk -= chunk % 2
        for off in range(0, len(pcm), chunk):
            self._audio.PlayStream(self._app, stream_id, pcm[off : off + chunk])
            self._stop.wait(self._chunk_ms / 1000.0)
        self._audio.PlayStop(self._app)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def make_g1_playstream_announcer(
    network_interface: str | None = None,
    *,
    init_dds: bool = True,
    volume: int | None = 100,
    rate: int = _RATE,
) -> G1SpeakerAnnouncer:
    """Build a :class:`G1SpeakerAnnouncer` connected to the G1 audio client.

    Args:
        network_interface: wired NIC to the G1 (required if ``init_dds``).
        init_dds: call ``ChannelFactoryInitialize``. Set False inside a running
            DimOS deployment where the DDS channel is already initialised.
        volume / rate: speaker volume (0-100) and PCM rate.

    ⚠️ Not unit-tested against the robot; verified manually via PlayStream.
    """
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    if init_dds:
        # Idempotent, process-wide DDS init (shared with arm_sdk / Dex1 / loco) so
        # the speaker can coexist with other DDS modules without a double-init crash.
        from dimos.robot.unitree.g1.act.dds_init import ensure_channel_factory

        if not network_interface:
            raise ValueError("network_interface is required when init_dds=True")
        ensure_channel_factory(network_interface)
    client = AudioClient()
    try:
        client.SetTimeout(10.0)
    except Exception:  # noqa: BLE001
        pass
    client.Init()
    return G1SpeakerAnnouncer(client, rate=rate, volume=volume)


__all__ = ["synth_pcm_jp", "G1SpeakerAnnouncer", "make_g1_playstream_announcer"]
