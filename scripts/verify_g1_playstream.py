#!/usr/bin/env python3
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

"""Verify Japanese audio via the G1 speaker using PlayStream (off-board synthesis).

The onboard TTS can't speak Japanese, so we synthesise the audio off-board (here
with espeak-ng — robotic but local/instant, just to prove the path) and push the
raw PCM to the G1 speaker with ``AudioClient.PlayStream``. ⚠️ Connects to the real
G1 and plays sound (no motion). Run on the laptop with the robot connected.

    ROBOT_INTERFACE=<nic> .venv/bin/python scripts/verify_g1_playstream.py
    # custom text / a pre-made wav:
    ... scripts/verify_g1_playstream.py --text "オクラを収穫します。" --volume 90
    ... scripts/verify_g1_playstream.py --wav some_japanese.wav

Goal: confirm (a) PlayStream actually plays our PCM and (b) the format (16 kHz
mono s16le is the assumption). If it plays intelligibly, we then pre-render the
fixed harvest phrases with a nicer LOCAL engine (VOICEVOX) and PlayStream those.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time

_DEFAULT_TEXT = "こんにちは。オクラ収穫ロボットです。日本語が聞こえますか？"


def _synth_pcm(text: str, wav_path: str | None, voice: str, speed: int | None, rate: int) -> bytes:
    """Return raw s16le mono PCM at ``rate`` Hz for ``text`` (or a given wav)."""
    tmp_wav = wav_path
    if tmp_wav is None:
        tmp_wav = tempfile.mktemp(suffix=".wav")
        cmd = ["espeak-ng", "-v", voice, "-w", tmp_wav]
        if speed:
            cmd += ["-s", str(speed)]
        cmd.append(text)
        print(f"[verify] synth (espeak-ng {voice}): {text!r}")
        subprocess.run(cmd, check=True)
    pcm_path = tempfile.mktemp(suffix=".pcm")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            tmp_wav,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(rate),
            pcm_path,
        ],
        check=True,
    )
    with open(pcm_path, "rb") as f:
        return f.read()


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify Japanese audio on the G1 via PlayStream.")
    ap.add_argument("--nic", default=os.getenv("ROBOT_INTERFACE", ""))
    ap.add_argument("--text", default=_DEFAULT_TEXT)
    ap.add_argument("--wav", default=None, help="play this wav instead of synthesising")
    ap.add_argument("--voice", default="ja", help="espeak-ng voice")
    ap.add_argument(
        "--speed", type=int, default=None, help="espeak-ng words/min (slower = clearer)"
    )
    ap.add_argument("--rate", type=int, default=16000, help="PCM sample rate [Hz] (G1 assumption)")
    ap.add_argument("--volume", type=int, default=None, help="0-100")
    ap.add_argument("--chunk-ms", type=int, default=200, help="PlayStream chunk size [ms]")
    ap.add_argument("--app-name", default="okra-harvest")
    args = ap.parse_args()

    if not args.nic:
        raise SystemExit("Set --nic or ROBOT_INTERFACE to the wired NIC to the G1.")

    pcm = _synth_pcm(args.text, args.wav, args.voice, args.speed, args.rate)
    dur = len(pcm) / (args.rate * 2)
    print(f"[verify] PCM: {len(pcm)} bytes, {args.rate} Hz mono s16le, ~{dur:.1f}s")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    print(f"[verify] ChannelFactoryInitialize on nic={args.nic!r}")
    ChannelFactoryInitialize(0, args.nic)
    client = AudioClient()
    try:
        client.SetTimeout(10.0)
    except Exception as exc:
        print(f"[verify] SetTimeout n/a ({exc})")
    client.Init()
    if args.volume is not None:
        print(f"[verify] SetVolume({args.volume}) -> {client.SetVolume(args.volume)}")

    # Clear any leftover stream from a previous run, then use a UNIQUE stream_id
    # (reusing a finished stream_id can be ignored by the robot -> no sound).
    client.PlayStop(args.app_name)
    time.sleep(0.2)
    stream_id = f"okra_jp_{os.getpid()}_{int(time.monotonic() * 1000) % 100000}"

    chunk = max(2, int(args.rate * 2 * args.chunk_ms / 1000))
    chunk -= chunk % 2  # keep 16-bit sample alignment
    print(f"[verify] streaming {len(pcm)} bytes in {chunk}-byte chunks (stream_id={stream_id})...")
    n = 0
    for off in range(0, len(pcm), chunk):
        ret = client.PlayStream(args.app_name, stream_id, pcm[off : off + chunk])
        code = ret[0] if isinstance(ret, tuple) else ret
        n += 1
        if code not in (0, None):
            print(f"[verify] chunk {n}: PlayStream code={ret}")
        time.sleep(args.chunk_ms / 1000.0)
    time.sleep(0.3)
    client.PlayStop(args.app_name)
    print(f"[verify] DONE ({n} chunks, all code=0). Did the G1 speaker play intelligible Japanese?")


if __name__ == "__main__":
    main()
