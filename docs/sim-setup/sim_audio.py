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

"""sim 用 音声 shim（§7-A / F-12）: 収穫アナウンスを手元PCで鳴らす / WAV 化する。

実機では ``g1_speaker.G1SpeakerAnnouncer`` が pyopenjtalk 合成 → ``AudioClient.PlayStream``
で G1 スピーカーへ送る。sim には G1 スピーカーが無いので、**合成（``synth_pcm_jp``）はそのまま
再利用**し、出力先を手元PCのスピーカー（``sounddevice``）に差し替える。これで音声の
「フレーズ内容・順番・タイミング」を sim で検証できる（実際の音圧・聞こえは実機, 計画書 §10）。

使い方:
  # 全アナウンスを WAV 化（音声デバイス不要・確実。これが検証の成果物）
  .venv/bin/python docs/sim-setup/sim_audio.py --dump docs/sim-setup/audio_samples
  # ライブ再生（音声デバイスが要る）
  .venv/bin/python docs/sim-setup/sim_audio.py --play
  # 収穫グラフに繋ぐ:
  from sim_audio import make_host_speaker_announcer
  voice = make_host_speaker_announcer()
  app = build_harvest_graph(skills, cfg, announcer=voice)
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from dimos.robot.unitree.g1.harvest import announce
from dimos.robot.unitree.g1.harvest.g1_speaker import _RATE, synth_pcm_jp

# 収穫中に流れる代表フレーズ（announce.py の全関数を網羅。引数つきは代表値）。
# 実際にグラフが言う順序に近い並びで列挙。
PHRASES: list[tuple[str, str]] = [
    ("01_start", announce.start()),
    ("02_detect_3", announce.detect_result(3)),
    ("03_detect_0", announce.detect_result(0)),
    ("04_grasping", announce.grasping()),
    ("05_approach_forward", announce.approaching("forward")),
    ("06_approach_back", announce.approaching("back")),
    ("07_approach_left", announce.approaching("left")),
    ("08_approach_right", announce.approaching("right")),
    ("09_regrasp", announce.regrasp()),
    ("10_verify_ok", announce.verify_ok()),
    ("11_verify_fail", announce.verify_fail()),
    ("12_picked_1", announce.picked(1)),
    ("13_picked_10", announce.picked(10)),
    ("14_skip_height", announce.skip_height()),
    ("15_ripeness_skip", announce.ripeness_skip(2)),
    ("16_searching", announce.searching()),
    ("17_revisiting", announce.revisiting()),
    ("18_next_station", announce.next_station()),
    ("19_basket_swap", announce.basket_swap()),
    ("20_give_up", announce.give_up()),
    ("21_safety_stop", announce.safety_stop("人を検知")),
    ("22_safety_resume", announce.safety_resume()),
    ("23_done_10", announce.done(10)),
]


def write_wav(path: str, pcm: bytes, rate: int = _RATE) -> None:
    """s16le mono PCM を WAV ファイルに書き出す。"""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(rate)
        w.writeframes(pcm)


def dump_wavs(out_dir: str) -> int:
    """全フレーズを WAV 化して ``out_dir`` に保存。生成数を返す。"""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for name, text in PHRASES:
        pcm = synth_pcm_jp(text)
        path = os.path.join(out_dir, f"{name}.wav")
        write_wav(path, pcm)
        dur = len(pcm) / 2 / _RATE  # [s]
        print(f"  {os.path.basename(path):28s} {dur:4.1f}s  「{text}」")
        n += 1
    print(f"[sim-audio] {n} ファイルを {out_dir} に保存")
    return n


class HostSpeakerAnnouncer:
    """Announcer 実装: pyopenjtalk 合成 → 手元PCの sounddevice 再生（非ブロッキング＋キャッシュ）。

    収穫ループを止めないよう ``say`` は即返し、バックグラウンドで順に合成・再生する。
    実機の :class:`G1SpeakerAnnouncer` と同じ Announcer プロトコルなので差し替え可能。
    """

    def __init__(self, rate: int = _RATE) -> None:
        self._rate = rate
        self._cache: dict[str, bytes] = {}
        self._q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="host-speaker")
        self._thread.start()

    def say(self, text: str) -> None:
        self._q.put(text)

    def _run(self) -> None:
        import numpy as np
        import sounddevice as sd

        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                pcm = self._cache.get(text)
                if pcm is None:
                    pcm = synth_pcm_jp(text, self._rate)
                    self._cache[text] = pcm
                arr = np.frombuffer(pcm, dtype="<i2")
                sd.play(arr, self._rate)
                sd.wait()
            except Exception as exc:
                print(f"[sim-audio] play failed: {exc}", file=sys.stderr)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def make_host_speaker_announcer() -> HostSpeakerAnnouncer:
    """収穫グラフの announcer= に渡す手元PCスピーカー実装を返す。"""
    return HostSpeakerAnnouncer()


def play_all(gap_s: float = 0.4) -> None:
    """全フレーズを順に手元PCで鳴らす（ライブ確認用。音声デバイス必須）。"""
    import time

    spk = HostSpeakerAnnouncer()
    for _name, text in PHRASES:
        spk.say(text)
    # キューが捌けるまで待つ（各フレーズ長＋間）。
    time.sleep(sum(len(synth_pcm_jp(t)) / 2 / _RATE + gap_s for _, t in PHRASES))
    spk.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dump",
        nargs="?",
        const=os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_samples"),
        help="全アナウンスを WAV 化して保存（既定: docs/sim-setup/audio_samples）",
    )
    ap.add_argument("--play", action="store_true", help="全アナウンスを手元PCで順に再生")
    args = ap.parse_args()

    if args.play:
        play_all()
        return 0
    out = args.dump or os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_samples")
    dump_wavs(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
