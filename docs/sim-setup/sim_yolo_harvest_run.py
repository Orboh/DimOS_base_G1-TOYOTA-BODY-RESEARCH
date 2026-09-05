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

"""YOLO 一連プロセス検証: 実検出→LangGraph→IK→**VLM判断**→把持 を sim で通す（.venv・知覚も本物）。

GT 座標注入を使わず、bridge の ego_view を **実 YOLO-seg で検出** して LangGraph harvest を駆動する。
detect(YOLO)→select→grasp(IK→arm_sdk → **cut_ok VLM ゲート** → dex1 close, bridge最近傍把持)→
verify(**moondream**)→record→loop。検出も判断も本物（GT/True固定なし）。

VLM（F-02 / SS-02 / 計画書 §8 7-C, exp_01KW1VR255KBKBD58C1M261E4A）:
  - 閉じる(=切断+把持)直前に ego_view を moondream に送り、応答が返れば把持へ進む（cut_ok ゲート）。
  - verify は把持中フレームを moondream で判定（既定 liveness=非空応答→True）。
  - **判定の正しさは実機**（sim 画像は OOD・§10）。sim では「VLM が在ループで応答する」配管を確認。
  - 前提: **Jetson で moondream(ollama) 稼働** ＝ tailscale http://100.113.43.64:11434。
    未到達だと cut_ok が安全側 False＝切らない→picks=0（起動ログに reachable= を出す）。
  - env: SIM_VLM=0 で VLM 無効化（従来 verify=True/ゲート無し）、SIM_VLM_HOST / SIM_VLM_MODEL /
    SIM_VLM_VERIFY_MODE(liveness|caption) / SIM_VLM_GRAB_SECS で調整。

前提 bridge（near クリップ修正済・カメラ配信・最近傍把持）:
  SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 SIM_HEADLESS=0 SIM_LOAD_ROOM=1 SIM_TABLE=1 SIM_OKRA=10 \
  SIM_GRAVITY=1 SIM_SELF_COLLISION=1 SIM_GRASP_NEAREST=1 SIM_GRASP_KINEMATIC=1 \
  SIM_PUB_CAMERA=1 SIM_CAM_MODE=torso SIM_CAM_NEAR=0.03 SIM_CAM_LOOK_WORLD=0.40,0.0,0.78 \
  SIM_VIEWPORT_CAM=1 ... sim_dds_bridge.py

実行（別ターミナル, VLM は既定 ON）:
  SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 .venv/bin/python docs/sim-setup/sim_yolo_harvest_run.py
"""

from __future__ import annotations

import os
import sys

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "docs/sim-setup"))

from sim_yolo_harvest_skills import SimYoloHarvestSkills

from dimos.robot.unitree.g1.harvest.blackboard import Box3D, HarvestConfig, initial_state
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph


def main() -> int:
    iface = os.getenv("SIM_DDS_IFACE", "lo")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "127.0.0.1").split(",") if p.strip()]
    conf = float(os.getenv("YOLO_CONF", "0.25"))
    out = os.getenv("SIM_YOLO_OUT")  # 注釈画像保存（任意）

    skills = SimYoloHarvestSkills(iface=iface, peers=peers, conf=conf, save_dir=out)
    # reach box は graph 空間モデル系（x=lateral(+右), y=depth(+前), z=高さ）で記述する。
    #   x(右): -0.30..0.30（左右±30cm）, y(前): 0.28..0.62（前方の届く奥行）, z: -0.25..0.15。
    # ※ detect が pos_3d を graph 系で出すよう統一済（軸ねじれ修正）。reposition/advance_left は
    #   この系で正しく横移動になる。広い畝(SIM_OKRA_LAT で±0.45等)では端が x(右)±0.30 を超え
    #   reposition→base 横移動で寄ってから収穫する。
    cfg = HarvestConfig(
        reach=Box3D(-0.30, 0.30, 0.28, 0.62, -0.25, 0.15),
        basket_capacity=99,  # この demo は交換を起こさない（S3 は Tier0 済）
        max_empty_advances=0,  # 全部見えている（探索移動しない）
        ripeness_threshold=0.5,
        max_grasp_retries=1,  # 検出由来の重複/誤りはリトライせず次へ
    )

    # 音声アナウンス（M1・F-12 / §7-A）: 手元PCスピーカーへ。各場面で日本語フレーズが鳴る。
    # SIM_VOICE=0 で無効。音声デバイス不在でも収穫を止めない（NullAnnouncer にフォールバック）。
    announcer = None
    if os.getenv("SIM_VOICE", "1") not in ("0", "false", "False", ""):
        try:
            from sim_audio import make_host_speaker_announcer

            announcer = make_host_speaker_announcer()
            print("[run] 音声アナウンス=ON（手元PCスピーカー, SIM_VOICE=0 で無効）", flush=True)
        except Exception as exc:
            print(f"[run] 音声アナウンス無効化（{exc}）→ 無音で続行", flush=True)

    app = build_harvest_graph(skills, cfg, announcer=announcer)
    print("[run] YOLO 検出で LangGraph harvest を sim 駆動（実検出→IK→VLM→把持）", flush=True)
    final = app.invoke(initial_state(), {"recursion_limit": 300})
    print(
        f"\n[run] 完了: picks={final.get('picks')} records={len(final.get('records', []))}",
        flush=True,
    )
    for line in final.get("log", [])[-12:]:
        print(f"   {line}", flush=True)
    return 0 if final.get("picks", 0) >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
