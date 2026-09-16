#!/usr/bin/env python
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

"""座標変換だけを単独検証するためのダミー推論サーバー（ニューラルネット不使用）。

umi_policy_server.py と全く同じ ZMQ REQ/REP + msgpack プロトコルに応答するが、
中身はカメラもモデルも使わず「指定した1方向に、毎回一定量だけ進み続ける」絶対
ウェイポイントを返すだけ（2026-09-16、座標変換の符号検証のため追加）。
UmiDiffusionBridge 側（IK・ワークスペース安全ゲート・joint delta制限等）は
一切変更せず本物のまま動かすので、「モデルの出来」と「実際に効いている座標系」
を切り分けて検証できる。

⚠️ 2026-09-16 訂正: 当初は「UMI/CAMフレーム（X=右,Y=下,Z=前方）の単位ベクトルを
送り、_R_TIP_TO_UMI がそれをG1のROOT/torsoフレームへ回転変換する」という設計
意図だった。しかし実機検証で判明した通り、現在の既定設定
（UMI_POSITION_ONLY=1 かつ UMI_TIP_TO_TCP_XYZ=0,0,0、いずれも既定値）では
_umi_wp_to_tip() 内の SE3合成が並進オフセット0のため常に恒等になり、
_R_TIP_TO_UMI の回転は位置に一切効かない（数値検証済み）。つまりこのサーバーが
送る数値は、UMIフレームとして回転されることなく、そのままG1のROOT/torsoフレーム
の差分として使われる。そのため以下の --axis は「UMI/CAMフレーム」ではなく
「G1のROOT/torsoフレーム」の単位ベクトルとして定義し直した（pinocchioでのFK
実測: X=前方が正, Y=左が正, Z=上が正、を根拠とする）。

検証したい問いはただ一つ:
  「G1のROOT/torsoフレームで"上（+Z）"方向を指令したとき、実機のグリッパーは
    本当に上に上がるか？」を6方向（±X,±Y,±Z）それぞれ単独で確認する。
このテストは position_only モードの実際の位置制御パイプライン（IK側）を直接
検証するもので、UMI/CAMフレーム変換(_R_TIP_TO_UMI)自体は経由しない
（position_onlyでは位置に対して無効なため）。

安全性: このサーバー自体は座標変換の検証目的で「motionを止める」ロジックを
持たない（＝意図的に無限に同方向へ進み続ける）が、UmiDiffusionBridge 側の
既存の安全ゲート（ws_x/ws_y/ws_z ワークスペース箱、max_joint_delta_deg、
joint limit clamp、max_duration_s）はそのまま効いているため、ワークスペース
境界に達した時点でそれ以上の waypoint は棄却され腕は保持される（衝突に向かって
無制限に動き続けることはない）。それでも実機テストなので e-stop は手に持つこと。

軸の指定（--axis）は G1 ROOT/torsoフレーム基準（pinocchio実測確認済み）:
  up/down      -> Z軸（up = +Z, down = -Z）      ※上下
  forward/back -> X軸（forward = +X, back = -X） ※前後（前進が接近方向）
  left/right   -> Y軸（left = +Y, right = -Y）   ※左右

Run:
  .venv/bin/python oda/umi_diffusion/axis_probe_server.py --axis up --step-mm 3.0
  # 別ターミナルで通常通り:
  IK_REACH_LIVE=1 OKRA_MODEL_NAME=axis_probe_up ROBOT_INTERFACE=<NIC> \
    .venv/bin/dimos run unitree-g1-model-no-kensho
  # 3枚目のターミナルで Enter を押せばいつでも安全に打ち切れる:
  .venv/bin/python oda/press_enter_to_cut.py

umi conda env は不要（zmq/msgpack/numpy のみ、DimOSの.venvで動く）。
"""

from __future__ import annotations

import time

import click
import msgpack
import numpy as np
import zmq

# G1 ROOT/torsoフレームでの単位ベクトル（pinocchio FK実測: X=前方が正, Y=左が正,
# Z=上が正）。position_onlyモードではこの数値がそのままIK targetのROOT座標として
# 使われる（UMI/CAMフレーム回転は経由しない、上のdocstring参照）。
_AXES = {
    "up": np.array([0.0, 0.0, 1.0]),
    "down": np.array([0.0, 0.0, -1.0]),
    "forward": np.array([1.0, 0.0, 0.0]),
    "back": np.array([-1.0, 0.0, 0.0]),
    "left": np.array([0.0, 1.0, 0.0]),
    "right": np.array([0.0, -1.0, 0.0]),
}


@click.command()
@click.option("--addr", default="tcp://127.0.0.1:5599")
@click.option(
    "--axis",
    type=click.Choice(sorted(_AXES)),
    default="up",
    help="G1 ROOT/torsoフレーム基準でどちらへ進み続けるか",
)
@click.option(
    "--step-mm",
    default=3.0,
    type=float,
    help="1 waypointあたりの移動量[mm]。n_exec_per_infer(既定2)×これが実際の"
    "1推論あたりの前進量。先程のdiffusion policy実測（数mm/waypoint）に合わせた既定値",
)
@click.option("--chunk-n", default=16, type=int, help="1推論で返すwaypoint数")
def main(addr: str, axis: str, step_mm: float, chunk_n: int) -> None:
    direction = _AXES[axis]
    step_m = step_mm / 1000.0

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(addr)
    print(
        f"axis_probe_server ready on {addr}: axis={axis} (ROOT/torso unit vector={direction}) "
        f"step={step_mm:.2f}mm/waypoint chunk_n={chunk_n}. Ctrl-C to stop.",
        flush=True,
    )
    n_req = 0
    try:
        while True:
            req = msgpack.unpackb(sock.recv(), raw=False)
            n_req += 1
            if req.get("cmd") == "health":
                sock.send(
                    msgpack.packb(
                        {"ok": True, "cam_age_ms": 0.0, "black_frac": 0.0}, use_bin_type=True
                    )
                )
                continue
            if req.get("reset"):
                print(f"RESET req#{n_req} eef_pos={req['eef_pos']}", flush=True)
            eef_pos = np.asarray(req["eef_pos"], dtype=np.float64)
            eef_aa = np.asarray(req["eef_rot_aa"], dtype=np.float64)
            # 毎回「今の実測位置」から direction*step_m*(i+1) だけ進んだ絶対
            # waypointを chunk_n 本返す。回転はposition_only=True前提で使われない
            # ため測定値をそのまま返す。
            actions = [
                [*(eef_pos + direction * step_m * (i + 1)).tolist(), *eef_aa.tolist()]
                for i in range(chunk_n)
            ]
            rep = {
                "ok": True,
                "actions": actions,
                "n": chunk_n,
                "infer_ms": 0.0,
                "cam_age_ms": 0.0,
                "black_frac": 0.0,
            }
            if n_req % 5 == 0:
                print(
                    f"req#{n_req} obs eef_pos={np.round(eef_pos, 4)} "
                    f"-> a0(abs)={np.round(actions[0][:3], 4)}",
                    flush=True,
                )
            sock.send(msgpack.packb(rep, use_bin_type=True))
    except KeyboardInterrupt:
        pass
    finally:
        sock.close(0)


if __name__ == "__main__":
    main()
