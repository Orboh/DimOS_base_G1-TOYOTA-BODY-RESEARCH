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

"""``HarvestModule.cmd_vel`` (Twist) を ``docs/sim-setup/sim_dds_bridge.py`` の
``SIM_BASE_MOVE`` が読むファイル("vx,vy" テキスト、既定 ``/tmp/sim_base_move.txt``)へ
橋渡しする sim 専用モジュール。

実機では ``G1HighLevelDdsSdk`` が同じ ``cmd_vel`` を受けて
``LocoClient.Move(vx, vy, vyaw, continuous_move=True)`` を呼ぶが、これは
Unitree 独自の JSON-RPC 形式 DDS トピック(``rt/api/loco/request``)に送信される
プロプライエタリなプロトコルで、sim 側では解釈できない。そのため sim 実行時は
このモジュールで ``G1HighLevelDdsSdk`` を置き換える
（``unitree_g1_okra_honban.py`` の ``OKRA_MOVE_SOURCE=sim`` 切替、
2026-09-12 GUIデモでの要望を受けて追加）。

``sim_dds_bridge.py`` 側の SIM_BASE_MOVE は「ファイルの mtime が
``SIM_BASE_CMD_TIMEOUT``(既定0.3s) 以内なら新鮮」という watchdog 方式のため、
それより速い間隔で書き続ける必要がある。呼び出し元
（``nav_skills.make_twist_move_cmd``）は既定10Hzで publish し続ける設計なので、
受信するたびに書けば自然に間に合う。

⚠️ ``vyaw``(angular.z) は ``SIM_BASE_MOVE`` 側が読まない（"vx,vy" の2値のみ）ため
無視される — 純横移動/前後移動のみ検証可能。回頭を含む検証をしたい場合は
sim_dds_bridge.py 側の対応が別途必要。
"""

from __future__ import annotations

from typing import Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class SimCmdVelBridgeConfig(ModuleConfig):
    # sim_dds_bridge.py の SIM_BASE_MOVE_FILE と一致させること（数値ドリフト防止）。
    base_move_file: str = "/tmp/sim_base_move.txt"


class SimCmdVelBridge(Module):
    """HarvestModule.cmd_vel (Twist) -> sim_dds_bridge.py SIM_BASE_MOVE ファイル。"""

    config: SimCmdVelBridgeConfig
    cmd_vel: In[Twist]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._warned_yaw = False

    def _on_cmd_vel(self, msg: Twist) -> None:
        vx = float(msg.linear.x)
        vy = float(msg.linear.y)
        vyaw = float(msg.angular.z)
        if vyaw != 0.0 and not self._warned_yaw:
            logger.warning(
                "SimCmdVelBridge: SIM_BASE_MOVE は yaw 非対応（vx,vyのみ読む）。"
                "vyaw は無視されます（回頭込みの検証は未対応）。"
            )
            self._warned_yaw = True
        try:
            with open(self.config.base_move_file, "w") as f:
                f.write(f"{vx},{vy}\n")
        except OSError as exc:
            logger.warning("SimCmdVelBridge: file write failed", error=str(exc))

    @rpc
    def start(self) -> None:
        self.register_disposable(Disposable(self.cmd_vel.subscribe(self._on_cmd_vel)))
        logger.info("SimCmdVelBridge started", base_move_file=self.config.base_move_file)

    @rpc
    def stop(self) -> None:
        super().stop()


__all__ = ["SimCmdVelBridge", "SimCmdVelBridgeConfig"]
