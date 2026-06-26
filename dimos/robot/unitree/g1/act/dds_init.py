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

"""Process-global, thread-safe initialisation of the Unitree DDS channel factory.

``ChannelFactoryInitialize`` configures a process-wide singleton. When more than
one DDS module lives in the same dimos process (e.g. G1ArmSdkConnection AND the
Dex1 gripper), each module's ``start()`` would otherwise call it independently —
and if the runtime starts modules on separate threads, those calls can race.

This helper serialises the call behind a lock and runs it exactly once per
process, so any number of sibling modules can ask for the factory without
ordering assumptions. The NIC is taken from the first caller (modules in one
blueprint share the same ``ROBOT_INTERFACE``).
"""

from __future__ import annotations

import threading

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_lock = threading.Lock()
_initialized = False

# Serialises unitree DDS channel/topic CREATION across modules. cyclonedds type
# registration (building the XTypes TypeObject for a topic's IDL type) is NOT
# thread-safe: two modules calling Channel*.Init() concurrently on parallel start
# threads — e.g. G1ArmSdkConnection (unitree_hg LowCmd_/LowState_) and the Dex1
# gripper (unitree_go MotorCmds_/MotorStates_) — corrupt it and raise
# "Failed to encode union ... DDS.XTypes.TypeObject". Hold this around
# ensure_channel_factory + every Channel*.Init() so sibling DDS modules serialise.
channel_lock = threading.Lock()


def ensure_channel_factory(network_interface: str = "") -> None:
    """Initialise the Unitree DDS channel factory once per process (thread-safe).

    Args:
        network_interface: NIC bound to the robot LAN (e.g. ``enx6c1ff771dc67``).
            unitree_sdk2py ignores ``CYCLONEDDS_URI``, so the interface must be
            passed here; an empty string lets the SDK auto-select (wifi/default
            route — usually NOT the robot LAN).
    """
    global _initialized
    import os

    import unitree_sdk2py.core.channel as _ch
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    with _lock:
        if _initialized:
            return
        nic = network_interface or ""
        # --- sim-in-the-loop: 非マルチキャスト経路（tailscale 等）向け unicast Peers 注入 ---
        # unitree_sdk2py は CYCLONEDDS_URI を無視し cyclonedds の config を固定する。
        # よって config テンプレ ``ChannelConfigHasInterface`` 自体に <Peers> を差し込む。
        # env ``DIMOS_DDS_PEERS``（相手の tailscale IP, カンマ区切り）設定時のみ有効＝
        # 通常の実機運用（env 未設定）には一切影響しない。nic（例: tailscale0）必須。
        _peers = os.getenv("DIMOS_DDS_PEERS", "").strip()
        if _peers and nic:
            _pj = "".join(
                f'<Peer address="{p.strip()}"/>' for p in _peers.split(",") if p.strip()
            )
            _ch.ChannelConfigHasInterface = (
                '<?xml version="1.0" encoding="UTF-8" ?><CycloneDDS><Domain Id="any">'
                '<General><Interfaces>'
                '<NetworkInterface name="$__IF_NAME__$" priority="default" multicast="false"/>'
                '</Interfaces><AllowMulticast>false</AllowMulticast>'
                '<EnableMulticastLoopback>false</EnableMulticastLoopback></General>'
                '<Discovery><ParticipantIndex>auto</ParticipantIndex>'
                f'<MaxAutoParticipantIndex>32</MaxAutoParticipantIndex><Peers>{_pj}</Peers>'
                '</Discovery></Domain></CycloneDDS>'
            )
            logger.info(f"DDS unicast Peers injected (sim-in-the-loop): {_peers}")
        logger.info(f"Initializing Unitree DDS channel factory interface={nic!r}...")
        try:
            ChannelFactoryInitialize(0, nic) if nic else ChannelFactoryInitialize(0)
        except Exception as e:  # tolerate an already-initialised factory
            logger.debug(f"ChannelFactoryInitialize raised (likely already init'd): {e}")
        _initialized = True


__all__ = ["channel_lock", "ensure_channel_factory"]
