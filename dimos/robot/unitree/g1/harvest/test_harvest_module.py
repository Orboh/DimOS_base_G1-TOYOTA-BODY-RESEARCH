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

"""HarvestModule の model_grasp_auto_start 分岐（2026-09-16 model_no_kensho用）
のオフラインテスト。

HarvestModule はModule基底クラス（DDS/LCM等）に依存する重いクラスのため、
フルインスタンス化はしない。対象メソッドはselfの属性しか読まない薄い
ディスパッチ・実行ロジックなので、SimpleNamespaceに直接バインドして
unbound method として呼び出す（dimos/hardware/sensors/camera/zed/test_zed.py
の _capture_loop テストと同じ手法）。
"""

from __future__ import annotations

from dataclasses import dataclass
import types
from types import SimpleNamespace

from dimos.robot.unitree.g1.harvest.harvest_module import HarvestModule


@dataclass
class _FakeGraspModule:
    ok: bool = True
    calls: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def run_episode(self, okra=None, force=None) -> bool:
        self.calls.append((okra, force))
        return self.ok


def test_run_model_auto_start_calls_grasp_module_once() -> None:
    grasp = _FakeGraspModule(ok=True)
    fake = SimpleNamespace(config=SimpleNamespace(model_grasp_auto_start=True), _grasp_module=grasp)
    HarvestModule._run_model_auto_start(fake)
    assert grasp.calls == [(None, None)]


def test_run_model_auto_start_without_grasp_module_does_not_raise() -> None:
    fake = SimpleNamespace(config=SimpleNamespace(model_grasp_auto_start=True), _grasp_module=None)
    HarvestModule._run_model_auto_start(fake)  # ログのみ、例外を出さない


def test_run_model_auto_start_swallows_exception_from_run_episode() -> None:
    class _Boom:
        def run_episode(self, okra=None, force=None):
            raise RuntimeError("boom")

    fake = SimpleNamespace(config=SimpleNamespace(model_grasp_auto_start=True), _grasp_module=_Boom())
    HarvestModule._run_model_auto_start(fake)  # 例外を握りつぶしログするだけ


def test_run_dispatches_to_auto_start_when_configured() -> None:
    grasp = _FakeGraspModule(ok=True)
    fake = SimpleNamespace(config=SimpleNamespace(model_grasp_auto_start=True), _grasp_module=grasp)
    # _run() は self._run_model_auto_start() を呼ぶ — SimpleNamespace には実体が
    # ないので、本物のメソッドを fake にバインドしてから呼び出す。
    fake._run_model_auto_start = types.MethodType(HarvestModule._run_model_auto_start, fake)
    HarvestModule._run(fake)
    assert grasp.calls == [(None, None)]


def test_run_uses_langgraph_app_when_auto_start_disabled() -> None:
    invoked: list = []

    class _FakeApp:
        def invoke(self, state, opts):
            invoked.append((state, opts))
            return {"picks": 3}

    fake = SimpleNamespace(
        config=SimpleNamespace(model_grasp_auto_start=False, recursion_limit=400),
        _app=_FakeApp(),
    )
    HarvestModule._run(fake)
    assert len(invoked) == 1
