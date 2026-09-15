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

"""Offline checks for the interim YOLO-backed detect_okra (no robot, no weights).

Uses a stub detector so the bbox→Okra mapping, class filter, 3D placement and
graph integration are verified without ultralytics weights or a camera. The live
YOLO path (real ``Yolo2DDetector``) is smoke-checked separately, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig, initial_state
from dimos.robot.unitree.g1.harvest.detect_yolo import (
    YoloOkraDetector,
    default_pixel_to_base,
)
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph


@dataclass
class _Det:
    name: str
    bbox: tuple[float, float, float, float]
    confidence: float = 0.9
    track_id: int | None = None


class _StubDetector:
    """Stands in for Yolo2DDetector: process_image returns a fixed detection list."""

    def __init__(self, dets):
        self._dets = dets

    def process_image(self, _frame):
        return self._dets


class _Frame:
    width = 640
    height = 480


def _detector(dets, **kw):
    return YoloOkraDetector(
        detector=_StubDetector(dets),
        frame_getter=lambda: _Frame(),
        target_classes={"okra", "banana"},
        **kw,
    )


def test_keeps_only_target_classes() -> None:
    dets = [
        _Det("banana", (300, 220, 340, 260), track_id=1),
        _Det("person", (10, 10, 50, 50), track_id=2),  # not a target -> dropped
    ]
    okra = _detector(dets).detect()
    assert len(okra) == 1
    assert okra[0].id == "okra_1"


def test_low_confidence_dropped() -> None:
    dets = [_Det("okra", (300, 220, 340, 260), confidence=0.2, track_id=5)]
    assert _detector(dets, min_confidence=0.5).detect() == []


def test_image_region_sign_from_pixel() -> None:
    # A box on the right half of the image -> +x (region R); left half -> -x (L).
    right = _detector([_Det("okra", (500, 220, 540, 260), track_id=1)]).detect()[0]
    left = _detector([_Det("okra", (60, 220, 100, 260), track_id=2)]).detect()[0]
    assert right.pos_3d["x"] > 0 and right.img_region == "R"
    assert left.pos_3d["x"] < 0 and left.img_region == "L"


def test_ripeness_fn_injected() -> None:
    dets = [_Det("okra", (300, 220, 340, 260), track_id=1)]
    okra = _detector(dets, ripeness_fn=lambda d: 0.83).detect()[0]
    assert okra.ripeness == 0.83


def test_default_pixel_to_base_center_is_forward() -> None:
    # Centre pixel -> straight ahead: ~zero lateral, positive depth.
    p = default_pixel_to_base(320, 240, image_w=640, image_h=480, depth_m=0.45)
    assert abs(p["x"]) < 1e-6
    assert p["y"] > 0


def test_detect_okra_drives_the_graph() -> None:
    """The YOLO detect_fn plugs into the harvest graph and a pick happens."""
    cfg = HarvestConfig()
    dets = [_Det("okra", (320, 240, 360, 280), track_id=7)]
    seen = {"done": False}

    def frame_once():
        if seen["done"]:
            return None  # field empties after the first look
        seen["done"] = True
        return _Frame()

    # Force the detection into the reach box so the graph grasps it.
    detector = YoloOkraDetector(
        detector=_StubDetector(dets),
        frame_getter=frame_once,
        target_classes={"okra"},
        pixel_to_base=lambda u, v, det: {"x": 0.30, "y": 0.45, "z": 0.80},
        ripeness_fn=lambda d: 0.9,
    )
    app = build_harvest_graph(_FakeSkills(detector.detect), cfg)
    final = app.invoke(initial_state(), {"recursion_limit": 400})
    assert final["picks"] == 1


class _FakeSkills:
    """Minimal HarvestSkills: real detect_fn, everything else trivial-success."""

    def __init__(self, detect_fn):
        self._detect = detect_fn
        self.grasps: list = []

    def detect_okra(self):
        return self._detect()

    def relative_move(self, lateral, forward=0.0, yaw=0.0):
        pass

    def go_to_next_station(self):
        return False

    def swap_basket(self):
        pass

    def grasp_okra(self, okra, force):
        self.grasps.append(okra.id)

    def verify_harvest(self):
        return True

    def record_harvest(self, record):
        pass


# --- 3D化できずに捨てた件数の可視化（2026-09-14） ------------------------------
# 「本当にオクラが無い」と「camera_info/深度がまだ来ていない」は、どちらも detect() が
# [] を返すため区別できなかった（音声も既定ログも同一）。last_dropped がその区別を担う。


def test_last_dropped_counts_undeprojectable_detections() -> None:
    dets = [
        _Det("okra", (300, 220, 340, 260), track_id=1),
        _Det("okra", (100, 220, 140, 260), track_id=2),
    ]
    det = _detector(dets, pixel_to_base=lambda u, v, d: None)  # 3D化が常に失敗
    assert det.detect() == []
    assert det.last_dropped == 2


def test_last_dropped_is_zero_when_all_deprojected() -> None:
    det = _detector([_Det("okra", (300, 220, 340, 260), track_id=1)])
    assert len(det.detect()) == 1
    assert det.last_dropped == 0


def test_last_dropped_resets_between_calls() -> None:
    """前回の破棄件数が残ると「カメラ準備中」を誤って言い続ける。"""
    dets = [_Det("okra", (300, 220, 340, 260), track_id=1)]
    fail = {"on": True}
    det = _detector(
        dets, pixel_to_base=lambda u, v, d: None if fail["on"] else {"x": 0.0, "y": 0.45, "z": 0.0}
    )
    det.detect()
    assert det.last_dropped == 1
    fail["on"] = False
    assert len(det.detect()) == 1
    assert det.last_dropped == 0


def test_class_filtered_detections_are_not_counted_as_dropped() -> None:
    """対象クラス外/低信頼は「3D化できなかった」ではないので数えない。"""
    dets = [
        _Det("person", (10, 10, 50, 50), track_id=1),
        _Det("okra", (300, 220, 340, 260), confidence=0.1, track_id=2),
    ]
    det = _detector(dets, min_confidence=0.5)
    assert det.detect() == []
    assert det.last_dropped == 0


def test_factory_exposes_detector_for_dropped_count() -> None:
    """``make_yolo_detect_okra`` の戻り値から破棄件数を辿れる（real_skills が使う）。"""
    from dimos.robot.unitree.g1.harvest.detect_yolo import make_yolo_detect_okra
    from dimos.robot.unitree.g1.harvest.real_skills import DimosHarvestSkills

    detect_fn = make_yolo_detect_okra(
        frame_getter=lambda: _Frame(),
        target_classes={"okra"},
        detector=_StubDetector([_Det("okra", (300, 220, 340, 260), track_id=1)]),
        pixel_to_base=lambda u, v, d: None,
    )
    assert detect_fn() == []
    assert detect_fn.detector.last_dropped == 1

    skills = DimosHarvestSkills(
        move_cmd=lambda *a, **k: None,
        detect_fn=detect_fn,
        grasp_fn=lambda *a, **k: None,
        verify_fn=lambda: True,
        next_station_fn=lambda: False,
        swap_fn=lambda: None,
        record_fn=lambda r: None,
    )
    assert skills.detect_okra() == []
    assert skills.last_detect_dropped() == 1


# --- conf しきい値の一元化（2026-09-15） ---------------------------------
# make_yolo_detect_okra(conf=...) は Yolo2DDetector 側の推論しきい値と
# YoloOkraDetector.min_confidence の両方に同じ値を流す一元窓口。ここでは
# detector を注入して後者への伝播だけを確認する（Yolo2DDetector 自体への
# 伝播は dimos.perception 側の責務・実重み不要のユニット範囲外）。


def test_make_yolo_detect_okra_conf_filters_low_confidence() -> None:
    """既定 conf=0.5 未満の検出は min_confidence として弾かれる。"""
    from dimos.robot.unitree.g1.harvest.detect_yolo import make_yolo_detect_okra

    detect_fn = make_yolo_detect_okra(
        frame_getter=lambda: _Frame(),
        target_classes={"okra"},
        detector=_StubDetector([_Det("okra", (300, 220, 340, 260), confidence=0.3, track_id=1)]),
        pixel_to_base=lambda u, v, d: {"x": 0.30, "y": 0.45, "z": 0.80},
    )
    assert detect_fn() == []


def test_make_yolo_detect_okra_conf_override_lets_it_through() -> None:
    """conf を明示的に下げれば、既定なら弾かれる検出も通る。"""
    from dimos.robot.unitree.g1.harvest.detect_yolo import make_yolo_detect_okra

    detect_fn = make_yolo_detect_okra(
        frame_getter=lambda: _Frame(),
        target_classes={"okra"},
        detector=_StubDetector([_Det("okra", (300, 220, 340, 260), confidence=0.3, track_id=1)]),
        pixel_to_base=lambda u, v, d: {"x": 0.30, "y": 0.45, "z": 0.80},
        conf=0.1,
    )
    assert len(detect_fn()) == 1


def test_last_detect_dropped_is_zero_without_detector_attribute() -> None:
    """検出器をぶら下げない detect_fn（VLM経路など）でも壊れない。"""
    from dimos.robot.unitree.g1.harvest.real_skills import DimosHarvestSkills

    skills = DimosHarvestSkills(
        move_cmd=lambda *a, **k: None,
        detect_fn=lambda: [],
        grasp_fn=lambda *a, **k: None,
        verify_fn=lambda: True,
        next_station_fn=lambda: False,
        swap_fn=lambda: None,
        record_fn=lambda r: None,
    )
    assert skills.last_detect_dropped() == 0
