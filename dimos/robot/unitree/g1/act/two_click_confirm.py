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

"""TwoClickConfirm: shared two-click confirmation gate (phantom-click protection).

One instance = one independent click-stream gate. A click CONFIRMS (returns
True) only when it lands within ``radius_m`` of the currently-armed click AND
arrives between ``min_gap_s`` and ``window_s`` after it; anything else ARMS
(or re-arms) on that click and returns False.

Why ``min_gap_s`` (2026-07-22): viewer camera-drags emit /clicked_point
bursts. Spatial scatter alone (radius_m) blocks most of them, but a slow drag
can emit two nearby points -- requiring a minimum gap between the two clicks
kills burst pairs while a deliberate human "click ... click" (>=0.35 s apart in
every observed run) still fires. A click faster than min_gap_s RE-ARMS on the
new point, so a drag burst just keeps re-arming and never fires.

Used by BOTH IkReachBridge (arm) and GripperGraspOnReach (jaw pre-open) with
identical parameters so the two modules agree on which click fired: the same
click pair either moves both or neither. Keep the parameters in sync via the
blueprint (OKRA_CONFIRM_* env knobs).
"""

from __future__ import annotations

import math


class TwoClickConfirm:
    """Two-click confirmation gate. Not thread-safe: caller holds its own lock."""

    def __init__(self, radius_m: float = 0.03, window_s: float = 2.5, min_gap_s: float = 0.35):
        self.radius_m = float(radius_m)
        self.window_s = float(window_s)
        self.min_gap_s = float(min_gap_s)
        self._pt: tuple[float, float, float] | None = None
        self._t: float = 0.0

    def feed(self, x: float, y: float, z: float, now: float) -> bool:
        """Feed one click; True = confirmed (fire now), False = armed/re-armed."""
        if self._pt is not None:
            gap = now - self._t
            dist = math.dist((x, y, z), self._pt)
            if self.min_gap_s <= gap <= self.window_s and dist <= self.radius_m:
                self._pt = None  # consumed -- this click fires
                return True
        self._pt = (float(x), float(y), float(z))
        self._t = float(now)
        return False


__all__ = ["TwoClickConfirm"]
