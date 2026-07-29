#!/usr/bin/env python3
"""YOLOオクラ検出のライブ重畳ビュー(デモ用・表示のみ、ロボット制御には一切関与しない).

カメラのLCMカラー画像(/camera/color_image)を購読し、okra11n-seg(ファインチューン済み
セグメンテーションモデル)の検出結果(マスク+信頼度)を重ねて別ウィンドウに表示する。
金曜デモのTier-2: 「自動化の目は既にある」を安全に見せるための独立ビューア。

実行(アプリ/カメラpublisherが動いている状態で、別ターミナル):
  cd ~/Toyota-auto-body-PoC/DimOS_oda
  CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
  LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' DISPLAY=:0 \
  .venv/bin/python oda/yolo_overlay.py

オフライン検証(画像1枚, ロボット不要):
  .venv/bin/python oda/yolo_overlay.py --image path/to.jpg --out /tmp/overlay.png

環境変数:
  OKRA_YOLO_MODEL : モデルパス(既定: oda/ZED_M_Depth_check/finetune_V5/model/okra11n-seg.pt)
  OKRA_YOLO_CONF  : 信頼度しきい値(既定 0.4)
  OKRA_YOLO_HZ    : 推論レート上限(既定 3.0)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL = os.getenv(
    "OKRA_YOLO_MODEL",
    os.path.join(_REPO, "oda/ZED_M_Depth_check/finetune_V5/model/okra11n-seg.pt"),
)
_CONF = float(os.getenv("OKRA_YOLO_CONF", "0.4"))
_HZ = float(os.getenv("OKRA_YOLO_HZ", "3.0"))


def _annotate(model, bgr, conf):
    """1フレーム推論して注釈済みBGR画像と検出数を返す."""
    r = model.predict(bgr, conf=conf, verbose=False)[0]
    out = r.plot()  # masks+boxes+conf を描いたBGR
    n = len(r.boxes)
    import cv2

    cv2.putText(out, f"okra x{n}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (0, 255, 0) if n else (0, 0, 255), 2)
    return out, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="オフライン検証: この画像1枚だけ処理")
    ap.add_argument("--out", default="/tmp/yolo_overlay_test.png")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(_MODEL)

    if args.image:
        import cv2

        bgr = cv2.imread(args.image)
        if bgr is None:
            print(f"cannot read {args.image}", file=sys.stderr)
            return 1
        out, n = _annotate(model, bgr, _CONF)
        cv2.imwrite(args.out, out)
        print(f"detections={n} -> {args.out}")
        return 0

    # ---- ライブモード: LCM購読 → 推論 → 表示 --------------------------------
    import cv2
    import numpy as np

    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs.Image import Image

    latest = {}
    LCMTransport("/camera/color_image", Image).subscribe(
        lambda m, _=None: latest.update(img=m.data)
    )
    print(f"listening /camera/color_image; model={os.path.basename(_MODEL)} conf={_CONF}")
    cv2.namedWindow("YOLO okra", cv2.WINDOW_NORMAL)
    period = 1.0 / max(_HZ, 0.1)
    try:
        while True:
            t0 = time.time()
            rgb = latest.get("img")
            if rgb is not None:
                bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
                out, _ = _annotate(model, bgr, _CONF)
                cv2.imshow("YOLO okra", out)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
