#!/usr/bin/env python3
"""YOLO検出 → /clicked_point ブリッジ(人間のクリックの完全な代替、2026-07-22夜実装).

パイプライン:
  /camera/color_image → okra11n-seg 推論 → 最高信頼度のマスク重心(u,v)
  → /camera/camera_info の内部パラメータで /camera/pointcloud を画素面へ投影
  → 重心の近傍(既定8px)の点群の中央値 = 3D点(光学フレーム)
  → /clicked_point に PointStamped(frame='/world/camera/pointcloud') を発行

発行するメッセージは人間がビューアでクリックしたものと同一契約なので、
IkReachBridge 側は一切変更なし(confirm/固定向き/上からアプローチ全部そのまま)。
二重クリック確認モード対応: 同じ点を YOLO_BRIDGE_DOUBLE_S 秒空けて2回発行する。

安全設計:
- 既定は DRY-RUN: 検出と3D点をログに出すだけで発行しない。YOLO_BRIDGE_LIVE=1 で発行
- 連続自動発火はしない。**Enterを押すたびに1回だけ**発火する人間ゲート付き
  (--once で1回発火して即終了。デモのTier-2.5用)

実行(アプリ稼働中・別ターミナル):
  cd ~/Toyota-auto-body-PoC/DimOS_oda
  CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
  LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
  YOLO_BRIDGE_LIVE=1 .venv/bin/python oda/yolo_click_bridge.py

環境変数:
  OKRA_YOLO_MODEL      : モデルパス(既定: yolo_overlay.py と同じ okra11n-seg.pt)
  OKRA_YOLO_CONF       : 信頼度しきい値(既定 0.4)
  YOLO_BRIDGE_LIVE     : 1 で /clicked_point を実際に発行(既定 DRY-RUN)
  YOLO_BRIDGE_PX_RADIUS: 重心近傍とみなす画素半径(既定 8)
  YOLO_BRIDGE_DOUBLE_S : 2回発行の間隔秒(既定 0.6; confirm窓0.35-3.5s内)
  YOLO_BRIDGE_MAX_M    : これより遠い3D点は拒否(既定 0.8m; 誤検出の背景対策)
  YOLO_BRIDGE_BODY_FRAME: 1 = ZED用。発行前に光学→ボディ回転(x=z_o, y=-x_o, z=-y_o)。
                        ZEDパイプラインはビューアのクリックがTF経由でボディ座標に
                        なるため、IkReachBridge(click_in_camera_body_frame=True)も
                        ボディ座標を期待する。点群データ自体は両カメラとも光学
                        フレームなので投影計算は共通(2026-07-23 camera.py確認)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL = os.getenv(
    "OKRA_YOLO_MODEL",
    os.path.join(_REPO, "oda/ZED_M_Depth_check/finetune_V5/model/okra11n-seg.pt"),
)
_CONF = float(os.getenv("OKRA_YOLO_CONF", "0.4"))
_LIVE = os.getenv("YOLO_BRIDGE_LIVE", "").strip() == "1"
_PX_RADIUS = float(os.getenv("YOLO_BRIDGE_PX_RADIUS", "8"))
_DOUBLE_S = float(os.getenv("YOLO_BRIDGE_DOUBLE_S", "0.6"))
_MAX_M = float(os.getenv("YOLO_BRIDGE_MAX_M", "0.8"))
_BODY_FRAME = os.getenv("YOLO_BRIDGE_BODY_FRAME", "").strip() == "1"
_CLICK_FRAME = "/world/camera/pointcloud"  # same contract as a human viewer click


def detect_centroid(model, rgb: np.ndarray, conf: float):
    """最高信頼度のオクラのマスク重心(u,v)と信頼度を返す。なければ None."""
    import cv2

    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    r = model.predict(bgr, conf=conf, verbose=False)[0]
    if len(r.boxes) == 0:
        return None
    best = int(np.argmax(r.boxes.conf.cpu().numpy()))
    c = float(r.boxes.conf[best])
    u = v = None
    if r.masks is not None and r.masks.xy is not None:
        # masks.xy は元画像ピクセル座標のポリゴン(レターボックス補正済み)。
        # model-space の masks.data を自前スケールするとパディング分ずれるので使わない。
        poly = np.asarray(r.masks.xy[best], dtype=np.float32)
        if len(poly) >= 3:
            mom = cv2.moments(poly)
            if mom["m00"] > 1e-6:
                u, v = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    if u is None:  # セグなし/縮退ポリゴンのフォールバック: バウンディングボックス中心
        x1, y1, x2, y2 = (float(t) for t in r.boxes.xyxy[best])
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return float(u), float(v), c


def centroid_to_3d(points: np.ndarray, K: list[float], u: float, v: float,
                   px_radius: float):
    """点群(光学フレームxyz)を画素面へ投影し、(u,v)近傍の中央値3D点を返す."""
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]
    p = np.asarray(points, dtype=float)
    z = p[:, 2]
    ok = z > 0.05
    p = p[ok]
    if len(p) == 0:
        return None
    uu = fx * p[:, 0] / p[:, 2] + cx
    vv = fy * p[:, 1] / p[:, 2] + cy
    near = (np.abs(uu - u) <= px_radius) & (np.abs(vv - v) <= px_radius)
    if near.sum() < 3:
        return None
    sel = p[near]
    # 前景を取る: 最も手前の3cmスラブだけで中央値(人間のクリックと同じ「表面点」の意味。
    # 中央値±3cm方式は前景/背景が半々のとき谷間に落ちて空集合になるので不可)
    zmin = float(np.min(sel[:, 2]))
    fg = sel[sel[:, 2] <= zmin + 0.03]
    return np.median(fg, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="1回発火して終了(Enter不要)")
    args = ap.parse_args()

    from ultralytics import YOLO

    from dimos.core.transport import LCMTransport
    from dimos.msgs.geometry_msgs.PointStamped import PointStamped
    from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

    model = YOLO(_MODEL)
    latest: dict = {}
    LCMTransport("/camera/color_image", Image).subscribe(
        lambda m, _=None: latest.update(img=m.data, img_t=time.time())
    )
    LCMTransport("/camera/pointcloud", PointCloud2).subscribe(
        lambda m, _=None: latest.update(pc=m, pc_t=time.time())
    )
    LCMTransport("/camera/camera_info", CameraInfo).subscribe(
        lambda m, _=None: latest.update(K=list(m.K))
    )
    pub = LCMTransport("/clicked_point", PointStamped)

    print(f"[bridge] model={os.path.basename(_MODEL)} conf={_CONF} "
          f"{'LIVE' if _LIVE else 'DRY-RUN(YOLO_BRIDGE_LIVE=1で発行)'}")
    t0 = time.time()
    while not all(k in latest for k in ("img", "pc", "K")):
        if time.time() - t0 > 10:
            print("[bridge] カメラトピックが来ない — アプリ/カメラ起動を確認", file=sys.stderr)
            return 1
        time.sleep(0.2)
    print("[bridge] camera topics OK")

    def fire_once() -> bool:
        if (time.time() - latest["img_t"] > 2.0) or (time.time() - latest["pc_t"] > 2.0):
            print("[bridge] 画像/点群が古い(>2s) — 発火中止")
            return False
        det = detect_centroid(model, latest["img"], _CONF)
        if det is None:
            print("[bridge] オクラ検出なし")
            return False
        u, v, conf = det
        pts, _colors = latest["pc"].as_numpy()
        p3 = centroid_to_3d(pts, latest["K"], u, v, _PX_RADIUS)
        if p3 is None:
            print(f"[bridge] 重心(u={u:.0f},v={v:.0f})近傍に点群なし — 発火中止")
            return False
        if float(np.linalg.norm(p3)) > _MAX_M:
            print(f"[bridge] 3D点が遠すぎ({np.linalg.norm(p3):.2f}m>{_MAX_M}m) — 背景誤検出の疑い、中止")
            return False
        print(f"[bridge] okra conf={conf:.2f} centroid=({u:.0f},{v:.0f}) "
              f"-> optical xyz=[{p3[0]:.3f} {p3[1]:.3f} {p3[2]:.3f}]")
        if _BODY_FRAME:  # ZED: クリック契約はボディ座標(REP-103: x前, y左, z上)
            p3 = np.array([p3[2], -p3[0], -p3[1]])
            print(f"[bridge] -> body xyz=[{p3[0]:.3f} {p3[1]:.3f} {p3[2]:.3f}] (ZEDモード)")
        if not _LIVE:
            print("[bridge] DRY-RUN: 発行せず")
            return True
        msg = PointStamped(x=float(p3[0]), y=float(p3[1]), z=float(p3[2]),
                           frame_id=_CLICK_FRAME)
        pub.publish(msg)          # 1回目 = ARM (confirmモード時)
        time.sleep(_DOUBLE_S)
        pub.publish(msg)          # 2回目 = 確定発火
        print("[bridge] /clicked_point x2 発行 — リーチ開始するはず")
        return True

    if args.once:
        return 0 if fire_once() else 1
    print("[bridge] Enter=検出→収穫1回 / q+Enter=終了")
    while True:
        line = input()
        if line.strip().lower() == "q":
            return 0
        fire_once()


if __name__ == "__main__":
    raise SystemExit(main())
