# オクラ検出 → 3D重心算出パイプライン（YOLO-seg + ZED）

`unitree-g1-okra-harvest-ik` / `-zed` が実際に使っている、胸部ZEDカメラの映像から
オクラを検出し、把持目標となる重心3D座標をロボットのtorso座標系で得るまでの処理を
コードベースで説明する。**設計の正は `SS-01-オクラ検出` / `SS-04-粗アプローチIK`**
（Obsidian `toyota-body-orboh/G1収穫設計書/`）であり、本ドキュメントは実装がその設計と
どう対応しているかの技術リファレンス。

> ⚠️ 同じ「オクラ検出→3D化」でも、本ドキュメントが説明するのは
> `dimos/robot/unitree/g1/harvest/detect_yolo.py` 系（本パイプライン）だけである。
> リポジトリには目的の異なる実装がもう2つ存在するので混同しないこと（§6参照）。

## 1. 全体データフロー

```
ZEDCamera (pyzed, depth_mode=NEURAL 推奨)
  ├─ color_image  ─┐
  ├─ depth_image  ─┤ (MEASURE.DEPTH, float32 [H,W] メートル値)
  └─ camera_info  ─┘ (K: fx,fy,cx,cy。1Hz配信)
        │
        ▼
Yolo2DDetector(model_name="okra11n-seg.pt" 等 okra-seg 重み)
  .process_image(color_image) → YOLO11n-seg .track(conf, iou, persist=True)
        │
        ▼
YoloOkraDetector.detect()  [detect_yolo.py]
  1. target_classes / min_confidence でフィルタ
  2. _mask_centroid(det): マスク画素の (u,v) 平均 = 「実の重心」ピクセル座標
     (マスク無し → bbox中心にフォールバック)
  3. _mask_median_depth(det, depth_getter):
       - マスク画素を最大200点に一様サブサンプル
       - 各画素の depth_image 値を depth_getter(u,v) で取得
       - 0.05 < d < 10.0 の有効値だけを集め、中央値(median)を採用
       (外れ値・茎/背景の混入に強い。単一点サンプリングより頑健)
  4. make_zed_pixel_to_base(): ZED実intrinsicsでピンホール逆投影
       x_opt = (u-cx)*depth/fx , y_opt = (v-cy)*depth/fy
       → {"x": x_opt, "y": depth, "z": -y_opt}   (harvest座標、カメラ原点)
  5. cam_to_torso 変換（校正済みなら）: カメラ原点→torso原点の座標に変換
        │
        ▼
Okra(id, pos_3d={x,y,z}(torso座標), ripeness, reachable=False)
        │
        ▼
graph.py の select: Box3D(reach box) 判定 → 候補のうち最も右(x最大)を target に
        │
        ▼
ik_approach.py: pos_3d を pinocchio/ROS標準座標へ変換して IK ソルバへ
```

## 2. 座標系の規約（最重要・混同厳禁）

このパイプライン全体は **harvest独自の座標規約** を一貫して使う。ロボット全体で
標準的なURDF/ROS規約（`x`=前, `y`=左, `z`=上）とは**軸の割り当てが違う**。

| 座標系 | x | y | z | 使用箇所 |
|---|---|---|---|---|
| ZED光学フレーム（生） | 右 | 下 | 前（視線方向） | `depth_image` の画素、`camera_info.K` |
| **harvest座標**（本パイプライン全体） | **右** | **前** | **上** | `pos_3d`, `Box3D`(reach/fov), `cam_to_torso_xyzquat` |
| URDF/ROS標準（IKソルバ, `ik_reach_bridge.py`） | 前 | 左 | 上 | `pinocchio` IK、main既存のクリック方式 |

harvest座標 → URDF標準への変換は `harvest_module.py` の `_ik_solve` で1回だけ
明示的に行われる：

```python
target_torso = [y_depth, -x_lat, z_height]  # harvest(x,y,z) -> URDF(x,y,z)
```

`make_zed_pixel_to_base` が出す `{x: x_opt, y: depth, z: -y_opt}` は、ZED光学フレーム
(x=右,y=下,z=前) を harvest座標 (x=右,y=前,z=上) へ**軸入れ替えのみ**（並進なし）で
変換したものである点に注意。これはカメラ原点における点であり、torso原点への変換は
別工程（§3）。

## 3. カメラ→torso変換（ハンドアイ校正）

`OKRA_CAM_TO_TORSO` 環境変数（`"x,y,z,qx,qy,qz,qw"`、**harvest座標系**、
`torso <- camera`）で与える。未設定（既定）だと `_parse_cam_to_torso` が `None` を
返し、**変換されずカメラ座標のまま** `pos_3d` に入る＝reach box判定・IKとも破綻する。

### 校正方法（2通り）

1. **テープ+IMU実測**（推奨、`scripts/compute_cam_to_torso.py`）
   torso原点からZEDレンズ中心までの並進[mm]と傾き[度]を定規・角度計で実測し、
   harvest座標系の規約で入力する（このスクリプト自身のdocstringが規約を明記）。

2. **URDF設計値の代替**（`docs/sim-setup/view_g1_cam_mount.py`）
   URDFのUNITREEロゴ（ZED取付面）位置をFKで計算する。**ただし出力される
   `torso相対pos` はURDF座標系のままで、スクリプト内のコメント
   （"← OKRA_CAM_TO_TORSOのx,y,zに使う"）は座標規約の違いを考慮しておらず誤り**。
   使う場合は必ず §2 の基底変換（`harvest.x=-urdf.y, harvest.y=urdf.x,
   harvest.z=urdf.z`）を適用してから `OKRA_CAM_TO_TORSO` に入れること。
   角度も「水平前向き」の場合はharvest座標系では**単位クォータニオン**
   `0,0,0,1` になる（`_OPTICAL_WXYZ` をそのまま使うのは誤り）。

いずれの方法で求めた値も **2026-09-01時点でコードにデフォルト値として反映されて
いない**（`OKRA_CAM_TO_TORSO=""` が既定）。運用前に必ず設定すること。

## 4. 主要パラメータ

| 名前 | 既定値 | 意味 |
|---|---|---|
| `OKRA_YOLO_MODEL` | `Kota0612/okra-seg-detector` | okra-seg重み |
| `OKRA_TARGET` | `"okra"` | 検出対象クラス名 |
| `_MASK_DEPTH_SAMPLES`（定数） | 200 | mask-median深度計算時の最大サンプル点数 |
| `depth_getter` の有効範囲 | `0.05 < d < 10.0` [m] | これ以外は無効値として除外 |
| `camera_info_fps`（ZEDCamera） | 1.0 Hz | 起動直後 最大1秒 intrinsics が無い窓がある |

## 5. フェイルセーフ設計

`fix/zed-intrinsics-no-guess-fallback` で導入された方針：**推測しない**。

```
camera_info 未受信 or 深度が無効値 → None を返す → その検出を丸ごと捨てる
（D435i定数などへのフォールバックは行わない。1m級の誤差を防ぐため）
```

## 6. リポジトリ内の類似実装との違い（混同注意）

同じ「オクラ検出→3D化」でも実装が3系統ある。

| 実装 | 場所 | 3D化の方式 | 状態 |
|---|---|---|---|
| **本パイプライン** | `harvest/detect_yolo.py` | マスク重心+mask-median深度+ピンホール逆投影 | 検出専用ブランチで開発、mainへ統合作業中 |
| 汎用点群パイプライン | `dimos/perception/detection/` (`Detection3DModule`) | Open3D点群 + bbox矩形フィルタ（マスク未使用） | person検出で実績あり。okra配線なし |
| クリック→YOLO半自動 | `oda/yolo_click_bridge.py` | マスクポリゴンのモーメント重心 + 点群逆投影 + 前景3cmスラブ中央値 | main既存。`/clicked_point` 発行、Enterキーで人間ゲート駆動 |

## 参考

- 設計書: `SS-01-オクラ検出.md` / `SS-04-粗アプローチIK.md`（Obsidian `toyota-body-orboh`）
- 距離算出精度の検証: `DEPTH_ACCURACY_EXPERIMENT.md`（同ディレクトリ）
