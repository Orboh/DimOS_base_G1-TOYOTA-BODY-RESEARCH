# 距離算出精度の実験計画（YOLO-seg + ZED 重心3D化）

`DETECTION_3D_PIPELINE.md` で説明した重心3D算出（マスク重心 + mask-median深度 +
ピンホール逆投影）の精度を、実機ZEDで定量的に検証するための実験計画。**未実施**。
実測結果が出たらこのファイルに追記し、`SS-01-オクラ検出.md` の「調査・確認項目」に
反映すること。

## 目的

1. mask-median深度が単純な1点サンプリングよりどれだけ外れ値に強いか定量化する
2. ZEDのdepth mode（PERFORMANCE〜NEURAL_PLUS）が重心3D精度・FPSに与える影響を測る
3. 既知距離のターゲットに対する誤差（バイアス・ばらつき）を測定し、実運用での
   許容誤差（IKワークスペース `0.75×0.60×1.20m` 相当）に収まるか判定する
4. `OKRA_CAM_TO_TORSO` 校正値（テープ実測 vs URDF代替、`DETECTION_3D_PIPELINE.md` §3）
   のどちらが実機での到達精度に効くかを比較する

## 実験A: mask-median深度 vs 単純中心点深度

**仮説**: マスク中心1点だけの深度は、葉の遮蔽やオクラのエッジ（背景に近い画素）を
拾うと外れ値になりやすい。mask-median（最大200点サブサンプル、中央値）はこれに
頑健なはず。

**手順**:
1. 固定位置に置いたオクラ（または模型）をZEDで撮影、YOLO-segでマスクを取得
2. 同一フレームに対し、以下3通りで深度を算出し比較:
   - (a) マスク重心1点の深度（`depth_image[v,u]` 単発）
   - (b) マスク全画素の深度の**平均**
   - (c) 現行実装: マスク画素を最大200点サブサンプル → **中央値**（`_mask_median_depth`）
3. 遮蔽あり／なしの両条件で実施（葉を意図的に一部かぶせる）
4. 各手法の「実測距離（メジャー）との誤差」「フレーム間のばらつき（std, 30フレーム）」を比較

**使えるツール**: `oda/ZED_M_Depth_check/finetune_V5/zed_pointcloud_detect_live.py`
（YOLO検出+NEURAL深度+3D点群のライブ表示。この上に深度サンプリング方式を切り替える
オプションを足すと3方式の比較がその場でできる）

## 実験B: ZED depth mode 比較

**使えるツール**: `oda/ZED_M_Depth_check/finetune_V5/zed_depth_mode_compare.py`
（既存。PERFORMANCE/QUALITY/ULTRA/NEURAL_LIGHT/NEURAL/NEURAL_PLUS の FPS・カバレッジ
・平均深度を自動比較する。90フレーム/モード、HD1080固定）

既存スクリプトはカバレッジとFPSのみ測定するため、以下を追加で測る:
- 各モードでの重心3D誤差（実験Cのターゲットを使う）
- SS-01設計書 §12 記載の推論ボトルネック（YOLO CPU 367ms問題）と合わせて
  「depth計算コスト + YOLO推論コスト」の合計レイテンシも記録し、実運用FPSの目安を出す

## 実験C: 既知距離ターゲットでの誤差測定（本命）

**手順**:
1. torso原点（またはロボット固定の基準点）からの相対位置が既知の点に、オクラ
   （または類似形状のターゲット）を複数距離・複数オフセットで設置する
   - 距離: 0.3 / 0.5 / 0.7 / 1.0 / 1.5 m（IKワークスペース `y: 0.05〜0.65m` を
     カバーしつつ、検出可能距離の限界も見る）
   - 横位置: 中央 / 左寄り / 右寄り（reach box `x: -0.20〜0.75m` の範囲）
   - 高さ: reach box `z: -0.35〜0.85m` の範囲で2〜3点
2. 各設置点でメジャー実測値（真値）を記録
3. `YoloOkraDetector.detect()` の出力 `pos_3d`（cam原点、torso変換前）と、
   ①の真値を比較し、以下を算出:
   - **バイアス**（系統誤差）: 平均誤差ベクトル
   - **ばらつき**（ランダム誤差）: 30フレーム分の標準偏差
4. `OKRA_CAM_TO_TORSO` を通した後のtorso座標でも同様に誤差を記録
   （§4のcam_to_torso校正の妥当性検証を兼ねる）

**評価基準の目安**: IKの `max_reach_pos_err_m`（`ik_reach_bridge.py` 参照、現行の
到達許容誤差）と同オーダー以下（数cm）に収まっているか。これを超える場合は
mask-median のサンプル数・前景スラブ幅（`yolo_click_bridge.py` の3cmスラブ方式）
などのチューニングが必要。

## 実験D: cam_to_torso校正値の比較

`DETECTION_3D_PIPELINE.md` §3 で導出した2つの候補値をどちらも実機に投入し、
実験Cと同じ既知距離ターゲットに対して、**IKが実際にどれだけ正確に到達するか**
（`unitree-g1-okra-harvest-ik` の `OKRA_ACT=0`、ACTなしでIK到達点だけ見るモードが
使える）を比較する。

| 候補 | `OKRA_CAM_TO_TORSO` |
|---|---|
| テープ実測ベース | `-0.0300,0.1090,0.2480,0.0105,0,0,0.9999` |
| URDF設計値ベース | `-0.0045,0.0746,0.2263,0,0,0,1` |

## 記録フォーマット（案）

```
timestamp, mode(A/B/C/D), depth_method, zed_depth_mode, distance_true_m,
lateral_true_m, height_true_m, pos_3d_x, pos_3d_y, pos_3d_z, error_mm, note
```
CSVで `oda/ZED_M_Depth_check/` 配下に蓄積し、後で `pandas` で誤差分布を可視化する
想定（既存の `finetune_V5/validate.py` / `test.py` の構成を流用できる）。

## 参考

- 重心3D算出の仕組み: `DETECTION_3D_PIPELINE.md`
- 設計書 §12（推論性能・GPU化）: `SS-01-オクラ検出.md`（Obsidian）
- IKワークスペース定義: `dimos/robot/unitree/g1/harvest/blackboard.py` の `HarvestConfig.reach`
