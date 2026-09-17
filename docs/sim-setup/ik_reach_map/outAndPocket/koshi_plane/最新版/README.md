> **このリポジトリには図（PNG / GIF）を含めていません。**
> このリポジトリは `*.png` `*.gif` が LFS 対象で、かつ LFS オブジェクトが全件 404 の状態のため、
> 図を入れると clone 時に smudge エラーを起こします。図は下記のスクリプトで再生成できます
> （CSV は入っているので、Isaac Sim を回さずに図だけ作り直すこともできます）。

# 【最新版】右腕IK 到達性 — pelvis基準・体に正対する平面・0.05m刻み・ポケット込み

G1 の胸に四次元ポケットを付けた状態で、ZED画角10×10 × 距離10段階の各点に右腕が届くかを、
Isaac Sim（headless）で実機モデルの幾何・自己干渉・ポケットとの当たりまで確認した結果。

**これが最新の条件です。** 過去の版（`../../koshi/` = 球面、`../..` 直下 = カメラ深度基準）は
距離の取り方が違うので、数字を混ぜないでください。

## 距離の取り方（過去版との違い）

| | 基準点 | 切り方 | 刻み |
|---|---|---|---|
| **本版** | **pelvis（骨盤）= IKの基準座標** | **体に正対する平面**（torso の X 一定） | **0.05 m** |
| `../../koshi/` | pelvis | pelvis 中心の球面 | 0.05 m |
| `../..` 直下 | ZEDカメラ | 光軸に垂直な平面 | 0.1 m |

`0.40 m` は「pelvis から前方 0.40 m にある、体に正対した平面」の意味です。

### 基準点は G1 の内部にあります

pelvis 原点は torso_link 座標で **(0.00396, 0, -0.044) m**（腰の関節より 4.4 cm 下）。
`pelvis.STL` を実測すると、原点は前面から 44 mm・背面から 44 mm 内側＝**骨盤のほぼ中心**です。
したがって**体表からの距離ではありません**。正面方向なら体表まで約 44 mm 差し引く形になります
（方向によって体表までの距離は変わるので厳密な換算はできません）。

## 結果

| 前方距離 | 到達 | | 前方距離 | 到達 |
|---|---|---|---|---|
| 0.20 m | 28/100 | | 0.45 m | 27/100 |
| 0.25 m | **34/100** | | 0.50 m | 26/100 |
| 0.30 m | 27/100 | | 0.55 m | 10/100 |
| 0.35 m | 24/100 | | 0.60 m | 0/100 |
| 0.40 m | 31/100 | | 0.65 m | 0/100 |

全1000点の内訳:

| 状態 | 点数 |
|---|---|
| 到達できる | **207** |
| ポケットに当たる | 263 |
| モデル干渉・姿勢が再現できない | 128 |
| IK が解けない | 143 |
| ワークスペース箱の外（IK未評価） | 259 |

- 使える帯は **0.25〜0.50 m の右寄り（c6〜c9）**。
- **近距離（0.20〜0.30 m）はポケットが支配的**、遠距離（0.55 m〜）は IK 不可。
- 0.60 m で 0 になるのはワークスペース箱 X ≤ 0.65 も効いている（pelvis基準0.65 m ＝ torso X=0.654）。
- 指先誤差は 598 点全体で中央値 **4.8 mm**。指令姿勢がそのまま入った点では 0.1 mm 台。

## ファイル

| ファイル | 内容 |
|---|---|
| `pocket_reach_score.png` | 10×10のメイン地図（10距離のうち何回通ったか） |
| `pocket_reach_by_distance.png` | 距離ごとの内訳。なぜ落ちたかを色と記号で6状態に分類 |
| `pocket_reach_space.png` | 空間分布（上面・側面・3D）。側面図にポケット断面つき |
| `reach_cloud.png` | **Isaac Sim 上で到達点=緑・不到達=赤の点群にしたもの** |
| `reach_0.4m_r4c9.gif` | 0.4 m の端（r4c9）へ腕を伸ばす動きのアニメーション |
| `pocket_reach_map.csv` | 1000点の最終判定（`state` / `final_ok`） |
| `isaac_verify_with_pocket.csv` | 598点の詳細（指先位置・押し戻し量・接触ペア・めり込み深さ・ポケット判定点数） |
| `reach_map.csv` | 段階1（IK解析）の生データ。IK解の関節角 q0..q6 を含む |
| `ik_only_reach_*.png` | **Isaac Sim を使わない段階1だけ**の図（区別のため改名） |
| `right_arm_*` | 段階1→段階2の受け渡し用 |

## Isaac Sim で見る

```bash
cd /isaac-sim/workspace/DimOS_base_G1-TOYOTA-BODY-
P="/isaac-sim/workspace/02_orita_tool/IK_doc/outAndPocket/koshi_plane/最新版"
T=/isaac-sim/workspace/02_orita_tool/IK_doc/isaacSim

# 到達=緑 / 不到達=赤 の点群
conda run -n env_isaaclab_2 python $T/show_reach_cloud.py --csv $P/pocket_reach_map.csv
#   --depths 0.3,0.35,0.4 …距離を絞る / --by-state …落ちた理由別の色 / --size 0.024 …点を大きく

# 1点だけ選んで腕を動かす
conda run -n env_isaaclab_2 python $T/show_point.py --csv $P/reach_map.csv \
    --depth 0.4 --row 4 --col 9 --animate --fov-plane 0.4 --highlight-cell
#   --headless --video out.gif …GIF に書き出す
```

GUI のときだけ G1 を z=0.8 m に持ち上げます（原点が骨盤なので地面に置くと脚が埋まるため）。
**判定結果は headless と完全に一致**することを確認済みです。

## 再現方法

```bash
# 段階1: IK解析（pelvis 基準・体に正対する平面）
.venv/bin/python /isaac-sim/workspace/02_orita_tool/IK_doc/zed_fov_reach_map.py \
    --planes 0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65 \
    --radius-center 0.00396,0,-0.044 --out "$P"

# 段階2+3: Isaac Sim headless（ポケット込み）※約1分
conda run -n env_isaaclab_2 python $T/isaac_reach_verify.py \
    --csv "$P/reach_map.csv" --out "$P/isaac_verify_with_pocket.csv" \
    --pocket --pocket-origin 0.0709,0,0.0770

# 図
DIST_LABEL="forward from the pelvis, plane facing the body" \
.venv/bin/python /isaac-sim/workspace/02_orita_tool/IK_doc/outAndPocket/plot_final_maps.py \
    --stage1 "$P/reach_map.csv" --stage2 "$P/isaac_verify_with_pocket.csv" --out "$P"
```

## 断定できないこと

- ポケット判定は**外形ボリュームに腕の点が入るか**で見ている。安全側の評価で、
  ポケットの中に手が入る状態も当たりと数える。
- 自己干渉 322 点のうち **311 点はめり込み 1mm 未満**（convex hull のかすり）。
  convex hull は実形状より膨らむので、207 点は**保守的な下限**。
- 腰・脚を固定し、基準姿勢から関節デルタ90°以内という条件付き。骨盤固定・重力0で評価しており、
  **バランスと重力たわみは見ていない。**
- アニメーションは IK 解へ関節角を線形補間しているだけで、**軌道計画ではない**。
  途中の姿勢が干渉するかは見ていない。
