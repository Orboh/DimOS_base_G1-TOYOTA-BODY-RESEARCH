> **このリポジトリには図（PNG / GIF）を含めていません。**
> このリポジトリは `*.png` `*.gif` が LFS 対象で、かつ LFS オブジェクトが全件 404 の状態のため、
> 図を入れると clone 時に smudge エラーを起こします。図は下記のスクリプトで再生成できます
> （CSV は入っているので、Isaac Sim を回さずに図だけ作り直すこともできます）。

# IK基準座標（pelvis）中心・0.05m刻み — Isaac Sim 検証つき

ZED画角10×10 × 距離12段階を、**IK が解く時の基準座標を中心**に取り直し、
Isaac Sim（headless）で実機モデルの幾何・自己干渉・胸ポケットとの当たりまで確認した結果。

## ★ 距離の基準について（要確認事項への回答）

**基準は「腰の関節」ではなく `pelvis`（骨盤）です。** これは IK ソルバ（pinocchio）の
ROOT フレームそのもので、`arm.torso_in_root` から実測しました。

| | 位置 |
|---|---|
| IK の基準座標 = pelvis 原点 | torso_link 座標で **(0.00396, 0, -0.044) m** |
| torso_link 原点（腰の関節 = waist_pitch_joint） | (0, 0, 0) |

つまり**腰の関節より 4.4 cm 下**です。一つ上の階層 `../../waist_radius/` にある結果は
torso_link 原点（腰の関節）中心なので、**別物**です。混同しないでください。

### 基準点は G1 の内部です

`pelvis.STL` を実測したところ、pelvis 原点はメッシュの bbox の内側にあります。

```
pelvis.STL（pelvis 座標系）
  bbox X -0.0488..+0.0488  Y -0.0645..+0.0645  Z -0.1475..+0.0025 [m]
  原点付近の断面: 前面 X=+0.044 / 背面 X=-0.044
  → 原点は前面から 44 mm 内側、背面から 44 mm 内側 ＝ 骨盤のほぼ中心
```

**したがって距離は体表からではなく、体の中の点から測っています。**
例えば `0.25 m` は骨盤中心から 0.25 m であり、体の前面からは約 **206 mm** です。
体表基準で言い直したい場合は、正面方向でおよそ **44 mm 差し引いて**ください
（ただし方向によって体表までの距離は変わるので、厳密な換算はできません）。

## 結果

| 距離 | 到達 | | 距離 | 到達 |
|---|---|---|---|---|
| 0.25 m | 0/100 | | 0.55 m | 30/100 |
| 0.30 m | 23/100 | | 0.60 m | 20/100 |
| 0.35 m | 31/100 | | 0.65 m | 10/100 |
| 0.40 m | 28/100 | | 0.70 m | 1/100 |
| 0.45 m | 29/100 | | 0.75 m | 0/100 |
| 0.50 m | **32/100** | | 0.80 m | 0/100 |

全1200点の内訳:

| 状態 | 点数 |
|---|---|
| 到達できる | **204** |
| ポケットに当たる | 220 |
| モデル干渉・姿勢が再現できない | 160 |
| IK が解けない | 186 |
| ワークスペース箱の外（IK未評価） | 360 |
| その距離に該当する点が無い（光線が球と交わらない） | 70 |

- **実用域は 0.35〜0.55 m**、右寄り（c6〜c9）に集中。
- **0.25 m は 0/100**。画角の上7割はそもそも該当点が無く、残りは全部ポケットに当たる。
  骨盤中心 0.25 m はポケットが占める空間そのものなので当然の結果。
- 指令姿勢がそのまま入った点では指先誤差 中央値 **7.0 mm**（584点全体）。

## ファイル

| ファイル | 内容 |
|---|---|
| `pocket_reach_score.png` | 10×10のメイン地図（12距離のうち何回通ったか） |
| `pocket_reach_by_distance.png` | 距離ごとの内訳。なぜ落ちたかを6状態で色分け |
| `pocket_reach_space.png` | 空間分布（上面・側面・3D）。側面図にポケット断面つき |
| `pocket_reach_map.csv` | 1200点の最終判定（`state` / `final_ok`） |
| `isaac_verify_with_pocket.csv` | 584点の詳細（指先位置・押し戻し量・接触ペア・めり込み深さ・ポケット判定点数） |
| `reach_map.csv` | 段階1（IK解析）の生データ。IK解の関節角 q0..q6 を含む |
| `right_arm_*` | 段階1→段階2の受け渡し用 |

一つ上の `../ik_only_reach_*.png` は **Isaac Sim を使わない段階1だけ**の図です（区別のため改名）。

## 再現方法

```bash
cd /isaac-sim/workspace/DimOS_base_G1-TOYOTA-BODY-
K=/isaac-sim/workspace/02_orita_tool/IK_doc/outAndPocket/koshi

# 段階1: IK解析（pelvis 中心の半径で点を作る）
.venv/bin/python /isaac-sim/workspace/02_orita_tool/IK_doc/zed_fov_reach_map.py \
    --radii 0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80 \
    --radius-center 0.00396,0,-0.044 --out $K/isaacsim

# 段階2+3: Isaac Sim headless（ポケット込み）※約1分
conda run -n env_isaaclab_2 python /isaac-sim/workspace/02_orita_tool/IK_doc/isaacSim/isaac_reach_verify.py \
    --csv $K/isaacsim/reach_map.csv --out $K/isaacsim/isaac_verify_with_pocket.csv \
    --pocket --pocket-origin 0.0709,0,0.0770

# 図
DIST_LABEL="from the IK reference frame = pelvis" \
.venv/bin/python /isaac-sim/workspace/02_orita_tool/IK_doc/outAndPocket/plot_final_maps.py \
    --stage1 $K/isaacsim/reach_map.csv --stage2 $K/isaacsim/isaac_verify_with_pocket.csv --out $K/isaacsim
```

GUI で様子を見る場合は段階2に `--gui --pause 1.2 --limit 50` を付けてください。
GUI のときだけ G1 を z=0.8 m に持ち上げます（原点が骨盤なので地面に置くと脚が埋まるため）。
**判定結果は headless と完全に一致**することを確認済みです（204/584 点）。

## 断定できないこと

- ポケット判定は**外形ボリュームに腕の点が入るか**で見ている。安全側の評価で、
  ポケットの中に手が入る状態も当たりと数える。
- 自己干渉 303 点のうち **295 点はめり込み 1mm 未満**（convex hull のかすり）。
  convex hull は実形状より膨らむので、204 点は**保守的な下限**。
- 腰・脚を固定し、基準姿勢から関節デルタ90°以内という条件付き。骨盤固定・重力0で評価しており、
  **バランスと重力たわみは見ていない。**
