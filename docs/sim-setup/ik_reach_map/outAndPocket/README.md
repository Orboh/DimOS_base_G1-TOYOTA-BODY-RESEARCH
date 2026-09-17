> **このリポジトリには図（PNG / GIF）を含めていません。**
> このリポジトリは `*.png` `*.gif` が LFS 対象で、かつ LFS オブジェクトが全件 404 の状態のため、
> 図を入れると clone 時に smudge エラーを起こします。図は下記のスクリプトで再生成できます
> （CSV は入っているので、Isaac Sim を回さずに図だけ作り直すこともできます）。

# 胸ポケットを付けた状態での 右腕IK 到達性（2026-09-15）

G1 の胸に四次元ポケット（半円柱 8.0 × 17.0 × 8.5 cm）を付けた状態で、
ZED画角10×10 × 距離5段階の各点に右腕が届くかを再評価した結果。

## 結果

| 段階 | 通過 | 差分 |
|---|---|---|
| IK解析だけ（pinocchio + URDF） | **171** | — |
| ＋ ロボットモデルの幾何・自己干渉（Isaac Sim） | **69** | -102 |
| **＋ ポケットとの当たり** | **65** | **-4** |

- **ポケットが原因で落ちたのは 4 点だけ。** ポケットに当たる点は 50 点あるが、
  うち 46 点は元々（自己干渉か指先誤差で）落ちていた点。
- 当たる場所は **c2〜c5（画角の左寄り〜中央）に集中**。ポケットが胸の中央にあるので、
  腕が体の前を横切るときに当たる。右寄り（c6〜c9）は無傷。
- 指先誤差は、指令姿勢がそのまま入った点では中央値 **0.1 mm**（URDF と USD の幾何は一致）。

## ポケットの取り付け

| 項目 | 値 |
|---|---|
| 外形 | 8.0 × 17.0 × 8.5 cm（肉厚 3mm、内寸 左右164 × 深さ82 mm） |
| 形状 | 正面から見て半円（直径17cm・深さ8.5cm）を前後8cm押し出した半円柱。前後とも板で塞ぎ、開口は上面のみ |
| 取付位置 | torso_link 相対 **xyz = (+0.0709, 0, +0.0770) m**、回転なし |
| 胴体との関係 | 背面の**上端だけが接触**（めり込み 0 mm）。胴体が下に行くほど細くなるため、下側は最大 129 mm 離れる |

## ファイル

| ファイル | 内容 |
|---|---|
| `reach_with_pocket.png` | 4パネル図。a)IKのみ → b)+ロボットモデル → c)+ポケット → d)散布図 |
| `isaac_verify_with_pocket.csv` | 171点の生データ（Isaac上の指先位置・押し戻し量・接触ペア・めり込み深さ・ポケット当たり） |
| `pocket_front/close/side/top.png` | 取り付けた見た目 |
| `g1_with_pocket.usd` | ポケット付き G1（元USDは無改変。reference で参照して足しているだけ） |
| `pocket_halfcyl.stl` | ポケット単体（mm単位、X=0が取付面） |

CSV の `pocket_hit` が 1 の行がポケットに当たる点。`pocket_pts` は腕の判定点のうち何点が
ポケットの空間に入ったか（大きいほど深く入っている）。

## 判定の性格（断定できないこと）

- ポケット判定は**外形ボリュームに腕の点が入るか**で見ている。安全側の評価で、
  「ポケットの中に手が入っている」状態も当たりとして数える。壁の肉厚3mmだけを厳密に見たい場合は要調整。
- 自己干渉 74 点のうち **71 点はめり込み 1mm 未満**（convex hull 近似でのかすり）。
  convex hull は実形状より膨らむので、パネル b/c は保守的なフィルタであって
  「必ずぶつかる」という意味ではない。
- 骨盤固定・重力0で評価している。**バランスと重力たわみは見ていない。**

## 途中で出した誤った結果（同じ間違いを避けるため記録）

| 症状 | 原因 |
|---|---|
| 「ポケットに当たる 0/171 点」と表示 | Mesh prim は**リンクの配下**（`visuals/...`）にあるのに、リンク名で直接 Mesh を探していた。腕メッシュが0個だった |
| 「すべて良好 97 点」と過大表示 | `dd` は m 単位なのに `< 5`（mm のつもり）で比較していた。正しくは `< 0.005` |
| 「自己干渉 0 件」と表示 | contact report の `actor0/1` は prim パスではなく**整数ID**。`PhysicsSchemaTools.intToSdfPath` での復号が必要 |

**自己干渉やポケット当たりが「0件」と出たら、まず検出が壊れていないか疑うこと。**
`isaac_reach_verify.py --selftest` がわざと干渉させて検出を確認するモード。

## 再現方法

```bash
cd /isaac-sim/workspace/DimOS_base_G1-TOYOTA-BODY-
# 段階1（IK解析。先に実行して reach_map.csv を作る）
.venv/bin/python /isaac-sim/workspace/02_orita_tool/IK_doc/zed_fov_reach_map.py
# 段階2（Isaac Sim headless。ポケット込み）
conda run -n env_isaaclab_2 python /isaac-sim/workspace/02_orita_tool/IK_doc/isaacSim/isaac_reach_verify.py \
    --pocket --pocket-origin 0.0709,0,0.0770 \
    --out /isaac-sim/workspace/02_orita_tool/IK_doc/outAndPocket/isaac_verify_with_pocket.csv
# 図
.venv/bin/python /isaac-sim/workspace/02_orita_tool/IK_doc/isaacSim/plot_isaac_verify.py \
    --verify-csv .../outAndPocket/isaac_verify_with_pocket.csv \
    --out .../outAndPocket/reach_with_pocket.png
```

ポケットの位置を変えたら `--pocket-origin` も合わせて変えること（`mount_pocket.py` が出す値をそのまま渡す）。
ポケット本体の生成・取り付けは `/isaac-sim/workspace/sub/四次元ポケット_CAD/` 側。
