# ZED画角 × G1右腕 IK到達性マップ

G1 の胸に付けたポケットを含めて、「ZEDカメラの画角のどこに、どれだけの距離なら右腕が届くか」を
IK と Isaac Sim で評価した一式です。

> **図（PNG / GIF）はこのリポジトリに含めていません。** `*.png` `*.gif` が LFS 対象で、
> LFS オブジェクトが全件404のため、入れると clone 時に smudge エラーになります。
> CSV は入っているので、`plot_*.py` を回せば図だけ再生成できます。

## 構成

```
ik_reach_map/
├── zed_fov_reach_map.py        段階1: IK解析（dimos .venv / Isaac Sim 不要・数秒）
├── plot_reach_3d.py            段階1の空間分布図
├── isaacSim/
│   ├── isaac_reach_verify.py   段階2: Isaac Sim headless で幾何・自己干渉・ポケットを確認
│   ├── show_reach_cloud.py     到達=緑 / 不到達=赤 の点群を Isaac Sim に表示・回転動画
│   ├── show_point.py           1点を選んで腕を動かす（赤丸・画角グリッド・GIF書き出し）
│   └── plot_isaac_verify.py    段階1と段階2の突き合わせ図
├── outAndPocket/
│   ├── koshi_plane/最新版/     ★採用条件の結果（pelvis基準・体に正対する平面・0.05m刻み）
│   ├── koshi/                  同じ pelvis 基準だが球面で切った版
│   ├── 0916/                   ポケット0.9倍の比較実験（ランダム2000点×3シード）
│   └── plot_final_maps.py      最終判定の図（6状態に色分け）
└── pocket_cad/                 ポケット本体の生成・取り付けスクリプトと STL
```

## 2段構成である理由

pinocchio は dimos の `.venv`、Isaac Sim は conda の `env_isaaclab_2` と**環境が分かれている**ため、
1つのスクリプトで両方は使えません。CSV 経由で受け渡します（既存の `verify_m2_reach_ik.py` と同じ流儀）。

段階1だけでは「その座標に指先を置ける関節角があるか」しか分からず、**自己干渉もポケットとの当たりも
判定できません**。段階2でそこを詰めています。

## 使い方

```bash
cd <このリポジトリ>
T=docs/sim-setup/ik_reach_map

# 段階1（先に実行）
.venv/bin/python $T/zed_fov_reach_map.py \
    --planes 0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65 \
    --radius-center 0.00396,0,-0.044 --out <出力先>

# 段階2（Isaac Sim headless・約1分）
conda run -n env_isaaclab_2 python $T/isaacSim/isaac_reach_verify.py \
    --csv <出力先>/reach_map.csv --out <出力先>/isaac_verify_with_pocket.csv \
    --pocket --pocket-origin 0.0709,0,0.0770

# 図（CSV があれば Isaac Sim 不要）
DIST_LABEL="forward from the pelvis, plane facing the body" \
.venv/bin/python $T/outAndPocket/plot_final_maps.py \
    --stage1 <出力先>/reach_map.csv --stage2 <出力先>/isaac_verify_with_pocket.csv --out <出力先>
```

各フォルダの `README.md` に、その条件での結果・数値・解釈の注意を書いてあります。
まず `outAndPocket/koshi_plane/最新版/README.md` を見てください。

## 主な結果（採用条件）

- 全1000点中、**到達できるのは 207 点**（ポケット263 / モデル干渉128 / IK不可143 / 箱外259）
- 使える帯は **pelvis から前方 0.25〜0.50 m の右寄り**
- ランダム2000点×3シードでの成功率は **18.78 % ± 0.28**
- ポケットを 0.9 倍にすると **19.70 % ± 0.39**（+0.92 ポイント）。ただし
  「ポケットに当たる」が減った分の多くは「モデル干渉」に移るだけで、効果は頭打ち

## 落とし穴（同じ間違いを避けるため）

| 症状 | 原因 |
|---|---|
| 自己干渉が「0件」と出る | contact report の `actor0/1` は prim パスでなく**整数ID**。`PhysicsSchemaTools.intToSdfPath` で復号が必要 |
| ポケット当たりが「0件」と出る | Mesh prim は**リンクの配下**（`visuals/...`）にある。リンク名で直接 Mesh を探すと0個になる |
| ポケットを変えたのに判定が変わらない | **`--pocket-size`（判定寸法）と `--pocket-stl`（表示形状）は別引数**。両方合わせる |
| DOF の対応がおかしい | Isaac の DOF 順は正準順と違う（index 22 が肘）。**必ず関節名で対応付ける** |
| GUI で姿勢が崩れる / 脚が地面に埋まる | 表示上の問題。`--pause` 中は毎ステップ指令し直す、G1 は z=0.8m に持ち上げる。**判定結果は headless と一致** |

**自己干渉やポケット当たりが 0 件と出たら、まず検出の壊れを疑ってください。**
`isaac_reach_verify.py --selftest` が、わざと腕を体にめり込ませて検出が生きているか確認するモードです。

## 評価の前提（断定できないこと）

- ポケット判定は**外形ボリュームに腕の点が入るか**で見る安全側の評価。中に手が入る状態も当たりと数える
- 自己干渉の大半はめり込み1mm未満（convex hull のかすり）。convex hull は実形状より膨らむので、
  この到達点数は**保守的な下限**
- 腰・脚を固定し、基準姿勢から関節デルタ90°以内という条件付き。**骨盤固定・重力0で、バランスと
  重力たわみは見ていない**
- 距離の基準は **pelvis（IK の ROOT）＝骨盤のほぼ中心で、体の内部**。体表からの距離ではない
