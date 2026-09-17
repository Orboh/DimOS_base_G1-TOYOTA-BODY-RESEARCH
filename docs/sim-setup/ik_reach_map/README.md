> **このリポジトリには図（PNG / GIF）を含めていません。**
> このリポジトリは `*.png` `*.gif` が LFS 対象で、かつ LFS オブジェクトが全件 404 の状態のため、
> 図を入れると clone 時に smudge エラーを起こします。図は下記のスクリプトで再生成できます
> （CSV は入っているので、Isaac Sim を回さずに図だけ作り直すこともできます）。

# ZED画角 × G1右腕 IK 到達性マップ

ZEDカメラの画角を N×N に分割し、各セルの3D座標へ G1 の右腕が IK で手を伸ばせるかを総当たりで判定して図にする。

**Isaac Sim は使わない。** pinocchio + `g1.urdf` の運動学計算だけで「その座標に指先を置ける関節角が
存在するか」を解く。500点が数秒で終わる。物理エンジンを回さないので、**自己干渉やバランス崩れは判定できない**
（それを見たいなら Isaac Sim 実走が要る）。

`DimOS_base_G1-TOYOTA-BODY-` のコードは読み取るだけで、一切変更しない。

## フォルダ構成

```
IK_doc/
├── zed_fov_reach_map.py      段階1: IK解析（dimos .venv / Isaac Sim 不要）
├── plot_reach_3d.py          段階1: 空間分布図
├── reach_*.png / reach_map.csv / right_arm_*  段階1の出力
└── isaacSim/                 段階2: Isaac Sim headless（env_isaaclab_2）
    ├── isaac_reach_verify.py
    ├── plot_isaac_verify.py
    └── isaac_verify.csv / stage1_vs_stage2.png
```

## 使い方

```bash
cd /isaac-sim/workspace/DimOS_base_G1-TOYOTA-BODY-
.venv/bin/python /isaac-sim/workspace/02_orita_tool/IK_doc/zed_fov_reach_map.py   # 判定 + 地図
.venv/bin/python /isaac-sim/workspace/02_orita_tool/IK_doc/plot_reach_3d.py       # 空間分布図
```

`.venv`（dimos側 Python 3.12）で動かすこと。`env_isaaclab_2` ではなく dimos の venv に pinocchio が入っている。

### よく使うオプション（`zed_fov_reach_map.py`）

| オプション | 既定 | 意味 |
|---|---|---|
| `--grid` | 10 | 画角の分割数 N（N×N） |
| `--depths` | `0.3,0.4,0.5,0.6,0.7` | カメラからの距離[m]。**画素は光線なので距離を与えないと3D点が決まらない** |
| `--hfov` | 90 | 水平画角[deg]（sim の `SIM_CAM_HFOV` 既定値） |
| `--cam-pos` | `0.08,0,0.20` | torso相対のカメラ取付位置（`SIM_CAM_LOCAL_POS`） |
| `--cam-fwd` | `1,0,-0.35` | torso相対の視線方向（`SIM_CAM_LOCAL_FWD`） |
| `--standoff` | 0.0 | 目標手前で止める量[m]。0 = その点そのものに指先を置く |
| `--out` | このフォルダ | 出力先 |

## 出力

| ファイル | 内容 |
|---|---|
| `reach_score.png` | 10×10 の各セルが何距離で届くか（メインの地図） |
| `reach_by_distance.png` | 距離ごとの可否。灰＋「/」= ワークスペース箱の外でIK未評価、濃いほど不足が大きい |
| `reach_space.png` | 到達/未到達の空間分布（上面図・側面図・3D） |
| `reach_map.csv` | 全点の生データ（500行）。列の意味は下表 |
| `right_arm_limits.csv` / `right_arm_joint_order.txt` | 段階1→段階2の受け渡し用 |
| `isaacSim/isaac_verify.csv` | 段階2の結果（Isaac上の指先位置・押し戻し量・接触ペア・めり込み深さ） |
| `isaacSim/stage1_vs_stage2.png` | 段階1と段階2の突き合わせ |

### `reach_map.csv` の列

| 列 | 意味 |
|---|---|
| `depth` / `row` / `col` | 距離[m] と 画角セル番号（row0=画像上端, col0=画像左端） |
| `u`, `v` | セル中心の画素座標[px] |
| `tx`, `ty`, `tz` | その画素＋距離が指す3D点の torso 座標[m]（X前・Y左・Z上）＝ IK の目標点 |
| `reach_ok` | **1 = 届く**（結論。図の ○× はこの列） |
| `in_ws_box` | 0 = ワークスペース箱の外なので IK を解いていない（「届かない」とは別物） |
| `converged` | IK ソルバの収束フラグ |
| `err_m` | 位置誤差[m] ＝ 指先があと何 m 足りないか |
| `joint_margin_rad` | 関節リミットまでの最小余裕[rad] |
| `max_joint_rad` | 解の関節角の絶対値の最大 |
| `q0`〜`q6` | IK 解の右腕7関節角[rad]。並びは `right_arm_joint_order.txt`。**段階2への受け渡し用** |

## 判定のしくみ

判定の正本は既存の `IkApproachSkill`（オクラ収穫パイプラインの Phase B と同一）。`None` を返したら「届かない」。
合格条件は **IK収束 かつ 基準姿勢からの関節デルタ ≤ 90° かつ 関節リミット内 かつ ワークスペース箱の中**。

IKソルバは `position_only=True`（`right_arm_model.py:235`）なので、返る残差 `err` は**純粋な位置誤差[m]**
＝「指先があと何m足りないか」。そのまま余裕の指標に使える。

画素→torso 変換は `docs/sim-setup/sim_dds_bridge.py` の胸カメラ規約に合わせてある
（L600 画角 / L617 取付 / L635-674 optical基底 / L710 焦点距離）。bridge が publish する `cam_to_torso` と同じ構成。

## 結果（既定値での実行, 2026-09-15）

| カメラからの距離 | 到達 | 箱外で即NG | IKで届かずNG | NG時の平均不足 |
|---|---|---|---|---|
| 0.3 m | **80/100** | 20 | 0 | — |
| 0.4 m | **74/100** | 20 | 6 | 5.1 cm |
| 0.5 m | **17/100** | 30 | 53 | 9.3 cm |
| 0.6 m | 0/100 | 65 | 35 | 22.6 cm |
| 0.7 m | 0/100 | 94 | 6 | 46.6 cm |

- **最も余裕があるのは画像中央〜やや右（c5〜c7）の中段（r2〜r6）、距離0.3〜0.5m**。
  最上位は r6c5 = torso (0.435, -0.040, 0.004) @0.4m。
- **画像左2列（c0,c1）は全距離で不可**。右腕の担当範囲（Y ≤ 0.20 m）の外。左側を扱うには左腕IKモデルが要るが、
  リポジトリには `right_arm_model.py` しか無い。
- 効くのは主に左右位置と距離。上下（行）の差は小さい。

## 段階2: Isaac Sim（headless）での裏取り

段階1は「その座標に指先を置ける関節角が存在するか」しか見ていない。**自己干渉とモデル整合は分からない。**
そこで段階1が出した関節角を実際の G1（USD）に入れて確かめるのが `isaac_reach_verify.py`。

段階2のファイルは **`isaacSim/` サブフォルダ**にまとめてある（段階1と混ざらないように分離）。
段階1の出力（`reach_map.csv` 等）を1つ上の階層から読むので、**先に段階1を走らせておくこと**。

```bash
cd /isaac-sim/workspace/DimOS_base_G1-TOYOTA-BODY-
conda run -n env_isaaclab_2 python /isaac-sim/workspace/02_orita_tool/IK_doc/isaacSim/isaac_reach_verify.py
conda run -n env_isaaclab_2 python /isaac-sim/workspace/02_orita_tool/IK_doc/isaacSim/isaac_reach_verify.py --selftest
.venv/bin/python /isaac-sim/workspace/02_orita_tool/IK_doc/isaacSim/plot_isaac_verify.py
```

**環境が段階1と違う**（pinocchio は dimos `.venv`、Isaac Sim は `env_isaaclab_2`）ため2段構成。
段階1が `reach_map.csv` に書いた関節角 q0..q6 を段階2が読む、という受け渡しになっている。

### 実装上つまずいた点（再実行時の注意）

| 事象 | 原因と対処 |
|---|---|
| 位置制御の追従誤差 0.15 rad | `set_joint_positions` で直接入れる。PD追従誤差が幾何誤差に混ざるのを避ける |
| 静定させるほど姿勢が崩れる（settle=30 で誤差 85 mm） | 駆動が指令姿勢を保持しきれない。**`--settle` は小さく保つ**（既定3） |
| DOF順が正準順と違う（index 22 が肘） | **必ず関節名で対応付ける**。インデックス直打ちは誤対応する |
| 接触が1件も出ない | contact report の `actor0/1` は**パス文字列でなく整数ID**。`PhysicsSchemaTools.intToSdfPath` で復号が必要 |
| 常時接触しているペアが出る | ハンドの指どうし等 convex hull の重なり。**ホーム姿勢の接触をベースラインとして差し引く** |
| 動かしていない左腕・脚が接触する | 姿勢保持の崩れによる偽陽性。**右腕が関与する接触だけを数える** |

`--selftest` はわざと腕を体にめり込ませて接触検出が機能しているかを確認するモード。
**自己干渉0件という結果を信用する前に必ず実行すること**（実際、上記の整数IDバグで一度「0件」を誤って出した）。

### 段階2の結果（171点）

| 指標 | 値 |
|---|---|
| 指先と目標の距離 | 中央値 6.0 mm / 最大 176.5 mm |
| **うち指令姿勢がそのまま入った100点** | **中央値 0.1 mm** ← URDF と USD の幾何は一致している |
| 指令姿勢が入らなかった71点 | 中央値 50.5 mm |
| 押し戻し量と指先誤差の相関 | r = 0.82 |
| 接触が検出された点 | 74/171。ただし**71点はめり込み1mm未満**（convex hull のかすり） |
| 実質的なめり込み（1mm以上） | 3点のみ。いずれも**左手と右手の干渉**（画像左寄りへ腕を伸ばした場合） |
| 段階1・2ともに良好（誤差<5mm かつ 接触なし） | **69/171 点** |

`isaacSim/stage1_vs_stage2.png` のパネルa→bで、**段階2を通すと左寄りのセルが大きく減り、右側（c5〜c9）が残る**のが見える。

### 段階2の解釈で断定できないこと

- 接触の大半（71/74）はめり込み1mm未満で、**convex hull 近似は実形状より膨らむ**ため、実機では余裕がある可能性がある。
  パネルbは「安全側に倒したフィルタ」であって「腕がぶつかると確定した」わけではない。
- 指先誤差が大きい点と接触の間には関連がある（誤差50mm超の78%が接触あり、r=0.82）が、
  **接触が唯一の原因とは言えない**。接触なしで誤差が大きい点も8点あり、駆動の姿勢保持が不完全なことも寄与している。
- 骨盤は固定し重力を0にしている。**バランスや重力たわみは見ていない。**

## 解釈の注意

- 0.6 m 以上が 0% の主因は**ワークスペース箱 `ws_x=(0.05, 0.65)` で先に弾かれている**こと（65/100）。
  これは収穫パイプラインの安全フィルタであって腕の物理限界そのものではない。ただし 0.5 m の時点で
  平均9.3cm 届いていないので、箱を外しても 0.6 m が実用になるとは考えにくい。
- **腰・脚を固定し、基準姿勢（全関節0）から関節デルタ90°以内**という条件付きの結果。実機は腰をひねる・
  前傾するぶん実際の到達範囲はこれより広いはず。この地図＝ロボットの限界、ではない。
- 到達率はカメラの取付・向きに強く依存する。今回は sim 既定値での結果で、実機ZEDの取付が違えば地図も変わる。

関連: 開発ログ `toyota-body-orboh/Free-space/orita/log/dimOS/2026-09-15_zed_fov_ik_reach_map.md`
