# 折田さん向け ハンドオフ — G1オクラ収穫 sim（Isaac Sim）

作成: 2026-07-13 / 引き継ぎ元: 上田 / 対象: 折田さん（強化学習）

このドキュメントは「①タスク列挙 → ②環境移行 → ③レポ/ファイル地図」の順で、折田さんが自分の PC で
sim を再現し、5つのタスクに着手できるようにするための引き継ぎ資料です。
既存の環境構築手順は [`TEAM_SETUP.md`](/docs/sim-setup/TEAM_SETUP.md)、実行カタログは [`docs/sim-setup/README.md`](/docs/sim-setup/README.md) が正。本書はそれを
「折田さんの担当タスク」に接続する地図として使ってください。

---

## 0. いま何が動いていて、何が課題か（30秒サマリ）

- **パイプライン**: 上位=LangGraph（`detect→select→grasp→verify→record` の固定シーケンス）、
  下位=dimos skills（ZED/YOLO検出 → IK粗アプローチ → 摩擦把持 → 籠収納）。全部 **ローカル1台** で回る
  （Isaac Sim + DDS `lo` ループバック。Jetson/Tailscale は実機用で sim には不要）。
- **歩行**: `unitree_rl_lab` の g1_29dof velocity policy を ONNX 化して Isaac 上で推論（`sim_walk_lib.py`）。
  立位・前進は成立するが**腕を前に出す（把持姿勢）と訓練分布外でふらつく**。← タスク2の核心。
- **把持**: 立位では摩擦把持成立を実証（`okra_z=0.945`）。**歩行モードでは腕がテーブル高のオクラまで
  あと ~5cm 届かず**（腰固定で胴体が高い）成功率が出ない。← タスク1/3の核心。
- 直近で潰した罠（2026-07-12）: 「摩擦で掴めない」の真因は摩擦でなく z 高さ／机ジャム（接近の過剰前進）
  ／グリッパ開放規約（sim は 0.0）／倒れオクラのスキップ分岐。詳細は
  [`Free-space/ueda/2026-07-12.md`（Obsidian）] と後述「既知の落とし穴」。

---

## 1. タスク列挙（折田さん担当・優先度つき）

> 記法: **触るファイル** / **現状** / **ゴール** / **RL・実装メモ**

### T1. Dimos フルプロセスの確認 + 改善（Isaac Sim 上） 〔優先: 高〕
- **触る**: `docs/sim-setup/sim_walk_harvest_run.py`（歩行E2Eランナ）, `sim_dds_bridge.py`（Isaac側・物理/DDS）,
  `sim_walk_harvest_skills.py`（歩行版skills）, `dimos/robot/unitree/g1/harvest/graph.py`（LangGraph本体）。
- **現状**: `detect→select→grasp→verify→record` は全自動で1周する。立位は把持成立、**歩行は picks=0**
  （z不足）。机ジャム/カゴなぎ倒し/グリッパ開放は 2026-07-12 に一部対処済み（後述）。
- **ゴール**: 前列5本を歩行モードで安定収穫（成功率を上げる）。まず「立位で成立している把持」を
  歩行モードでも成立させるのが最短。
- **メモ**: 立位検証は `friction_pick_servo.py`（位置ずれを排除した純把持テスト）。ここで通る設定を
  歩行へ移すのが定石。E2E は §「実行手順」参照。

### T2. 歩行ポリシー強化：前ならえ姿勢でふらつかず歩く 〔優先: 高 / RL本丸〕
- **触る**: `~/Desktop/unitree_rl_lab`（IsaacLab 学習側・**ここで再学習**）, `docs/sim-setup/sim_walk_lib.py`
  （ONNX推論・obs構成・`PolicyWalker`）, `usd_file/walk_policy/{policy.onnx,deploy.yaml}`（成果物差し替え先）。
- **現状**: 現ポリシーは**腕14関節を obs からマスク**して default 姿勢前提で学習（`sim_walk_lib.py` の
  `w_arm_mask`）。把持のため腕を前方（前ならえ）へ出すと、policy が訓練分布外の腕状態を見て
  バランスを崩す。今は「事前挙手→骨盤ピンで踏ん張り」で誤魔化しているが本質解決でない。
- **ゴール**: **腕を前方に出した状態でも安定して歩く/立つ** policy。
- **RLメモ**: (a) 学習時に腕姿勢を obs に含める / arm を task 化して domain randomization、
  (b) 上半身に外乱（把持腕の質量移動）を加えた reward で頑健化、(c) 学習後 ONNX 吐き直し →
  `usd_file/walk_policy/` を差し替え、`deploy.yaml`（gains/default_q/obs項）も対で更新。
  ★ **観測構成は per-term 履歴連結が正**（per-step だと発散。メモリ `g1-isaac-policy-walk-floating-base` 参照）。

### T3. 把持と移動の統合制御 〔優先: 中〕
- **触る**: `sim_walk_harvest_skills.py`（`_align_to` 接近停止 / `grasp_okra` reach→servo→close→lift）,
  `sim_dds_bridge.py`（`SIM_WALK_GRASP_HOLD` 骨盤ピン）。
- **現状の設計**: 「歩いて接近 → 実測 base_x で停止（机ジャム防止）→ 骨盤ピンで土台固定 → Δサーボで
  指の隙間をオクラ中心へ → close → lift」。**骨盤ピンは必須**（外すと reach で前へドリフトし机にラム、
  2026-07-12 実証）。残課題は**ピン中は胴体が高く z が ~5cm 届かない**こと。
- **ゴール**: 「安定した土台」と「オクラ高さへ届く」を両立。案: 把持中に軽く腰を落とす（crouch）
  動作を policy/教示で入れる、または T2 の arm-aware policy で立ったまま低リーチを許容。
- **メモ**: 座標は torso 相対。オクラ中心 torso z≈-0.066。立位は隙間が z=-0.10 まで届いて成立、
  歩行は -0.16 でも降りきらない（=腰高が効いている）。

### T4. 移動系スタック：オクラと一定距離・オクラを踏まない 〔優先: 中〕
- **触る**: `dimos/robot/unitree/g1/harvest/graph.py`（`reposition`/`advance_left`/`revisit`）,
  `nav_skills.py`, `sim_dds_bridge.py`（base 移動）, 設計 SS-07（移動と足配置）。
- **現状**: `standoff_min=0.25`（畝安全＝一定距離）、接近は base_x 実測キャップで停止。横 sweep は
  `advance_step=0.30`。ただし**「オクラを踏まない」フットプランニングは未実装**（足配置制約なし）。
- **ゴール**: (a) 畝と一定距離を保って前進、(b) オクラ株を踏まない足配置。sim 上で株位置を
  コストマップ/接地禁止領域として与え、footstep を制約。
- **メモ**: 現状の base 移動は policy への `vx,vy` 指令（`/tmp/sim_base_move.txt`）。足配置制御は
  velocity policy の外側に footstep 制約層が要る（RL or 探索計画）。SS-07 の設計を参照。

### T5. LangGraph タスク定義の妥当性確認 〔優先: 中 / 設計レビュー〕
- **触る**: `dimos/robot/unitree/g1/harvest/graph.py`, `blackboard.py`（`HarvestConfig`）,
  `sim_walk_harvest_run.py`（cfg 上書き箇所）。
- **確認すべき論点（既に見つかっている不整合）**:
  1. **reach box の二重定義**: `HarvestConfig.reach` の既定は実機 torso 前提 `Box3D(0.10,0.50, 0.30,0.60, 0.40,1.10)`
     （z=0.4〜1.1m）。だが歩行ランナ `sim_walk_harvest_run.py` は `Box3D(0.0,1.5, -0.6,0.6, -0.3,0.3)` に
     **上書き**（torso 相対・広い箱）。→ select の in-reach 判定が実質ザルになり、到達可否は skills 側の
     `_align_to`/base_x キャップが担保している。**座標系（world/torso）と reach 判定の責務を整理**すべき。
  2. **再把持ループ**: `route_after_verify` は失敗時に**同じ target へ最大 `max_grasp_retries=3` 回** GRASP
     を繰り返す（detect を通らない）。倒れたオクラへ突っ込むのを 2026-07-12 に「倒れ検知スキップ」で
     暫定回避。retry の是非・上限・倒れ判定閾値（`SIM_WALK_FALLEN_Z=0.74`）を設計として妥当か確認。
  3. `ripeness_threshold=0.6` / `grasp_force=0.3` / `advance_step=0.30` / `standoff_min=0.25` /
     `max_empty_advances=2` / `max_revisits=5` などの数値が sim/実機で妥当か。**確定値は Obsidian 設計書が正**
     （CLAUDE.md 参照。数値ドリフト防止のためコードにハードコードしない方針）。

---

## 2. 環境移行（折田さんの PC で再現する手順）

移行は **3系統 + git非管理の重量物**。gitに入るのはコードだけ、それ以外は別経路で運ぶ。

### 2-A. コード（git）
```bash
git clone https://github.com/Orboh/DimOS_base_G1-TOYOTA-BODY-.git dimos-hackathon
cd dimos-hackathon
git checkout feat/g1-sim-neural-grasp      # 現行の作業ブランチ（把持/歩行の最新）
```
> ⚠️ Orboh org のリポジトリへのアクセス権が折田さんに必要（要付与）。

### 2-B. Isaac Sim 環境（conda env `isaac-sim`, py3.10）＝物理/GUI/bridge を動かす
`TEAM_SETUP.md` §2 の通り。要点:
```bash
conda create -n isaac-sim python=3.10 -y && conda activate isaac-sim
pip install --upgrade pip
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
pip install scipy psutil pyyaml pillow boto3 gymnasium
pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu118
```
実行時は必ず `PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES` を付与（user-site 汚染遮断 + EULA）。

### 2-C. dimos 本体 / 収穫ランナ環境（`.venv`, py3.12, **uv 管理**）＝ LangGraph 側
```bash
# uv 未導入なら: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                       # pyproject.toml + uv.lock から .venv を再現
# lerobot は 0.4.1 ピン必須（act_service の正規化処理まわり。CLAUDE.md）
```
実行は `.venv/bin/dimos` ではなく、sim では `.venv/bin/python docs/sim-setup/sim_walk_harvest_run.py`。

### 2-D. 外部リポ（`~/Desktop/` に clone。コード内で絶対パス参照あり）
| リポ | 用途 | clone |
|---|---|---|
| `unitree_sdk2_python` | DDS IDL（`rt/arm_sdk`, `rt/dex1/right/cmd`）。sim 通信に必須 | `github.com/unitreerobotics/unitree_sdk2_python` |
| `unitree_rl_lab` | **歩行ポリシーの学習元（T2 でここを触る）** | `github.com/unitreerobotics/unitree_rl_lab` |
| `unitree_mujoco` | 別シムでの歩行単体検証（任意） | `github.com/unitreerobotics/unitree_mujoco` |
> ⚠️ スクリプトは `/home/kota-ueda/Desktop/...` を絶対パスで参照する箇所がある（`sys.path.insert`）。
> 折田さんのユーザ名が違う場合、`docs/sim-setup/*.py` 冒頭の `REPO`/`sys.path` と
> `PYTHONPATH=.../unitree_sdk2_python` を自分のパスに置換が必要（要 grep 置換）。

### 2-E. USD 資産（git 非管理・**S3**, 計 ~1.5GB）
```bash
aws s3 sync s3://orboh-datasets/g1-okra-sim/usd_file/ usd_file/ --exclude "okra_field_full.usd"
# 8GB GPU では okra_field_full.usd(1.15GB/63M面) は開けない。16GB+ 機のみ個別取得。
```
`walk_policy/{policy.onnx,deploy.yaml}` も `usd_file/` 配下なのでこの sync に含まれる。
> ⚠️ `aws configure` 済み・orboh アカウントアクセスが折田さんに必要（要付与）。

### 2-F. GraspGen（任意・T1/把持改善の neural 化を試す場合。docker image 24.9GB）
```bash
# branch の修正済み Dockerfile から build（torch2.7/CUDA12.8/Blackwell 対応・upstream issue #61/#44 潰し済）
docker build -f dimos/manipulation/grasping/docker_context/Dockerfile -t dimos-graspgen:latest .
# 点群→6DoF把持: docs/sim-setup/graspgen_pc_capture.py → graspgen_infer.py
```

### 2-G. 移行しないもの
- `dex1_1_service`（切断ハードの cross-repo）: sim には不要。実機切断のみ。
- Jetson / Tailscale / ZED 実機経路: sim では未使用。

---

## 3. 実行手順（最短で E2E を1周させる）

```bash
# ① bridge（Isaac 側・GUI・chinou室+机+オクラ+摩擦モード・歩行policy）
cd ~/Desktop/dimos-hackathon
PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
PYTHONPATH=$HOME/Desktop/unitree_sdk2_python \
SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 SIM_HEADLESS=0 \
SIM_WALK_POLICY=1 SIM_LOAD_ROOM=1 SIM_TABLE=1 SIM_OKRA=10 SIM_GRASP_FRICTION=1 \
SIM_WALK_GRASP_HOLD=1 SIM_WALK_SPAWN_X=-0.10 SIM_WALK_SPAWN_Y=0.22 \
SIM_OKRA_BREAK_N=1.0 SIM_LOG_EVERY=1 \
  ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/sim_dds_bridge.py > /tmp/bridge.log 2>&1 &
# 「loop start」が出れば準備完了

# ② 歩行収穫 E2E（.venv 側・LangGraph）
BRIDGE_LOG=/tmp/bridge.log SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 \
  SIM_WALK_PICK_IDS=0,1,2,3,4 SIM_WALK_SPAWN_X=-0.10 SIM_WALK_SPAWN_Y=0.22 \
  SIM_WALK_GRASP_HOLD=1 SIM_WALK_SERVO_Z_MIN=-0.16 SIM_WALK_BASE_X_MAX=0.08 \
  .venv/bin/python docs/sim-setup/sim_walk_harvest_run.py
# 停止: touch /tmp/sim_bridge_stop
```
立位（位置ずれ排除）の純把持テスト（T1/把持デバッグの基準）:
```bash
# bridge を SIM_GRAVITY=1 SIM_SELF_COLLISION=0（立位・重力ON）で起動した上で
BRIDGE_LOG=/tmp/bridge.log PICK_IDX=0 SERVO_Z_MIN=-0.10 \
  .venv/bin/python docs/sim-setup/friction_pick_servo.py
```

### 主な env（bridge / skills 調整ノブ）
| env | 既定 | 意味 |
|---|---|---|
| `SIM_WALK_POLICY` | 0 | 1で本物ポリシー歩行（base移動を脚歩行に） |
| `SIM_GRASP_FRICTION` | 0 | 1で摩擦把持（磁石不使用・純物理） |
| `SIM_WALK_GRASP_HOLD` | 0 | 1で把持中に骨盤ピン（倒れない土台）。**現状必須** |
| `SIM_WALK_SERVO_Z_MIN` | -0.12 | Δサーボ z 下限（低いほど深く降りる。歩行は要 -0.16） |
| `SIM_WALK_BASE_X_MAX` | 0.08 | 接近停止の実測 base_x スタンス（机ジャム防止） |
| `SIM_WALK_FALLEN_Z` | 0.74 | 倒れオクラ判定 z（以下ならスキップ） |
| `SIM_WALK_SPAWN_X/Y` | 0/0 | スポーン位置（左端収穫は Y=0.22 で左端okraを即リーチ） |
| `SIM_GRAVITY` / `SIM_SELF_COLLISION` | — | 立位検証用（walk時は重力ON固定・指定無視） |

---

## 4. レポ / ファイル地図

### 4-A. 収穫ロジック（`dimos/robot/unitree/g1/harvest/`）
| ファイル | 役割 |
|---|---|
| `graph.py` | **LangGraph 本体**（固定シーケンス・ルータ）。T5 の主対象 |
| `blackboard.py` | `HarvestConfig`（reach box/閾値/上限）・状態型。T5 |
| `ik_approach.py` | pinocchio IK 粗アプローチ（reach 解） |
| `place_basket.py` | F-07 籠収納（開放角 q_open。sim は 0.0 に上書き必須） |
| `act_grasp.py` / `grasp_sequence.py` | ACT 精密把持（実機/学習系。neural 把持の受け皿） |
| `detect_yolo.py` | YOLO11-seg オクラ検出 |
| `nav_skills.py` | 移動スキル。T4 |
| `skills.py` / `real_skills.py` / `dummy_skills.py` | skills 抽象と実装（実機/モック） |

### 4-B. sim ハーネス（`docs/sim-setup/`, 52スクリプト）
| ファイル | 役割 |
|---|---|
| `sim_dds_bridge.py` | **Isaac 側の心臓**（USD構築・物理・DDS購読・歩行tick・把持固定・骨盤ピン）。73KB |
| `sim_walk_lib.py` | 歩行ポリシー推論（`PolicyWalker`・obs構成・floating base）。**T2** |
| `sim_walk_harvest_skills.py` | 歩行版 HarvestSkills（接近/Δサーボ/把持/倒れ検知）。**T1/T3** |
| `sim_harvest_skills.py` | 立位版 skills（親クラス） |
| `sim_walk_harvest_run.py` | 歩行 E2E ランナ（LangGraph 起動）。**T1** |
| `friction_pick_servo.py` | 立位・純把持テスト（位置ずれ排除の基準器）。2026-07-12 新規 |
| `friction_grasp_probe.py` | 把持の接触力プローブ（法線/摩擦力計測）。2026-07-12 新規 |
| `view_chinou.py` | シーン確認ビューア（`ROOM_USD` で畑へ切替可） |
| `graspgen_pc_capture.py` / `graspgen_infer.py` | GraspGen 点群→6DoF把持（T1 neural化） |
| `README.md` / `TEAM_SETUP.md` / `OKRA_FIELD.md` / `CHINOU_ROOM.md` | 実行カタログ / 環境構築 / 畑・室の資産メモ |

### 4-C. USD 資産（`usd_file/`, S3）
`chinou_center.usd`（実測室）, `okra_field.usd`（実測畑2M面）, `okra_field_full.usd`（63M面/16GB+機用）,
`g1-29dof-dex1-base-fix-usd/g1bag.usd`（収穫構成G1・basket_physics.usd と同居必須）, `okra.usd`,
`walk_policy/{policy.onnx,deploy.yaml}`（歩行 policy 成果物・**T2 の差し替え先**）。

### 4-D. 設計書（Obsidian `toyota-body-orboh` vault・**着手前に必読**）
`G1収穫設計書/`: `00-全体設計書` / `11-開発計画` / `SS-01`〜`SS-08`。
特に **SS-04 粗アプローチIK / SS-05 精密把持ACT / SS-07 移動と足配置** が T1/T3/T4 に直結。
**パイプラインの確定パラメータ（把持時間・切断角・ブレード上限等）はこの設計書が正**（コードにハードコードしない）。

---

## 5. 既知の落とし穴（2026-07-12 デバッグで判明）

- **「摩擦で掴めない」は z 高さの問題**（摩擦μ=1.6は十分）。把持不良を見たらまず `friction_pick_servo.py`
  で位置ずれを排除した純把持テストをして、摩擦 vs 配置を切り分ける。
- **接近の過剰前進（机ジャム/なぎ倒し）**: `_okra_now` の位置推定が spawn 依存でズレる（未修理）。
  実測 base_x キャップ（`SIM_WALK_BASE_X_MAX`）で実害回避中。推定そのものの修理は未着手。
- **グリッパ開放規約が実機と sim で逆**: 実機 Dex1 は開放 q=5.2、sim bridge は 0.0（5.2 は全閉にマップ）。
- **カゴなぎ倒し**: 左手カゴが未収穫オクラを掃く → **左端から収穫**（右へ進めばカゴは収穫済み側へ退く）。
- **bridge が長時間(~300s)で articulation 不正化**し `apply_action`/`get_joint_positions` で落ちることがある。
  `sim_walk_lib.tick` にガードを入れたが根治せず（要因未特定・spawn位置の障害物近接の疑い）。
- **user-site 汚染**: isaac-sim env は `PYTHONNOUSERSITE=1` 必須（torch 等がスキップされ起動失敗する）。
- **観測は per-term 履歴連結**（歩行 policy。per-step だと発散）。T2 で再学習する際の必須事項。

---

## 6. 折田さんへ確認 / 共有が必要なもの（上田 → 折田／要手配）

- [ ] **PC スペック**、特に **GPU VRAM**。sim 実行は 8GB で可、だが **T2 の RL 再学習は大 GPU 推奨**
      （学習は別マシン=EC2 g5 等の選択肢も。`okra_field_full.usd` は 16GB+ 必須）。
- [ ] **Orboh org のリポジトリ**アクセス権付与（`DimOS_base_G1-TOYOTA-BODY-`）。
- [ ] **AWS / S3** アクセス（`aws configure`・orboh アカウント）＝USD 資産取得に必須。
- [ ] 絶対パス（`/home/kota-ueda/...`）の置換方針（ユーザ名差異。sed 一括 or シンボリックリンク）。
- [ ] このハンドオフの**共有形式**（本 md を折田さんに渡す / Obsidian 化 / 別途ミーティング）。
```
