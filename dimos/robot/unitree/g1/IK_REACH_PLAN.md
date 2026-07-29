# IK Reach — DimOS IK でオクラに腕を近づける 計画

> 位置づけ：**ACT 非依存の独立実験**。学習ポリシーも自動検出器も使わず、**人間が点群ビューア上でゴール点をクリック指定**し、
> その 3D 点へ DimOS の Pinocchio IK（`CartesianIKTask` / `PinocchioIK`）で右腕を**片道リーチ**（pre-grasp 接近、把持はしない）。
> 既存の安全な送信継ぎ目 `G1ArmSdkConnection`（Stage B で実機検証済の 250Hz clip-to-measured / weight ランプ）を再利用。

決定事項（2026-06-16, ユーザー確認済）:
1. **ゴール**：知覚 → IK 一回リーチ（点群からゴール点を 1 回確定し、右腕を pre-grasp 姿勢へ）。
2. **ゴール点の取得＝人間が指定**：自動検出器（YOLOE 等）は**使わない**。点群ビューア上で人間がオクラの点をクリック → `PointStamped`。
   - ※方針転換の理由：検出器ベースの全モデル重み（YOLOE/YOLO/SAM2/graspnet）が Orboh フォークの git LFS で **404（blob 未 push）** で取得不可。人間指定にすると検出器・重み問題が**丸ごと不要**になる（後述 §6）。
3. **ACT との関係**：ACT 非依存の独立ライン。

---

## 0. ゴールと非ゴール

- **ゴール**：右腕（Dex1 側, 7DOF）の手先を、人間が点群上で指定したオクラ近傍の **pre-grasp ポーズ**へ IK で移動。誤差数 cm で停止。
- **非ゴール**：自動でのオクラ検出・定位、把持・持ち上げ（ACT の領分）、左腕、歩行・台座移動、視覚サーボ閉ループ（今回は一回リーチ）。

---

## 1. ICD（システム間インターフェース）— 着工前に埋める

| Pair (A → B) | 経路 | Transport | Message format | Rate / QoS | SDK / lib | Fault behavior | Smoke-test |
|---|---|---|---|---|---|---|---|
| D435i → RealSenseCamera | USB (NX) | pyrealsense2 | depth16 + color + intrinsics | depth 30Hz / PC 5Hz | `pyrealsense2`（要導入確認） | フレーム欠落→直近保持 | R0a: `enable_pointcloud=True` で `pointcloud` が出るか |
| RealSenseCamera → Rerun viz | in-proc / rerun | `Out[PointCloud2] pointcloud` + `Out[Image] color_image` | PointCloud2（camera/optical frame）+ RGB | 5–30Hz | `dimos.visualization.rerun` | viz 落ち→腕制御に無影響 | R0b: dimos-viewer に点群が見えるか |
| 人間 → RerunWebSocketServer | WebSocket :rerun_ws_port | JSON `{type:"click",x,y,z,entity_path,ts}` | クリック点 | 人手・任意 | websockets | 未クリック→target 出ず hold | 既存 test_websocket_server.py で検証済 |
| RerunWebSocketServer → IkReachBridge | in-proc (Stream) | `Out[PointStamped] clicked_point` | x,y,z + `frame_id=entity_path` | クリック毎 | dimos | 不正フレーム→無視しログ | R1: クリック点の xyz/frame をログ |
| IkReachBridge → G1ArmSdkConnection | in-proc (Stream) | `Out[JointState] arm_target` | 14-vec[rad]（左7=hold, 右7=IK解） | リーチ時 1〜数Hz | dimos | IK 不収束/デルタ超過→target 出さず hold | R2: arm_target をログ（DRY） |
| G1ArmSdkConnection → robot | DDS `enx…` NIC | CycloneDDS | `rt/arm_sdk`(LowCmd_), `rt/lowstate`(LowState_) | 250Hz | unitree_sdk2py, cyclonedds 0.10.5 | target 停止→最終姿勢 hold, stop で weight 1→0 | 既存（Stage B B1 検証済） |

⚠️ **未確定の Smoke-test 行**：(R0a) `pyrealsense2` 導入と D435i から `pointcloud` が出ること、(R1) クリック点の `entity_path`/frame が何系か（→ torso への変換式が決まる）。

---

## 2. 再利用マッピング表（新規コード前の必須チェック）

| 書こうとしている機能 | 既存ソース (file:line) | Port action |
|---|---|---|
| IK ソルバ（DLS, FK, Jacobian） | `dimos/manipulation/planning/kinematics/pinocchio_ik.py:72` `PinocchioIK` | **direct import** |
| Pose→IK→関節, warm-start, デルタ安全制限 | `dimos/control/tasks/cartesian_ik_task.py:80` `CartesianIKTask` | **direct import**（右腕モデルを渡す） |
| D435i → 点群/カラー/内部パラメータ（**検出器不要**） | `dimos/hardware/sensors/camera/realsense/camera.py:75` `RealSenseCamera`（`Out` color_image/depth_image/pointcloud/camera_info, `enable_pointcloud`） | **direct import** |
| 点群可視化（人間が見る） | `dimos/visualization/rerun/`（dimos-viewer / rerun init） | **direct import / 設定** |
| 人間のクリック点 → `PointStamped` | `dimos/visualization/rerun/websocket_server.py:74` `RerunWebSocketServer`（`Out[PointStamped] clicked_point`, `:154` click→publish, `frame_id=entity_path`） | **direct import** |
| PointStamped 型 | `dimos/msgs/geometry_msgs/PointStamped.py` | **direct import** |
| arm_target→rt/arm_sdk 安全送信（clip/weight/250Hz） | `dimos/robot/unitree/g1/act/g1_arm_sdk_connection.py:117` `G1ArmSdkConnection` | **direct import / 再利用** |
| 入力購読→計算→arm_target 発行の Module 雛形 | `dimos/robot/unitree/g1/act/act_bridge.py`（ActBridge） | **inline copy ベース**（ACT 呼び出しを IK 呼び出しへ差し替え） |
| 右腕 7DOF サブモデル | `g1.urdf`（29関節フル） + Pinocchio `buildReducedModel` | **lazy-import refactor**（フルURDFをロードし右腕以外をロック→nq=7 縮約） |
| カメラ→torso 外部パラメータ | `g1.urdf` `d435_joint`（torso_link→d435_link, xyz=(0.0576,0.0175,0.4299), rpy=(0,0.8308,0)） | **direct read**（既知 SE3） |

**新規作成は実質 1 つ**：`IkReachBridge`（ActBridge 雛形 + PinocchioIK、入力を `clicked_point` に）。＋右腕縮約モデルのロード関数（小）。

---

## 3. 目標アーキテクチャ（モジュールグラフ）

```
D435i ──> RealSenseCamera ─┬─PointCloud2[pointcloud]──> Rerun viz（dimos-viewer）
                           └─Image[color_image]───────>      │
                                                              │ 人間がオクラの点をクリック
                                                              ▼
                                            RerunWebSocketServer.clicked_point ─PointStamped(frame=entity_path)─┐
                                                                                                                ▼
G1ArmSdkConnection.motor_states ──JointState(29)──> IkReachBridge
                                                    1) clicked point を torso_link 系へ変換
                                                    2) target SE3 = point + approach offset + 固定姿勢
                                                    3) PinocchioIK(右腕7DOF, warm-start=現右腕角) → q_sol(7)
                                                    4) 安全チェック（収束・関節デルタ）→ 14-vec(左hold + 右q_sol)
                                                    └ Out[JointState] arm_target
                                                                          ▼
                                                    G1ArmSdkConnection ──CycloneDDS──> rt/arm_sdk（250Hz, clip-to-measured, weight ramp）
```

autoconnect で `pointcloud` / `color_image` / `clicked_point` / `motor_states` / `arm_target` を名前一致結線。
**ACT 経路（ActBridge）も自動検出器も起動しない**＝二重司令・重み依存なし。

---

## 4. コンポーネント設計

### ① 右腕 IK モデル（7DOF, torso 基準）
- `g1.urdf` を `pinocchio.buildModelFromUrdf` でロード → 右腕 7 関節以外をロックして `buildReducedModel`（`right_shoulder_pitch/roll/yaw, right_elbow, right_wrist_roll/pitch/yaw`）。
- base = `torso_link`、`ee_joint_id` = `right_wrist_yaw_joint`（必要なら `right_hand_palm` 手先点へ FK オフセット）。結果 `nq==7`。
- 検証 R0c：`forward_kinematics(現右腕角)` の手先が実機の見た目と一致。

### ② クリック点 → torso 系 目標
- `clicked_point: PointStamped`（x,y,z, `frame_id=entity_path`）。**entity_path が示す座標系**が肝（R1 で実測）。
  - 点群を camera/optical frame で Rerun に流しているなら xyz は camera 系 → `d435_joint` の既知 SE3 で torso へ。
  - world/robot 系で流しているならその TF を使う。
- target_position = clicked_point_torso（必要なら approach_offset：手前 -X に数 cm 等）。
- target_orientation = 固定（手のひらがオクラを向く定数 quaternion）。最初は安全側に「現在の手先姿勢を保ち位置だけ寄せる」でも可。

### ③ IkReachBridge（新規 Module, ActBridge 雛形）
- `In[PointStamped] clicked_point`, `In[JointState] motor_states` → `Out[JointState] arm_target`。
- 内部に `PinocchioIK`（右腕）。1) クリック点を torso 系へ変換、2) target SE3 計算、3) `solve(target, q_current=現右腕角)`、4) `check_joint_delta`／収束判定、5) 14-vec 組立（左 7=現値 hold, 右 7=q_sol）。
- **一回リーチ方針**：クリックを受けたら 1 度だけ目標を確定し、その目標へ向かう（毎フレーム解き直しは閉ループ＝非ゴール）。
- 未クリック・IK 失敗時は target を出さない（`G1ArmSdkConnection` は最終姿勢 hold）。

### ④ 送信（既存再利用）
- `G1ArmSdkConnection`：`arm_target`(14) を 250Hz で clip-to-measured 追従、weight 0→1 ランプ。**改修不要**。
- `arm_velocity_limit` はリーチ用に保守的（5–10 rad/s）に下げて検証開始。

### ⑤ Blueprint
- 新 blueprint `unitree_g1_ik_reach.py`：RealSenseCamera + Rerun viz/RerunWebSocketServer + IkReachBridge + G1ArmSdkConnection を合成 → `all_blueprints.py` 登録。DRY-RUN 版（`publish_cmd=False`）も用意。

---

## 5. 段階検証（安全第一・Stage B に倣う）

| 段階 | 内容 | 安全 |
|---|---|---|
| **R0** | 部品単体：(a) `pyrealsense2` + D435i `pointcloud` 出力, (b) dimos-viewer に点群表示, (c) 右腕 FK が実機姿勢と一致 | 動かない |
| **R1** | クリック dry：ビューアでオクラをクリック → `clicked_point` の xyz/frame をログ。frame→torso 変換式を確定 | 動かない |
| **R2** | IK dry-run（`publish_cmd=False`）：クリック点→IK 解→`arm_target` をログ。収束・関節デルタ・到達性を確認 | 動かない |
| **R3** | LIVE・周囲クリア・低速（vel_limit 小, weight ランプ, **追従誤差監視**）：安全な固定点へ片道リーチ | 自立・e-stop |
| **R4** | LIVE：実オクラの点をクリック → pre-grasp 接近（把持なし）。停止距離・誤差確認 | 自立・e-stop |

各段で `g1_arm_sdk_connection.py` の track-err ログ（max|target-measured|）でドリフト無しを確認。

---

## 6. リスク / 未確定

1. ~~オクラ検出器~~ **【方針転換で解消】**：自動検出を**人間クリックに置換**。これにより検出器・モデル重みが一切不要に。
   - 背景（検証済 2026-06-16）：`get_data("models_yoloe")` 等の実体（`data/.lfs/models_*.tar.gz`）は git LFS 追跡済だが **Orboh フォークのサーバーに blob が無く `git lfs pull`→[404]**（YOLOE 185MB / YOLO 72MB / EdgeTAM 52MB / contact_graspnet / graspgen すべて）。ナビ地図(`hk_village*`,`markers_go2`)は `*`＝取得済で LFS 自体は正常＝モデル blob だけ未 push。ACT が動くのは重みが HF(`sotata/act-okura-pick`) にあるため。
   - → 将来自動検出に戻すなら、別途「モデル weight 入手経路（ultralytics 直 DL→`data/models_yoloe/` 手置き 等）」の確立が前提。本計画では回避。
2. **D435i 深度パイプライン**：`RealSenseCamera`（`enable_pointcloud=True`）で point cloud は出る想定。残課題は (a) `pyrealsense2` の導入、(b) D435i のカメラ排他（Stage B の teleimager 一本化方針と同一 D435i を奪い合う→起動 publisher の整理＝NX インフラ調整／Sota 案件）。
3. **クリック点の frame**：`clicked_point.frame_id`(=entity_path) が camera/world/torso のどれか → torso への変換式。R1 で実測確定。
4. **右腕縮約モデルの ee と実手先**：`right_wrist_yaw` vs `right_hand_palm`（Dex1 指先）。pre-grasp 距離は手先定義に依存。
5. **可達性**：クリック点が右腕ワークスペース外だと IK 不収束。台/オクラ配置の前提を決める。

---

## 7. 着手順序（推奨）

1. **R0a/R0b**：`pyrealsense2` 導入確認 → `RealSenseCamera` で D435i 点群を dimos-viewer に表示。
2. **R0c**：右腕縮約モデル + FK 検証。
3. **R1**：ビューアでクリック → `clicked_point` の frame を確定し torso 変換を実装。
4. **R2**：`IkReachBridge` を dry で実装（クリック点→IK→arm_target ログ）。
5. **R3**：`G1ArmSdkConnection` 結線、低速 LIVE で固定点リーチ。
6. **R4**：実オクラをクリックして接近（把持なし）。

---

## 8. 参照
- IK：`pinocchio_ik.py` / `cartesian_ik_task.py` / `cartesian_ik_jogger.py`（PoseStamped 送出の手本）
- カメラ：`hardware/sensors/camera/realsense/camera.py`（D435i 点群源）
- 点クリック：`visualization/rerun/websocket_server.py`（`clicked_point`）/ `test_websocket_server.py`（click→PointStamped 検証）
- 送信継ぎ目：`g1_arm_sdk_connection.py`（Stage B 実機検証済）
- モデル：`dimos/robot/unitree/g1/g1.urdf`（29関節, d435_joint extrinsic 記載）
- 関連計画：`act/STAGE_B_PLAN.md`（ACT 経路・arm_sdk の前提知識）
