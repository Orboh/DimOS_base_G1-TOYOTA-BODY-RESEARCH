# Stage B — オクラ ACT の DimOS 移植 計画

> 前提：Stage A（公式 `unitree_lerobot/eval_g1.py` での実機検証）は **2026-06-11 に成功**。
> このドキュメントは、その動作を **DimOS フレームワーク経由**で再現するための実装計画。
> 開始時はまず本ブランチの WIP 3ファイルを再読込して現状確認すること。

---

## 0. ゴールと位置づけ

- **Stage A（完了）**：公式 `eval_g1.py`（単体スクリプト）で「学習済みオクラ ACT が実機 G1 上半身で正しく動く」ことを確認したリファレンス実装。
- **Stage B（本計画）**：同じ動作を **DimOS（Module/Stream/Blueprint＋エージェント層）** で再現する製品統合。ACT を 1 つの skill として扱い、ナビ/知覚/coordinator と合成可能にし、Jetson 単体で動かす本番形にする。

### Stage A で確証が取れた事実（Stage B の設計根拠）
1. **モデルは正しい**（公式経路でオクラ把持成功）。問題が出れば DimOS 統合側を疑う。
2. **制御経路は `rt/arm_sdk`（motion control mode）**：脚はオンボード自立制御に委譲。Unitree 仕様で LowCmd の有効 index は **12–28（腰+腕）と 29（weight）のみ、脚 0–11 は無視**。→ 脚に司令を送る必要も競合もない（`rt/lowcmd` 全身経路だと脚 kp_high 保持で発振＝振動）。
3. **追従速度が鍵**：`eval_g1.py` の `G1_29_ArmController` は **250Hz** publish、毎周期 `clip_arm_q_target(target, velocity_limit=20 rad/s)`。ACT は 30Hz。
4. **カメラは単眼 RGB 480×640 `cam_left_high`**（頭部 D435i）。
5. **右手 Dex1 のみ**：`rt/dex1/right/state` のみ存在。学習データでも state[14]（左 grip）は **mean=std=0 の定数**＝左は未使用。

---

## 1. 目標アーキテクチャ（DimOS モジュールグラフ）

```
G1Connection (lowstate源) ──JointState──┐
TeleimagerCamera (teleimager) ─Image────┤
                                         ▼
                                    ActBridge ──(ZMQ REQ)──> ACT service (別venv, lerobot ACTPolicy)
                                    ├ state[16] 組立(腕14 + 右grip1 + 左grip=0)        └ action[16]
                                    └ RGB 480x640（teleimager形式・変換不要）
                                         │ Out[JointState] arm_target
                                         ▼
                              G1ArmSdkConnection ──CycloneDDS── rt/arm_sdk (motor_cmd[15-28], weight@29)
                              (+ gripper) ─────────────────────  rt/dex1/right/cmd (MotorCmds_)
```
autoconnect で Out↔In を名前一致で結線。

### 既存ファイル（本ブランチ `feat/g1-act-arm-sdk`）
- `act_bridge.py` … ActBridge（In[Image] color_image / In[JointState] motor_states → Out[JointState] arm_target、ZMQ REQ で外部 ACT サービス、30Hz worker）。**dry-run 済**（DDS 書込なし）。
- `g1_arm_sdk_connection.py` … G1ArmSdkConnection（rt/arm_sdk publisher、weight ramp、slew limit、kp_arm=80/kd=3・kp_wrist=40/kd=1.5）。**WIP・以前ドリフト**。
- `blueprints/manipulation/unitree_g1_act_arm.py` … 統合 blueprint。**WIP**。
- 参考：`~/act-okura/act_service.py`（ZMQ REP, lerobot ACTPolicy）、`~/act-okura/act_g1_direct.py`（dimos 非経由の単一プロセス ACT→arm_sdk 閉ループ＝**ほぼ正解形のロジック**）。
- リファレンス：Orboh/unitree_lerobot **PR #1** / **Issue #2**（Stage A の修正一式）。

---

## 2. 主要タスク

### ① 追従ループの修正（最重要・ドリフト根治）
- **原因**：`g1_arm_sdk_connection.py` の slew=0.5 rad/s が遅すぎ → 腕が ACT ターゲットに追従できず → 観測 OOD → 暴走。
- **修正**：
  - 速度上限を **~20 rad/s** に（`eval_g1.py` の clip 同等）。
  - arm_sdk publish を **250Hz** に上げ、ACT(30Hz) とループ分離（最新ターゲットへ高速補間）。
  - weight は 0→1 を数百 ms ランプ後 1 固定。
  - ゲインは現状値で可（バグは slew であってゲインではない）。
  - `~/act-okura/act_g1_direct.py` の閉ループ実装を DimOS モジュールへ移植するのが近道。

### ② カメラ：`TeleimagerCamera` モジュール新設（`color_image` 互換）＋起動時 teleimager 化
**方針**：DimOS のカメラ源を GEAR-SONIC から **teleimager に一本化**。DimOS の下流（nav / ActBridge）は `color_image`（`Out[Image]`）ストリームしか見ないので、**源モジュールを差し替えても出力を `color_image` のままにすれば下流は無改修**で動く。これで「nav が壊れる」ジレンマ（旧案A/B）が解消し、源を teleimager に統一しつつ nav を温存できる。

- **`TeleimagerCamera` モジュール新設**（`ZmqCamera` と同形＝`Out[Image] color_image` を出す）：
  - 中身は teleimager の `ImageClient`（eval_g1.py と同じ。:60000 で設定取得 → :55555 購読 → `get_head_frame()`）でフレーム取得 → DimOS `Image` msg に詰めて emit。
  - 実装は (a) teleimager を DimOS venv に入れて `ImageClient` 再利用、または (b) teleimager のワイヤ形式を読む薄い ZMQ サブスクライバを内製（依存を増やさない）。
  - 色順(BGR→RGB)・解像度を確認（ACT は RGB 480×640。teleimager `head_camera` は 480×640 単眼で一致）。
- **blueprint 置換**：`unitree_g1_act_arm.py`（および nav 系）で `ZmqCamera` → `TeleimagerCamera`（env で切替可能にしておくと安全）。下流は `color_image` を受けるだけなので無改修。
- **起動時 publisher を teleimager 化**：NX の `g1-cam-publisher.service`（`uvc_zmq_publisher.py`, GEAR-SONIC `ego_view`:5555）を無効化し、teleimager image_server（conda `teleimager_relobot`、`teleimager-server --rs`、:55555 PUB + :60000 REQ-REP）を systemd で boot 自動起動に。
  - ⚠️ **D435i は1台＝排他**。teleimager 一本化で GEAR-SONIC publisher は廃止（同時起動不可）。`color_image` を保つので nav は源が teleimager になるだけ。
  - **共有NXインフラの変更**＝本来 Sota に Discord 調整（openclaw 壊れ中）。NX は ssh 必須で直接触れないため、`g1-teleimager.service` ＋ conda 起動 wrapper ＋ installer ＋ 旧サービス無効化手順を用意し、実行はオペレータ（NX 側）。
- **利点**：起動 publisher 1本化（D435i 競合解消）／ACT が学習時と同一の teleimager 形式を受領（変換の当て推量不要）／nav 無改修。
- （別案）DimOS 無改修で「teleimager 購読→`ego_view`:5555 再 publish」ブリッジを1個挟む手もあるが、1ホップ＋JPEG再エンコード増。**本筋は `TeleimagerCamera` 新設**。

### ③ 観測 state[16] の組立（実機グリッパー対応）
- 腕14（恒等写像、dimos 順＝G1_29_JointArmIndex で検証済）＋ **左grip=0 固定**（右手のみ・学習の定数0と一致）＋ **右grip=`rt/dex1/right/state`**。
- dry-run では grip=0 だったので、**右 grip 状態の購読を追加**。

### ④ グリッパー駆動経路
- `action[15]` → `rt/dex1/right/cmd`（`MotorCmds_`, `cmds[0].q`, kp=5/kd=0.05）。左 cmd は送らない。

### ⑤ arm_sdk 送信モジュール（送信の継ぎ目）
- `rt/arm_sdk` に `motor_cmd[15-28]=腕target`、`motor_cmd[29].q=weight`、**脚(0-11) は触らない**（仕様上無視）。`mode_machine` は lowstate から、CRC 付与、250Hz。
- **sport mode 解除しない**（脚はオンボード loco）。motion control mode 前提（Stage A と同じ）。

### ⑥ Blueprint ＋ 配線
- `unitree_g1_act_arm.py`：lowstate源(coordinator系) ＋ ZmqCamera ＋ ActBridge ＋ G1ArmSdkConnection ＋ gripper を合成 → `all_blueprints.py` 登録。
- ⚠️ 既存 blueprint の lowcmd publisher と arm_sdk が**二重に司令を出さない**ことを確認。

### ⑦ ACT 推論サービス（別venv）
- `act_service.py`（ZMQ REP, lerobot ACTPolicy）。lerobot `select_action` は chunk queue＋temporal ensembling のステートフル処理のため、当面 **ONNX 化せず ZMQ サービスで隔離**（dimos venv との依存衝突回避）。loopback で REQ-REP レイテンシ **<30ms** を確認。

---

## 3. 検証（段階的・安全第一）

| 段階 | 内容 | 安全 |
|---|---|---|
| **B0** | dry-run（DDS書込なし）：state[16]組立・ACT出力・カメラ形式をログ確認 | 動かない |
| **B1** | arm_sdk LIVE・**周囲クリア**・weight ランプ・**追従誤差(commanded vs measured)監視** → ドリフト無しを確認 | 自立・e-stop |
| **B2** | motion control mode 自立で**オクラ把持フル試行**（Stage A を DimOS で再現） | 自立・e-stop |

各段で `eval_g1.py` のリファレンス挙動と比較する。

---

## 4. リスク / 未確定

- teleimager `head_camera` の解像度・色順が ACT 期待(RGB 480×640)と一致するか要確認。`TeleimagerCamera` 実装時に teleimager 依存を DimOS venv に入れるか薄い購読を内製するかの判断。
- ZMQ サービスのレイテンシが 30Hz ループに乗るか。
- coordinator 系 blueprint の既存送信経路（lowcmd 等）と arm_sdk の競合。
- ACT 観測の closed-loop：腕が追従しないと obs が OOD 化（①が効くか B1 で要確認）。

---

## 5. 着手順序（推奨）

1. **送信の継ぎ目だけ単体テスト**：ACT 無しで「現在姿勢→指定ターゲットへ arm_sdk 追従」させ、**追従誤差が小さくドリフトしない**ことを確認（`act_g1_direct.py` ロジック移植）。
2. ②カメラ変換を B0 dry-run で検証。
3. ③④⑤を結線し B1。
4. ⑥⑦で blueprint 化し B2。

---

## 6. 参照

- Stage A 実装：Orboh/unitree_lerobot PR #1（`feat/okura-act-realrobot-eval`）/ Issue #2
- 手順書：unitree_lerobot `docs/OKURA_ACT_REALROBOT.md`
- FleetSeek：debug exp_01KTV1MCPZ8NFP6VPEYGY6TMY5 / skill exp_01KTV4TDQRW9WRC77QMZ1GEN1Y
- モデル：`sotata/act-okura-pick-06102026`、データセット `Orboh/okura-sub-lerobot`
- 実機制約：右手 Dex1 のみ / motion control mode / NIC `enx6c1ff771dc67` / cyclonedds 0.10.5
