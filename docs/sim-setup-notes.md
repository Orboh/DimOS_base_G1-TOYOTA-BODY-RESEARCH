# Isaac Sim オクラ収穫 sim 環境 セットアップ手順メモ（再現用）

> 対応タスク: `docs/sim-setup-task.md`（第1歩＝Isaac Sim を入れて room+G1 を立て腕を動かす）
> 実施: 2026-06-25 / 担当マシン: Kota ラップトップ
> ステータス: **完了（全 DoD 達成）**
> 成果物: 本ファイル ＋ `docs/sim-setup/`（スクリプト・スクリーンショット）

---

## 0. 結論サマリ（先に読む）

- 採用構成: **Isaac Sim 4.5.0 + Python 3.10（conda env `isaac-sim`）／ pip インストール**。
- 8GB VRAM 機（RTX 2000 Ada Laptop）で **DoD 1〜5 すべて達成**。
  - 起動 ✅／room.usd ロード ✅（外部参照の一部欠落あり=§6）／G1 直立 ✅（fix_base）／右腕1関節を指令駆動 ✅／カメラRGB取得 ✅。
- **VRAM 実測: 起動 ~1.7GB、room+G1+カメラ描画でピーク ~3.1GB / 8GB**。→ **この第1歩の範囲では 8GB で十分実用に耐える**（§7 に判断詳細）。
- 前提として GPU を 3.5GB 占有していた音声入力ツール `vocalinux` を削除して VRAM を確保した（§2）。

### DoD 結果一覧

| DoD | 結果 | 証拠 |
|---|---|---|
| 1. Isaac Sim 起動 | ✅ | 初回シェーダコンパイル 261s で `app ready`。`docs/sim-setup/minimal_start.py` |
| 2. room.usd ロード | ✅ | **`open_stage` で部屋全体（3.0×6.4m）をロード**。`add_reference` は defaultPrim(`/World`)しか読まず壁/奥床が落ちるため不可（§6-8）。`Jalapeno_Plant`等の外部参照は欠落（§6-6）。`01_standing.png` |
| 3. G1 が物理で立つ | ✅ | fix_base。原点=骨盤のため reset 後 `set_world_pose(z=+0.792)` で足を床へ接地。base world pos=[0,0,0.792]、発散なし。`01_standing.png` |
| 4. 右腕1関節を指令駆動 | ✅ | `right_shoulder_pitch_joint` 0.019→0.619 rad 指令→0.600 追従（0.58 rad 移動）。`02_joint_moved.png` |
| 5. カメラRGB取得 | ✅ | viewport キャプチャ＋Camera センサ `get_rgba()` 両方で非黒RGB。`*_sensor.png` |
| 6. 再現メモ | ✅ | 本ファイル |

---

## 1. 対象マシンの環境（実測 2026-06-25）

| 項目 | 値 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA RTX 2000 Ada Generation Laptop, **VRAM 8188 MiB (8GB)** |
| Driver / CUDA | 580.159.04 / CUDA 13.0 capable |
| GLIBC | 2.35（pip Isaac Sim 要件 2.34+ を満たす） |
| RAM | 31GB（実行中 RSS は最大 ~15GB） |
| Disk(/) 空き | 約 260GB（env は ~13GB 消費） |
| conda | miniconda3 / conda 26.1.1 |

> Isaac Sim 推奨は VRAM 16GB+、最小は RTX 3070 級/8GB。本機は最小ラインだが、第1歩の軽量シーンでは余裕あり（ピーク 3.1GB）。

---

## 2. 事前作業: VRAM 確保（vocalinux 削除）

着手時 GPU は使用 4828 MiB / 空き 3.3GB で、うち **3.5GB を音声入力ツール `vocalinux`** が占有していた。Kota の指示で削除:

```bash
bash ~/.local/share/vocalinux-install/uninstall.sh -y      # 公式アンインストーラ
rm -f ~/.local/bin/vocalinux ~/.local/bin/vocalinux-gui    # 取りこぼし
rm -rf ~/.local/share/vocalinux-ibus
```

結果: GPU 使用 4828→**1264 MiB** / 空き 3.3GB→**6.5GB**。autostart も解除済み（再起動で復活しない）。

---

## 3. Isaac Sim インストール（採用手順・再現コマンド）

### バージョン選定理由
- Isaac Sim **4.x → Python 3.10** / 5.x → Python 3.11。
- `unitree_sim_isaaclab`（チームで DDS 駆動 G1 sim が稼働した足場）は **4.5.0 / 5.x 両対応**。
- 5.0 は install 失敗 Issue 複数（isaac-sim/IsaacSim #84, #115）。**4.5 が最も枯れ**、カスタムG1 USD の生成時期（2025-06, Isaac Lab UrdfConverter）とも整合。
- → **Isaac Sim 4.5.0 + Python 3.10** を採用。将来 Isaac Lab 2.1 + unitree_sim_isaaclab を上に載せられる。

### 手順（コピペ可）
```bash
# 1. conda env (py3.10) 専用。普段の .venv/bin/dimos とは別環境
conda create -n isaac-sim python=3.10 -y
conda activate isaac-sim
pip install --upgrade pip

# 2. Isaac Sim 4.5.0 本体（~10-15GB DL）
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com

# 3. ★重要★ 足りない依存を明示導入（§6 の落とし穴参照）
pip install scipy psutil pyyaml pillow boto3 gymnasium
pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu118

# 4. EULA 同意（初回必須）
export OMNI_KIT_ACCEPT_EULA=YES

# 5. 起動確認（GUI で見たい場合）
isaacsim isaacsim.exp.full
```

### 実行時のお作法
```bash
# user-site 汚染を遮断（§6）＋ EULA。スクリプトは env の python で直接叩く
PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
  ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/sim_smoke_test.py
# GUI で見る場合は末尾に --gui
```

---

## 4. アセット

`/home/kota-ueda/Desktop/dimos-hackathon/usd_file/`（.gitignore 済＝ローカルのみ）

| ファイル | 役割 | 形式 | 備考 |
|---|---|---|---|
| `room.usd` (37K) | 研究室環境 | USDC crate 0.9.0 | **外部参照欠落あり**（§6） |
| `g1-…/g1_29dof_with_dex1_base_fix1.usd` (26M) | カスタムG1(29DoF+右手Dex1, **fix_base:true**) | USDC crate 0.8.0 | 自己完結。Articulation root=`/World/G1/root_joint`、**33 DoF** |
| `…/config.yaml` | UrdfConverter 設定 | — | stiffness100/damping1, drive=force/position, fix_base:true, collider=convex_hull |

### G1 の DOF（33個、実測 `dof_names`）
脚×12 / 腰×3（waist yaw/roll/pitch）/ 腕×14（shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw を左右）/ 手×4（`left/right_hand_Joint1_1`, `Joint2_1`＝Dex1グリッパ）。
右腕＝`right_shoulder_pitch/roll/yaw_joint`, `right_elbow_joint`, `right_wrist_roll/pitch/yaw_joint`。

---

## 5. 検証スクリプト（`docs/sim-setup/`）

- `minimal_start.py` … SimulationApp の最小起動テスト（DoD-1 切り分け用）。
- `sim_smoke_test.py` … 本番。room+G1 ロード→立位確認→右腕1関節を `apply_action(ArticulationAction)` で駆動→viewport/Camera センサで RGB 保存。`--gui` で GUI 表示。

### 要点（API メモ, Isaac Sim 4.5）
- `from isaacsim import SimulationApp` を**最初に**生成してから他の isaac/omni を import。
- World: `from isaacsim.core.api import World`（旧 `omni.isaac.core` は deprecated だが残存）。
- USD 配置: `from isaacsim.core.utils.stage import add_reference_to_stage`。
- Articulation: `from isaacsim.core.prims import SingleArticulation`。**関節指令は `set_joint_position_targets` ではなく `apply_action(ArticulationAction(joint_positions=...))`**（旧APIと混同しやすい＝§6）。
- カメラ画角: `from isaacsim.core.utils.viewports import set_camera_view`＋`omni.kit.viewport.utility.capture_viewport_to_file`（capture 後 `sim_app.update()` を数フレーム回す）。
- Camera センサ: `from isaacsim.sensors.camera import Camera` → `initialize()` → 数十フレーム step 後 `get_rgba()`。姿勢は persp カメラの実 transform を流用すると規約ズレを避けられる。

---

## 6. つまずきメモ（重要）

1. **user-site 汚染で依存が無言スキップされる（最大の罠）**
   `~/.local/lib/python3.10/site-packages` に gr00t/voyager 等の editable インストールがあり、py3.10 の新 env がそれを user-site 経由で参照。pip が `torch/scipy/psutil/pyyaml/pillow/boto3` を「既にある」と誤認して env に入れず、起動時に `No module named 'scipy'` 等で失敗。
   - 対策: §3-3 の依存を**明示インストール**＋実行時 `PYTHONNOUSERSITE=1` で user-site を遮断。
2. **`torch` が未導入**（上記の派生）。Isaac Sim 4.5 推奨は `torch==2.5.1+cu118`。driver 580 で cu118 動作 OK（`torch.cuda.is_available()=True`）。
3. **関節指令 API**: `SingleArticulation.set_joint_position_targets` は**存在しない**→ `AttributeError`→segfault(exit139)。正しくは `apply_action(ArticulationAction(joint_positions=...))`。
4. **EULA**: 未設定だと import 時に `input()` で停止。`OMNI_KIT_ACCEPT_EULA=YES`。
5. **初回起動が遅い**: シェーダコンパイルで **261秒**。2回目以降は warm cache で短縮（数十秒〜）。
6. **room.usd の外部参照欠落**: `room.usd` は別マシンで作成され、`usd_file/Jalapeno_Plant/scene.usdc`（payload）と材質テクスチャ `./WhatsApp Image 2026-06-23 ….jpeg` を相対参照するが、これらが usd_file に同梱されていない。→ 植物オブジェクトと一部テクスチャが欠落（箱がデフォルトのピンク材質に）。床・壁・箱のジオメトリは描画されるので第1歩の目的は満たすが、**room を完全表示するには元マシンから欠落アセットを受領して `usd_file/` に配置する必要**。
7. 非致命の起動エラー（`isaacsim.asset.browser` / `replicator_yaml` の python extension load 失敗、各種 deprecation 警告）はヘッドレス実行に影響なし。
8. **★room が「縮尺おかしい」に見えた真因★**: room.usd は**壁(`/wall`,`/wall_01`,`/wall_02`)と奥の嵩上げ床(`/floor`,`/floor_01`)をルート直下に持ち、defaultPrim `/World` の外**に置いている。`add_reference_to_stage` は **defaultPrim サブツリーしか読まない**ため、これらが丸ごと欠落し、`/World` 内の小さな床片（3×3m, 高さ0.34m）だけが表示されていた。
   - 部屋全体の実寸は **3.0 × 6.4 × 0.74m**（現実的な部屋サイズ。metersPerUnit=1.0・全 prim の scale=1＝スケール値の誤りではない）。`Sketchfab_model` 由来＋Omniverse Kit で手組み（作成者: `saumy`、テクスチャは WhatsApp 写真）。
   - 対策: **`open_stage(room.usd)` で部屋全体を開いてから** G1 を `/G1`（ルート）に追加する。G1 を `/World/G1` に入れると room の `/World` と混ざるので避ける。
9. **G1 の床下沈み込み**: G1 USD は原点が骨盤（bbox z=[-0.79, +0.53]）。fix_base が骨盤を z=0 に固定するため脚が床下に沈む。prim の translate を reset 前に設定しても固定ジョイントに上書きされる。**reset 後に `robot.set_world_pose(position=[x,y,0.792])`** で base を持ち上げると接地する（足が z=0）。

---

## 7. 報告事項（task §6 の成果物・判断）

### 8GB で実用に耐えるか / 別機要否 / MuJoCo 切替要否
- **この第1歩（room+G1+単一カメラ、低解像度、headless）では 8GB で十分**。VRAM ピーク ~3.1GB、空き 6.5GB に対し余裕大。**別機・MuJoCo 切替は不要**。
- ただし今後 **複数カメラ高解像度・オクラ植物多数・Replicator 合成データ・GUI 常用** に拡張すると 8GB は厳しくなる可能性。次フェーズで描画負荷が増えたら 16GB+ 機（D-007 Omniverse 担当機）への移行を再検討。
- MuJoCo は USD 固有アセット（room/dex1 G1）を扱えないため、本タスク（USD 検証）の代替にはならない。dimos 本線の軽量フロー検証用としては引き続き有効。

### fix_base のままでよいか
- 第1歩（立位・腕動作の確認）には fix_base:true が**安全で適切**（base が原点固定＝発散しない。実測 base pos≈[0,0,0]）。
- **歩行・足配置（SS-07）の sim 検証には base-free 版が必須**。`configuration/` の `waist_fix` 派生も fix_base 系。歩行をやる段では URDF から `fix_base:false` で再変換、または floating-base 版 USD を用意する。

### unitree_sim_isaaclab の再利用可否
- Unitree 公式 `unitree_sim_isaaclab` は Isaac Sim 4.5/5.x 対応で、**DDS 駆動の G1+dex1 sim 実績がチーム内にある**（横手環境）。今回 4.5 を入れたので**バージョン整合は取れる**見込み。
- ただし当該リポは**このマシンには無い**（横手 `/home/techshare/...`）。次フェーズで clone + Isaac Lab 2.1 導入が必要。DDS 駆動足場としての再利用は**有望**。
- 補足: dimos 側の Isaac 統合（`dimos/simulation/isaac/`）は薄いラッパのみ。dimos 本線 sim は MuJoCo（`g1-sim-connection`）。sim-in-the-loop は「unitree_sim_isaaclab(Isaac) ↔ DDS ↔ dimos」構成が現実的。

---

## 8b. room の「正解」寸法と正規セットアップ（XR Teleop レポートより）

`~/Downloads/G1_Teleop_Project_Report (1).pdf`（G1-29 XR Teleoperation, 2026-06-23）§6 が決定打。

- **room.usd は意図的に「~3m × 3m interior、壁 ~0.74m tall」**（Up=Z, metersPerUnit=1.0, 550 prims, prim path `/World/Room`）。
  → 壁0.73mは正常。「壁が低い＝バグ」「実寸より縮小」という当初の見立ては**誤り**。小さな囲い部屋が正。
- **正規ロボット spawn: position (-0.15, -0.47895, 0.76m)、rot=(0.7071,0,0,0.7071)=Z軸90°回転**。
- **正規の読み込み = unitree_sim_isaaclab のタスク `Isaac-G1-Custom-Room-Joint`**（`sim_main.py --task ... --device cpu --enable_cameras --enable_dex1_dds --robot_type g129`）。**Isaac Sim 5.1 + Isaac Lab** 構成（本セットアップは 4.5 standalone）。レポート側ロボットは g1bag.usd(31関節)。
- 「別PCで正しかった」= techshare 環境で上記タスク経由でロード＆documented spawn していたから。当機の自作 standalone は原点・無回転で置いていたためズレていた。
- 当機 `~/Desktop/xr_teleoperate` は **teleop クライアント側のみ**で `unitree_sim_isaaclab` は無い（techshare にのみ存在）。
- 当機の room.usd には `/World` 外のルート直下に**余分な嵩上げ床(y=2.5〜4.92)**が紛れており、全体 bbox が 3×6.4m に伸びる（囲い本体は ~3×3m で PDF と一致）。canonical 版は `/World/Room` 単一。
- 対応: `sim_smoke_test.py` の spawn を PDF 記載姿勢に合わせ済み（`SPAWN_POS=(-0.15,-0.47895,lift)`, `SPAWN_ORI=(0.7071,0,0,0.7071)`）。

## 8. 次フェーズへの引き継ぎ

1. 元マシンから room の欠落アセット（`Jalapeno_Plant/`, テクスチャ jpeg）を受領して `usd_file/` に配置 → room 完全表示。
2. 歩行検証用に G1 の base-free 版 USD を用意（fix_base:false 再変換）。
3. `unitree_sim_isaaclab` を clone ＋ Isaac Lab 2.1 導入 → DDS 駆動足場を構築。
4. sim のカメラ/関節状態を DDS/LCM で配信 → Jetson の dimos が購読、`arm_target`/`cmd_vel`/`gripper_target` を sim が駆動（task §8）。

---

## 9. sim-in-the-loop: dimos(Jetson) ⇄ Isaac Sim DDS ブリッジ（2026-06-25 検証）

狙い: dimos の `rt/arm_sdk`(unitree_hg LowCmd) を Isaac の仮想G1へ、Isaac の関節状態を `rt/lowstate` へ。
「接続先を実機→sim に差し替えるだけ」。

### 成果（tailscale 越しで末端成立）
- **Jetson → tailscale unicast → laptop bridge → Isaac 仮想G1の腕が指令角へ追従**（測定 r_shoulder_pitch=0.582/指令0.6, r_elbow=-0.786/指令-0.8）。loopback も成立。
- 成果物: `docs/sim-setup/sim_dds_bridge.py`（ブリッジ）, `dds_test_pub.py`（dimos代用の試験送信機）。

### 構成・実装
- bridge は **raw cyclonedds** で transport を自前制御（unitree_sdk2py の ChannelFactoryInitialize は config固定・CYCLONEDDS_URI無視のため使わない）。IDL型は unitree_sdk2py から借用（`sys.path` に `/home/kota-ueda/Desktop/unitree_sdk2_python`）。
- laptop isaac-sim env に `cyclonedds==0.10.5` 導入済み。
- 正準G1_29関節順→Isaac dof名でマップ（29/29 一致）。腕=idx15-28、weight=motor_cmd[29]。
- bridge は `rt/lowstate` に **mode_machine=1** + 29関節 q/dq を発行（dimos arm controller 起動に必須）。

### ハマりどころ（重要）
1. **cyclonedds-python の Domain は参照保持必須**: `Domain(id, cfg)` を変数に入れないと即GC→domainごと破棄→socketが生成直後にrelease→discovery不成立。loopbackは速くて偶発成功、tailscaleはrelay遅延でGCが先行し0件。→ `dom = Domain(id, cfg)` で保持して解決。
2. **orboh AP は client isolation**: 同一WiFi(192.168.0.x)でも端末間DDS(マルチキャスト/ユニキャスト)が届かない → same-LAN不可、**tailscaleが唯一の実機間経路**。生UDPは tailscale を通る（確認済）。
3. **tailscale DDS は unicast 設定**: `<Interfaces><NetworkInterface name="tailscale0" multicast="false"/>` + `<AllowMulticast>false` + `<Discovery><Peers><Peer address="相手100.x"/></Peers>`。ParticipantIndex auto/MaxAuto32。

### 残（実 dimos 駆動まで）
- Jetson dimos venv(py3.12) に unitree_sdk2py + cyclonedds 導入（現状無し）。
- dimos 側 tailscale unicast 化: unitree_sdk2py の `ChannelConfigHasInterface` に `<Peers>`+`AllowMulticast=false` 注入 + Domain参照保持。
- 最小の腕パス（full okra は ZED/ACT/moondream 必須で sim 不可 → arm-only サブ構成）。
- dimos が bridge の rt/lowstate(mode_machine) を受けて起動するか検証。

### 実行コマンド（検証時）
```bash
# laptop bridge（remote=tailscale）
PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=/home/kota-ueda/Desktop/unitree_sdk2_python \
  SIM_DDS_IFACE=tailscale0 SIM_DDS_PEERS=<jetson 100.x> SIM_HEADLESS=1 \
  ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/sim_dds_bridge.py
# Jetson 試験送信（dimos代用）
SIM_DDS_IFACE=tailscale0 SIM_DDS_PEERS=<laptop 100.x> \
  /home/tbr/workspace_ssd/unitree_mujoco/.venv/bin/python /tmp/dds_test_pub.py
```

### 実 dimos を sim に向ける手順（2026-06-25 検証で確立）

**到達点**: dimos の実 DDS 機構（`unitree_sdk2py` の `ChannelFactoryInitialize`+`ChannelPublisher("rt/arm_sdk", LowCmd_)`）が **tailscale 越しに sim の仮想G1を駆動**することを実証（`docs/sim-setup/dimos_style_pub.py`、Jetson の py3.10 稼働env `unitree_mujoco/.venv` から）。これは G1ArmSdkConnection が出す経路と同一。

**dimos を tailscale unicast 化するパッチ（要点）**: `unitree_sdk2py` は cyclonedds の config を固定し CYCLONEDDS_URI を無視するので、`ensure_channel_factory` 前に config テンプレへ Peers を注入する:
```python
# dimos/robot/unitree/g1/act/dds_init.py の ensure_channel_factory 冒頭などで（env で opt-in）
import os, unitree_sdk2py.core.channel as ch
peers = os.getenv("DIMOS_DDS_PEERS", "")  # 例: laptop の 100.x
if peers:
    pj = "".join(f'<Peer address="{p}"/>' for p in peers.split(","))
    ch.ChannelConfigHasInterface = (
      '<?xml version="1.0" encoding="UTF-8" ?><CycloneDDS><Domain Id="any">'
      '<General><Interfaces><NetworkInterface name="$__IF_NAME__$" priority="default" multicast="false"/></Interfaces>'
      '<AllowMulticast>false</AllowMulticast><EnableMulticastLoopback>false</EnableMulticastLoopback></General>'
      f'<Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>32</MaxAutoParticipantIndex><Peers>{pj}</Peers></Discovery>'
      '</Domain></CycloneDDS>')
# そして ensure_channel_factory(network_interface="tailscale0") で起動
```
（unitree_sdk2py 側は Domain を class 変数で保持するため、bridge で必要だった Domain-GC 対策は不要。）

**dimos venv の前提**: 当機 Jetson の dimos venv(py3.12) は cyclonedds が空 namespace・unitree_sdk2py 無し＝実 dimos の DDS は現状動かない。導入には py3.12/aarch64 用の cyclonedds と unitree_sdk2py が必要（unitree_mujoco venv は py3.10 で別物）。**ただし full okra blueprint は ZED/ACT/moondream 必須で sim 起動不可**（設計上 ACT/切断は実機のみ＝設計書11 §5）。→ sim 検証は「arm-only の最小 dimos パス」を別途用意するのが筋。

**sim 検証の現実的ゴール**: IK 粗アプローチ・歩行/左掃引・LangGraphフロー等「knチ覚に依存しない上位制御」を arm_sdk/cmd_vel 経由で sim に出して検証。ACT 把持・切断・実画像依存の検出は sim 対象外（実機のみ）。

---

## 10. カメラ配信ブリッジ（2026-06-25 tailscale越しで実証）

**到達点**: Isaac sim のカメラ画像を **dimos の `ZmqCamera` 互換形式**で配信し、Jetson が tailscale(DERP relay)越しに 187フレーム受信・デコード成功（room 内のG1が写る）。

### 形式（dimos `ZmqCamera` = GEAR-SONIC ego_view 互換）
- ZMQ **PUB が `tcp://0.0.0.0:5555` に bind**、`msgpack.packb({"images":{"ego_view": <base64-JPEG>}, "timestamps":{"ego_view": ts}})` を送る。
- dimos の `ZmqCamera`（`dimos/robot/unitree/g1/camera/zmq_image_source.py`）が SUB で connect→ `color_image`(RGB) を emit → 検出モジュールへ。

### ブリッジ起動（カメラ有効）
```bash
# 既存の arm/state DDS に加えてカメラ ZMQ も配信
PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=/home/kota-ueda/Desktop/unitree_sdk2_python \
  SIM_DDS_IFACE=tailscale0 SIM_DDS_PEERS=<jetson 100.x> SIM_HEADLESS=1 SIM_LOAD_ROOM=1 \
  SIM_PUB_CAMERA=1 SIM_CAM_EYE=3,3,2 SIM_CAM_TARGET=0,0,0.6 \
  ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/sim_dds_bridge.py
# 主な env: SIM_CAM_PORT(5555) SIM_CAM_W/H(640/360) SIM_CAM_FPS(15) SIM_CAM_TOPIC(ego_view) SIM_CAM_EYE/TARGET(look-at)
```

### dimos 側で受ける（実 blueprint）
- dimos の `ZmqCamera` を **host=手元PCの tailscale IP(100.100.126.61), port=5555, topic=ego_view** で構成 → sim 画像が `color_image` に乗る。検出(YOLO-seg)はそこを購読。
- 検証用スタンドアロン受信機: `docs/sim-setup/dds_cam_sub.py`（`SIM_CAM_HOST=<laptop 100.x>` で接続→PNG保存）。

### 残課題
- **構図**: 現状は俯瞰の look-at（SIM_CAM_EYE/TARGET）。実際の頭部/胸部カメラとして G1 のリンクに追従させる、または planter を正面に捉える配置が必要。
- **オクラ模型欠落**: room.usd の `Jalapeno_Plant/scene.usdc`（外部payload）が未同梱でオクラが写らない。元マシンから `usd_file/Jalapeno_Plant/` を受領要。
- **検出は OOD**: YOLO-seg は実写学習のため sim 画像では精度低（設計どおり）。sim では構図/配置の検証用と割り切る。

### #1 完了: torso_link 追従カメラ + intrinsics + depth（2026-06-25）
- **リンク追従**: カメラ prim を `/G1/torso_link` の子として生成しローカル変換を設定 → G1 が動くと視点連動。`SIM_CAM_MODE=torso`(既定)。torso 未検出時は固定 look-at にフォールバック(`SIM_CAM_MODE=fixed`)。
- **取付(extrinsic)**: `SIM_CAM_LOCAL_POS`(既定 0.08,0,0.20)＋`SIM_CAM_LOCAL_FWD`(既定 1,0,-0.35=前方やや下)。sim では取付が既知＝**校正不要**。ブリッジが `cam_to_torso`(torso<-optical, x,y,z,qx,qy,qz,qw)を算出し ZMQ で配信 → dimos の `OKRA_CAM_TO_TORSO` にそのまま渡せる（実機は要校正だが sim は ground-truth）。
- **intrinsics**: `SIM_CAM_HFOV`(既定90°)から焦点距離設定。K=[fx,0,cx,0,fy,cy,0,0,1] を配信（cx=W/2,cy=H/2,fx=(W/2)/tan(hfov/2)）。
- **depth**: `cam.add_distance_to_image_plane_to_frame()`＋`get_depth()` → 16bit PNG(mm) で `msg["depth"][topic]`＋`depth_scale`(0.001)。検証で 1.00/1.29/1.56m・3D構造明瞭。
- 配信メッセージ: `{images:{ego_view:b64jpg}, depth:{ego_view:b64png16}, intrinsics:{ego_view:K}, cam_to_torso, depth_scale, timestamps}`。dimos `ZmqCamera` は images のみ読む（後方互換）。full okra 検出には depth/intrinsics/cam_to_torso を読む取り込み側（SimZedCamera 相当）が別途必要。
- 検証受信機 `dds_cam_sub.py` を color+depth+intrinsics 対応に拡張。
- **残**: color が暗い（room 照明弱＋オクラ模型欠落=#2）。検出を sim でやるなら室内ライト追加＋オクラ模型同梱＋（OOD前提で）配線確認 or ground-truth 注入。

### #1 仕上げ修正（2026-06-25, GUI実行で発覚した不具合）
- **InvalidSample クラッシュ**: pub 停止時 cyclonedds が `InvalidSample`(motor_cmd 無し)を配信→`AttributeError`→segfault。→ `reader.take()` を `[s for s in ... if hasattr(s,"motor_cmd")]` でフィルタ。
- **カメラ xform 精度クラッシュ**: `Camera.initialize()` の `xformOp:orient` は quatd。`AddOrientOp()`(既定float)で精度不一致 Tf例外→segfault・`cannot find xform op chest_cam` 警告。→ `AddTranslateOp/AddOrientOp(precision=PrecisionDouble)`+`Gf.Quatd` に統一（警告0・GUIで chest_cam が正しく表示）。
- **照明追加**: room が暗くカメラ像が真っ黒だったため `SIM_ADD_LIGHT=1`(既定) で DomeLight+DistantLight を追加 → カメラ平均輝度 ~5→140。`SIM_LIGHT_INTENSITY` で調整、`SIM_ADD_LIGHT=0` で無効。
- **GUIで torso カメラ視点を見る**: ビューポート左上カメラ切替 → Cameras → `/G1/torso_link/chest_cam`。G1 が動くと視点も追従。
