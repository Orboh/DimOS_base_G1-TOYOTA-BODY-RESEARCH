# G1 オクラ収穫 — Isaac Sim 検証スイート

dimos の収穫スキル／オーケストレータ（LangGraph）を、**実機を使わずに Isaac Sim 上の仮想 G1 で検証**するためのスクリプト群と実行カタログ。

- **環境・USDアセットの用意** → [`TEAM_SETUP.md`](./TEAM_SETUP.md)（isaac-sim env / `.venv` / S3 からの USD 取得）。**先にこちらを完了**すること。
- **検証計画の全体像** → Obsidian `G1収穫設計書/12-検証計画-sim.md`（何を本物/GT/mock にして全プロセスを通すか）。
- このディレクトリは git 管理、**USD 実体は S3**（`usd_file/` は gitignore）。

---

## 0. 全体像（sim-in-the-loop）

```
 dimos（収穫制御: IK/LangGraph/DDS）        Isaac Sim（仮想 G1）
   rt/arm_sdk  / rt/dex1/right/cmd  ──DDS──▶  sim_dds_bridge.py
   cmd_vel                                     ├ 指令を仮想G1へ適用
   ZmqCamera  ◀───────ZMQ───────────────────  └ 状態(rt/lowstate)・カメラを返す
```
- **中心 = `sim_dds_bridge.py`**：DDS を受けて仮想 G1 を動かし、状態・カメラを返す橋渡し。
- 経路は **ローカル loopback**（同一PC）か **tailscale**（Jetson↔手元PC）。
  - loopback: `SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1`（**lo はマルチキャスト不可 → 必ず unicast peer 指定**）
  - tailscale: `SIM_DDS_IFACE=tailscale0 SIM_DDS_PEERS=<相手の 100.x>`
- IK は dimos 側（`.venv`：pinocchio+unitree_sdk2py 完備）で実行＝実機と同じ分担。

到達状況: **M0 配管 → M1 音声 → S3/S4 ロジック → 7-C VLM → M2 検出→IK → M3 机上ピック → M4 LangGraph 自律収穫 → F-07 籠収納（収穫サイクル完結）**。

---

## 1. クイックスタート（ローカル loopback で収穫サイクル）

### ① bridge を起動（isaac-sim env, GUI, 机+オクラ+籠）
GUI は X セッションのある端末から。私の環境では `DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority`。
```bash
cd ~/Desktop/dimos-hackathon
PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
PYTHONPATH=/home/kota-ueda/Desktop/unitree_sdk2_python \
SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 SIM_HEADLESS=0 SIM_LOAD_ROOM=1 SIM_TABLE=1 SIM_OKRA=10 \
  ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/sim_dds_bridge.py
```
`[bridge] loop start` ＋ `mapped 29/29` ＋ `basket world pos=...` が出れば準備完了。

### ② LangGraph 収穫を実行（.venv）
```bash
SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 \
  .venv/bin/python docs/sim-setup/sim_harvest_run.py
```
→ 実 `graph.py` が detect→select→IK→掴む→持ち上げ→**籠へ投入** を手前右4本で自律実行（`picks=4`）。
停止: 別端末で `touch /tmp/sim_bridge_stop`（または Ctrl-C）。

---

## 2. 検証カタログ（段階別）

各行: 何を確認するか / コマンド（bridge は §1①、別 env 指定は明記）。

| 段階 | 確認内容 | 実行 |
|---|---|---|
| **M0/S0** | 配管: 指令→仮想G1 右腕が追従 | bridge（`SIM_TABLE` 無しでも可）＋ `dimos_style_pub.py`（Jetson の DDS venv, `PUB_OSC=1`）|
| **M1** | 音声 F-12: 全アナウンス合成・再生 | `.venv/bin/python docs/sim-setup/sim_audio.py --dump`（WAV化）/ `--play`（再生）|
| **S3** | カゴ満杯=10個ロジック F-13 | `.venv/bin/python docs/sim-setup/verify_s3_basket10.py`（sim不要）|
| **S4** | 安全 F-11: FileEStop/把持中断/BladeGuard | `.venv/bin/python docs/sim-setup/verify_s4_safety.py`（sim不要）|
| **7-C** | VLM 切断可否 liveness F-02 | `SIM_VLM_HOST=http://<jetson>:11434 .venv/bin/python docs/sim-setup/sim_vlm_liveness.py`（moondream 要）|
| **M2(数値)** | 検出GT→IK が解けるか（A配置 7/10） | ① `M2_OUT=/tmp/m2.json … _dump_torso_m2.py`（isaac-sim, headless）→ ② `M2_OUT=/tmp/m2.json .venv/bin/python … verify_m2_reach_ik.py` |
| **M2(可視)** | 右腕がオクラへ到達 | bridge（§1①）＋ `SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 .venv/bin/python docs/sim-setup/sim_ik_reach_pub.py` |
| **M3** | 机上ピック: 掴む→持ち上げ | bridge ＋ `… .venv/bin/python docs/sim-setup/sim_pick_demo.py` |
| **M4/F-07** | LangGraph 自律収穫＋籠収納 | §1 のとおり（`sim_harvest_run.py`）|

> 物理（落下/把持の動力学）系の単体検証: `verify_physics_drop.py` / `verify_basket_catch.py`（物理トラック）。

---

## 3. bridge の主な環境変数（`sim_dds_bridge.py`）

| env | 既定 | 意味 |
|---|---|---|
| `SIM_DDS_IFACE` | `lo` | DDS NIC。loopback=`lo` / remote=`tailscale0` |
| `SIM_DDS_PEERS` | （空） | unicast peer。loopback=`127.0.0.1` / remote=相手 100.x。**lo では必須** |
| `SIM_LOAD_ROOM` | `0` | `1` で部屋(`SIM_ROOM_USD`=既定 chinou_center.usd)を読む |
| `SIM_TABLE` / `SIM_OKRA` | `0` / `10` | 机＋直立オクラ N 本（A配置）を載せる |
| `SIM_GRAVITY` | `0`(OFF) | 既定 OFF（弱PDの腕を指令角で保持）。`1` で重力ON（動力学） |
| `SIM_HEADLESS` | `1` | `0` で GUI |
| `SIM_PUB_CAMERA` | `0` | `1` で ego_view(color+depth+K+cam_to_torso) を ZMQ 配信 |
| `SIM_GRASP_OFFSET` | `0,0,0` | 把持位置（手リンク原点からのオフセット） |
| `SIM_GRASP_TARGET_FILE` | `/tmp/sim_grasp_target.txt` | graph が次に掴む okra index を書くファイル |

---

## 4. ファイルマップ

**sim-in-the-loop 基盤**
- `sim_dds_bridge.py` — ★中心。DDS(rt/arm_sdk, rt/dex1, rt/lowstate)＋ZMQカメラ橋渡し。机/オクラ/籠/把持処理。
- `sim_scene.py` — 机＋直立オクラ A 配置の共有ビルダー（配置の正本）。
- `view_chinou.py` — chinou+g1bag シーンの GUI ビューア（`--table --okra N`）。

**収穫の sim 駆動（LangGraph 接続）**
- `sim_harvest_skills.py` — `HarvestSkills` 実装（detect=GT/grasp=IK+dex1→sim/verify/record）。
- `sim_harvest_run.py` — 実 `graph.py` を sim で invoke するランナ。

**送信・受信ユーティリティ**
- `dimos_style_pub.py` / `dds_test_pub.py` — 試験送信（合成スイープ, dimos流/raw）。
- `sim_ik_reach_pub.py` — IK 1本リーチ送信。`sim_pick_demo.py` — reach→閉じ→持ち上げ。
- `dds_cam_sub.py` — sim カメラ(ego_view)受信デバッグ。

**検証スクリプト**
- `verify_s3_basket10.py`(カゴ10) / `verify_s4_safety.py`(安全) / `sim_vlm_liveness.py`(VLM) /
  `_dump_torso_m2.py`＋`verify_m2_reach_ik.py`(検出→IK) / `verify_physics_drop.py` / `verify_basket_catch.py`。

**アセット recipe（USD は S3、これらが「作り方」の正本）**
- `convert_chinou.py`(FARO→USD) / `build_chinou_phys.py`(コライダー) / `make_okra_usd.py`(オクラ) /
  `setup_physics_materials.py`(摩擦/弾性) / `add_basket_collider.py`(籠 collider)。

---

## 5. ハマりどころ

- **loopback は unicast 必須**：`lo` はマルチキャスト不可 → `SIM_DDS_PEERS=127.0.0.1` を両端に。
- **重力 OFF が既定**：g1bag の関節 PD が弱く重力下で腕が垂れる。運動学検証は OFF のまま。実把持/実落下は `SIM_GRAVITY=1`＋PD調整が要る。
- **`OKRA_CAM_TO_TORSO`（ハンドアイ校正）は実機のみ**：sim は bridge が GT の cam_to_torso を出すため不要。
- **IK は `.venv` で実行**：isaac-sim env には dimos の依存（structlog/langgraph）が無い。M2 数値検証は isaac で座標 dump → `.venv` で IK の2段。
- **isaac-sim env の警告**：`isaacsim.asset.browser: No module named 'requests'` は無害（カメラ配信や asset browser を使うなら `pip install requests`）。
- **初回 DDS 取りこぼし**：discovery 確立前の最初の指令が落ちることがある（`sim_harvest_skills.py` は起動時 warm-up 済）。
- **GUI が出ない**：自分の X セッション端末から起動（or `DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority`）。

---

## 6. リンク
- 検証計画（正本）: Obsidian `G1収穫設計書/12-検証計画-sim.md`
- 環境/アセット: [`TEAM_SETUP.md`](./TEAM_SETUP.md) ／ Isaac Sim 構築詳細: `../sim-setup-notes.md`
- 遠隔(Jetson↔PC)再接続: [`REMOTE_ACCESS.md`](./REMOTE_ACCESS.md)
- 設計正本: `G1収穫設計書/00-全体設計書.md`, `SS-01`〜`SS-08`
