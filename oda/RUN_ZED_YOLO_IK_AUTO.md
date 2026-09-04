# ZED × YOLO × IK 自動オクラ把持 — 農場オフライン実行手順

胸ZEDの点群からYOLOがオクラを検出し、人間のクリックを代替して右腕がIKでリーチ→自動全閉で把持する
「全自動収穫」の通し手順。畑などネット無し環境向け。

- ブループリント本体: `unitree-g1-okra-ik-only-grasp-zed`
- 自動化の肝: **YOLOブリッジが `/clicked_point` を発行して人間のクリックを代替**する。
  ZEDアプリ側（IKリーチ・固定向き・自動全閉）は無改造。
- 詳細・トラブル対処: [RUN_ZED_IK.md](RUN_ZED_IK.md) / [RUN_CHEATSHEET.md](RUN_CHEATSHEET.md)

> ⚠️ **実機通し検証は未達**。オフライン計算は検証済みだが、実機ではYOLOブリッジが未通電・
> ZEDの自動クローズ把持も未達・ZED運用中のPCフリーズが4回発生。**必ず下の「段階を踏む」に従うこと。**

---

## 全体構成（3ターミナル）

| 端末 | 役割 | 起動するもの |
|---|---|---|
| **A** | ZED点群 + IKリーチ + グリッパ本体 | `dimos run unitree-g1-okra-ik-only-grasp-zed` |
| **B** | ビューア（監視用・任意だが推奨） | `dimos-viewer` |
| **C** | YOLO検出 → クリック発行 | `python oda/yolo_click_bridge.py` |

前提ハード:
- **ラップトップ**（このPC）: ZEDをUSB3ポート**4-4**へ直結（4-1は不良）。ロボットLANに有線（NIC `enp46s0` / IP .222）
- **G1**: 電源ON・LAN接続・周囲クリア・**リモコンe-stop（L2+B）を手元に**
- Jetson NXは不要（ZEDはPC直結）

---

## 持ち出し前チェック（屋内・ネットありで必ず）

畑でやると詰むのでここで済ませる:

1. **ZEDキャッシュ温め**: 一度 `dimos run unitree-g1-zed-ik-view` を起動し点群が出るのを確認。
   工場キャリブ／NEURAL深度モデルはシリアル毎に初回1回だけDLされる → **ZED個体を替えたら必須**。
2. **YOLOモデル確認**: `oda/ZED_M_Depth_check/finetune_V5/model/okra11n-seg.pt` の存在確認。
   ultralytics初回推論のキャッシュも屋内で温めておく。
3. **Wi-Fiを切った状態で一度通しで動かす** = 唯一のオフライン動作証明（オンラインだと経路問題が隠れる）。

実行時にインターネットは一切使わない（DimOS/IK/DDS/LCMは全ローカル、キャッシュ済みなら深度モデルもDL不要）。

---

## 手順

### 0. 毎セッション事前（自分のターミナルで直接。sudoはtty必須）

```bash
sudo ip link set lo multicast on
sudo ip route replace 224.0.0.0/4 dev enp46s0        # ← オフラインでは特に必須（無いとLCM全滅）
sudo sysctl -w net.core.rmem_max=67108864 net.core.rmem_default=67108864
```
確認: `ip link show lo` の1行目に `MULTICAST` があればOK。
※この3行はPC再起動・G1電源OFFのたびに消える。

グリッパ生存確認（爪が動けばOK・開き状態でスタンバイ）:
```bash
cd ~/Toyota-auto-body-PoC/DimOS_oda
CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
DEX1_NIC=enp46s0 DEX1_OPEN_Q=3.7 .venv/bin/python oda/gripper_open.py
```
動かない（位置固定・トルク0）→ モーター無効。G1電源OFF→ハンドコネクタ挿し直し→刃閉じて電源ON→再確認。

### A. ZED+IKアプリ（把持フル構成）

```bash
cd ~/Toyota-auto-body-PoC/DimOS_oda
CYCLONEDDS_HOME=~/cyclonedds-noshm \
LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
DIMOS_SKIP_COORDINATOR_RPC=1 PYTEST_VERSION=1 \
ROBOT_INTERFACE=enp46s0 \
OKRA_NOACT_KP_ARM=160 OKRA_NOACT_KD_ARM=6.0 \
IK_REACH_LIVE=1 OKRA_NOACT_GRIP_LIVE=1 \
OKRA_GRIP_KP=20 OKRA_NOACT_CLOSE_Q=1.7 \
OKRA_OPEN_Q=3.7 \
OKRA_TIP_OFFSET_XYZ="0.25,-0.003,0" \
OKRA_APPROACH_ABOVE_M=0.08 \
.venv/bin/dimos run unitree-g1-okra-ik-only-grasp-zed
```
- `OKRA_NOACT_CLOSE_Q` はゼロ点次第。刃を手で全閉して q を読み、その値−0.1（直近実測 1.7）。
- `OKRA_APPROACH_ABOVE_M=0.08` = 上からアプローチ（コの字軌道）。株が高ければ 0.12。
  消すと従来の直行リーチ（そのとき tip offset の y も `-0.003` に戻す）。
- PC不調時は `ZED_DEPTH_MODE=PERFORMANCE` を足す（点群粗くなるがGPU負荷1/10・フリーズ対策）。

### B. ビューア（別端末・**アプリ再起動のたびに開き直す**）

```bash
~/Toyota-auto-body-PoC/DimOS_oda/.venv/bin/dimos-viewer \
  --connect rerun+http://127.0.0.1:9877/proxy --ws-url ws://127.0.0.1:3030/ws
```

### C. YOLOブリッジ（別端末・**ZEDモード**）

**まずDRY-RUN**（`YOLO_BRIDGE_LIVE` を付けない = 検出と3D点をログに出すだけ）:
```bash
cd ~/Toyota-auto-body-PoC/DimOS_oda
CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
YOLO_BRIDGE_BODY_FRAME=1 \
OKRA_YOLO_CONF=0.25 YOLO_BRIDGE_PX_RADIUS=20 \
.venv/bin/python oda/yolo_click_bridge.py
```
- `YOLO_BRIDGE_BODY_FRAME=1` は **ZED必須**（光学→ボディ座標変換 x=z_o, y=-x_o, z=-y_o）。
- `OKRA_YOLO_CONF=0.25` / `YOLO_BRIDGE_PX_RADIUS=20` はZED画角での調整値。
- **Enter を押すたびに1回だけ**検出発火（連続自動発火はしない人間ゲート）。
  ログに `okra conf=... -> body xyz=[...]` が出る。DRY-RUNなので発行はしない。

座標が的と合うのを確認できたら `YOLO_BRIDGE_LIVE=1` を足して再起動 → Enter で実発火:
```bash
cd ~/Toyota-auto-body-PoC/DimOS_oda
CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
YOLO_BRIDGE_BODY_FRAME=1 YOLO_BRIDGE_LIVE=1 \
OKRA_YOLO_CONF=0.25 YOLO_BRIDGE_PX_RADIUS=20 \
.venv/bin/python oda/yolo_click_bridge.py
```
Enter → 腕がリーチ → 到達後1〜2秒でグリッパ自動全閉。次のEnterまで発火しない。

YOLOブリッジの主な環境変数:

| 変数 | 意味 | 本手順の値 |
|---|---|---|
| `YOLO_BRIDGE_BODY_FRAME` | ZED用の光学→ボディ座標変換 | `1`（ZED必須） |
| `YOLO_BRIDGE_LIVE` | `/clicked_point` を実発行 | DRY-RUN段は付けない / 本番 `1` |
| `OKRA_YOLO_CONF` | YOLO信頼度しきい値 | `0.25`（既定0.4） |
| `YOLO_BRIDGE_PX_RADIUS` | 重心近傍とみなす画素半径 | `20`（既定8） |
| `YOLO_BRIDGE_MAX_M` | これより遠い3D点は背景誤検出として拒否 | 既定 `0.8`m |
| `YOLO_BRIDGE_DOUBLE_S` | 二重クリック確認の間隔秒 | 既定 `0.6` |
| `OKRA_YOLO_MODEL` | モデルパス | 既定 `okra11n-seg.pt` |

---

## 段階を踏む（必ず）

全自動は実機未検証。現地では下記の順で escalate する:

1. **DRY-RUN**（端末C・LIVE無し）: Enterで検出座標が的と合うか確認
2. **ホバー検証**: 端末Aに `OKRA_NOACT_STANDOFF_M=0.05` を足して起動 → 刃が狙い点の5cm手前で
   止まる（接触しない）。端末Cを `YOLO_BRIDGE_LIVE=1` にしてEnter1発 → 奥行き・横ズレを実測
   （刃先〜的が5cmちょうど＝合格。4cmなら刃が1cm長い→tip offset x を +1cm）
3. **本把持**: standoff を外して端末Aを起動し直し、Enterで1回ずつ収穫

---

## 安全・停止

- **リモコンe-stopを手元に**: L2+B（ダンピング＝機体脱力、吊り/支え必須）／復帰 L2+↑
- 通常停止: 端末A で Ctrl-C → ログに `G1ArmSdkConnection disconnected` を確認 → 電源OFF可
- **Ctrl-Cで固まった場合**: グリッパは自動解放されない →
  `DEX1_NIC=enp46s0 DEX1_OPEN_Q=3.7 .venv/bin/python oda/gripper_open.py`、または リモコン L2+B
  （腕は通信途絶タイムアウトで自動脱力する）
- 掴んだ物を取りたい（アプリ稼働中）: 別端末で開き指令 → [RUN_CHEATSHEET.md](RUN_CHEATSHEET.md)「掴み解除」参照
- **PCフリーズ対策**: AC電源接続必須・`ZED_DEPTH_MODE=PERFORMANCE`

---

## 現地で詰まりやすい所

| 症状 | 対処 |
|---|---|
| 点群がビューアに出ない | 手順0の sudo 3行を再実行（G1電源OFFのたびに経路が消える） |
| YOLOがオクラを検出しない | `OKRA_YOLO_CONF` を下げる（0.25→0.15）／明るさ・距離を調整 |
| `3D点が遠すぎ` で発火中止 | `YOLO_BRIDGE_MAX_M` を上げる or 対象に近づく（既定0.8m） |
| 重心近傍に点群なし | `YOLO_BRIDGE_PX_RADIUS` を上げる（20→30）／ZEDは35cm以上離す |
| 腕が的とズレた所へ伸びる | `ZED_MOUNT_XYZRPY` の再計測（ZED付け直したら必ずズレる） |
| ビューアが古い画面のまま | ビューアを閉じて開き直す（アプリ再起動とセット） |
| ZED映像が帯状に乱れる | USBポート4-4に挿す（4-1不良） |

---

## 関連ファイル

| ファイル | 役割 |
|---|---|
| `dimos/robot/unitree/g1/blueprints/manipulation/unitree_g1_okra_ik_only_grasp_zed.py` | ZEDブループリント本体（環境変数定義） |
| `oda/yolo_click_bridge.py` | YOLO検出→`/clicked_point`発行（クリック代替） |
| `dimos/robot/unitree/g1/act/ik_reach_bridge.py` | クリック→IK本体 |
| `oda/RUN_ZED_IK.md` | ZED版クリック運用の詳細＋トラブル対処 |
| `oda/RUN_CHEATSHEET.md` | 全知見のコピペ一覧 |
