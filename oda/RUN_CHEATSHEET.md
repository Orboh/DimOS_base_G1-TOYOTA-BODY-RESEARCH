# 実行チートシート — クリック→IK切断把持（2026-07-21時点の全知見）

このPC（RTX3070ラップトップ, NIC=`enp46s0`, IP=.222）用。
詳細理論は `RUN_ZED_IK.md` / `FARM_QUICKSTART.md`。ここはコピペ用の一覧。

---

## 0. 毎セッション共通の事前チェック（順番どおり）

```bash
# (a) PCをAC電源に接続。ロボットLANに有線接続
# (b) G1電源ON（刃は閉じてから、の習慣は維持）→ リモコン L2+↑ → R1+Y
# (c) ネットワーク準備（PC再起動後 と G1電源OFFのたびに 消える）:
sudo ip link set lo multicast on
sudo ip route replace 224.0.0.0/4 dev enp46s0
sudo sysctl -w net.core.rmem_max=67108864 net.core.rmem_default=67108864
```

```bash
# (d) グリッパ生存確認（毎回必須！ モーター無効は普通の電源投入でも起こる）:
cd ~/Toyota-auto-body-PoC/DimOS_oda
CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
DEX1_NIC=enp46s0 DEX1_OPEN_Q=3.0 .venv/bin/python oda/gripper_open.py
```
- **爪が動けばOK**（そのまま開き状態でスタンバイ）
- **動かない（位置固定・トルク0）** → モーター無効。G1電源OFF → ハンドコネクタ挿し直し → 刃を閉じて電源ON → 再確認

```bash
# (e) close_q の決定（アタッチメント交換後・ゼロ点不安時のみ）:
#     刃を手で完全に閉じた状態で q を読む → close_q = その値 − 0.1
#     ※ゼロ点は電源サイクルでも引き継がれる(2026-07-20/21実測)。
#     直近実測: 刃全閉 q≈1.80 → close_q=1.7 / 刃付き全開の機械限界 q≈3.85 → 標準全開 3.7
```

---

## 1. D435i版（頭カメラ・Jetson NX中継）— 全部入りスクリプト

### 1-A. 上からアプローチ（コの字軌道：リフト→水平→垂直降下）

株を蹴らずに上から刃を入れる。`OKRA_APPROACH_ABOVE_M` の行が有効化スイッチ。

```bash
cd ~/Toyota-auto-body-PoC/DimOS_oda
OKRA_DEX1_PREFIX=rt/dex1/left \
OKRA_NOACT_GRIP_LIVE=1 \
OKRA_NOACT_CLOSE_Q=1.7 \
OKRA_GRIP_KP=20 \
OKRA_NOACT_KP_ARM=160 OKRA_NOACT_KD_ARM=6.0 \
OKRA_OPEN_Q=3.7 \
OKRA_TIP_OFFSET_XYZ="0.25,-0.003,0" \
OKRA_APPROACH_ABOVE_M=0.08 \
bash oda/start_okra_ik_only_grasp.sh --live
```
- 数字＝移動高度（目標の何m上を通るか）。株の頭に当たるなら `0.12` に上げる
- 動きはゆっくりめ（手先約0.2m/s の直線なぞり）

### 1-B. 従来どおり（直行リーチ：現在位置→目標を一発）

`OKRA_APPROACH_ABOVE_M` の行を**消しただけ**。7/16実証時代と同じ動き。速いが低い姿勢から株を掃くことがある。

```bash
cd ~/Toyota-auto-body-PoC/DimOS_oda
OKRA_DEX1_PREFIX=rt/dex1/left \
OKRA_NOACT_GRIP_LIVE=1 \
OKRA_NOACT_CLOSE_Q=1.7 \
OKRA_GRIP_KP=20 \
OKRA_NOACT_KP_ARM=160 OKRA_NOACT_KD_ARM=6.0 \
OKRA_OPEN_Q=3.7 \
OKRA_TIP_OFFSET_XYZ="0.25,-0.003,0" \
bash oda/start_okra_ik_only_grasp.sh --live
```
- ※工具オフセットのyが**1-Aと違う**ことに注意：`-0.023`は上から降りる姿勢用の暫定校正、
  直行リーチでは元の `-0.003`（横ズレが出たら§3のホバー法で測り直し）

共通:
- スクリプトが順に: Jetsonカメラ起動 → sudo3行 → 点群到達チェック → ビューア自動起動(12秒後) → アプリ
- `--live` を外す＝ドライラン（腕動かない確認モード）
- 終了: Ctrl-C → `G1ArmSdkConnection disconnected` を確認 → 電源OFF可

## 2. ZED版（胸カメラ・PC直結）— FARM_QUICKSTART方式

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
別ターミナルでビューア（**アプリ再起動のたびにビューアも開き直すこと** — 古い画面は静止画のまま）:
```bash
~/Toyota-auto-body-PoC/DimOS_oda/.venv/bin/dimos-viewer \
  --connect rerun+http://127.0.0.1:9877/proxy --ws-url ws://127.0.0.1:3030/ws
```
- ZEDはUSB3ポート**4-4**へ（4-1は不良）。フリーズ対策は `ZED_DEPTH_MODE=PERFORMANCE`（点群粗くなる）
- **上から/従来の切替は§1と同じ**：上記は上からアプローチ版。従来の直行リーチにするには
  `OKRA_APPROACH_ABOVE_M=0.08 \` の行を消す（そのときは工具オフセットyも `-0.003` に戻す。§1-B参照）

## 3. 計測モード（ズレを測るとき）

上のコマンドに1行足して再起動:
```bash
OKRA_NOACT_STANDOFF_M=0.05 \
```
→ 刃が**狙い点の5cm手前でホバー**（的に触れない）。
- 奥行き検証: 刃先〜的の実測がちょうど5cm＝合格。4cmなら刃が1cm長い→xを+1cm
- 横ズレ: 真後ろから的方向に目線を通して左右を読む
- **接触したまま測らない**（的が動いてアーティファクトになる — 2026-07-21実証）
- 的は棒に貼ったテープ等の固定点。**右寄り・胸の高さ**に置く（正面・低位置は腕が的を掃く＋手首が捻れる）

## 4. 環境変数リファレンス

| 変数 | 意味 | 現在の推奨値 | 消すとどうなる |
|---|---|---|---|
| `OKRA_DEX1_PREFIX` | dex1トピック接頭辞 | `rt/dex1/left`（本機はケーブルが左挿し） | rightを探して起動失敗 |
| `OKRA_NOACT_CLOSE_Q` | 切り閉じ位置 | 1.7（刃全閉1.8−0.1） | 0.0＝1.8rad過剰押し込み注意 |
| `OKRA_GRIP_KP` | 閉じ力（力≒kp×残差） | 20（切断実績値） | 5.0＝弱い保持 |
| `OKRA_OPEN_Q` | クリック毎の自動開き幅 | 3.7（全開3.85の手前） | 3.0標準開き / `""`で自動開き無効 |
| `OKRA_TIP_OFFSET_XYZ` | 刃の工具オフセット(手首系) | `"0.25,-0.003,0"`※(7/22再製作個体・暫定、7/23朝に校正) | 素のDex1(18.45cm)＝4cm奥に刺さる |
| `OKRA_APPROACH_ABOVE_M` | 上からアプローチ(コの字軌道) | 0.08（株が高ければ0.12） | **消す＝従来の直行リーチ** |
| `OKRA_TARGET_Z_OFFSET` | クリック点からの上オフセット | （なし＝0、クリック点直狙い） | 0.05で旧「+5cm上」に戻る |
| `OKRA_CUT_BELOW_CENTROID_M` | クリック点の下を狙う | 必要時のみ（0.03=3cm下） | 0 |
| `OKRA_NOACT_STANDOFF_M` | 手前ホバー(計測用) | 計測時0.05 / 本番は消す | 0＝刃が点まで行く |
| `OKRA_NO_GRIPPER` | グリッパ無し運用 | ハンド未接続時のみ `1` | — |

※ yの-0.023は「上からアプローチ姿勢」用の暫定値。向き固定（下記5-3）後に再校正予定。

## 5. 既知の課題と次にやること（2026-07-21時点）

1. **アタッチメント破損中** → 修理/交換後、**§3のホバー法で奥行き再検証**（+4cmは旧アタッチメント実測値）
2. 「1発目は必ずズレる」= クリック瞬間の手首の向きが目標になる仕様のため。
   **修正手順**: 数クリック→ドンピシャの回を申告→そのリーチの関節角からFKで向きを算出→
   `fixed_orientation_xyzw` に固定（Claudeに「今の！」と言えば拾ってくれる）
3. 切断後の低い姿勢から遠い点をクリックすると90°ガードで拒否される → もう一度クリックすれば通る

## 6. トラブル対処 トップ6

| 症状 | 対処 |
|---|---|
| グリッパが指令無視（位置固定・トルク0） | モーター無効。G1電源OFF→ハンドコネクタ挿し直し→刃閉じ電源ON |
| カメラ映らない(D435i) | NXで `tail ~/ik_cam_standalone.log` → "No device connected"ならUSB挿し直し→スクリプト再実行 |
| 点群がビューアに出ない | §0(c)の3行を再実行（G1電源OFFのたびに経路が消える） |
| ビューアが古い画面のまま | ビューアを閉じて開き直す（アプリ再起動とセット） |
| ZED映像が帯状に乱れる | USBポート4-4に挿す（4-1不良）。`journalctl -k`のEPROTO(-71)で確認 |
| PCフリーズで腕が伸びたまま固まった | ①`DEX1_NIC=enp46s0 python oda/gripper_open.py`で解放 ②`python oda/arm_release.py`で腕降ろし |

## 掴み解除（アプリ稼働中に挟んだ物を取りたいとき・別ターミナル）

```bash
cd ~/Toyota-auto-body-PoC/DimOS_oda
CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
.venv/bin/python -c "
import time
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.JointState import JointState
t = LCMTransport('/g1/gripper_target', JointState)
for _ in range(3):
    t.publish(JointState(name=['g1/right_gripper'], position=[3.7], velocity=[0.0], effort=[0.0]))
    time.sleep(0.3)
print('open sent')"
```
- アプリ停止中は代わりに: `DEX1_NIC=enp46s0 DEX1_OPEN_Q=3.7 .venv/bin/python oda/gripper_open.py`
- 次のクリックで自動で開くので、通常サイクルではこのコマンド不要（物を取るときだけ）

## 終了手順（毎回）

1. Ctrl-C → ログに `G1ArmSdkConnection disconnected` を確認
2. グリッパに何か挟まってたら開きスクリプトで解放
3. G1電源OFF

## 関連ドキュメント

- `FARM_QUICKSTART.md` — ZED版の農場向け最短手順（Zオフセット既定変更前の記述あり、環境変数は本書優先）
- `RUN_ZED_IK.md` — ZED版の詳細＋トラブル対処
- `AGX_ORIN_PORT_SPEC.md` — AGX Orin移植仕様（別スレッド）
