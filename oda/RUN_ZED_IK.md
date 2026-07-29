# 胸ZEDカメラ クリック→IKリーチ 実行手順

G1の胸に付けたZED Miniの点群をクリックすると、右腕がその点までIKで伸びる
(オプションでグリッパを閉じる)デモの起動手順とトラブル対処。
ブループリント: `unitree-g1-okra-ik-only-grasp-zed`
(2026-07-16 実機検証済み: LIVEリーチ8/8収束)

## 構成(前提)

- **ラップトップ**(このPC): ZED MiniをUSB3(青い端子)に直結、
  ロボットLANに有線接続(NIC `enp46s0` / IP 192.168.123.222)
- **G1**: 電源ON、LAN接続。腕を使うので周囲クリア+**リモコンe-stop(L2+B)を手元に**
- **Jetson NXは不要**(D435i版と違いカメラ中継なし)

## オフライン実行(ネット無し環境への持ち出し)

**このシステムは実行時にインターネットを一切使わない。** 畑などネット無し環境でも
このドキュメントの手順がそのまま動く。内訳:

| 要素 | ネット要否 | 備考 |
|---|---|---|
| DimOS / dimos run / IK / ビューア | 不要 | 全部ローカル実行 |
| G1とのDDS・LCM通信 | 不要 | ロボットLAN(192.168.123.x)+ループバックのみの隔離網 |
| ZED工場キャリブレーション | **初回オープン時に1回だけDL** | このPCはキャッシュ済み(`/usr/local/zed/settings/SN17243330.conf`) |
| NEURAL深度のAIモデル | **初回に1回だけDL+GPU最適化** | このPCはキャッシュ済み(`/usr/local/zed/resources/neural_depth_*.model*`) |

**持ち出し前チェックリスト**(現地でハマらないために):
1. **手順1の3行を必ず実行**(特に `ip route replace 224.0.0.0/4 dev enp46s0`。Wi-Fi接続中は
   無くても動いてしまうが、オフラインではこれが無いとLCM通信が全滅する —
   2026-07-16にWi-Fiを切って実証済みの落とし穴)
2. 屋内(ネットあり)で一度 `dimos run unitree-g1-zed-ik-view` を起動し、点群が出るのを確認
   しておく(=キャッシュが温まっている証明。ZED個体を替えた場合はこのステップ必須 —
   キャリブファイルはシリアル番号ごとに1回DLされる)
3. **別のPC**(AGX Orin等)で初めて動かす場合も同様に、屋内で初回起動を済ませておく
4. `sudo`のネットワーク設定3行(手順1)は再起動ごとに必要だが、オフラインで問題なく実行可
5. NTP同期は不要(タイムスタンプは全てローカル時計基準で、鮮度チェックも同一マシン内比較)
6. **オフラインでの動作確認は「Wi-Fiを切った状態で一度通しで動かす」のが唯一の証明**
   (オンラインだと上記の経路問題が隠れる)

## 手順

### 1. ネットワーク準備(PC再起動のたびに必要)

```bash
sudo ip link set lo multicast on
sudo ip route replace 224.0.0.0/4 dev enp46s0
sudo sysctl -w net.core.rmem_max=67108864 net.core.rmem_default=67108864
```

確認: `ip link show lo` の1行目に `MULTICAST` が入っていればOK。

**2行目(マルチキャスト経路)は特にオフライン時に必須。** Wi-Fiが繋がっていると
Wi-Fi側の経路が代役をするため無くても動いてしまうが、Wi-Fiを切る(=オフライン運用)と
LCM通信(点群・クリック・関節状態の全て)が経路無しで死ぬ(2026-07-16に実際に発生)。
最初から必ず入れておくこと。

### 2. アプリ起動

```bash
cd ~/Toyota-auto-body-PoC/DimOS_oda
CYCLONEDDS_HOME=~/cyclonedds-noshm \
LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
DIMOS_SKIP_COORDINATOR_RPC=1 PYTEST_VERSION=1 \
ROBOT_INTERFACE=enp46s0 \
OKRA_NOACT_KP_ARM=160 OKRA_NOACT_KD_ARM=6.0 \
IK_REACH_LIVE=1 \
.venv/bin/dimos run unitree-g1-okra-ik-only-grasp-zed
```

モード切り替え(環境変数の足し引き):

| やりたいこと | 変更 |
|---|---|
| ドライラン(腕を動かさず座標確認だけ) | `IK_REACH_LIVE=1` を外す |
| グリッパ未装着で腕だけ | `OKRA_NO_GRIPPER=1` を足す |
| グリッパを実際に閉じる | `OKRA_NOACT_GRIP_LIVE=1` を足す(装着+LIVE時のみ) |
| 閉じ位置 | `OKRA_NOACT_CLOSE_Q`(本来0.0=全閉。**2026-07-16現在はゼロ点シフトのため-1.5** — 下のトラブル対処参照) |
| 握力 | `OKRA_GRIP_KP`(既定5.0=柔らか把持。強く握り込む/カッター用途は20程度。停止時の握力≈kp×位置誤差) |
| カメラ取付位置の微調整 | `ZED_MOUNT_XYZRPY="x,y,z,roll,pitch,yaw"`(既定 `0.109,0.030,0.248,0.0,-0.0209,0.0`) |
| Dex1トピック | `OKRA_DEX1_PREFIX`(既定`rt/dex1/left` — この機体の配線都合。右ポート配線に直したら`rt/dex1/right`) |
| 点群を軽くする(PC不調時) | `ZED_DEPTH_MODE=PERFORMANCE`(粗くなるがGPU負荷1/10) |
| 点群の密度/範囲 | `ZED_PC_VOXEL`(既定0.002m)/ `ZED_DEPTH_TRUNC`(既定0.8m=これより遠くは非表示) |

2026-07-16の実績構成(オクラ握りつぶし把持に成功した組み合わせ):
`IK_REACH_LIVE=1 OKRA_NOACT_GRIP_LIVE=1 OKRA_GRIP_KP=20 OKRA_NOACT_CLOSE_Q=-1.5`

### 3. ビューア起動(別ターミナル)

```bash
~/Toyota-auto-body-PoC/DimOS_oda/.venv/bin/dimos-viewer \
  --connect rerun+http://127.0.0.1:9877/proxy --ws-url ws://127.0.0.1:3030/ws
```

### 4. 操作

- 3Dビューで**点群の粒の上を直接クリック** → 腕がその点へ伸びる(LIVE時)
- 目標は「前方20〜60cm・やや右側・腰〜胸の高さ」が右腕の得意範囲
- 腕は次のクリックまで到達姿勢を**保持し続ける**(自動で戻らない)
- ログ確認(別ターミナル): `[LIVE->arm_sdk] reach #N ... converged=True` が成功の印

### 4.5 把持の標準手順(クリック→全閉で掴む)

前提: グリッパは**起動前に手で開いておく**(アプリは起動時のグリッパ位置を
そのまま保持する設計。全開でなくてよい、対象が入る幅+余裕があれば十分)。

1. グリッパを開いた状態で、手順2のコマンドを**把持フル構成**で起動:
   ```bash
   cd ~/Toyota-auto-body-PoC/DimOS_oda
   CYCLONEDDS_HOME=~/cyclonedds-noshm \
   LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
   LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
   DIMOS_SKIP_COORDINATOR_RPC=1 PYTEST_VERSION=1 \
   ROBOT_INTERFACE=enp46s0 \
   OKRA_NOACT_KP_ARM=160 OKRA_NOACT_KD_ARM=6.0 \
   IK_REACH_LIVE=1 OKRA_NOACT_GRIP_LIVE=1 \
   OKRA_GRIP_KP=20 OKRA_NOACT_CLOSE_Q=-1.5 \
   .venv/bin/dimos run unitree-g1-okra-ik-only-grasp-zed
   ```
   意味: 腕LIVE + グリッパLIVE、握力kp=20、閉じ目標-1.5(ゼロ点シフト補正込みの
   「閉じ切り」— 2026-07-16のオクラ握りつぶし把持の実績値)。
2. ビューア(手順3)を開き、**対象の点群をクリック**
3. 自動で進む: 腕がリーチ → 到達後1〜2秒で `reach_done` → **グリッパが-1.5まで自動全閉**
   (ログ: `GripperGraspOnReach: reach_done -> closing gripper q=-1.500`)
4. 掴んだまま保持され続ける。**次のクリックも即動作指令になる**ので、
   把持後はビューア内で不用意にクリックしないこと
5. 掴み直す場合: 開き指令(トラブル対処「把持のやり直し」のスニペット)→
   対象を置き直す → 再クリック

注意: ゼロ点を正しく直した後(爪を閉じた状態で電源投入)は、`OKRA_NOACT_CLOSE_Q`を
**0.0に戻す**こと(-1.5のままだと正しいゼロに対して過剰に押し込む)。

### 5. 停止(必ずこの順で)

1. アプリのターミナルで **Ctrl-C**
2. ログに **`G1ArmSdkConnection disconnected`** が出るのを確認
   (weightを1→0に下げて腕を機体側コントローラに返した証拠)
3. 確認できてから電源OFF・LAN抜線など

---

## トラブル対処

### 起動が「No LowState」で止まる/落ちる
G1の電源が入っていない、LANが繋がっていない、またはNICが違う。
`ROBOT_INTERFACE`が実際の有線NIC名(`ip a`で確認、既定`enp46s0`)か確認。
※これは安全設計(ロボット不在時に指令を出さない)なので異常ではない。

### 起動が「No rt/dex1/.../state」で止まる
- グリッパ(Dex1)が付いていない/サービス未起動 → 腕だけ試すなら
  `OKRA_NO_GRIPPER=1` を足して再起動
- **付けているのに出ない** → 左右のサービス取り違えの可能性。この機体は
  右手首に付けたDex1が**左サービス(`rt/dex1/left/*`)として認識される**
  (ケーブルが左手用ポート挿し、2026-07-16確認)。そのためこのブループリントの
  既定は `rt/dex1/left`。配線を正しい右ポートに直した場合は
  `OKRA_DEX1_PREFIX=rt/dex1/right` を指定すること。
  どちらのトピックが生きているかの確認:
  ```bash
  # DDS上の全トピック列挙(dex1で絞る)
  CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
  .venv/bin/python -c "
  import time
  from cyclonedds.domain import DomainParticipant
  from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication
  dr = BuiltinDataReader(DomainParticipant(0), BuiltinTopicDcpsPublication)
  seen=set(); t0=time.time()
  while time.time()-t0<8:
      seen |= {s.topic_name for s in dr.take(N=100)}; time.sleep(0.2)
  print([t for t in sorted(seen) if 'dex1' in t])"
  ```

### ZEDが見つからない(Failed to open ZED camera)
```bash
lsusb | grep -i stereolabs
```
- 何も出ない → USBを挿し直す。**USB3(青い端子)**に挿すこと
- `HID Interface` しか出ない(`ZED-M camera`が無い) → カメラ機能が
  列挙されていない。**抜いて挿し直す**(PCクラッシュ後によく起きる)
- pyzedで確認: `.venv/bin/python -c "import pyzed.sl as sl; print(sl.Camera.get_device_list())"`

### 点群がビューアに出ない
- ほぼ確実に **lo multicastがオフ**(PC再起動でリセットされる)。手順1をやり直す
- `ip link show lo` で `MULTICAST` の有無を確認

### ビューアが勝手に開かない
アプリからの自動起動はDISPLAYの無いシェルでは失敗する(ログに
"Rerun native viewer not available" と出る)。手順3のコマンドで手動起動すればよい。

### クリックしても腕が動かない
- ログに `click frame '/world/camera/color_image' != expected` 等が出る →
  **画像やカメラ枠をクリックしている**。点群の粒の上を狙う
- `joint ... delta XX° exceeds 90.0°` → 今の姿勢から遠すぎる目標。
  もう少し近い/自然な位置をクリック(安全ゲートによる正常な拒否)
- `outside workspace box` → 腕の届く範囲外(後方・左側・遠すぎ等)
- `within debounce interval` → 連打しすぎ。2秒あけて再クリック

### グリッパが「閉じ切らない」(kpを上げても変わらない)
まず実測値を見る(下のコマンド)。**実測q≈0でトルク≈0なのに爪が開いている**なら、
力不足ではなく**Dex1のゼロ点シフト**(ハンドを半開きの状態で電源投入すると、その位置が
q=0として記録される。2026-07-16に実際に発生)。対処:
- **応急**: マイナスの閉じ目標を使う(`OKRA_NOACT_CLOSE_Q=-1.5`。サービスは負の目標を
  受け付け、偽ゼロを突き抜けて閉じる — 実測済み)
- **根本**: **爪を完全に閉じた状態でハンド(G1)の電源を入れ直す** → ゼロ点が正しい位置に
  戻り、close_qを0.0に戻せる
```bash
# グリッパ実測q/トルクの確認
CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
.venv/bin/python -c "
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
ChannelFactoryInitialize(0, 'enp46s0')
s=[]
sub=ChannelSubscriber('rt/dex1/left/state', MotorStates_)
sub.Init(lambda m: s.append((m.states[0].q, m.states[0].tau_est)), 10)
time.sleep(4); print('q=%.3f tau=%.3f' % s[-1] if s else 'no state')"
```

### 把持のやり直し(爪を開き直したい)
専用UIは無いので、アプリの正規入力トピックに開き目標を送る(アプリ稼働中に別ターミナルで):
```bash
CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
.venv/bin/python -c "
import time
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.JointState import JointState
t = LCMTransport('/g1/gripper_target', JointState)
for _ in range(3):
    t.publish(JointState(name=['g1/right_gripper'], position=[3.0], velocity=[0.0], effort=[0.0]))
    time.sleep(0.3)
print('open q=3.0 sent')"
```
開いたら対象を置き直して再クリック(閉じは自動)。

### 掴んだ後に腕が勝手に動く
**ビューア操作中の誤クリックがそのまま動作指令になる**(2026-07-16に発生 —
「勝手に下りた」の正体は視点操作中の意図しないクリック2回)。把持後は
ビューアの3Dビュー内でのクリックに注意。視点回転はドラッグ、クリック(押して離す)だけが
指令になる。デモ運用ではクリック確認ステップの追加が今後の改善候補。

### 腕が実際の目標とズレた場所に伸びる
カメラ取付変換(`ZED_MOUNT_XYZRPY`)の誤差。現在値はメジャー+IMU実測の
仮値(±1〜2cm精度)。数cmのズレなら値を微調整
(x=前方向, y=左, z=上, 単位m。回転はroll,pitch,yaw[rad]、pitch正=下向き)。
大きくズレる場合はハンドアイ校正が必要(未実装、次の課題)。
※ZEDを取り付け直した/マウントを動かした場合は必ずズレる → 再計測。

### PCが突然落ちる(電源断)
2026-07-16に2回発生(いずれもNEURAL深度=GPU高負荷中、原因未特定)。対処:
1. **ACアダプタ接続を確認**(バッテリー駆動を避ける)
2. `ZED_DEPTH_MODE=PERFORMANCE` に落とす(粗くなるがクリックには十分)
3. 再発調査用にGPU監視を回す:
   `nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw --format=csv -l 3 >> oda/gpu_watch.log`
   (これまでの記録では最高64°C/65W程度で熱の証拠なし)

### Ctrl-Cで止まらない/ログにdisconnectedが出ずに死んだ
既知の未解決問題(D435i版から継続)。この状態では:
- **腕**: 通信途絶タイムアウトで自動的に機体側に戻る(脱力する)
- **グリッパ(Dex1)**: **自動解放されない**。固まったままになる →
  **リモコンのL2+B(ダンピングモード)で解放**(吊り/支え必須、機体が脱力する)
- 別シェルから止める場合:
  `kill -INT $(pgrep -f "python .venv/bin/dimos run unitree-g1-okra-ik-only-grasp-zed" | head -1)`
  ※ `pgrep -f "dimos run"` だけだとbashラッパーに当たって本体が残るので、
  必ず `python` を含むパターンで

### sudoがコマンド経由で失敗する
このドキュメントのsudo行は**自分のターミナルで直接**実行すること
(パスワード入力が要るため、tty無しのツール経由では失敗する)。

---

## 参考: 関連ファイル

| ファイル | 役割 |
|---|---|
| `dimos/robot/unitree/g1/blueprints/manipulation/unitree_g1_okra_ik_only_grasp_zed.py` | 本体ブループリント(環境変数の定義もここ) |
| `dimos/robot/unitree/g1/blueprints/perceptive/unitree_g1_zed_ik_view.py` | カメラ+ビューアのみ(G1不要の表示確認用): `dimos run unitree-g1-zed-ik-view` |
| `dimos/robot/unitree/g1/act/ik_reach_bridge.py` | クリック→IK本体(`camera_mount_xyzrpy`/`click_in_camera_body_frame`) |
| `dimos/hardware/sensors/camera/zed/camera.py` | ZEDドライバ(`pointcloud_voxel`/`pointcloud_depth_trunc`) |
| `oda/inspect_camera_transform.py` | カメラ→torso変換の数値を表示する調査スクリプト |
| D435i(頭部カメラ)版 | `dimos run unitree-g1-okra-ik-only-grasp` + `oda/start_okra_ik_only_grasp.sh`(Jetson中継が必要、今回の変更の影響なし) |
