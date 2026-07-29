# 農場クイックスタート — 昨日(2026-07-16)の把持デモをそのまま再現する

上から順に実行するだけ。理屈・選択肢は省略(詳細は `RUN_ZED_IK.md`)。
インターネット不要。Wi-Fiはオフでよい。

## 0. G1の電源を入れる【最重要・順番厳守】

**グリッパの爪を手で完全に閉じてから、G1の電源を入れる。**
(理由: 電源投入時の爪の位置が「閉じ位置ゼロ」として記録されるため。
これを守ると下のコマンドの `OKRA_NOACT_CLOSE_Q=0.0` が正しい全閉になる)

電源投入後、リモコンで **L2+↑ → R1+Y** で運動制御モードまで入れる
(この機体で実機確認済みの組み合わせ。入れないと腕が指令を無視して動かない)。

## 1. 配線

- **このPCをACアダプタに接続(必須!)** — バッテリー駆動だとCPU/GPUが間引かれ、
  クリック→動作の全段が目に見えて遅くなる(2026-07-17農場で実証、AC接続で解消)。
  PC突然死の原因候補でもある
- ZED Mini → このPCの **USB3ポート(青い端子)**
- このPC → ロボットLAN(有線、いつものポート)
- G1 → 同じLAN

## 2. PC側ネットワーク準備(PCを起動するたびに毎回)

```bash
sudo ip link set lo multicast on
sudo ip route replace 224.0.0.0/4 dev enp46s0
sudo sysctl -w net.core.rmem_max=67108864 net.core.rmem_default=67108864
```

## 3. グリッパの開き(2026-07-17から自動化)

**クリックするたびに、グリッパが標準開き位置(q=3.0)より閉じていれば自動で開く**
ようになった(`OKRA_OPEN_Q`、空文字で無効化)。籠リリース後・異常終了後など、
どんな状態から始めても毎サイクル同じ開きで掴みに行く。
手で開いておく必要はもう無いが、初回起動時に開き動作が正しいか目視確認すること。

## 4. アプリ起動(ターミナル1)

```bash
cd ~/Toyota-auto-body-PoC/DimOS_oda
CYCLONEDDS_HOME=~/cyclonedds-noshm \
LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
DIMOS_SKIP_COORDINATOR_RPC=1 PYTEST_VERSION=1 \
ROBOT_INTERFACE=enp46s0 \
OKRA_NOACT_KP_ARM=160 OKRA_NOACT_KD_ARM=6.0 \
IK_REACH_LIVE=1 OKRA_NOACT_GRIP_LIVE=1 \
OKRA_GRIP_KP=20 OKRA_NOACT_CLOSE_Q=0.0 \
.venv/bin/dimos run unitree-g1-okra-ik-only-grasp-zed
```

起動成功の目印(ログに全部出ること):
```
LAUNCHING **LIVE**
Camera successfully opened
Dex1 right gripper ready
arm_sdk ready (mode_machine=5)
```

## 5. ビューア起動(ターミナル2)

```bash
~/Toyota-auto-body-PoC/DimOS_oda/.venv/bin/dimos-viewer \
  --connect rerun+http://127.0.0.1:9877/proxy --ws-url ws://127.0.0.1:3030/ws
```

## 6. 収穫

1. **e-stop(リモコンL2+B)を手に持つ**
2. 3Dビューで**オクラの点群の粒を直接クリック**
3. 腕が伸びる → 1〜2秒後にグリッパが自動で閉じ切る → 保持
4. **掴んだ後はビューア内で不用意にクリックしない**(クリック=即動作指令)
5. 掴み直し: 下の「爪を開き直す」→ オクラ置き直し(or 別のオクラ)→ 再クリック

爪を開き直す(ターミナル3、アプリは動かしたまま):
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
    t.publish(JointState(name=['g1/right_gripper'], position=[3.0], velocity=[0.0], effort=[0.0]))
    time.sleep(0.3)
print('open sent')"
```

## 7. 終了(順番厳守)

1. ターミナル1で **Ctrl-C**
2. ログに **`G1ArmSdkConnection disconnected`** が出るのを確認
3. それから電源OFF・ケーブル抜き

Ctrl-Cで止まらない/上の行が出ないまま死んだ → **L2+Bで解放**(腕は勝手に脱力するが
グリッパは固まったままなので、L2+Bが必要)。

## うまくいかないとき トップ4

| 症状 | 対処 |
|---|---|
| 点群がビューアに出ない | 手順2の3行をやり直す(再起動で消える)。ZEDを挿し直す |
| 起動が `No LowState` で落ちる | G1の電源・LANケーブル確認 |
| 起動が `No rt/dex1/.../state` で落ちる | ハンドのケーブル確認。腕だけなら起動コマンドに `OKRA_NO_GRIPPER=1` を足す |
| グリッパが閉じ切らない | 手順0を守らず電源を入れた可能性。`OKRA_NOACT_CLOSE_Q` を `-0.5` → `-1.0` → `-1.5` と下げて再起動(昨日は-1.5で閉じ切った)。または爪を閉じてG1を電源入れ直し→0.0に戻す |
| クリックしても腕が全く動かない(指令は出ている) | リモコンで **L2+↑ → R1+Y**(運動制御モード、この機体で実機確認済み)。入っていないとモーターが指令を無視する(2026-07-17実証。※マニュアル類のR1+X表記はFW違い) |
| グリッパが指令を完全無視(位置不動・トルク0) | **モーター無効状態(mode=0)**。原因=**電源が入ったままハンドのケーブルを抜き挿しした**(活線挿抜。基板が無効状態で再起動し、サービスは再有効化しない — 2026-07-17実証)。対処: G1電源OFF→コネクタ挿し直し→刃を閉じて電源ON。確認: `RUN_ZED_IK.md`のグリッパ実測コマンドで `mode` が1になっていること。**予防: ハンドのケーブル抜き挿しは必ずG1電源OFFで行う** |

それ以外の症状は `RUN_ZED_IK.md` のトラブル対処を参照。

## カッターの設定(標準構成 — 2026-07-17以降、IKは常時カッター装着で運用)

刃を付ける前は正確だった=ズレの原因は**カッターの切断ポイントがDex1指先から
ずれている**こと。工具オフセットを設定すれば直る:

1. カッターを装着した状態で2つ測る:
   - **A(前)**: 素のDex1爪先端から刃の切断ポイントまで、指の向きにさらに何cm先か
   - **B(横)**: 指の中心線から切断ポイントが左右どちらに何cmずれているか
2. 起動コマンドに1行足す(例: A=3cm, B=3cm の場合):
   ```bash
   OKRA_TIP_OFFSET_XYZ="0.215,-0.03,0" \
   ```
   (x = 0.1845+A[m], y = ±B[m]。**yの符号は1回リーチして確認** — ズレが倍に
   なったら符号を反転する。素のDex1に戻したらこの行を消すだけ)

## 農場での既知の制約(2026-07-17時点)

- 上記工具オフセット設定後もズレが残る場合: 仮変換(未校正)+地面の傾き
  (畑でG1が傾くと1°≈1cm@50cm)が原因候補。ハンドアイ校正が根本対策(未実装)
