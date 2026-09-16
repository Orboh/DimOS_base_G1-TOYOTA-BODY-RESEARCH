# 頭部D435i × YOLO × IK — 胸ZEDとの対照実験 手順

2026-09-07 に成立した **胸ZED × YOLO × IK**（[RUN_ZED_YOLO_IK_AUTO.md](RUN_ZED_YOLO_IK_AUTO.md) /
`oda/zed_*.sh`）に対し、**カメラだけ頭部D435iに差し替えた**対照実験。

- ブループリント: `unitree-g1-okra-ik-only-grasp`（ZED版 `-zed` の兄弟。IK・腕・軌道は同一コード）
- スクリプト: `oda/head_status.sh` / `head_up.sh` / `head_arm_down.sh` / `head_down.sh` / `head_yolo.sh`
- **既定値は zed_*.sh と同一**にしてある（軌道・ゲイン・重力FF・駐機姿勢）。触らないこと。

---

## この実験で切り分けたいこと

着地点のズレ（横4件・下2件、[okra-grasp-failure-breakdown]）のうち、
**カメラの外部パラメータ由来の分**がどれだけあるか。

| | 胸ZED（9/7） | 頭D435i（今回） |
|---|---|---|
| 外部パラメータ | `ZED_MOUNT_XYZRPY` = **メジャー実測の未校正値** (±1cm) | **URDF `d435_joint` の設計値** xyz=[0.0576,0.0175,0.4299]・pitch=47.6°うつむき |
| 配信経路 | PC直結USB3・アプリ内プロセス | NX の `ik_camera_standalone.py` → LCMマルチキャスト → PC |
| クリック座標系 | TFあり → **ボディ系**で届く (`click_in_camera_body_frame=True`) | TFなし → **光学系の生値**（既定） |
| 深度の計算場所 | このPCのGPU（VRAM 8GBを食い合う） | NX側（このPCのGPUはYOLOだけ） |
| 解像度 | 1280×720 | 640×480 |

**読み方**: ズレの向きと量が2構成で**変わる**なら、外部パラメータ／ハンドアイ校正が効いている。
**ほぼ同じ**なら、カメラは主犯ではなく腕側（追従・たわみ・FK）に残っている。

> ⚠️ **ログの `[TIP] tip_diff_mm` では判定できない。** あれは「指令値 vs 実測関節角からのFK」の差で、
> **カメラ誤差は原理的に入らない**（[ik-tracking-error-measurement-blindspot]）。
> カメラを比べるなら測るのは **実際のオクラと指先の物理的なズレ（定規）** の一択。
> 記録するのは (1) 定規で測った 前後/左右/上下 のズレ[mm] (2) ログの `okra_target`（torso系の目標）
> (3) `[TIP]` の実測手先位置。(2)と(3)が一致していて(1)がズレていれば、それがカメラ由来。

---

## 手順

### 0. 毎セッション事前（自分のターミナルで。sudoはtty必須）

```bash
sudo ip link set lo multicast on
sudo ip route replace 224.0.0.0/4 dev enp2s0    # ← ZED版と違い**必須**（点群がNXから有線で来る）
sudo sysctl -w net.core.rmem_max=67108864 net.core.rmem_default=67108864
```

### 1. 生死確認

```bash
oda/head_status.sh
```
`NX 応答あり` と `点群受信 >0パケット` が出ること。NXは**有線 .164** を使う
（WiFi .0.211 はセッション中に切れて中継ごと落ちる）。

### 2. 起動 → 駐機姿勢

```bash
oda/head_up.sh          # NX中継を起こす → 点群到達を確認 → LIVE起動 → ビューア
oda/head_arm_down.sh    # A案駐機姿勢（床上0.70m・前方0.24m）へ
```
起動直後は腕を注視（weightランプ中の暴れが過去に1回）。

### 3. まず人間クリックで通す（YOLOはまだ使わない）

ビューアで点群の同じ点を2回クリック → 必ずログを確認:

```bash
grep -aE 'leg done|leg stopped' oda/head_run_out/head_reach_*.log | tail
```
`leg done` が **3つ** 出れば3段階アプローチが効いている。出ずに1回で届いてしまう場合は
「下からすくう」旧動作にフォールバックしている（`ws_x` 下限 / `max_joint_delta_deg` /
向き固定 の3つの門を疑う。[RUN_ZED_IK.md](RUN_ZED_IK.md)）。

### 4. YOLOのimgszを測り直す（**ZEDの値は流用できない**）

D435iは640×480でZEDと画角も解像度も違う。DRY-RUNでconfが最大になる値を探す:

```bash
for s in 640 960 1280; do OKRA_YOLO_IMGSZ=$s oda/head_yolo.sh --once; done
```

### 5. YOLO自動発火

```bash
oda/head_yolo.sh                       # まずDRY-RUNで3D点が妥当か見る
OKRA_YOLO_IMGSZ=<採用値> oda/head_yolo.sh --live --once
```

### 6. 停止（**順番厳守**）

```bash
oda/head_arm_down.sh    # 先に腕を下ろす
oda/head_down.sh        # あとからアプリ+ビューア（HEAD_STOP_NX=1 でNX中継も止める）
```

---

## 落とし穴

1. **`YOLO_BRIDGE_BODY_FRAME` を立てない。** ZED版(`zed_yolo.sh`)は=1だが、頭カメラで立てると
   光学→ボディ回転が**二重に掛かって腕が明後日に飛ぶ**。`head_yolo.sh` は明示的に `unset` 済み。
2. **ZED版と同時に起動しない。** `rt/arm_sdk` を取り合って異音＋暴走。`head_up.sh` が検知して止める。
3. **クリック点オフセットは 0 のまま。** 生の誤差を測るのが目的（`HEAD_X/Y/Z_OFFSET`）。
4. **距離ゲート**は `YOLO_BRIDGE_MAX_M=0.9` に上げてある（頭カメラは43cm上・47.6°うつむきで
   ZEDより被写体が遠い）。「遠すぎ」で蹴られたらさらに上げるが、背景誤検出の防波堤なので上げすぎない。
5. **グリッパ無し（リーチのみ）が既定** `OKRA_NO_GRIPPER=1`。9/7のZED実験と同条件。
   この環境変数は 2026-09-08 に頭カメラ版ブループリントへ追加した（ZED版には元からあった）。

---

## NX側（頭カメラ中継）のセットアップ — 2026-09-08 に判明した3点

胸ZEDはPC直結なので不要だったが、頭D435iは中継が要る。9/8に実機で詰まった順に:

### 1. `~/run_ik_camera.sh` が消えていた → 復元済み

最後の稼働は `~/ik_cam_standalone.log` の 9/4 13:58。同ログのヘッダ行から設定値を復元した:
`D435i 347622073233 / 640x480@15 / K=[607.3,607.2,323.7,259.1] / pc 3.0Hz`。

実行環境（NXでの実測）: python は **`~/miniconda3/envs/teleimager/bin/python3.10`**
（系の `python3` は 3.8 で pyrealsense2 が入っていない）。pyrealsense2 2.50.0 は
`~/librealsense/build/wrappers/python` に cpython-310 向けでビルド済み。dimos は pip 未導入なので
**リポ `~/workSpace/dimos_min` を `PYTHONPATH` に通す**（スクリプトをパス指定で実行すると
`sys.path[0]` は `examples/` になり cwd は入らない ＝ `ModuleNotFoundError: dimos` の原因）。

### 2. D435iは `videohub_pc4` に掴まれていて、kill しても復活する

`pyrealsense2` が `xioctl(VIDIOC_S_FMT) failed: Device or resource busy` で開けない。
単に kill すると **`master_service` が数秒で復活させる**（実測 pid 2160 → 4563）。先にこれを止める:

```bash
sshpass -p 123 ssh unitree@192.168.123.164 'echo 123 | sudo -S systemctl stop master_service'
```

止まるのは**このJetson上の video hub 2本だけ**（`/unitree/module` には master_service と
video_hub_pc4 しか無い）。**腕の制御(`rt/arm_sdk`/`rt/lowstate`)は 192.168.123.161 側なので無関係。**
復帰は `systemctl start master_service`（G1再起動でも戻る）。

### 3. NXがマルチキャストをWiFiに流していた ← **0パケットの真犯人**

NXは wlan0(192.168.50.235, metric 600) の既定経路が eth0(metric 20100) より強く、
`ip route get 239.255.76.67` が **wlan0** を返していた。中継は動いているのに点群が
ラップトップに1つも来ない、という症状になる。ラップトップ側と同じ経路を**NXにも**入れる:

```bash
sshpass -p 123 ssh unitree@192.168.123.164 'echo 123 | sudo -S ip route replace 224.0.0.0/4 dev eth0'
```

**経路を直したら中継を再起動すること**（LCMはソケット生成時の経路を掴む）。
どちらもNX再起動で消えるので毎回必要。

### おまけ: 受信テストは有線IPを明示する

`IP_ADD_MEMBERSHIP` を `INADDR_ANY` で張ると既定経路側(WiFi)を選んでしまい、**中継が生きていても
0パケットに見える**。`head_status.sh` / `head_up.sh` は `enp2s0` のIPv4を明示するよう修正済み。
