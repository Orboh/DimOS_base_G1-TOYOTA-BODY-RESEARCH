# Unitree G1 — ラップトップ単体でのクリック移動（Orboh版手順）

**検証済み: 2026-06-05（実機G1）** — PCのdimosだけでG1をクリックナビゲーションさせる手順。
公式の [`docs/platforms/humanoid/g1/index.md`](/docs/platforms/humanoid/g1/index.md#L83) はロボットのNX上でdimosを動かす構成（`unitree-g1-nav-onboard`）だが、本手順は **G1側に何もインストールしない**。

使用ブループリント: **`unitree-g1-nav-laptop`**
（`unitree-g1-nav-onboard` のバリアント。LiDAR受信IPとDDSのNICをPC側で自動検出する）

## 構成

```
Mid-360 LiDAR (192.168.123.120) ──UDP(~5MB/s)──> PC: FastLIO2 (SLAM)
                                                      ↓
                                  navスタック (TerrainAnalysis / SimplePlanner /
                                   LocalPlanner / PathFollower / PGO)
                                                      ↓ cmd_vel
PC: Rerun viewer クリック → MovementManager → G1HighLevelDdsSdk ──DDS──> G1歩行
```

## 前提条件

- Unitree G1 EDU、PC は Ubuntu 22.04/24.04
- PC を G1 に **Ethernet 直結**し、固定IPを設定（例: `192.168.123.212/24`）
  - 確認: `ping 192.168.123.164`（NX）と `ping 192.168.123.120`（LiDAR）が両方通ること
- dimos のインストール（このリポジトリ、`uv sync` 済みの venv）

## 1. nix のインストール（初回のみ）

ネイティブモジュール（C++）のビルドに必要：

```bash
curl -fsSL https://install.determinate.systems/nix | sh -s -- install
# インストール後、新しいターミナルを開くか:
. /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
```

## 2. ネイティブモジュールの事前ビルド（初回のみ）

⚠️ **5モジュール全部**が必要。`dimos run` 内の自動ビルドに任せると、nixがPATHにないシェルでは
`nix: not found (exit 127)` で全モジュール巻き添えクラッシュする
（[FleetSeek exp_01KTASFAEVY95HY2CFZ13VNWFE 参照](https://web-ebon-zeta-33.vercel.app/experience/exp_01KTASFAEVY95HY2CFZ13VNWFE)）。

```bash
. /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
cd dimos/hardware/sensors/lidar/fastlio2/cpp && nix build .#fastlio2_native && cd -
cd dimos/navigation/nav_stack/modules
(cd local_planner    && nix build "github:dimensionalOS/dimos-module-local-planner/v0.6.0"    --no-write-lock-file)
(cd terrain_analysis && nix build "github:dimensionalOS/dimos-module-terrain-analysis/v0.1.1" --no-write-lock-file)
(cd path_follower    && nix build "github:dimensionalOS/dimos-module-path-follower/v0.2.0"    --no-write-lock-file)
(cd pgo/cpp          && nix build .#default --no-write-lock-file)
```

各モジュールの**自分のディレクトリ内**で実行すること（`result/` シンボリックリンクの位置を
`dimos/core/native_module.py` がモジュールファイル基準で解決するため）。
ビルド済みなら以後 `dimos run` にnixは不要。

## 3. G1 をバランス立ちさせる

コントローラーで: **L2+B → L2+Up →（立たせて）R2+A**
起動ログに `Current motion mode: ai` が出ればOK。立っていないと歩行コマンドは効かない。

## 4. 起動

```bash
source .venv/bin/activate
dimos run unitree-g1-nav-laptop
# 初回はLCM multicast / socket bufferのシステム設定を聞かれるので y
```

ネットワークは自動検出される。手動で上書きしたい場合：

```bash
ROBOT_INTERFACE=<NIC名> LIDAR_HOST_IP=<PCのIP> LIDAR_IP=192.168.123.120 dimos run unitree-g1-nav-laptop
```

**正常起動の目印（ログ）:**
- `FastLio2 network check passed host_ip=<PCのIP>`
- `Initializing DDS on interface: <NIC名>` → `Motion switcher initialized` → `G1 DDS SDK connection started`
- 起動直後の `No direct transform found between 'map' and 'body'` 警告は最初のスキャン処理までの一過性のもの

## 5. クリックで移動

dimosが**Rerun viewerを自動で開く**。3Dビューに点群と地図が出たら、床をクリック → 経路が引かれてG1が歩く。

viewerを別途開く場合（Python 3.13のwheelが無いので3.12指定が必須）：

```bash
uvx --python 3.12 dimos-viewer --connect rerun+http://localhost:9877/proxy --ws-url ws://localhost:3030/ws
```

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| `nix: not found (exit 127)` で全NativeModuleがクラッシュ | 手順1のprofileをsourceするか新ターミナル。または手順2の事前ビルド |
| `uvx dimos-viewer` が `cp313` ABIエラー | `--python 3.12` を付ける |
| 地図が出ない / 点群が来ない | `cat /sys/class/net/<NIC>/statistics/rx_bytes` を2回見て差分確認（Mid-360は約5MB/s）。0ならLiDARへの疎通(.120へping)と固定IPを確認 |
| クリックしても歩かない | ロボットがバランス立ちか確認（手順3）。ログの `MovementManager` が `Ignored out-of-range click` を出していないか確認 |
| viewerクラッシュ | ブループリント内 `vis_throttle=0.5` を 0.3 に下げる |

## 頭部カメラ（WebRTC）— ❌ 検証の結果、現行ファームでは不可（2026-06-05）

`unitree-g1-nav-laptop-cam` で実機検証した結果、**G1備え付けカメラはPC単体では取得できない**：

| 確認項目 | 結果 |
|---|---|
| WebRTCシグナリング | `.161:8081`（制御ボード、旧方式/offer）で接続成功。**NXの`.164`にはサービス無し** |
| 接続確立 | ICE/Peer/DataChannel全て🟢 |
| SDPの映像トラック | `m=video a=sendonly` あり（Go2共通ファームの名残） |
| **映像データ(RTP)** | **一切流れない**。`track.recv()`直叩きでもタイムアウト |

**根本原因:** 頭部D435iはNX（`.164`）にUSB接続。WebRTCサービスがいる制御ボード（`.161`）には
カメラの映像ソースが物理的に存在しない。
詳細: [FleetSeek exp_01KTBBHAV0QD2RR3XPMZJ8VZNR](https://web-ebon-zeta-33.vercel.app/experience/exp_01KTBBHAV0QD2RR3XPMZJ8VZNR)

**代替手段:**
1. **USBウェブカメラをPCに挿す**（推奨・ロボット側ゼロ変更）— 上流dimos自身がG1でこの構成
   （`uintree_g1_primitive_no_nav.py` の `Webcam(camera_index=0)`、コメント "height of camera on G1 robot"）。
   既にEthernetテザーがあるのでUSBケーブルを並走させてG1頭部にマウントすればよい
2. NX上に最小限の配信プロセス（GStreamer/RTSP か pyrealsense2+LCMの小スクリプト）を置く —
   「G1に何も入れない」方針を緩める場合のみ

`G1Connection.enable_video`（WebRTC用）はデフォルトFalseのまま残置。
補足: `UnitreeWebRTCConnection` はホスト不達時にタイムアウト無しで永久ブロックする
（connection refusedなら即失敗するので、`.161`へのpingが通る限り起動は固まらない）

## 頭部カメラ（ZMQ / UVC-on-NX）— ✅ これが動く方法（2026-06-06実機検証済み）

**NX上で配信スクリプトを1本動かす**（cv2/V4L2 `/dev/video4` → JPEG → msgpack → ZMQ `tcp://*:5555`、
NVIDIA GEAR-SONICスキーマ準拠）。スクリプトは `scripts/uvc_zmq_publisher.py`。

> ⚠️ 旧手順（`realsense_zmq_publisher.py` + `~/.venv_cam` + tmux）は現NXでは動かない：
> pyrealsense2 は librealsense2.so.2.56 欠落でimport不可、venvは存在せず、NXにtmuxも無い。
> cv2版はNXのシステムpython3（cv2/pyzmq/umsgpack同梱）でpipインストール不要。

### セットアップ（初回のみ・電源ONで自動起動になる）

```bash
# ラップトップから1コマンド（NXのパスワードを2回聞かれる: ssh/sudo）
scripts/install_nx_cam_service.sh
```

systemdサービス `g1-cam-publisher` として登録され、以後はG1の電源を入れるだけで配信が始まる。
D435iの認識前に起動しても `Restart=always` が3秒ごとに再試行する。

```bash
# PC側
dimos run unitree-g1-nav-laptop-cam     # ZMQ_CAMERA_HOST / ZMQ_CAMERA_PORT で上書き可
```

### 運用コマンド（NX上）

```bash
journalctl -u g1-cam-publisher -f       # ログ確認（送信fpsが5秒ごとに出る）
sudo systemctl restart g1-cam-publisher # 再起動
sudo systemctl stop g1-cam-publisher    # 停止
```

- PC側の受信は `ZmqCamera` モジュール（`dimos/robot/unitree/g1/camera/zmq_camera_module.py`）→ `/color_image`(LCM)。
  受信状況は起動ログに自己申告される（first frame / 5秒ごとfps / 無受信警告）
- 購読はZMQ CONFLATE（最新フレームのみ保持）— 遅い購読者でも映像遅延が蓄積しない（実機検証: 蓄積+7.96s/12s → ±0）
- 歩行コマンドはDDS一本のまま（カメラモジュールに制御経路なし）
- ⚠️ PCのwebcamを使う他のスタックと併用するとカメラ混流に注意（FleetSeek exp_01KQEZY97E）
- カメラ全経路のデバッグ記録: FleetSeek exp_01KTDF72WF4R211XYK1AW9VGNT

## 安全上の注意

- 最初のクリックは**1〜2m先**、コントローラーを手元に
- 有線テザー接続のままロボットが歩くので、ケーブルの取り回しに注意

## 記録

- FleetSeek (skill): [exp_01KTATP3CEFK59MMZ25XWTP81H](https://web-ebon-zeta-33.vercel.app/experience/exp_01KTATP3CEFK59MMZ25XWTP81H)
- FleetSeek (debug_note): [exp_01KTASFAEVY95HY2CFZ13VNWFE](https://web-ebon-zeta-33.vercel.app/experience/exp_01KTASFAEVY95HY2CFZ13VNWFE)
