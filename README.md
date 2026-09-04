<div align="center">

<img width="1000" alt="banner_bordered_trimmed" src="https://github.com/user-attachments/assets/64f13b39-da06-4f58-add0-cfc44f04db4e" />

<h2>The Agentive Operating System for Physical Space</h2>

[![Discord](https://img.shields.io/discord/1341146487186391173?style=flat-square&logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/dimos)
[![Stars](https://img.shields.io/github/stars/dimensionalOS/dimos?style=flat-square)](https://github.com/dimensionalOS/dimos/stargazers)
[![Forks](https://img.shields.io/github/forks/dimensionalOS/dimos?style=flat-square)](https://github.com/dimensionalOS/dimos/fork)
[![Contributors](https://img.shields.io/github/contributors/dimensionalOS/dimos?style=flat-square)](https://github.com/dimensionalOS/dimos/graphs/contributors)
![Nix](https://img.shields.io/badge/Nix-flakes-5277C3?style=flat-square&logo=NixOS&logoColor=white)
![NixOS](https://img.shields.io/badge/NixOS-supported-5277C3?style=flat-square&logo=NixOS&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-supported-76B900?style=flat-square&logo=nvidia&logoColor=white)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)

<a href="https://trendshift.io/repositories/23169" target="_blank"><img src="https://trendshift.io/api/badge/repositories/23169" alt="dimensionalOS%2Fdimos | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<big><big>

[Hardware](#hardware) •
[Installation](#installation) •
[Agent CLI & MCP](#agent-cli-and-mcp) •
[Blueprints](#blueprints) •
[Development](#development)

⚠️ **Pre-Release Beta** ⚠️

</big></big>

</div>

# Intro

Dimensional is the modern operating system for generalist robotics. We are setting the next-generation SDK standard, integrating with the majority of robot manufacturers.

With a simple install and no ROS required, build physical applications entirely in python that run on any humanoid, quadruped, or drone.

Dimensional is agent native -- "vibecode" your robots in natural language and build (local & hosted) multi-agent systems that work seamlessly with your hardware. Agents run as native modules — subscribing to any embedded stream, from perception (lidar, camera) and spatial memory down to control loops and motor drivers.
<table>
  <tr>
    <td align="center" width="50%">
      <a href="docs/capabilities/navigation/native/index.md"><img src="assets/readme/navigation.gif" alt="Navigation" width="100%"></a>
    </td>
    <td align="center" width="50%">
      <img src="assets/readme/perception.png" alt="Perception" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <h3><a href="docs/capabilities/navigation/native/index.md">Navigation and Mapping</a></h3>
      SLAM, dynamic obstacle avoidance, route planning, and autonomous exploration — via both DimOS native and ROS<br><a href="https://x.com/stash_pomichter/status/2010471593806545367">Watch video</a>
    </td>
    <td align="center" width="50%">
      <h3>Perception</h3>
      Detectors, 3d projections, VLMs, Audio processing
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="docs/capabilities/agents/readme.md"><img src="assets/readme/agentic_control.gif" alt="Agents" width="100%"></a>
    </td>
    <td align="center" width="50%">
      <img src="assets/readme/spatial_memory.gif" alt="Spatial Memory" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <h3><a href="docs/capabilities/agents/readme.md">Agentive Control, MCP</a></h3>
      "hey Robot, go find the kitchen"<br><a href="https://x.com/stash_pomichter/status/2015912688854200322">Watch video</a>
    </td>
    <td align="center" width="50%">
      <h3>Spatial Memory</a></h3>
      Spatio-temporal RAG, Dynamic memory, Object localization and permanence<br><a href="https://x.com/stash_pomichter/status/1980741077205414328">Watch video</a>
    </td>
  </tr>
</table>


# Hardware

<table>
  <tr>
    <td align="center" width="50%">
      <h3>Humanoid</h3>
      <img width="245" height="1" src="assets/readme/spacer.png">
    </td>
    <td align="center" width="50%">
      <h3>Misc</h3>
      <img width="245" height="1" src="assets/readme/spacer.png">
    </td>
  </tr>

  <tr>
    <td align="center" width="50%">
      🟨 <a href="docs/platforms/humanoid/g1/index.md">Unitree G1</a><br>
      🟨 <a href="docs/platforms/humanoid/g1/index_orboh_make.md">Unitree G1 (Orboh セットアップ)</a><br>
    </td>
    <td align="center" width="50%">
      🟥 <a href="https://github.com/dimensionalOS/openFT-sensor">Force Torque Sensor</a><br>
    </td>
  </tr>
</table>
<br>
<div align="right">
🟩 stable 🟨 beta 🟧 alpha 🟥 experimental

</div>

> [!IMPORTANT]
> 🤖 Direct your favorite Agent (OpenClaw, Claude Code, etc.) to [AGENTS.md](AGENTS.md) and our [CLI and MCP](#agent-cli-and-mcp) interfaces to start building powerful Dimensional applications.

# Installation

## Interactive Install

```sh skip
curl -fsSL https://raw.githubusercontent.com/dimensionalOS/dimos/main/scripts/install.sh | bash
```

> See [`scripts/install.sh --help`](scripts/install.sh) for non-interactive and advanced options.

## Manual System Install

To set up your system dependencies, follow one of these guides:

- 🟩 [Ubuntu 22.04 / 24.04](docs/installation/ubuntu.md)
- 🟩 [NixOS / General Linux](docs/installation/nix.md)
- 🟧 [macOS](docs/installation/osx.md)

> Full system requirements, tested configs, and dependency tiers: [docs/requirements.md](docs/requirements.md)

## Python Install

### Quickstart

```bash
uv venv --python "3.12"
source .venv/bin/activate
uv pip install 'dimos[base,unitree]'

# Run the humanoid stack in simulation (no hardware needed)
# NOTE: First run downloads the MuJoCo scene from LFS
dimos --simulation run unitree-g1-sim
```

```bash
# Install with simulation support
uv pip install 'dimos[base,unitree,sim]'

# Humanoid in MuJoCo simulation
dimos --simulation run unitree-g1-sim

# Humanoid + LLM agent + MCP server in simulation
dimos --simulation run unitree-g1-agentic-sim
```

```bash
# Control a real robot (Unitree G1 over WebRTC)
export ROBOT_IP=<YOUR_ROBOT_IP>
dimos run unitree-g1-basic
```

# Featured Runfiles

| Run command | What it does |
|-------------|-------------|
| `dimos --simulation run unitree-g1-sim` | Humanoid in MuJoCo simulation |
| `dimos --simulation run unitree-g1-agentic-sim` | Humanoid agentic + MCP server in simulation |
| `dimos run unitree-g1-okra-ik-only-grasp` | Okra harvesting — click a point, IK reach, grasp |
| `dimos run unitree-g1-okra-ik-only-grasp-zed` | Okra harvesting — ZED + YOLO detection instead of clicking |
| `dimos run unitree-g1-okra-ik-diffusion` | Okra harvesting — UMI diffusion policy |
| `dimos run unitree-g1-mid360-fastlio` | Mid-360 LiDAR + FastLIO odometry |
| `dimos run unitree-g1-nav-laptop` | G1 nav stack driven from the laptop |
| `dimos run demo-camera` | Webcam demo — no hardware needed |

> Full blueprint docs: [docs/usage/blueprints.md](docs/usage/blueprints.md)

# Agent CLI and MCP

The `dimos` CLI manages the full lifecycle — run blueprints, inspect state, interact with agents, and call skills via MCP.

```bash
dimos run unitree-g1-agentic --daemon    # Start in background
dimos status                              # Check what's running
dimos log -f                              # Follow logs
dimos agent-send "explore the room"       # Send agent a command
dimos mcp list-tools                      # List available MCP skills
dimos mcp call relative_move --arg forward=0.5  # Call a skill directly
dimos stop                                # Shut down
```

> Full CLI reference: [docs/usage/cli.md](docs/usage/cli.md)


# Usage

## Use DimOS as a Library

See below a simple robot connection module that sends streams of continuous `cmd_vel` to the robot and receives `color_image` to a simple `Listener` module. DimOS Modules are subsystems on a robot that communicate with other modules using standardized messages.

```py skip
import threading, time, numpy as np
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs import Twist
from dimos.msgs.sensor_msgs import Image, ImageFormat

class RobotConnection(Module):
    cmd_vel: In[Twist]
    color_image: Out[Image]

    @rpc
    def start(self):
        threading.Thread(target=self._image_loop, daemon=True).start()

    def _image_loop(self):
        while True:
            img = Image.from_numpy(
                np.zeros((120, 160, 3), np.uint8),
                format=ImageFormat.RGB,
                frame_id="camera_optical",
            )
            self.color_image.publish(img)
            time.sleep(0.2)

class Listener(Module):
    color_image: In[Image]

    @rpc
    def start(self):
        self.color_image.subscribe(lambda img: print(f"image {img.width}x{img.height}"))

if __name__ == "__main__":
    autoconnect(
        RobotConnection.blueprint(),
        Listener.blueprint(),
    ).build().loop()
```

## Blueprints

Blueprints are instructions for how to construct and wire modules. We compose them with
`autoconnect(...)`, which connects streams by `(name, type)` and returns a `Blueprint`.

Blueprints can be composed, remapped, and have transports overridden if `autoconnect()` fails due to conflicting variable names or `In[]` and `Out[]` message types.

A blueprint example that connects the image stream from a robot to an MCP-backed LLM agent for reasoning and action execution.
```py skip
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs import Image
from dimos.robot.unitree.g1.connection import G1Connection
from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer

blueprint = autoconnect(
    G1Connection.blueprint(),
    McpServer.blueprint(),
    McpClient.blueprint(),
).transports({("color_image", Image): LCMTransport("/color_image", Image)})

# Run the blueprint
if __name__ == "__main__":
    blueprint.build().loop()
```

## Library API

- [Modules](docs/usage/modules.md)
- [LCM](docs/usage/lcm.md)
- [Blueprints](docs/usage/blueprints.md)
- [Transports](docs/usage/transports/index.md) — LCM, SHM, DDS, ROS 2
- [Data Streams](docs/usage/data_streams/README.md)
- [Configuration](docs/usage/configuration.md)
- [Visualization](docs/usage/visualization.md)

## Demos

<img src="assets/readme/dimos_demo.gif" alt="DimOS Demo" width="100%">

## G1 初期セットアップ（Orboh — 新しい個体・NX再フラッシュ後は必須）

> ⚠️ **G1の頭部カメラ配信は、ロボットのオンボードコンピュータ（NX, `192.168.123.164`）への
> ワンタイムセットアップが必要です。** 新しいG1個体を触るとき、またはNXが再フラッシュされた後は、
> SSH鍵・カメラ配信サービスがすべて消えているため、以下を一度実行してください：

```bash
# ラップトップから1コマンド（NXのパスワードを聞かれる。デフォルト: 123）
scripts/install_nx_cam_service.sh
```

これで `g1-cam-publisher` がsystemdサービスとして登録され、**以後はG1の電源を入れるだけで
頭部カメラ（D435i → ZMQ `tcp://*:5555`）が自動配信されます**（実機で再起動2回検証済み 2026-06-06）。

- 動作確認: `dimos run unitree-g1-nav-laptop-cam` → Rerun viewerのCameraパネルに映像
- NX側ログ: `ssh unitree@192.168.123.164 journalctl -u g1-cam-publisher -f`
- 詳細・トラブルシュート: `docs/platforms/humanoid/g1/index_orboh_make.md`
- 任意（SSH快適化）: `ssh-copy-id -i ~/.ssh/id_ed25519_g1.pub unitree@192.168.123.164`

## JetsonをWiFi AP化する手順（Orboh — 現場で無線直結したい時）

**目的**: JetsonにノートPCを直接無線接続してSSHしたい場合、Jetson自身をWiFiアクセスポイント(AP)にする。
社内WiFi / DHCPに依存せず、APモード時のJetsonは常に固定IP `192.168.12.1`。
再起動後も自動でAPが立つよう常駐化する。OSSの [`oblique/create_ap`](https://github.com/oblique/create_ap) を使用。

> [!WARNING]
> **APと通常WiFi（STA）は同時使用不可。** WiFiチップは1個なので、AP化するとそのJetsonは
> **インターネットに繋がらなくなる**（社内WiFi経由のネットを失う）。

> [!WARNING]
> **作業中のロックアウトに注意。** WiFi経由でSSH中にWiFiをAPに切り替えると接続が切れる。
> **WiFi以外の入口（有線 or モニタ直結）を必ず確保してから作業すること。**
> AGX Orinには有線LANポートがあるので、下記「有線直結」を先に済ませるのが最も安全。

### 1. インストール（Jetson上）

```bash
git clone https://github.com/oblique/create_ap && cd create_ap
sudo make install
sudo apt install -y hostapd dnsmasq network-manager
```

> [!NOTE]
> `apt install` 時に `hostapd.service failed to start` と出るのは**無害**。
> create_ap は独自のプロセスとして hostapd を起動するため、systemd サービスとして上がる必要はない。

### 2. WiFiインターフェース名とAPモード対応を確認

```bash
iw dev | grep Interface                          # AGX Orin: wlP1p1s0 / NX: wlan0
iw list | grep -A8 "Supported interface modes"   # "* AP" があればOK
```

### 3. `/etc/create_ap.conf` を編集

以下は SSID `agx` / パスワード `agx12345` の例（AGX Orin, IF名 `wlP1p1s0`）。
IF名は手順2で確認した値に合わせること。

```bash
sudo sed -i \
  -e 's/^WIFI_IFACE=.*/WIFI_IFACE=wlP1p1s0/' \
  -e 's/^SSID=.*/SSID=agx/' \
  -e 's/^PASSPHRASE=.*/PASSPHRASE=agx12345/' \
  -e 's/^GATEWAY=.*/GATEWAY=192.168.12.1/' \
  -e 's/^SHARE_METHOD=.*/SHARE_METHOD=none/' \
  -e 's/^NO_VIRT=.*/NO_VIRT=1/' \
  -e 's/^INTERNET_IFACE=.*/INTERNET_IFACE=/' \
  /etc/create_ap.conf
```

- `SHARE_METHOD=none` — インターネット共有なし
- `NO_VIRT=1` — AP仮想IF非対応アダプタ向け（安全側）
- `INTERNET_IFACE=` — 空のまま

### 4. WiFiを切断（STA/AP同時不可）

```bash
sudo nmcli device disconnect wlP1p1s0   # NXの場合は wlan0
```

STAのまま create_ap を起動しようとすると `can not be a station and an AP at the same time` エラーが出る。

### 5. NMが起動時にWiFiを先取りしないよう全wifiプロファイルの自動接続をOFF

```bash
for c in $(nmcli -t -f NAME,TYPE connection show | grep ":802-11-wireless$" | cut -d: -f1); do
  sudo nmcli connection modify "$c" connection.autoconnect no
done
```

### 6. create_ap を有効化・起動

```bash
sudo systemctl enable --now create_ap
```

再起動後も自動でAPが立ち上がる。

### 接続方法

ノートPCのWiFiを `agx`（パスワード `agx12345`）に繋ぎ、SSHする。

```bash
ssh tbr@192.168.12.1    # AGX Orin の例。ユーザー名は各機体に合わせる
```

### 有線直結（強く推奨 — ロックアウト防止・救出用）

AGX Orinの有線 `eno1` には NetworkManager プロファイル「Wired connection 1」で
**静的 `192.168.123.222`** が設定済み（再起動後も維持）。

```bash
# ノートPC側NICを 192.168.123.50/24 に設定してから:
ssh tbr@192.168.123.222    # WiFiの状態に関わらず常に到達できる
```

- AP化作業中はこの有線でSSHしながら作業すると安全
- AP切り替え失敗時の救出にも使える

### インターネットが必要になったとき（AP ⇄ 通常WiFi切替）

```bash
# 一時的に社内WiFiに戻す（ネット復活。DHCPでIPは変わる）
sudo systemctl stop create_ap
sudo nmcli connection up <wifi-profile-name>

# APに戻す
sudo systemctl start create_ap
```

完全にAPをやめる場合:

```bash
sudo systemctl disable create_ap
# wifiプロファイルの autoconnect を yes に戻す
for c in $(nmcli -t -f NAME,TYPE connection show | grep ":802-11-wireless$" | cut -d: -f1); do
  sudo nmcli connection modify "$c" connection.autoconnect yes
done
```

### 動作確認（実機検証済み 2026-06-10）

AGX Orinを電源OFF→ONしても `create_ap` が自動起動することを確認済み。

```bash
systemctl is-active create_ap    # → active
systemctl is-enabled create_ap   # → enabled
ip addr show wlP1p1s0            # → inet 192.168.12.1/24 が割り当て済み
```

`agx` SSID が信号強度100で発信されていること、ノートPCから `ssh tbr@192.168.12.1` で到達できることを確認済み。

# Development

## Develop on DimOS

```sh skip
export GIT_LFS_SKIP_SMUDGE=1
git clone https://github.com/dimensionalOS/dimos.git
cd dimos

# Run the default test suite (uv run syncs deps on demand; --all-groups
# only needed for self-hosted tests / mypy — see docs/development/testing.md)
uv run pytest --numprocesses=auto dimos
```


## Multi Language Support

Python is our glue and prototyping language, but we support many languages via LCM interop.

Check the language interop examples upstream:
- [C++](https://github.com/dimensionalOS/dimos/tree/main/examples/language-interop/cpp)
- [Lua](https://github.com/dimensionalOS/dimos/tree/main/examples/language-interop/lua)
- [TypeScript](https://github.com/dimensionalOS/dimos/tree/main/examples/language-interop/ts)
