# 遠隔アクセス & sim-in-the-loop 再接続ガイド（明日用クイックリファレンス）

> Jetson(dimos) ⇄ 手元PC(Isaac Sim) を Tailscale で繋いで sim を駆動する手順。
> 2026-06-25 に DERP relay 越し（完全リモート相当）でも動作確認済み。

---

## 0. 構成の地図（誰がどこ）

| ノード | Tailscale IP | 何が動く | 内部の繋ぎ方 |
|---|---|---|---|
| **Jetson**(AGX Orin リュック, `jetson-orin-dimos`) | **100.113.43.64** | dimos（収穫制御, DDS発行） | SSH user `tbr` / 鍵 `~/.ssh/id_ed25519_g1` |
| **手元PC**(`kotaueda-thinkpad-p16-gen-2`) | **100.100.126.61** | Isaac Sim + DDSブリッジ | conda env `isaac-sim` |

- どちらも tailnet **orboh.com**（login `koutaueda@orboh.com`）。
- 経路: 同一LAN/有線があれば直結(~2ms)、無ければ自動で **DERP relay(~30-120ms)**。どちらでもsimは動く。

---

## 1. 明日まずやること: Jetson が生きているか確認

手元PCがどこにあっても（別ネットでもOK。PCにネットさえあれば）:
```bash
tailscale status | grep jetson      # "jetson-orin-dimos ... active/idle" なら生存（offline なら §4）
ping 100.113.43.64                  # 到達確認
ssh tbr@100.113.43.64 'whoami'      # → tbr が返ればSSH-OK（鍵で無パスワード）
```
> `ssh tbr@100.113.43.64` が確実（IP直）。エイリアス `ssh jetson-ts` は HostName が古い tailnet 名で解決しない恐れ。

**前提**: Jetson は電源ON＋`orboh` WiFi 圏内であること（Jetsonの唯一の uplink）。`create_ap` は無効化済み・`orboh` 自動接続・tailscaled 自動起動なので、**再起動しても自動復帰**する。

---

## 2. Jetson 内のファイルを編集する（遠隔）

そのまま SSH で編集できる（tailscale 越しで確認済み）:
```bash
ssh tbr@100.113.43.64                       # 入ってから vim 等
# or 1行
ssh tbr@100.113.43.64 'sed -i ... <file>'
# ファイル送受信
scp -i ~/.ssh/id_ed25519_g1 <local> tbr@100.113.43.64:/path/   # 送る
scp -i ~/.ssh/id_ed25519_g1 tbr@100.113.43.64:/path/ <local>   # 取る
```
> `sudo` は `tbr` のパスワード必須・非対話SSH不可 → sudo が要る操作は Jetson の実端末で。

---

## 3. sim-in-the-loop を起動する（Jetson dimos → Isaac Sim）

### 3-1. 手元PC: Isaac Sim DDS ブリッジを起動
```bash
cd ~/Desktop/dimos-hackathon
# headless（数値確認）。GUIで見るなら SIM_HEADLESS=0 と DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority を付ける
PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=/home/kota-ueda/Desktop/unitree_sdk2_python \
  SIM_DDS_IFACE=tailscale0 SIM_DDS_PEERS=100.113.43.64 SIM_HEADLESS=1 SIM_LOAD_ROOM=1 \
  ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/sim_dds_bridge.py
# 終了: 別端末で touch /tmp/sim_bridge_stop（または Ctrl-C）
```

### 3-2. Jetson: arm 指令を発行（今は dimos の代用＝試験送信機）
```bash
scp -i ~/.ssh/id_ed25519_g1 docs/sim-setup/dimos_style_pub.py tbr@100.113.43.64:/tmp/   # 初回のみ
ssh tbr@100.113.43.64 'SIM_DDS_IFACE=tailscale0 SIM_DDS_PEERS=100.100.126.61 PUB_SECS=15 \
  /home/tbr/workspace_ssd/unitree_mujoco/.venv/bin/python /tmp/dimos_style_pub.py'
```
→ 手元PCのブリッジに `cmds_rx` が増え、仮想G1の右腕が動く。

### 3-3. 実 dimos（okra blueprint）を sim に向ける場合
- Jetson で起動時に **`DIMOS_DDS_PEERS=100.100.126.61`（手元PCのTS IP）と `ROBOT_INTERFACE=tailscale0`** を設定すれば、
  既に入れた `dds_init.py` パッチが unicast 化する（env 未設定なら通常運用に無影響）。
- ただし full okra blueprint は ZED/ACT/moondream（実機・実画像依存）が必須で sim 単独起動は不可。
  sim 検証は「知覚を外した arm-only / cmd_vel の最小構成」を別途用意する必要がある（→ `sim-setup-notes.md` §9）。
- 前提: Jetson dimos venv(py3.12) に現状 unitree_sdk2py / 機能する cyclonedds が無い（要導入）。

---

## 4. Jetson が offline のとき（復旧）

| 症状 | 対処（Jetson 実端末 or 圏内で） |
|---|---|
| `tailscale status` に jetson が出ない/offline | Jetson が電源ON・`orboh`圏内か確認。`sudo nmcli connection up orboh` で uplink 再取得 → `tailscale status` 確認 |
| ping 通るが SSH が固まる | Tailscale SSH の check が再有効化されてないか。`sudo tailscale set --ssh=false`（実端末で） |
| WiFi 繋いだのにネット無し | IP/GW 未適用の既知症状 → `sudo nmcli connection up orboh`（IP 192.168.0.x と default route が入る） |

---

## 5. 既知のハマり（DDS over tailscale）

- **cyclonedds-python の `Domain` は参照保持必須**（`dom = Domain(...)`）。GCされると discovery が落ちる（ブリッジ/試験送信機は対策済み）。
- **orboh AP は client isolation** → 同一WiFiでも端末間 DDS 不可。tailscale が唯一の経路（DERP relay でも sim は動く）。
- **unitree_sdk2py は CYCLONEDDS_URI 無視**。tailscale unicast 化は config テンプレ `ChannelConfigHasInterface` に `<Peers>` 注入（dimos 側は `dds_init.py` パッチ＋`DIMOS_DDS_PEERS`、ブリッジ側は raw cyclonedds で対応済み）。

---

## 6. 関連ファイル
- ブリッジ: `docs/sim-setup/sim_dds_bridge.py`
- 試験送信機: `docs/sim-setup/dds_test_pub.py`（raw cyclonedds）/ `dimos_style_pub.py`（unitree_sdk2py＝dimos流）
- 詳細メモ: `docs/sim-setup-notes.md`（Isaac Sim 構築 §1-8、DDSブリッジ §9）
