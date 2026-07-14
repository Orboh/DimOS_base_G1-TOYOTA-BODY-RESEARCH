# G1 オクラ収穫 sim 環境 チームセットアップ手順

別メンバーが自分の PC で、知能センター内に収穫構成 G1（左腕バスケット＋右手 Dex1）が立つ
Isaac Sim 環境を再現するための通し手順。詳細・つまずき集は [`sim-setup-notes.md`](../sim-setup-notes.md) を参照。

> ⚠️ **USD アセットは git に入っていません**（`.gitignore` で `usd_file/` 除外、サイズ大）。
> コードは git、**アセットは S3** から取得する 2 系統構成です。

---

## 0. 前提

- Ubuntu 22.04、NVIDIA GPU **VRAM 8GB 以上**（RTX 2000 Ada 8GB で実用確認済）、ドライバ + CUDA
- `conda`（miniconda 等）、`aws` CLI（`aws configure` 済 / orboh アカウントにアクセス可）
- ディスク空き ~20GB（Isaac Sim 本体 10–15GB + アセット ~70MB）

## 1. リポジトリ取得

```bash
git clone <repo> dimos-hackathon
cd dimos-hackathon
git checkout feat/g1-okra-sim-verification
```

## 2. Isaac Sim 環境（conda env `isaac-sim`）

詳細は [`sim-setup-notes.md` §3](../sim-setup-notes.md)。要点のみ:

```bash
# 専用 env（普段の .venv/bin/dimos とは別。py3.10）
conda create -n isaac-sim python=3.10 -y && conda activate isaac-sim
pip install --upgrade pip

# Isaac Sim 4.5.0 本体（~10-15GB DL）
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com

# ★重要★ user-site 汚染で torch 等がスキップされ起動失敗するので明示導入（§6）
pip install scipy psutil pyyaml pillow boto3 gymnasium
pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu118
```

実行時は必ず **`PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES`** を付ける（user-site 汚染の遮断 + EULA 同意）。

## 3. USD アセットを S3 から取得（★git に無い）

```bash
# 知能センター室 + オクラ畑 + 収穫構成G1 + バスケット を usd_file/ にミラー展開
# 巨大なフル解像度畑(okra_field_full.usd 1.15GB, 16GB+ GPU機専用)は既定で除外。
aws s3 sync s3://orboh-datasets/g1-okra-sim/usd_file/ usd_file/ --exclude "okra_field_full.usd"
# フル解像度版が要る場合のみ個別取得:
#   aws s3 cp s3://orboh-datasets/g1-okra-sim/usd_file/okra_field_full.usd usd_file/
```

取得物（`usd_file/`）:
| ファイル | 中身 |
|---|---|
| `chinou_center.usd` | FARO 実測の知能センター室（8.24×8.40×2.73m, 床z=0, 床/壁コライダー + 照明 + 床に floor_mat） |
| `okra_field.usd` | FARO 実測のオクラ畑（鹿児島県農業総合開発センター, 13.14×7.60×2.30m, 床z=0, **屋外＝地面コライダーのみ + Dome+Sun照明**, 頂点色, 2M面） |
| `okra_field_full.usd` | 上記の**間引きなしフル解像度版**（63M面 / 1.15GB）。⚠️ **8GB機では開けない・16GB+ GPU機用**。通常は 2M 版を使う。詳細 [`OKRA_FIELD.md`](OKRA_FIELD.md) |
| `g1-29dof-dex1-base-fix-usd/g1bag.usd` | 収穫構成 G1（左手首バスケット直付け / 右手 Dex1 / 足・指に物理マテリアル / GroundPlane無効） |
| `g1-29dof-dex1-base-fix-usd/basket_physics.usd` | バスケット形状（g1bag が相対参照。**同ディレクトリ必須**） |
| `usd_file/basket.3mf` | バスケット元データ（mm, 3MF） |
| `okra.usd` | 把持対象オクラ（10cm/12g, RigidBody+convexHull+grippy素材 okra_mat） |
| `usd_file/Okra01.3mf` | オクラ元データ（mm, 3MF スライサープロジェクト） |

> g1bag.usd は `./basket_physics.usd` を相対パスで参照するので、**2ファイルは必ず同じフォルダに**置くこと（sync すれば自動でそうなる）。

## 4. 起動

```bash
cd dimos-hackathon
PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
  ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/view_chinou.py --gui
```

- 知能センター室に G1 が床に接地して立ち、左手首にバスケット、右手 Dex1、天井照明 3×3 が点く。
- 視点操作（Omniverse）: 右ドラッグ=見回す / Alt+左ドラッグ=旋回 / 中ドラッグ=パン / ホイール=ズーム。
- フラグ: `--gravity`（通常重力で動力学）/ `--ceil-lights N` `--ceil-intensity V`（天井照明）/ なしで重力OFF・初期姿勢固定表示。

**オクラ畑を見る場合**（同じ `view_chinou.py` を環境変数で畑に向ける。畑は Sun+Dome 焼き込み済なので天井灯 OFF）:
```bash
PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES ROOM_USD="$PWD/usd_file/okra_field.usd" \
  ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/view_chinou.py --gui --ceil-lights 0
```

## 5. （任意）物理マテリアルを作り直す場合

物理マテリアル（摩擦・弾性）は **配布 USD に焼き込み済み**。値を変えたい/再生成したい時のみ:

```bash
# pxr(usd-core) があれば isaac-sim env でなくても可
~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/setup_physics_materials.py
```
摩擦・弾性の値はスクリプト冒頭の定数（`FLOOR`/`FOOT`/`GRIP`）で調整。床×5・足×2・Dex1指×7 に bind。

---

## 補足

- アセットを更新したら S3 を更新: `aws s3 sync usd_file/ s3://orboh-datasets/g1-okra-sim/usd_file/ --exclude "*.bak" --exclude "*-bak"`
- 軽い解析（USD構造ダンプ・身長計測・配置確認の3D描画）は Isaac Sim を起動せず
  `pip install usd-core numpy matplotlib` した別 venv の `Usd.Stage.Open` で十分速い。
- 詳細・既知の落とし穴（user-site 汚染 / room スケール / DDS over tailscale 等）は [`sim-setup-notes.md`](../sim-setup-notes.md)。
