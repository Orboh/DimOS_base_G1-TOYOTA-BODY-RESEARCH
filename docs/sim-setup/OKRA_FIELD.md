# オクラ畑 実環境（FARO スキャン → okra_field.usd）

> 2026-07-06 / Kota 依頼。新規 3D スキャン（鹿児島県農業総合開発センターのオクラ畑）を
> 知能センター室（[`CHINOU_ROOM.md`](/docs/sim-setup/CHINOU_ROOM.md)）と**同じパイプライン**で Isaac Sim に取り込む。

## 結論（先に）

- 元データ = FARO Focus スキャン **`20260702__鹿児島県農業総合開発センター`（OBJ, 単位 mm, 頂点色 `v x y z r g b`）**。
  構成は知能センターと同一（`OBJ_10/` 10%間引き ~386MB / `OBJ_100/` フル ~4.1GB）。
- 実寸 = **13.14 × 7.60 × 2.48m(H)**（footprint 13.1×7.6m の畝、支柱・オクラ列で高さ ~2.3m）。
- **知能センターの変換ロジック（`convert_chinou.py`）がそのまま流用可能**。up軸自動判定（最小スパン軸=Z=2.48m）も正しく縦を取る。
- 屋内との唯一の本質差 = **屋外なので壁/天井が無い** → 物理/照明ステップだけ屋外版に差し替え。
- → **`usd_file/okra_field.usd`（40.7MB / 1.18M頂点 / 2.0M面 / 頂点色, bbox 13.14×7.60×2.30m・地面 z=0・XY中心）**。

## 屋内(chinou)との違い＝ここだけ差し替えた

| 工程 | 知能センター（屋内） | オクラ畑（屋外） |
|---|---|---|
| OBJ→USD 変換 | `convert_chinou.py` | **同じ**（`MESH_PATH=/World/OkraField` を渡すだけ） |
| 床/コライダー/照明 | `build_chinou_phys.py`（床+**4枚壁**box + Dome/Distant） | **`build_field_phys.py`**（**地面box のみ・壁なし** + Dome + **Sun**） |
| 床高さ検出 | 決め打ちバンド 0.45–0.54m | **z ヒストグラムで支配的スラブを自動検出**（どの畑でも通す） |
| 確認ローダー | `view_chinou.py`（既定 chinou） | **同じ**（`ROOM_USD=okra_field.usd` + `--ceil-lights 0`） |

## パイプライン（AWS EC2 `orboh-sim` g5.12xlarge `i-0db430a2eff6cd78c`）

知能センターと同一。EC2 は変換のみ（8GB ラップトップ回避）、物理/照明はローカル usd-core。

```bash
# 1) 元データ(OBJ_10)を S3 へ（laptop）
aws s3 cp --recursive "20260702__鹿児島県農業総合開発センター/OBJ_10/" \
  s3://orboh-datasets/okra_field/OBJ_10/
# 2) 共有SGへ laptop IP 許可（ユーザーが ! 実行）＋ EC2 起動
aws ec2 authorize-security-group-ingress --group-id sg-0cbc698627eeb3da2 \
  --protocol tcp --port 22 --cidr <laptop-ip>/32
aws ec2 start-instances --instance-ids i-0db430a2eff6cd78c
# 3) EC2: 既存 chinou_venv を再利用（trimesh/open3d/usd-core/fast-simplification/scipy 済）
ssh -i ~/.ssh/orboh-sim-key.pem ubuntu@<ip>
aws s3 cp --recursive s3://orboh-datasets/okra_field/OBJ_10/ ~/field/OBJ_10/
IN_DIR=~/field/OBJ_10 OUT_USD=~/field/okra_field.usd TARGET_TRIS=2000000 \
  MESH_PATH=/World/OkraField ~/chinou_venv/bin/python convert_chinou.py
# 4) 回収（laptop へ scp）＋ EC2 停止（★課金 $5.67/h、必ず止める）
scp -i ~/.ssh/orboh-sim-key.pem ubuntu@<ip>:~/field/okra_field.usd usd_file/okra_field.usd
aws ec2 stop-instances --instance-ids i-0db430a2eff6cd78c

# 5) ローカルで 地面=z0 + 地面コライダー + 太陽光 を焼き込み（GPU不要, usd-core）
SRC=usd_file/okra_field.usd python docs/sim-setup/build_field_phys.py

# 6) Isaac Sim で確認（G1 を畑に置いてスケール目視・スクショ）
PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES ROOM_USD="$PWD/usd_file/okra_field.usd" \
  ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/view_chinou.py --gui --ceil-lights 0

# 7) 配布用 S3 へ（チームは aws s3 sync で取得）
aws s3 cp usd_file/okra_field.usd s3://orboh-datasets/g1-okra-sim/usd_file/okra_field.usd
```

## 実測ログ（2026-07-06 実施）

- native bbox span (mm) = [13140.5, 7602.0, 2483.7]、min z ≈ 52715mm（原点から ~52m オフセット、FARO 座標系）。
- up-axis(min span) = **Z (2.484m)** ＝知能センターと同じ FARO 軸規約 → 軸入替なし。
- decimation: 5,659,498 → 2,000,000 tris（fast-simplification、頂点色を cKDTree 最近傍で再マップ）。
- 地面自動検出: **z=0.178m** の密な水平スラブ（peak bin n=32019, band n=140979）を地面と判定 → −0.178m シフトで地面 z=0。
  その下 0.178m はスキャンノイズ（知能センターの反射ゴースト床 0.496m と同種）。
- 確認: Isaac Sim で world size (13.139, 7.6, 2.482)m、G1 lift +0.792m で地面に接地（base z 発散なし）。
  スクショ `docs/sim-setup/okra_field_scale_check.png`（G1 ~1.27m が ~2m のオクラ列に対し妥当な比率）。

## フル解像度版（間引きなし, 2026-07-06 追加）

デフォルトの `okra_field.usd`（2M面）とは別に、**OBJ_100 を間引きなし（`TARGET_TRIS=0`）で変換した高精細版** `okra_field_full.usd` も生成済み。

- **V=32,431,119 / F=63,033,985（63M面）/ USD 1.15GB**、頂点色付き、地面 z=0（自動検出 0.177m シフト）・footprint 13.14×7.60m・高さ2.31m。
- 物理/照明も 2M 版と同じく焼き込み済（地面コライダーのみ + Dome+Sun）。**焼き込みも EC2 で実施**（1.15GB を 8GB 機で再ロードするのは重いため。EC2 ロールは S3 read-only なので S3 配置は laptop 経由）。
- 配布 = `s3://orboh-datasets/g1-okra-sim/usd_file/okra_field_full.usd`。ローカルは `usd_file/okra_field_full.usd`。
- 手順は 2M 版と同一。変換で `TARGET_TRIS=0`、焼き込みで `SRC=...okra_field_full.usd` を指定するだけ。**フル版の変換/焼き込みは EC2 側で回すこと**（ローカルは非推奨）。
- ⚠️ **8GB ラップトップの Isaac Sim では開けない見込み**（2M 版で VRAM ピーク 3.1GB → 63M 面は桁違い）。**16GB+ の GPU ワークステーション**で開く前提。通常の確認・開発は 2M 版 `okra_field.usd` を使う。

## 残・注意

- **屋外なので壁コライダーは無い**。畝の外へ出さないナビ制約が要るなら、`build_field_phys.py` に
  footprint 外周ボックスを追加するか、ナビ側の境界で担保する。
- 地面コライダーは footprint 全体 + 2m マージンの平面ボックス（畝の凹凸は視覚メッシュのみ）。
  足配置で畝の起伏を踏ませたいなら低ポリ三角コライダーを別途。
- テクスチャは頂点色（スキャン RGB）のみ。フォトリアルが要れば OBJ_100 + UV テクスチャで再生成（大RAM機）。
- 元 OBJ は S3 `s3://orboh-datasets/okra_field/OBJ_10/` に温存。OBJ_100 は未変換。
