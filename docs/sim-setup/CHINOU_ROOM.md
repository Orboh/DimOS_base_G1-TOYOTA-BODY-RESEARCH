# 知能センター実環境 room（FARO スキャン → chinou_center.usd）

> 2026-06-25 / Kota 依頼。Isaac Sim の部屋が「G1 に対して小さすぎる」件の調査と、実環境の正しいスケールでの USD 生成。

## 結論（先に）

- **`usd_file/room.usd` は FARO スキャンとは無関係なダミー部屋**だった（Sketchfab 由来・手組み、作成者 `saumy`、WhatsApp 写真テクスチャ、~3×3m・壁 0.74m）。埋め込み文字列に `chinou/知能/focus/20260618` 参照ゼロで確定。
- 実環境の元データは **FARO Focus スキャン `20260618_Focus測定_知能センター`（OBJ, 単位 mm）**。真の実寸 = **8.24 × 8.40 × 3.22m(H)**。
- 「縮尺バグ」ではなく **別アセット（ダミー小部屋）が読まれていた**のが「小さい」の真因。壁 0.74m は G1(1.27m)の腰以下＝極端に小さく見えた。
- → FARO スキャンを正しいスケールで USD 化 = **`usd_file/chinou_center.usd`（39MB / 1.04M頂点 / 2.0M面 / 頂点色付き、bbox 8.24×8.40×3.22m・床 z=0・XY中心）**。

## 元データ

USB `/.../20260618_Focus測定_知能センター/`:
- `OBJ_100/` … フル解像度（数億面 / 計~10GB）
- `OBJ_10/`  … 90%間引き（計~886MB / 14.76M面）← 変換に使用（8mの部屋には十分高密度）
- 単位 = **mm**（頂点 `v x y z r g b`、座標が数千〜28000、Z≈25118–28342mm が床〜天井）

## 変換の要点（寸法を壊さない3条件）

1. **mm→m: ×0.001 を頂点にベイク**（USD の `metersPerUnit=1.0` のまま。metersPerUnit は参照時に自動適用されないため、ベイクが安全）。
2. **recenter**: 最小スパン軸（Z=天井 3.22m）を鉛直とし、床を z=0、XY を原点中心へ。
3. **up軸 = Z**（最小スパン軸＝天井高）。

これを外すと: ×0.001 忘れ→8m が 8000m、二重掛け→0.008m、recenter 忘れ→原点から ~25m 離れて G1 が室外。

## パイプライン（AWS EC2 `orboh-sim` g5.12xlarge）

ローカル 8GB を避けて実施。EC2 ロールは **S3 read のみ（PutObject 不可）**＝結果は scp で laptop へ直接回収。

```bash
# 1) 元データを S3 へ（laptop）
aws s3 cp --recursive "OBJ_10/" s3://orboh-datasets/chinou_center/OBJ_10/
# 2) EC2 起動・SSH（共有 SG に laptop IP 許可が必要。SSM 未登録）
aws ec2 start-instances --instance-ids i-0db430a2eff6cd78c
# 3) EC2: venv + deps
python3 -m venv ~/chinou_venv
~/chinou_venv/bin/pip install trimesh numpy open3d usd-core fast-simplification scipy
aws s3 cp --recursive s3://orboh-datasets/chinou_center/OBJ_10/ ~/chinou/OBJ_10/
# 4) 変換（detached 推奨。run_convert.sh が convert.log に集約）
IN_DIR=~/chinou/OBJ_10 OUT_USD=~/chinou/chinou_center.usd TARGET_TRIS=2000000 \
  ~/chinou_venv/bin/python convert_chinou.py
# 5) 回収（laptop へ直接 scp）＋ EC2 停止
scp ubuntu@<ip>:~/chinou/chinou_center.usd usd_file/chinou_center.usd
aws ec2 stop-instances --instance-ids i-0db430a2eff6cd78c
```

スクリプト: `convert_chinou.py`（このディレクトリ）。decimation は **fast-simplification（主）/ open3d vertex-clustering（予備）**、色は scipy cKDTree 最近傍で再マップ。

## ハマりどころ

- **真のクラッシュ原因 = `Gf.Vec3f(*np_arr.astype(np.float32))`**（numpy.float32 を受けず Boost.Python ArgumentError）。→ `Gf.Vec3f(float(x),float(y),float(z))`。当初 open3d quadric を疑ったが無実。
- detached 起動は `setsid ... & disown; exit 0` 経由 ssh が rc=255 で不安定 → **plain ssh を local background で保持**＋`run_convert.sh`→`convert.log` 集約が安定。
- EC2 SSH は共有 SG への laptop IP 追加が必要（auto モードは共有インフラ変更を拒否＝ユーザーが `!` 実行）。

## 使い方 / 残作業

- **確認ローダー**: `view_chinou.py` … `open_stage(chinou_center.usd)` + ground plane(z=0) + G1 配置 + スクショ。
  ```bash
  PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
    ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/view_chinou.py [--gui]
  ```
- **コリジョン**: chinou_center.usd は視覚メッシュのみ。G1 が立つには床コライダー（ローダーが ground plane を z=0 に追加）。壁・什器との衝突が要るナビ検証では静的 triangle collider 付与 or 壁の簡易ボックス化（次段）。
- **テクスチャ**: 現状は頂点色（スキャン RGB）のみ。フォトリアルが要るなら UV+テクスチャ変換を別途。
- **フル解像度**: 必要なら OBJ_100 から同パイプラインで再生成（EC2 大RAM機）。
- `sim_smoke_test.py` を chinou_center に切替えるなら `ROOM_USD` 差し替え＋spawn 姿勢を room 中心基準で再調整。

## 11. 床補正 + コライダー + ライト 焼き込み（2026-06-25, `build_chinou_phys.py`）

### 床が 0.49m ズレていた問題
当初 recenter は「Z 最小値=0」にしていたが、Z 分布解析で **真の床は z≈0.496m**（密な水平面 42万頂点）と判明。最小値側(z=0〜0.49)は**光沢テラゾー床の反射ゴースト/ノイズ**(疎)で、実床より下。このため ground plane(z=0) が実床より 0.49m 下＝**G1 の足が約0.49m 埋もれていた**。

### 対処（ローカル usd-core で再生成。GPU/EC2 不要）
`docs/sim-setup/build_chinou_phys.py`:
1. 既存 USD の幾何を読み、**支配的水平スラブの中央値=床(0.496m)を検出 → 全頂点を −0.496m シフト**して床を z=0 に。
2. **不可視 static コライダー**を追加: 床ボックス(上面 z=0) + 4枚壁ボックス(室外周, 高さ ~2.8m)。`UsdPhysics.CollisionAPI`、剛体なし=動かない衝突体。視覚はスキャンメッシュ、物理はクリーンなボックス（スキャンノイズのスパイクに引っかからない）。
3. **DomeLight(1000) + DistantLight(2500)** 追加、メッシュ `doubleSided=true`(裏面の暗さ解消)。
4. `chinou_center.usd` を上書き(視覚色は保持)。視覚のみ版は S3 とローカル .bak に退避。

### 結果 / 注意
- 床 z=0・天井 ~2.73m(室内高)。bbox min z=-0.49 はゴースト層(床コライダー下に隠れる)。
- `view_chinou.py` は ground plane 追加を停止(USD 内蔵床を使用)。
- **G1 USD は `fix_base:true`** で base 固定＝落下しないので床コライダーの動的検証は別途(floating-base 版 or 落下物)で。視覚上は足が床面に一致。
- 壁は bbox 外周のボックス近似(「室外に出ない」ナビ用)。什器との厳密衝突が要るなら低ポリ三角コライダーを追加。
