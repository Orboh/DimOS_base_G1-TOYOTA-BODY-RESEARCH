# UMI Diffusion EE micro-adjustment — RUN guide (branch IK_dex1_umi)

IK で pre-grasp まで接近 → **UMI Diffusion policy が手首GoPro映像でEEを閉ループ微調整** →
収束したら `/g1/adjust_done` を発火。**グリッパー開閉はユーザーの別プログラム**（`/g1/adjust_done` を購読）。

構成: `[GoPro/UVC]→[umi_policy_server (umi conda env)]` ⇄ ZMQ ⇄ `[UmiDiffusionBridge (DimOS)]→arm_target`

---

## 0. 前提（このPC/RTX3070でセットアップ済み 2026-07-24）
- `umi` conda 環境: `oda/umi_diffusion/environment_g1.yaml` から構築済。**要固定**: `huggingface_hub==0.25.2`(timm0.9.7の`cached_download`), `wandb`(workspaceがimport), `pyzmq`/`msgpack`(IPC)。再構築時は `conda env create -f oda/umi_diffusion/environment_g1.yaml` 後に上記pipを確認。
- DimOS `.venv`: `pyzmq`/`msgpack` 追加済。
- DDS: `export CYCLONEDDS_HOME=~/cyclonedds-noshm; export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH`

---

## 1. オフライン検証（ロボット/GoPro不要・実施済 ✅）
```bash
# Step1: ckptロード+推論(88ms<100ms, action 9次元=pos+rot6d, gripper無)
conda run -n umi --no-capture-output python oda/umi_diffusion/smoke_policy.py
# Step3: IPC+9→6デコード(dummy-cam)。サーバをdummyで起動→クライアント
conda run -n umi --no-capture-output python oda/umi_diffusion/umi_policy_server.py --dummy-cam &
.venv/bin/python oda/umi_diffusion/smoke_ipc.py        # -> SMOKE_IPC OK
```

## 2. GoPro前処理の確定（実機GoPro必要・**未実施**）

**学習時の前処理は確定済み**（`inspect_dataset_frames.py` で `dataset.zarr` の `camera0_rgb` を抽出・目視）:
- **生の魚眼画像（樽型歪み）／魚眼補正なし** → `--fisheye` は**付けない**（既定）。
- `draw_predefined_mask(mirror=False, gripper=True, finger=False)` 適用 → **黒画素 約21%**（既定 `--no-mirror` 無しで一致）。
- RGB・224²・bgr→rgb。学習ドメインは明るい室内/白机（末尾フレームに緑オクラ2本）。
→ **`umi_policy_server` / `smoke_gopro.py` の既定フラグが学習と一致**。原則フラグ追加は不要で、以下は「一致の確認」作業。

### 手順
1. **学習フレーム抽出（実施済・再実行可, GoPro不要）**
   ```bash
   conda run -n umi python oda/umi_diffusion/inspect_dataset_frames.py
   # -> oda/umi_diffusion/train_frame_{00000,05705,11409}.png（比較の基準画像）
   ```
2. **GoPro接続 → デバイス特定**
   ```bash
   ls -l /dev/video*            # 接続前後の差分で特定
   v4l2-ctl --list-devices      # 名前で確認（無ければ: sudo apt install v4l-utils）
   v4l2-ctl -d /dev/videoN --list-formats-ext   # 対応解像度/FPS
   ```
   - GoProは**手持ち収集時と同じレンズ画角（Wide/ネイティブ魚眼）**にする。UVC(Webcam)モード or キャプチャカード経由。
   - ⚠️ **`/dev/videoN` の番号は再接続で入れ替わる。必ず `by-id` の固定パスを使う**（2026-07-29に
     ZED-M と Elgato が 4↔6 で実際に入れ替わり、ZED-M の映像を GoPro と誤認した）。
     ```bash
     ls -l /dev/v4l/by-id/          # 固定パス一覧（シリアル番号ベース）
     ```
     このラップトップの GoPro（HDMIキャプチャ）経路:
     ```
     /dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB3442072PE-video-index0
     ```
     （`-index1` はメタデータなので使わない。OpenCV は `cv2.VideoCapture` にこの
     シンボリックリンクのパス文字列を渡してそのまま開ける＝実測確認済み。）
   - ⚠️ **HDMIキャプチャ経由は「無信号でも黒フレームが正常に取れる」**：デバイスは open でき
     1920x1080 のフレームを返すが、GoPro が電源OFF / HDMI未接続 / HDMI出力OFF だと全画素0で
     `black-pixels=100.0%` になる（学習は約21%）。100%が出たらソフトではなく **GoPro側の給電・
     HDMI結線・HDMI出力設定** を確認する。
   - **HERO9 は本体にHDMI端子が無く Media Mod (ADFMD-001) が必須。** GoPro側の設定は
     UMI公式の正典手順に従う（README「Real-world Deployment > Hardware Setup > 4. Setup GoPro」）:
     GoPro Labs ファームを入れ、`~/umi/universal_manipulation_interface/assets/QR-MHDMI1mV0r27Tp60fWe0hS0sLcFg1dV.png`
     をカメラでスキャンしてクリーンHDMI出力に設定する（ファイル名のトークンが設定内容: `MHDMI1`=クリーンHDMI,
     `r27`=2.7K, `p60`=60fps, `fW`=Wide。学習時のintrinsics `gopro_intrinsics_2_7k.json` も
     2704x2028 / 59.94fps で一致）。
   - **アスペクトは 4:3 で出るのが正常**（実測: 1920x1080枠の中に左右240pxの黒帯＋実効1440x1080=4:3）。
     `get_image_transform` はアスペクト維持リサイズ＋中央クロップなので、この黒帯は224²に入らない。
     ライブは幅の56.1%（=実効4:3幅の74.8%）を切り出し、学習は2704幅の75.0%を切り出すので**相対クロップが一致**する。
3. **重ね比較（キモ）**
   ```bash
   # --cam-device の既定が上記 by-id パスなので、通常は引数不要
   conda run -n umi python oda/umi_diffusion/smoke_gopro.py
   # -> oda/umi_diffusion/gopro_vs_train.png  [左=ライブ前処理 | 右=学習フレーム]
   #    oda/umi_diffusion/gopro_live_224.png  （ライブ224²単体）

   # ライブ版（構え・画角合わせ用。[生 | 前処理224 | 学習] を実時間表示, s=保存 q=終了）
   conda run -n umi python oda/umi_diffusion/preview_gopro.py
   ```
   `gopro_vs_train.png` を開いて **3点を照合**:
   - **魚眼の曲がり方 / 像円の有無**: 直線（机縁・柱）が左右で同じように湾曲するか。ライブが直線的なら GoPro が Linear モード → **Wide/魚眼に変更**。
     定量判定: **学習フレームは上隅がビネットで潰れている**（24px角パッチで mean≈12 / max≈23-32。明るい照明下でこの暗さ＝魚眼の像円の外）。
     ライブ側の上隅に普通に被写体が写っている（例: mean 108 / max 129）なら **像円が見えていない＝画角が学習より狭い** → GoPro側のレンズ設定/レンズMod不一致。
   - **黒マスクの位置**: マスク(黒)が**ライブ画像で実際にグリッパーが写る領域を覆っているか**。ズレていれば実機dex1-1のGoPro取付が手持ち収集時と違う（=視覚ドメインシフト）→ マウント位置を合わせる。
   - **明るさ/色/RGB**: 植物や肌が自然色か（色が反転なら bgr/rgb 問題）。`black-pixels ≈ 21%` がログに出るか。
4. **不一致時のみ**フラグ調整（通常不要）:
   - どうしても曲率が合わない（学習が実は補正済み等の疑い）→ `--fisheye --sim-fov <FOV>`（intrinsics: `~/umi/.../example/calibration/gopro_intrinsics_2_7k.json`）を試し再比較。
   - ミラー処理差 → `--no-mirror`。
   確定したフラグは **§3 のサーバ起動にも同じものを付ける**。
5. 一致OK → GoProの **by-id パス**と（もしあれば）確定フラグを控え、§3へ。

## 3. 起動（2プロセス）
> ⚠️ **以下はそのままコピペして動くコマンドだけを書いてある。**
> このファイル中の `[ ]` や `<...>` は「省略可」「値を入れる場所」を表す**記法**なので、
> 打ち込んではいけない。特に `<FOV>` は bash のリダイレクト演算子と解釈され、
> `bash: FOV: No such file or directory` で python が起動すらしない。

```bash
# (1) 推論サーバ（umi env・先に起動）
#     --cam-device / --ckpt はいずれも既定値でよい（既定 = by-id Elgatoパス、~/umi/epoch=0110-*.ckpt）
conda run -n umi --no-capture-output python oda/umi_diffusion/umi_policy_server.py

# (2) DimOSアプリ（別ターミナル, リポジトリroot = DimOS_oda/ で実行）
#     `dimos` は .venv 内の実行ファイル。素の `dimos` は conda (base) の PATH に無く
#     `dimos: command not found` になるので、.venv/bin/dimos を直接叩く（activate不要:
#     shebangが venv の python を絶対パスで指しているため）。
export CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib
.venv/bin/dimos run unitree-g1-okra-ik-diffusion          # DRY-RUN（arm動かない）
IK_REACH_LIVE=1 .venv/bin/dimos run unitree-g1-okra-ik-diffusion   # LIVE（arm駆動）
```
§2 の照合で不一致があった場合のみ、(1) に確定したフラグを足す（角括弧は書かない）:
```bash
# 例: 魚眼補正を入れる場合。<FOV> は実数に置き換える
conda run -n umi --no-capture-output python oda/umi_diffusion/umi_policy_server.py \
  --fisheye --sim-fov 120 --no-mirror
```

ビューアでオクラをクリック → IK が standoff(0.05m)手前まで接近 → reach_done → **Diffusionが微調整** → 収束で `/g1/adjust_done`。

## 4. グリッパー（ユーザーの別プログラム）
本アプリはグリッパーに触れない。別プログラムで `/g1/adjust_done`(Bool, LCM) を購読し、Dex1(`rt/dex1/left/cmd`)を閉じる。
（参考: `oda/gripper_pinch.py` / `oda/gripper_open.py` が直接DDSの最小例）

---

## 主要な環境変数（新規分）
| 変数 | 既定 | 意味 |
|---|---|---|
| `UMI_SERVER_ADDR` | `tcp://127.0.0.1:5599` | 推論サーバ |
| `UMI_CONTROL_HZ` | `10.0` | 閉ループ制御周波数 |
| `UMI_N_EXEC_PER_INFER` | `2` | 1推論あたり実行ウェイポイント数（88msレイテンシに対する余裕確保） |
| `UMI_PREDICT_TIMEOUT_MS` | `300` | 1リクエストの締切。**バッテリー駆動時は推論440〜500msでこれを全リクエスト超過**する（下記「既知の留意点」）→ 電源を繋ぐか値を上げる |
| `UMI_POSITION_ONLY` | `1` | v1=位置のみ追従(安全)。`0`で6-DOF(姿勢も追従) |
| `UMI_CONVERGE_EPS_M` | `0.004` | 収束判定: 連続する指令 tip の距離がこれ未満なら settled |
| `UMI_CONVERGE_HOLD_TICKS` | `8` | settled が連続これ回で `adjust_done` 発火 |
| `OKRA_STANDOFF_M` | `0.05` | IKが手前で止まる距離（Diffusionの微調整余地） |
| `OKRA_UMI_TIP_OFFSET_XYZ` | =`OKRA_TIP_OFFSET_XYZ` | **Step6**: UMI TCP点に一致させる tip frame |

### 推論ログ（ブリッジ側）
| 変数 | 既定 | 意味 |
|---|---|---|
| `UMI_LOG_EVERY_N` | `1` | `adj` 行を N tick ごとに出す（1=毎tick）。ログを絞るなら `5` |
| `UMI_LOG_JOINTS` | `1` | `adj` 行に `q_meas` / `q_sol` / `Δq(deg)` ブロックを含める |
| `UMI_LOG_CHUNK_MAX` | `4` | 1推論あたり print する waypoint 数（`-1`=全部, `0`=出さない） |
| `UMI_TRACE_PATH` | `auto` | JSONL トレース。`auto` = run ログdir の `umi_diffusion_trace.jsonl`、`""` で無効 |

### 推論ログ（サーバ側 CLI）
| オプション | 既定 | 意味 |
|---|---|---|
| `--log-every N` | `1` | N リクエストごとに診断行（obs / cam_age / infer_ms / 生9次元）|
| `--quiet` | off | リクエスト行を一切出さない |
| `--trace PATH` | off | リクエストごとに生 `(N,9)` net 出力 + デコード後 `(N,6)` を JSONL |

## 4.5 ログの読み方（「UMIが何を推論したか」の追い方）

3階層。すべて実測値（オフライン dummy-cam 検証 2026-07-30）。

**① 推論ごと — サーバに送った観測と、返ってきた chunk 全体**
```
[DRY] umi-infer[ep1 tick2] obs tip(torso)[ 0.114 -0.264 -0.261] aa[-0.24 1.114 -0.097]
    q_meas=[ 0.246 -0.215 -0.001  0.949 -0.064 -0.079 -0.002]
    server n=16 infer=440.8ms exec=2 reset=False
    chunk Δtip(torso,mm) #0[-0.9 +0.8 -0.5]=1.3 #1[+1.7 +1.4 +0.1]=2.2 #2[+3.6 +1.1 +1.1]=4.0
                         #3[+5.7 +2.0 +2.5]=6.6 … #15[+19.0 +13.6 +12.7]=26.6 span=27.1
```
- `obs tip / q_meas`: ポリシーに渡した実測 EE と実測7関節。**ここが reach 姿勢でなければ座標以前の問題**。
- `n=16 exec=2`: 16本返ってきて先頭2本だけ実行（`UMI_N_EXEC_PER_INFER`）。残り14本も③のトレースに全部残る。
- `chunk Δ`: 各 waypoint が実測 tip から**どれだけ動かそうとしているか**(mm)。
  `#0` がほぼ 0 で `#15` が 20〜30 mm = 「ゆっくり寄せる」正常な chunk。**全部 0 なら推論が動きを出していない。**

**② tick ごと — 推論した関節データ**
```
[DRY] adj[1] tick=3 wp0/2of16 t=0.92s tgt(torso)[ 0.113 -0.264 -0.261] tip(torso)[ 0.113 -0.264 -0.261]
    q_meas =[ 0.246 -0.215 -0.001  0.949 -0.064 -0.079 -0.002]
    q_sol  =[ 0.247 -0.213 -0.     0.95  -0.064 -0.079 -0.001]
    Δq(deg)=[ 0.08  0.08  0.04  0.03 -0.    0.01  0.03] worst=right_shoulder_pitch_joint 0.1°
    conv=True err=0.0001 step=2.8mm track=1.3mm settled=2/8
```
- `q_sol` = **IKが解いて `arm_target` に載る7関節指令**（LIVEなら実際に出る値）。`Δq` は実測との差[deg]。
- `tgt` vs `tip`: 指令 waypoint と、その `q_sol` の FK tip。乖離が大きい = IK が届いていない。
- `step`: 連続する**指令**間距離（収束判定に使う値）。`track`: 指令 vs 実測（PDたわみ。判定には使わない）。

**③ エピソード終了 — 「調整できたのか」の一行**
```
UmiDiffusionBridge[1] episode END reason=converged dur=2.38s ticks=9 infers=5
    infer_ms avg=465.2 max=475.9 timeouts=0 skips{ws=0 ik=0 delta=0 lim=0}
    tip(torso) start[ 0.116 -0.265 -0.262] end[ 0.114 -0.264 -0.261] net=2.0mm path=21.5mm
```
`reason` は `converged` / `max_duration` / `server_misses` / `state_stale` / `no_state` / `stopped` / `exception`。

**失敗シグネチャ**
| 症状 | 読み方 |
|---|---|
| `net`≈0 かつ `path` 大 | 前後に震えているだけ。寄せていない（上の例も静止腕なので net 2 mm） |
| `chunk Δ` が全部 ≈0 | ポリシーが動きを出していない（obs/前処理/reset を疑う） |
| `skips{ws=…}` が大きい | workspace box で全弾き。`ticks=0` なら1本も実行していない |
| `timeouts` > 0 | `UMI_PREDICT_TIMEOUT_MS` < 実 `infer_ms`。サーバ側 `req#` 行と突き合わせ |
| サーバ側 `cam_age` 大 | GoPro/Elgato が止まっている（ブリッジ側からは絶対に見えない） |
| サーバ側 `pose_buf=1` | pose履歴1本で horizon-2 obs を組んでいる（エピソード先頭のみ正常） |

**④ トレース（機械可読）** — 既定 `~/.dimos/logs/<run>/umi_diffusion_trace.jsonl`（起動ログに実パスが出る）。
`kind` は `infer`(chunk全16本) / `exec`(q_sol・err・step) / `skip` / `end`。
```bash
python -c "
import json,collections,numpy as np
rows=[json.loads(l) for l in open('umi_diffusion_trace.jsonl')]
print(collections.Counter(r['kind'] for r in rows))
print(np.array([r['q_sol'] for r in rows if r['kind']=='exec']).round(3))"
```

## 5. 座標系整合（Step 6・実機校正・**未実施**）
`~/umi/okra_20260723_ishimaru/dataset_plan.pkl` の `grippers[0].tcp_pose`（手持ちSLAM追跡のTCP軌跡）が示す
TCP点/姿勢軸に、G1 FK tip frame（`OKRA_UMI_TIP_OFFSET_XYZ`）を一致させる。相対表現ゆえ絶対world不要だが
**tip点がズレると相対回転がレバーアーム誤差を生む**。不明箇所はteleop replayで経験的整合。

## 段階的ロールアウト & 安全
1. **DRY-RUN**（`arm_target`ログ健全性・突進/限界超え無し・収束で`adjust_done`発火）
2. **LIVE + `UMI_POSITION_ONLY=1`**（位置のみ追従で挙動確認）
3. **LIVE + `UMI_POSITION_ONLY=0`**（6-DOF姿勢追従）
- ⚠️ **Ctrl-Cクリーン停止は不安定**。リモートE-stop **L2+B**(damping)を常備、復帰 L2+UP。abort時はその場保持（突進しない設計）。
- サーバtimeout時はその場保持→連続10回で自動abort（`adjust_done`は撃たない＝安全側）。

## 既知の留意点
- action は **9次元（pos3+rot6d, gripper無）**。学習データ`gripper_width`が定数0.0001（死）のため。→ グリッパーは別プログラム担当（本設計の前提）。
- obs前処理で `feature_aggregation(attention_pool_2d) is ignored -> CLS token` の警告。学習時も同一timm0.9.7で出ていたはずで整合（要最終確認）。
- 推論88ms（このPC・**AC電源時**）。10Hz予算100ms内だが余裕小 → `UMI_N_EXEC_PER_INFER>=2` 既定。厳しければ Orin 移設（サーバ側だけ移動）。
- ⚠️ **バッテリー駆動だと推論 440〜500 ms**（実測 2026-07-30、dummy-cam）。RTX3070 が負荷時に 0.8〜1.1 GHz へ落ち
  電力も約30Wで張り付く（AC時のブーストは 1.9 GHz 超）。既定 `predict_timeout_ms=300` を**全リクエストが超過**するため、
  10回連続ミスでエピソードが `reason=server_misses` で abort（腕は保持・`adjust_done` は撃たない）。
  **走らせる前に AC を繋ぐ**こと。繋げないなら `UMI_PREDICT_TIMEOUT_MS=800` 等に上げる（10Hzは出ない＝約2Hz動作）。
- ゲートで全 waypoint が弾かれ続けると `max_duration_s` に到達せず永久ループになるバグがあった（2026-07-30 修正）。
  現在は経過時間をループ先頭で判定するので、`skips` だけのエピソードも 30 秒で `reason=max_duration` として終わる。
