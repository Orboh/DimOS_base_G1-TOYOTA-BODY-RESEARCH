# ZED 3D検出を別PCのブラウザで見る（視聴者向け）

Jetson（`jetson-orin-dimos`）にUSB直結した ZED の映像を、rerun のWebビューアで
ライブ表示するための手順です。**カラー画像・3D点群・オクラ検出の3D位置**が同時に見られます。

> 見るだけなら **SSH もトンネルも不要**。Tailscale に入っていれば、ブラウザで URL を開くだけです。

---

## 前提

1. **orboh.com の Tailscale tailnet に参加している**こと
   （参加していないと Jetson の `100.113.43.64` に到達できません。管理者に招待を依頼してください）
2. **Jetson 上で配信が動いている**こと
   （誰か一人が起動していれば、複数人が同時に見られます → [配信の起動](#配信の起動配信担当者向け)）

## 見る手順

ブラウザ（Chrome 推奨）で次の URL を開くだけです:

```
http://100.113.43.64:9090/?url=rerun%2Bhttp://100.113.43.64:9876/proxy
```

- `+` は必ず `%2B` にエンコードしてください（素の `+` はクエリで空白化して繋がりません）。
- 数秒で、左に3D点群・右に ZED カメラ映像（オクラ検出に赤点＋距離ラベル）が表示されます。

## うまく映らないとき

| 症状 | 対処 |
|---|---|
| rerun のUIは出るが映像・点群が出ない | **ページをリロード**。配信が起動中の時間帯にタブを開くと空ソースに固着します |
| 左上 **Sources** に `okra_zed_live_local` が出ない | 配信が動いていません。配信担当者に起動を依頼 |
| そもそもページが開かない | Tailscale が繋がっているか確認（`tailscale status` に `jetson-orin-dimos` が見えるか） |
| 映像が止まって見える | タイムライン右端の再生（▶）が最新フレームに追従しているか確認 |

---

## 配信の起動（配信担当者向け）

配信を立てる/止める人だけ SSH が必要です（鍵 `~/.ssh/id_ed25519_g1`）。

```bash
# Jetson に SSH
ssh -i ~/.ssh/id_ed25519_g1 tbr@100.113.43.64

# 配信を起動（torch が CPU なので YOLO 負荷を落とした引数）
cd /mnt/ssd/workspace/DimOS_base_G1-TOYOTA-BODY-
setsid nohup ./.venv/bin/python scripts/zed_rerun_stream_local.py 3 6 768 0.25 NEURAL \
  < /dev/null > /tmp/zed_stream.log 2>&1 &
```

スクリプト実体: `scripts/zed_rerun_stream_local.py`

引数: `[stride] [yolo_every] [imgsz] [conf] [depth]`。上の例の `3 6 768 0.25 NEURAL` は
**Jetson の CPU torch 向けに負荷を落とした値**で、スクリプト既定値（`3 2 1280 0.25 NEURAL`）とは異なります。
GPU があるマシンなら既定値のままで構いません。NEURAL 深度に失敗すると PERFORMANCE に自動フォールバックします。

### YOLO 重みの入手

重みは `data/models_yolo/okra11n-seg.pt`（単一クラス okra）を読みます。
**この重みは git には入っていません**（`data/.lfs/models_yolo.tar.gz` は LFS ポインタのみで、
Orboh フォークには blob が無く `git lfs pull` は 404）。Hugging Face から取得してください。

```bash
hf download Kota0612/okra11n-seg-v5 model/okra11n-seg.pt --local-dir /tmp/okra-w
mkdir -p data/models_yolo && mv /tmp/okra-w/model/okra11n-seg.pt data/models_yolo/
```

配布元: [Kota0612/okra11n-seg-v5](https://huggingface.co/Kota0612/okra11n-seg-v5)（public / ultralytics / 学習データセット同梱）

### 停止

```bash
# ★ ローカルシェルから実行すること（下記の「罠」を参照）
ssh -i ~/.ssh/id_ed25519_g1 tbr@100.113.43.64 'pkill -f zed_rerun_stream_local.py'
```

### 罠: `pkill -f` を SSH の一行コマンドの中で使わない

`pkill -f zed_rerun_stream` を、それを含む SSH 一行コマンドの中に書くと、`-f` が
**送り込んだコマンド文字列自身**にマッチして自分のシェルごと殺します（起動が
`exit 255`・無出力で失敗する原因）。停止はローカルシェルから単独で `pkill` するか、
PID を調べて `kill` してください。

### 補足

- ポートは `0.0.0.0` 待受・CORS `*` なので、tailnet 内なら誰でも直接ブラウザで見られます（トンネル不要）。
- ZED は Jetson 上の 1 プロセスが 1 回だけ開くので、**視聴者が増えてもカメラ競合は起きません**。
- スクリプトは SSD（`scripts/`）に置いてあり、再起動で消えません（旧版は `/tmp` にあり揮発していました）。
- Jetson の torch は CPU 版のため、本体 venv のままでは YOLO が低速です。GPU 化は下の「GPU 推論サービス（Jetson）」を参照（`libcudss.so.0` は欠落しておらず、ラッパーが `LD_LIBRARY_PATH` を解決します）。

---

## GPU 推論サービス（Jetson）

Jetson の dimos venv は CPU 版 torch が入るため、重い推論はそのままでは GPU に載りません。
**重い依存だけを別 venv に隔離して ZMQ で繋ぐ**構成を使います（`scripts/act_service.py` と同型）。

```
dimos 本体 venv (3.12, CPU torch)
   │  ZED深度・IK・3D化・アプリ全体
   └──ZMQ(msgpack)──▶ /mnt/ssd/yolo_gpu_venv (3.10, CUDA torch)
                        YOLO-seg 2D検出＋mask のみ
```

### 起動

**必ず `run_yolo_service.sh` 経由で起動してください。** 素の `python yolo_service.py` は
`ImportError: libcudss.so.0` で落ちます。

```bash
bash scripts/run_yolo_service.sh --selftest --model data/models_yolo/okra11n-seg.pt  # 速度自己測定
bash scripts/run_yolo_service.sh --model data/models_yolo/okra11n-seg.pt             # serve
```

ラッパーが `LD_LIBRARY_PATH` を設定します（システムの CUDA 12.6 cublas + venv の cudss）。

> [!NOTE]
> `libcudss.so.0` は **欠落していません**（`.../nvidia/cu12/lib/` に存在）。
> リンカのパスに入っていないだけで、ラッパーがそれを解決します。
> 以前この補足に「`libcudss.so.0` 欠落の修理が必要」と書かれていましたが誤りでした
> （2026-07-29 に AGX Orin 実機で確認・訂正）。

### 実測値（2026-07-29, AGX Orin, JetPack R36.5 / CUDA 12.6）

```
[yolo_service] model=okra11n-seg.pt device=0 cuda=True Orin
[selftest] 1280x720 det=0 infer 90.4 ms (11.1 FPS)
```

環境: torch 2.11.0 (CUDA 12.6) / ultralytics 8.4.75 / Python 3.10。

> [!WARNING]
> 導入時のコミットメッセージには「CPU 367ms → GPU 30ms（約12倍）」とありますが、
> 上記条件では **30ms は再現せず 90.4ms** でした。当時の測定条件（解像度・モデル・
> ウォームアップ有無）が記録されていないため比較できません。速度が要件になる場合は
> 使う条件で測り直してください。

### ワイヤプロトコル（ZMQ REP, `tcp://127.0.0.1:5702`）

```
request  : {"image_jpeg": <jpeg bytes>, "conf": 0.5, "iou": 0.6,
            "classes": ["okra"], "reset": <bool>}     # conf 以降は任意
response : {"width": W, "height": H,
            "detections": [{"name", "class_id", "confidence", "track_id",
                            "bbox": [x1,y1,x2,y2], "mask_polygon": [[x,y],...]}]}
```

3D 化はしません（ZED 深度を持つのは dimos 本体側）。2D 検出＋mask 輪郭までを返します。

---
