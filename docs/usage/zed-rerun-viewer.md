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

配布元: [`Kota0612/okra11n-seg-v5`](https://huggingface.co/Kota0612/okra11n-seg-v5)（public / ultralytics / 学習データセット同梱）

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
- Jetson の torch は CPU 版のため YOLO は低速です。GPU 化は別課題（`libcudss.so.0` 欠落の修理が必要）。
