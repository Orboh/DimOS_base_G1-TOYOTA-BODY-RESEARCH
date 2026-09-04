# G1 × DimOS agentic blueprint — GPU portable container

`onnxruntime-gpu` 経由で perception の onnx 推論を GPU に逃がして、laptop 上で
CPU 飽和→徐々に重くなる問題を緩和する GPU コンテナです。

- **ベースイメージ** = `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04`。
  `--gpus all` を付けて起動する必要があり、ホスト側に NVIDIA Container Toolkit
  が要ります。
- **`requirements.txt`** は `onnxruntime`(CPU)を除外し `onnxruntime-gpu==1.26.0`
  のみを入れた状態のスナップショット。
  `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` の回避策はそのまま効きます。
- イメージのタグ名 `go2-agentic-gpu` は、既存のビルド済みイメージ / 配布 tar を
  そのまま使えるように残した歴史的名称です(中身は機体非依存)。

## 前提

ホスト側に以下が要ります:

- NVIDIA driver(CUDA 12.x 以上対応のもの — 通常 530 以降)
- Docker Engine
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  - `nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` 済み
- 動作確認: `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi`

## 起動 — 開発マシン上でビルドする場合

```bash
# .env(リポジトリ直下)に ROBOT_IP / OPENAI_API_KEY をセットしておく
./docker-gpu/run.sh                # agentic blueprint 起動
# 別ターミナルで:
./docker-gpu/run.sh humancli       # Textual TUI で自然言語入力
# あるいは:
./docker-gpu/say.sh "前に進んで"   # 単発コマンド送信
```

初回ビルドは ~15-30 分かかります(CUDA + cuDNN + torch + onnxruntime-gpu 等で
イメージは 10 GB 超になります)。

## 配布(tar)

`go2-agentic-gpu.tar.gz`(7.4 GB, gzip -1, sha256: `add3fa0971267f93618421616e2ce0328a6c4f568ef7dee111285a31eca0f489`)を同梱しています。配布先での復元:

```bash
gunzip -c go2-agentic-gpu.tar.gz | docker load
docker images | grep go2-agentic-gpu

# .env を用意したうえで:
docker run --rm -it \
    --network host --cap-add NET_ADMIN --gpus all \
    --env-file /path/to/.env \
    --name go2-agentic-gpu \
    go2-agentic-gpu:latest
```

humancli を使いたいときは `docker run ... go2-agentic-gpu:latest dimos humancli`。
既定の CMD は `dimos run unitree-g1-agentic` です。

## トラブルシュート

- `WARN: no NVIDIA GPU visible in the container.` が出る
  → `--gpus all` を渡してない or NVIDIA Container Toolkit が未設定。
- `RuntimeError: ... libcudart ... not found`
  → ベースイメージの CUDA バージョンとホスト driver の不一致。
  ホスト driver を上げるか、Dockerfile の `nvidia/cuda:12.6.3-cudnn-runtime-*`
  を 11.x や 12.4 に下げる。
- `Failed to get the position of the robot.`
  → SLAM の odom が収束する前に `relative_move` を呼んだ。少し待つ。
