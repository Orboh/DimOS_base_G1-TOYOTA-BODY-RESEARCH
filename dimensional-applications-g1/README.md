# Unitree G1 × DimOS — 自然言語で動かす

自然言語で G1 を動かすための起動グル(shell + docker)です。

- `.venv/` → リポジトリ直下の `.venv` への symlink（追跡していないので初回に作成:
  `ln -s ../.venv dimensional-applications-g1/.venv`）
- Docker image: `go2-agentic-gpu:latest` — タグ名は既存のビルド済みイメージ /
  配布 tar をそのまま使えるように残しています(中身は機体非依存)。
  ビルド定義は `docker-gpu/` に同梱。

## 使う blueprint

- `unitree-g1-agentic` / `unitree-g1-agentic-sim`(このディレクトリの既定)
- ほかにも dimos には `unitree-g1-basic`, `unitree-g1-joystick`, `unitree-g1-coordinator`,
  `unitree-g1-detection`, `unitree-g1-full`, `unitree-g1-shm` 等多数。

## セットアップ

1. `../.env` に `ROBOT_IP_G1=<G1の IP>` を追加
2. PC を G1 と同 LAN に置く

## 起動 — venv 直叩き

ターミナル A(エージェント起動):

```bash
./run_robot.sh           # 実機
# または
./run_sim.sh             # シミュレーション
```

ターミナル B(言語入力):

```bash
.venv/bin/dimos humancli            # Textual TUI(推奨)
# または
./say.sh "立って"
./say.sh "右手を上げて"
```

停止:

```bash
.venv/bin/dimos stop
```

## 起動 — Docker

```bash
./docker-gpu/run.sh                  # dimos run unitree-g1-agentic
./docker-gpu/run.sh humancli         # TUI 直結
./docker-gpu/run.sh shell            # 中で何でも
```

コンテナ名は `go2-agentic-gpu-g1`。`--network host` + LCM の都合で、同じ
イメージのコンテナを複数同時に起動することはできません。

## 注意

- G1 は二足歩行なので、四足機の sport コマンド相当はありません。
  `UnitreeG1SkillContainer` が公開するスキルは `stand_up`, `bow`, `wave_hand` 系。
- システムプロンプトは blueprint 側に同梱されているので、humancli 起動時に
  自動で G1 用のものに切り替わります。
- 初回接続で WebRTC のエラーが出たら `dimos/robot/unitree/g1/connection.py` を参照。
