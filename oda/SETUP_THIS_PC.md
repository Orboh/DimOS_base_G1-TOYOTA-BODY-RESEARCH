# このPC(rad907 / Ubuntu 24.04, RTX 5050)での実行メモ — 2026-07-20 環境構築

`oda/RUN_ZED_IK.md` / `oda/FARM_QUICKSTART.md` の手順を、このPC向けに読み替えるためのメモ。
環境構築時(2026-07-20)の差分と残作業を記録する。

## 元ドキュメントからの読み替え(このPC固有)

| 項目 | 元ドキュメント(ラップトップ) | このPC |
|---|---|---|
| リポジトリ | `~/Toyota-auto-body-PoC/DimOS_oda` | `~/workspace/DimOS_base_G1-TOYOTA-BODY-` |
| 有線NIC | `enp46s0` | `enp2s0`(`ROBOT_INTERFACE=enp2s0`、multicast経路も `dev enp2s0`) |
| ZEDキャリブキャッシュ | 済み | **未**(初回はネットありで `dimos run unitree-g1-zed-ik-view` を1回起動してキャッシュを温めること) |

## 構築済み(2026-07-20)

- ブランチ `oda/ik-only-grasp` チェックアウト済み
- `.venv`: `dimos[base,unitree,unitree-dds]` editable install
  (unitree-sdk2py-dimos 1.0.3 / cyclonedds 0.10.5 をソースビルド)
- `~/cyclonedds-noshm`: CycloneDDS 0.10.5 を `-DENABLE_SHM=NO` でビルド・配置済み
  (Python側 cyclonedds はこれに対してビルド済み。実行時は元ドキュメント通り
  `CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib` を付ける)

## ZED SDK(2026-07-20 導入済み)

- SDK 5.4.0 (CUDA13/TensorRT10.13) を `/usr/local/zed` にインストール済み
- NEURAL深度モデルのDL+このRTX 5050向けTensorRT最適化も実施済み
  (`/usr/local/zed/resources/.neural_depth_5.3.model_optimized-*`)
- pyzed 5.4 (cp312) を `.venv` に導入済み。wheelの控え:
  `~/Downloads/pyzed-5.4-cp312-cp312-linux_x86_64.whl`(オフライン再導入用)
- インストーラの控え: `~/Downloads/ZED_SDK_Ubuntu24_cuda13.0_tensorrt10.13_v5.4.0.zstd.run`
  (再実行時は **sudoを付けずに** 実行すること)

## 残作業(現地に行く前に)

```bash
# ZED工場キャリブレーションのキャッシュ温め
# (ZED Mini実機を繋いで、ネットありで1回。シリアル番号ごとに初回DLされる)
cd ~/workspace/DimOS_base_G1-TOYOTA-BODY-
PYTEST_VERSION=1 .venv/bin/dimos run unitree-g1-zed-ik-view
```

## 起動コマンド(このPC版・把持フル構成)

```bash
sudo ip link set lo multicast on
sudo ip route replace 224.0.0.0/4 dev enp2s0
sudo sysctl -w net.core.rmem_max=67108864 net.core.rmem_default=67108864

cd ~/workspace/DimOS_base_G1-TOYOTA-BODY-
CYCLONEDDS_HOME=~/cyclonedds-noshm \
LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
DIMOS_SKIP_COORDINATOR_RPC=1 PYTEST_VERSION=1 \
ROBOT_INTERFACE=enp2s0 \
OKRA_NOACT_KP_ARM=160 OKRA_NOACT_KD_ARM=6.0 \
IK_REACH_LIVE=1 OKRA_NOACT_GRIP_LIVE=1 \
OKRA_GRIP_KP=20 OKRA_NOACT_CLOSE_Q=0.0 \
.venv/bin/dimos run unitree-g1-okra-ik-only-grasp-zed
```

ビューア(別ターミナル):

```bash
~/workspace/DimOS_base_G1-TOYOTA-BODY-/.venv/bin/dimos-viewer \
  --connect rerun+http://127.0.0.1:9877/proxy --ws-url ws://127.0.0.1:3030/ws
```

以降の操作・トラブル対処は `RUN_ZED_IK.md` / `FARM_QUICKSTART.md` の通り
(NICとパスだけ上の表で読み替える)。

## このPC固有のフリーズ対策(2026-07-20 実機で発生)

腕リーチ自体は成功したが、NEURAL深度+ビューア+点群を1台で同時に回して
**数分でシステム全体がフリーズ**(要PC再起動)。前回boot kernel ログに
`NVRM: ... Out of memory [NV_ERR_NO_MEMORY]` = **GPU VRAM枯渇**が記録されていた。
このPCは VRAM 8GB(RTX 5050)/ RAM 14GB と控えめなのが原因。

**このPCで動かすときは NEURAL を避け、負荷を落として起動すること:**

```bash
ZED_DEPTH_MODE=PERFORMANCE \   # GPU負荷1/10・VRAM大幅減(クリック用途は十分)
ZED_PC_VOXEL=0.004 \           # 点群を粗く(既定0.002→データ半減)
ZED_DEPTH_TRUNC=0.6 \          # 表示を近距離に限定(既定0.8)
# ...（上の起動コマンドの他の環境変数はそのまま）
```

加えて **ブラウザ等の他アプリを閉じて RAM を空ける**。監視するなら別ターミナルで
`nvidia-smi -l 2` を回して VRAM 使用量を見張る。

## グリッパが mode=0 で無反応になったとき(2026-07-20 復帰実績)

7/17農場で **電源ONのままハンドのケーブルを抜いた(活線挿抜)** ことでモーター基板が
不正状態に入り、指令を無視するようになっていた。**G1電源OFF → コネクタ挿し直し →
爪を閉じて電源ON** の正規手順でリセットされ、開き指令で正常動作を確認。
教訓: **ハンドのケーブル抜き挿しは必ずG1電源OFFで**。

生死の判定は `oda/gripper_move_probe.py`(開き方向へ実際に動かして q が動くか見る)。
注意: DDSで読める `mode`/`temperature` は NX のサービスが埋めていない常時0の値なので
故障判定に使えない(サービスは q/dq/tau_est しか publish しない)。
