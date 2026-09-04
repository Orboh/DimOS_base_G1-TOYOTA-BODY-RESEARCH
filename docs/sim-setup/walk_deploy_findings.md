# 公式C++ deploy（unitree_rl_lab g1_29dof）ビルド/実行 調査メモ

調査日: 2026-06-29〜30 / 目的: 「公式C++ deployをビルドして本物の歩行を動かす」

## 結論サマリ（先に重要制約）
1. **FSM 遷移はジョイスティック必須**。`FSMState.h` の遷移チェックは
   `func(FSMState::lowstate->joystick)`。config.yaml の遷移は
   `Passive→FixStand: LT + up.on_pressed` / `FixStand→Velocity: RB + X.on_pressed`。
   キーボードは `keyboard_velocity_commands`（歩行中の速度指令の例）用で、**状態遷移には使われない**。
   → 遷移トリガには **無線コントローラ信号（wireless_remote in rt/lowstate）** が要る。
   sim では unitree_mujoco が `USE_JOYSTICK=1`＋実ゲームパッドで wireless_remote を埋める想定。
2. **歩行は unitree_mujoco（別シム）上**。我々の収穫パイプライン（Isaac Sim 知能センター＋
   オクラ＋ZED＋IK＋ACT）とは**別の物理世界**。C++ deploy が歩いても、それは Isaac の収穫シーン
   内のロボットを動かさない。→ まず「本物歩行の単体実証」、収穫ワークフローへの統合は別ステップ。

## ビルド依存（deploy/robots/g1_29dof/CMakeLists.txt）
- C++17, cmake>=3.12（cmake/make/g++ あり ✓）
- find_package: **Boost program_options（静的 libboost_program_options.a）**, **yaml-cpp（静的 libyaml-cpp.a）**, **fmt**
- include: /usr/include/eigen3, /usr/local/include/ddscxx, /usr/local/include/iceoryx/v2.0.2,
  deploy/include, thirdparty/onnxruntime-linux-x64-1.22.0/include
- link: **unitree_sdk2 ddsc ddscxx**（= unitree_sdk2 C++ のインストールが要る）, rt, pthread,
  onnxruntime（thirdparty に同梱 1.22.0 ✓）
- Types.h が `unitree/dds_wrapper/robots/g1/g1.h` を include → **dds_wrapper の所在要確認**
  （unitree_sdk2 同梱か別物か）。joystick パースもこの wrapper（unitree_joystick.hpp）。

## ローカル環境の現状
- unitree_rl_lab: **未clone**
- unitree_sdk2 (C++): **未インストール**（/opt/unitree_robotics 無し）
- cmake(~/.local), make, g++: あり
- libyaml-cpp.so.0.7 あり（ただし CMake は **静的 .a** を要求 → -dev/静的版要確認）
- onnxruntime: thirdparty 同梱で OK
- unitree_mujoco: ~/Desktop/unitree_mujoco（config.py 既に g1/DOMAIN_ID=1/lo に編集済、
  ENABLE_ELASTIC_BAND=False ← 公式フローは elastic band 前提なので要 True 検討）

## 実行フロー（README）
1. `cd deploy/robots/g1_29dof/build && ./g1_ctrl`
2. [L2 + Up] → FixStand（立つ）
3. mujoco ウィンドウをクリックして **8** → 足を接地（elastic band 緩める）
4. [R1 + X] → Velocity policy 実行（歩行）
5. mujoco ウィンドウで **9** → elastic band 無効化

## 解決済み（上記の確認事項）
- ゲームパッド: 無し → bridge をファイル注入式に最小改造（/tmp/sim_joy.bin）で代替。
- dds_wrapper: unitree_sdk2 に同梱（include/unitree/dds_wrapper/…）。
- unitree_sdk2 C++: **プリビルド配布**（libunitree_sdk2.a + ddsc/ddscxx .so 同梱）→ 重コンパイル不要。
- elastic band: 自作 launcher で torso 係留（length=0.445 で重量支持）→ 滑らか除荷で自動化。

## ✅ 成功した最終構成と再現手順（2026-06-30）
ビルド済み: unitree_sdk2→~/.local、g1_ctrl→unitree_rl_lab/deploy/robots/g1_29dof/build/g1_ctrl
一括起動（ゲームパッド・手動キー不要・完全自動）:
```
cd ~/Desktop/dimos-hackathon
WALK_SECS=15 WALK_VX=0.3 WALK_SETTLE=3 bash docs/sim-setup/run_walk.sh
# ログ: /tmp/walk_logs/{sim,g1ctrl,joy}.log
```
個別起動（公式3端末フロー相当）:
```
# 端末1: シム（MuJoCo + DDS domain0/lo + band）
.venv/bin/python docs/sim-setup/sim_walk_run.py
# 端末2: 公式C++ deploy
LD_LIBRARY_PATH=$HOME/.local/lib:~/Desktop/unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib \
  ~/Desktop/unitree_rl_lab/deploy/robots/g1_29dof/build/g1_ctrl --network lo
# 端末3: 注入オーケストレータ
.venv/bin/python docs/sim-setup/sim_walk_joy.py
```

### 結果
- 安定構成（settle 3s + vx 0.3）: **39秒間 転倒0回**、直立 z=0.75〜0.79 維持、累計 ~2.7m 歩行。
- 既知の課題: 進行方向が +y へドリフト（band解除直後に軽い yaw 混入 → body前方=world+y）。
  歩行安定性は問題なし。直進性は今後の微調整（解除手順 or yaw 補正）。

### 改造したファイル（外部リポ unitree_mujoco）
- simulate_python/config.py: DOMAIN_ID 1→0（g1_ctrl が domain0 ハードコード）。
- simulate_python/unitree_sdk2py_bridge.py: joystick未接続時 /tmp/sim_joy.bin から wireless_remote 注入。

### 重要な前提
- g1_ctrl は DDS domain **0** ハードコード（main.cpp）。`--network lo` 必須。
- check_mode_machine は lowstate.mode_machine==0 を「シム環境」として PASS（ブロッカー無し）。
- 歩行は unitree_mujoco（別シム）。Isaac 収穫シーンとは別世界 → 収穫統合は別フェーズ。
