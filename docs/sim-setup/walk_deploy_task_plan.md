# タスク計画: 公式C++ deploy で本物の歩行を動かす

## ゴール
unitree_rl_lab の g1_29dof velocity policy を **公式C++ deploy（g1_ctrl）** でビルドし、
unitree_mujoco sim2sim で **本物の脚歩行**を実証する。
（最終的には収穫ワークフローへ統合だが、まず単体実証が本タスク）

## 重要制約（findings.md 参照）
- FSM 遷移はジョイスティック必須・**ゲームパッド無し** → sim 側で wireless_remote 自動合成で代替。
- 歩行は unitree_mujoco（別シム）。Isaac 収穫シーンとは別世界 → 統合は後続フェーズ。

## フェーズ
- [ ] P1: unitree_rl_lab / unitree_sdk2 を clone（~/Desktop/）
- [ ] P2: unitree_sdk2 C++ を /usr/local に install（cmake && sudo make install）※sudo要
- [ ] P3: g1_ctrl をビルド（deploy/robots/g1_29dof/build で cmake .. && make）
- [ ] P4: ゲームパッド代替＝unitree_mujoco に wireless_remote 自動シーケンス注入
       ＋ elastic band 有効化（立ち上げ補助）
- [ ] P5: sim2sim 実行 → Passive→FixStand→Velocity 遷移 → 歩行確認
- [ ] P6:（後続）収穫ワークフローへの統合方針決定

## 依存（確認済み）
- apt: libyaml-cpp-dev/libboost-all-dev/libeigen3-dev/libspdlog-dev/libfmt-dev → 全て導入済み ✓
- 静的: libyaml-cpp.a / libboost_program_options.a ✓
- unitree_sdk2: プリビルド（libunitree_sdk2.a x86_64）+ dds_wrapper 同梱 + ddsc/ddscxx/iceoryx 同梱
- onnxruntime: rl_lab deploy/thirdparty/onnxruntime-linux-x64-1.22.0 同梱 ✓

## 解明した実装仕様（P4設計の根拠）
- g1_ctrl は **DDS domain 0 ハードコード**（main.cpp）→ unitree_mujoco config DOMAIN_ID=0 に合わせる。
- 引数 `--network lo`（unitree_mujoco INTERFACE=lo）。
- wireless_remote(40byte): head[0:2], btn=byte2(low)+byte3(high), lx[4:8],rx[8:12],ry[12:16],L2[16:20],ly[20:24]。
  - btn bits: byte2: bit0=R1,bit1=L1,bit2=Start,bit3=Select,bit4=R2,bit5=L2 / byte3: bit0=A,bit1=B,bit2=X,bit3=Y,bit4=up,bit5=right,bit6=down,bit7=left。
  - 遷移: FixStand=`LT(L2)+up.on_pressed`（byte2 bit5 + byte3 bit4）, Velocity=`RB(R1)+X.on_pressed`（byte2 bit0 + byte3 bit2）。
- 速度: velocity_commands 観測 = vx=ly, vy=-lx, yaw=-rx（範囲 vx[-0.5,1.0],vy[-0.3,0.3],yaw[-0.2,0.2]）。
  ※ deploy.yaml を keyboard_velocity_commands に変えれば w/s/a/d でも可（が、今回は wireless_remote 注入で全自動）。
- ElasticBand: torso_link を [0,0,3] へ吊る。force=stiffness(200)*(dist-length)-damping(100)*v。
  length=0→満吊り、length↑で降下、enable=False で解除。属性直叩きで自動制御可。

## エラー記録
| エラー | 試行 | 解決 |
|---|---|---|
| (なし) | | |

## 進捗ログ
- 2026-06-29/30 調査完了。
- ✅ P1 clone（unitree_sdk2 / unitree_rl_lab → ~/Desktop）
- ✅ P2 unitree_sdk2 を ~/.local に install（sudo回避・dds_wrapper同梱確認）
- ✅ P3 g1_ctrl ビルド成功（CPATH/LIBRARY_PATH=~/.local、ldd全解決）
- ✅ P4 ゲームパッド代替（wireless_remote ファイル注入）＋ elastic band 自動制御 完成
  - bridge: joystick未接続時 /tmp/sim_joy.bin から wireless_remote 注入（最小改造）
  - sim_walk_run.py（自作launcher）: band を /tmp/sim_band.txt + キー7/8/9 で制御
  - sim_walk_joy.py（orchestrator）: L2+up→FixStand / R1+X→Velocity / band滑らか除荷 / ly前進
  - band length: 立位支持=0.445（質量35.1kg実測, K=200, 重量344N）/ 除荷=2.167
- ✅ P5 **本物の歩行 達成（2026-06-30）**
  - g1_ctrl が DDS接続→FSM遷移（Passive→FixStand→Velocity）注入で成功
  - band解除後、直立(z≈0.78)のまま前進: x −0.02→+1.29m / 約4秒 ≈ 0.32 m/s（指令0.4）
  - 課題: 約5秒歩行後に横転（y→−1.27）。解除直後の残留横ずれ＋即前進が原因と推定。
    → 整定フェーズ追加 + vx 低減で安定化を試行中。
- ▶ P6 安定化（整定→前進, vx調整）→ その後 収穫ワークフロー統合
