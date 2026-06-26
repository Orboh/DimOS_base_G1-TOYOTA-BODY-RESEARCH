# タスク依頼: Isaac Sim でオクラ収穫シミュレーション環境を立ち上げる（第1歩）

> **依頼者**: Kota（CTO）／**担当**: 別セッション（シミュレーション環境セットアップ）
> **作成**: 2026-06-25

---

## 0. このタスクの位置づけ（背景）

G1 オクラ収穫パイプラインの **実機検証の前倒し**として、シミュレーション環境を立ち上げたい。最終的に目指す構成は:

```
PC (Isaac Sim = 仮想ロボット)  ──DDS/LCM (LAN)──  Jetson (dimos + 把持パイプライン)
   ・G1 29dof+dex1 を物理/描画         ・LangGraph / IK / 検出 / 歩行
   ・カメラ・関節状態を配信              ・arm_target / cmd_vel / gripper を送る
```
dimos は transport 非依存（DDS/LCM/SHM）で、把持ブループリント（`unitree-g1-okra-harvest-ik` 等）は既に DDS トピックで実機と喋る設計。**接続先を実機 → sim に差し替えるだけ**で sim-in-the-loop 検証ができる、というのが狙い。

**ただし本タスクはその第1歩**＝「**Isaac Sim を入れて、研究室環境USD と カスタムG1 USD をロードし、G1 が物理的に立って基本制御できる**」ところまで。dimos 接続や把持フロー全体は次のフェーズ（§7）。

> 関連設計: `G1収穫設計書/11-開発計画.md` §5（3Dマッピング→Omniverse, D-007）／`G1収穫設計書/00-全体設計書.md`。

---

## 1. ゴール（このタスクの完了条件 / DoD）

1. ✅ Isaac Sim が当該マシンで起動する
2. ✅ **研究室環境 `room.usd`** をシーンとしてロードできる
3. ✅ **カスタムG1 `g1_29dof_with_dex1_base_fix1.usd`** を room 内に配置し、**物理的に立つ**（落下・発散しない）
4. ✅ G1 の関節を**指令で動かせる**（手始めに右腕を1関節でも目標角へ動かす）
5. ✅ G1 のカメラ（ヘッド/手首相当）または少なくとも1つのRGBビューがレンダリングされ取得できる
6. ✅ 上記の起動手順を **再現可能な形でメモ**（コマンド・つまずき・GPU設定）

---

## 2. 入力アセット（このリポジトリ内・ローカル）

`/home/kota-ueda/Desktop/dimos-hackathon/usd_file/`（※128MB、`.gitignore` 済み＝ローカルのみ。同一マシンで作業すること）

| ファイル | 役割 |
|---|---|
| `room.usd`（37K）| **研究室環境**シーン。まずこれをワールドに |
| `g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd`（26M）| **カスタムG1**（29DoF + 右手Dex1-1、base_fix版）。これを room に置く |
| `g1-29dof-dex1-base-fix-usd/g1bag.usd`（26M）| G1+バックパック版（必要なら）|
| `g1-29dof-dex1-base-fix-usd/configuration/*.usd` | URDF派生（base/physics/sensor、waist_fix版）。物理・センサ設定の参照用 |
| `g1-29dof-dex1-base-fix-usd/config.yaml` | UrdfConverter 設定。**joint_drive: stiffness 100 / damping 1, drive=force/position, fix_base: true, collider=convex_hull, make_instanceable: true**（2025-06-13生成）|

> **注意（config.yaml より）**: このG1 USDは **`fix_base: true`**（土台固定）で生成されている。歩行させるには base を free にした版が要る可能性あり（`configuration/` に `waist_fix` 等の派生あり、または再変換）。まずは fix_base のまま「腕が動くか」を見るのが安全。

---

## 3. 前提・環境（重要：GPU要件）

- **Isaac Sim は強力なRTX GPUが必要**。推奨 VRAM 16GB以上。
- ⚠️ **Kota のラップトップは RTX 2000 Ada Laptop / VRAM 8GB**＝**最低ラインギリギリ**。room+G1+カメラ描画は **8GBだと重い/落ちる可能性**。
  - → **可能なら 16GB+ のRTX機で実施**を推奨（計画 D-007 の「Omniverse環境構築」担当機があれば最適）。8GBで試すなら描画品質を下げ、カメラ解像度・シーンを最小化して様子を見る。
- Isaac Sim バージョン: 最新安定版（Isaac Sim 4.x / Isaac Lab）を想定。**まず公式手順でインストール**。

---

## 4. 既存の参考実装（車輪の再発明をしない）

1. **Unitree公式 `unitree_sim_isaaclab`（最有力）**: 横手くんの `dex1_1_service` リポ（ブランチ `feat/drop-to-basket`）の `20260624/src/run_harvest.py` が、**`unitree_sim_isaaclab` を `--action_source dds --enable_dex1_dds --robot_type g129 --enable_cameras` で起動**している。**DDS駆動のIsaac Lab G1 sim（dex1含む）が既にチーム内で動いた実績**。まずこれが使えないか確認すると近道。
   - 参照: `/home/techshare/xr_teleoperate/unitree_sim_isaaclab`（横手環境）／ `Orboh/dex1_1_service` PR#1
2. **dimos のsim統合**: このリポに `dimos/simulation/{isaac,mujoco,genesis,...}` と sim ブループリント（`unitree-g1-basic-sim` / `-nav-sim` / `-agentic-sim` / `g1-sim-connection`）が実在。Isaac連携の足場になるか調査。
3. **MuJoCo代替**: `data/mujoco_sim/unitree_g1.xml` あり。**もしIsaac Simが8GBで厳しければ、軽量なMuJoCoでフロー検証に切替**という選択肢も報告してほしい（Kotaのラップトップでも回る）。

---

## 5. 進め方（推奨ステップ）

1. **GPU確認**: `nvidia-smi`。16GB未満なら §3 の縮小設定 or 別機検討を先に判断・報告。
2. **Isaac Sim インストール**（公式手順）。起動確認。
3. **`room.usd` を開く** → シーンが見えるか。
4. **`g1_29dof_with_dex1_base_fix1.usd` を room に配置** → 物理再生（Play）で立つか。発散したら `config.yaml` の gains（stiffness100/damping1）や collider を調整。
5. **関節制御**: 右腕1関節を目標角へ（Isaac の Articulation API or Action graph）。
6. **カメラ**: ヘッド/手首相当のRGBが取れるか。
7. **手順をメモ化**（再現可能に）。

---

## 6. 成果物（報告してほしいもの）

- 起動手順メモ（インストール・コマンド・GPU設定・つまずき）
- スクリーンショット（room+G1が立っている画、関節が動いた証拠）
- **判断**: このマシン（8GB）でIsaac Simが実用に耐えるか／別機が要るか／MuJoCoに切替すべきか
- `fix_base: true` のままでよいか、歩行には base-free 版/再変換が要るか の所見
- `unitree_sim_isaaclab` が再利用できそうか（DDS駆動の足場として）

---

## 7. このタスクの範囲外（やらないこと）

- dimos との DDS 接続（次フェーズ）
- 把持パイプライン全体（IK→ACT→cut）の sim 実行
- **ACT把持・切断の sim 検証**（ACTは実機画像で学習＝sim画像はOOD、切断はPhysX非対応。これらは実機でしか確かめない＝設計書11 §5）
- オクラ物体の配置・検出（まず G1 と room が動いてから）

→ まずは「**Isaac Sim が立ち上がり、研究室USDの中でカスタムG1が立って腕が動く**」ことだけを確実に。ここが取れたら次フェーズ（dimos接続・カメラ配信）の依頼を別途出す。

---

## 8. 次フェーズの見通し（参考）

本タスク成功後:
1. sim のカメラ/関節状態を DDS/LCM で配信 → Jetson の dimos が購読
2. dimos の `arm_target` / `cmd_vel` / `gripper_target` を sim が受けて駆動
3. `unitree-g1-okra-harvest-ik` の接続先を sim にした `-sim` 派生ブループリントで、**LangGraphフロー・IK到達・歩行/左掃引**を実機前に検証（ハンドアイ校正不要＝sim外部パラメータ既知が利点）
