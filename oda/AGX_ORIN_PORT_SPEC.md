# IK単独オクラ把持デモ — Jetson AGX Orin (ZED Mini) 移植仕様書

**対象ブループリント:** `unitree-g1-okra-ik-only-grasp`
(`dimos/robot/unitree/g1/blueprints/manipulation/unitree_g1_okra_ik_only_grasp.py`)
**現状:** ラップトップ + 頭部D435i(Jetson NX経由LCM中継)+ 人間クリックUI で実機LIVE確認済み(2026-07-15)。
**目標:** Jetson AGX Orin 64GB バックパック上で完結、カメラは胸部固定のZED Miniに切り替え。
**作成日:** 2026-07-15 — 作成時点の調査結果に基づく。実装前に③のブロッカーを解消すること。

---

## 0. スコープ

含む: カメラ入力の切り替え(D435i/頭部 → ZED Mini/胸部)、実行ホストの切り替え(ラップトップ → AGX Orin)、
それに伴うhand-eyeキャリブレーションのやり直し、`dimos run`での起動。

含まない: ACT統合・切断・カゴ収納・ナビゲーション・YOLO自動検出への切り替え(これらは別ドキュメント
`G1収穫設計書/SS-0*.md` および `feat/g1-okra-langgraph` ブランチのLangGraphパイプラインの領分であり、
今回のIK単独デモの移植とは独立)。人間クリックによるIK単独把持、という現在の設計はそのまま維持する。

---

## 1. 再利用マッピング表(CLAUDE.md Phase 1 準拠)

| 機能 | 既存ソース | 対応 |
|---|---|---|
| ZEDカメラ駆動(color/depth/pointcloud) | `dimos/hardware/sensors/camera/zed/camera.py` `ZEDCamera`(このworktreeに既に存在。汎用カメラドライバ、PR #935由来。project固有コードではない) | **direct import** — `ZEDCamera.blueprint(enable_depth=True, enable_pointcloud=True, ...)` |
| ZED→HarvestModule配線の実例 | `origin/feat/g1-okra-langgraph`の`unitree_g1_okra_harvest_zed.py`(このworktreeにはまだ無い。`git show origin/feat/g1-okra-langgraph:<path>`で参照可) | **設計参考のみ** — LangGraph自動検出向けの配線なので今回はコピー不可、`.remappings`/`.transports`パターンだけ流用 |
| IKリーチ本体・クリックUI・アーム/グリッパ制御 | `dimos/robot/unitree/g1/act/ik_reach_bridge.py`(`IkReachBridge`)、`g1_arm_sdk_connection.py`、`g1_gripper_connection.py`、`vis_module("rerun", ...)` | **direct import・無変更** — カメラソースが変わってもこの層は一切変更不要 |
| 頭部D435iのhand-eye定数 | `ik_reach_bridge.py:76-94`(`_D435_XYZ`/`_D435_RPY`/`_OPTICAL_WXYZ`/`_default_torso_from_optical()`) | **design from scratch** — チェストZED用の新しい定数が必要(実測キャリブレーションが前提、§5参照)。理由: URDFの取り付け位置・角度が頭部と胸部で全く異なるため流用不可 |
| Jetsonカメラ中継(D435i→LCM multicast) | `oda/start_okra_ik_only_grasp.sh`のJetson kick部分、Jetson NX上の`~/run_ik_camera.sh`/`ik_camera_standalone.py` | **不要になる** — ZED MiniはAGX Orinに直結(USB3)なので中継プロセス自体が要らない |
| AGX Orin接続情報 | vault `04-Context.md:233-246`, `Free-space/ueda/開発レポート_Jetson_WiFi_AP化_2026-06-10.md` | **reference** — 後述§2 |
| independent venv の教訓 | 本セッションの`DimOS_oda`修正(symlink venv → 独立venv) | **同じ手順を適用** — AGX Orin側でも共有venv/symlinkは避ける |

---

## 2. AGX Orin 接続情報(vault確認済み)

| 経路 | 詳細 |
|---|---|
| 自前WiFi AP経由 | SSID `agx` / pass `agx12345` → `ssh tbr@192.168.12.1`(pass `tbr123`, 自動起動) |
| 有線 | `ssh tbr@192.168.123.222`(PC側 `.50`)、NIC `eno1` |
| Tailscale | ホスト名 `jetson-orin-dimos` → `100.113.43.64` |
| OS | Ubuntu 22.04.5 (tegra), JetPack 6.2 / L4T 36.5 |
| DimOSの現状ブランチ | `feat/g1-okra-langgraph`(このIK単独デモの`oda/ik-only-grasp`とは別ブランチ・別系統) |

⚠️ 物理的な搭載・接続状態はセッションごとに変動している(2026-06-06時点で搭載・稼働確認済みだが、
2026-06-25には「ベンチ上で未接続」の記録もあり)。着手前に必ず現況を確認すること。

---

## 3. ICD表(システム間インターフェース)

| 系統ペア | ネットワーク経路 | トランスポート | メッセージ形式 | レート/QoS | SDK/lib版 | 認証 | 故障時挙動 | スモークテスト |
|---|---|---|---|---|---|---|---|---|
| G1 ↔ AGX Orin | G1内蔵Ethernet(NUC-バックパック間, Gigabit) | DDS(CycloneDDS no-shm) | `rt/arm_sdk`, `rt/lowstate`, `rt/dex1/right`(Unitree IDL) | arm_sdk 250Hz / lowstate ~500Hz | cyclonedds 0.10.2/0.10.5 + `unitree_sdk2py`(本セッションと同じ`/home/sota/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python`フォーク) | なし(物理LAN) | 既存の`G1ArmSdkConnection`/`G1GripperConnection`のfail-safe hold、変更不要 | ⚠️ 未検証 — AGX Orinから`dimos run unitree-g1-coordinator`等でDDS疎通を先に確認すること |
| ZED Mini ↔ AGX Orin | USB3直結 | ZED SDK | `ZEDCamera`が`color_image`/`depth_image`/`pointcloud`/`camera_info`を出力 | `pointcloud_fps`既定5Hz、`camera_info_fps`既定1Hz、`fps`既定15(config可変) | ZED SDK 5.4 + pyzed(2026-06-24時点でAGX Orin上に導入・動作確認済み、Python3.12venv) | なし | `ZEDCamera`モジュール自体のfail behavior(要コード確認、本ドキュメント範囲外) | 既存: `scripts/verify_zed_banana_detect.py`(`feat/g1-okra-langgraph`側)が動作実績あり。**okra重心クリック用途では未検証** |
| 操作者ラップトップ ↔ AGX Orin(ビューア) | AGX OrinのWiFi AP/有線/Tailscale(上表) | Rerun gRPC + WebSocket(現状と同じ、ホスト/クライアントの役割が逆転 = AGX Orinがサーバ) | Rerun protocol | インタラクティブ、クリック操作なので低遅延が望ましい | rerun-sdk(dimosの固定バージョンに追従) | なし(現状と同様) | ビューア切断時はクリック不可になるだけ、アームは最後の姿勢を保持(現状と同じ) | ⚠️ 未検証 — `dimos-viewer --connect rerun+http://<AGXのIP>:9877/proxy --ws-url ws://<AGXのIP>:3030/ws` を別マシンから接続して確認 |
| G1 NX ↔ AGX Orin | G1内蔵Ethernet | DDS/V4L2(Mid-360, 手首UVC) | 今回のIK単独デモでは未使用(harvest/navパイプライン専用) | — | — | — | 対象外(今回のブループリントはMid-360も手首カメラも使わない) |

---

## 4. 必要なコード変更

1. **新しいカメラブループリント**: `oda/`または`dimos/robot/unitree/g1/blueprints/manipulation/`に
   `unitree_g1_okra_ik_only_grasp_zed.py`のような新規ファイルを追加(既存の
   `unitree_g1_okra_ik_only_grasp.py`は編集せず、ZED版は別ファイルとして共存させる。
   `unitree_g1_okra_harvest.py` ⇔ `unitree_g1_okra_harvest_zed.py`(langgraphブランチ)の既存の
   命名慣習に合わせる)。中身は現行ファイルとほぼ同じで、`vis_module("rerun", ...)`への入力を
   Jetson-NX中継のD435i pointcloudではなく `ZEDCamera.blueprint(enable_depth=True,
   enable_pointcloud=True, depth_mode="PERFORMANCE" or "NEURAL", ...)` の`pointcloud`/`color_image`/
   `camera_info`出力に置き換える。`IkReachBridge`/`GripperGraspOnReach`/`G1ArmSdkConnection`/
   `G1GripperConnection`は無変更で流用。

2. **hand-eye定数の新規作成**: `ik_reach_bridge.py:76-94`の`_D435_XYZ`/`_D435_RPY`/`_OPTICAL_WXYZ`/
   `_default_torso_from_optical()`はD435i・頭部専用。ZED Mini・胸部用の新しい定数が必要
   ―― g1.urdfにチェストZEDマウントの joint 定義があるか確認し、無ければ実測が必要
   (§5のブロッカー①)。`ZEDCamera`モジュール自体は`base_transform`/`base_frame_id`という
   汎用マウント姿勢の設定フィールドを持っている(`camera.py:60-61`)ので、D435i方式の
   決め打ち定数をそのまま真似るより、こちらの仕組みで指定できないか設計時に検討する価値がある。

3. **`expected_click_frame`の更新**: 現行`"/world/camera/pointcloud"`のままで良いか、ZED用に
   別のRerun entity pathにするかは、上記の新ブループリントでのstream命名/remappings次第。

4. **起動スクリプト**: `oda/start_okra_ik_only_grasp.sh`の`[1/4] Jetson NX kick`と
   `[2/4] マルチキャストルート設定`は丸ごと不要(ZEDはAGX Orinに直結、laptop↔robot越しの
   LCM中継が無くなるため)。AGX Orin用に新しい起動スクリプトを作成し、`dimos run
   unitree-g1-okra-ik-only-grasp-zed`(仮の名前)を直接叩くだけのシンプルな形にする。

---

## 5. 未解決のブロッカー

1. **チェストZEDのhand-eyeキャリブレーション未実施**(`G1収穫設計書/SS-04-粗アプローチIK.md`に
   明記: 「胸部ZEDに置き換えるため、`T_base_camera`は取り直しが必要」「❌ 未」)。
   これが無いとLIVEでのリーチ精度が保証されない ―― 頭部D435iの時と同じ手順
   (実機での`[CALIB]`ログ比較、SS-04ドキュメント参照)をチェストZEDで再実施する必要がある。
   **LIVEテストの前提条件、最優先で潰すべき項目。**

2. **AGX Orin側の実行環境**: 現在`feat/g1-okra-langgraph`ブランチが動いている(このIK単独デモの
   `oda/ik-only-grasp`とは別系統)。同じマシンに両方を共存させる場合、本セッションで学んだ教訓
   (worktreeごとに独立venvを持つ・symlink venvは避ける)をAGX Orin側でも適用すること。
   `uv sync`だけだとCPU版torchが入る既知の罠がある(2026-06-24調査済み、YOLO用に別途
   Python3.10 GPU-torch venvをJetson向けpypiインデックスから構築 ―― ただし今回のIK単独
   デモ自体はtorch/YOLOを使わないので、この罠は直接は関係ない。将来ACT/YOLOを足すときに注意)。

3. **物理的な搭載・ネットワーク状態の確認**: 着手前にAGX OrinがG1に実装されていて、電源・
   ネットワークが生きているか確認する(§2の変動する状態を参照)。

4. **G1↔AGX Orin間のDDS疎通自体が未検証**(vaultのチェックリストにある
   `dimos run unitree-g1-coordinator`疎通確認はNUC/ラップトップからは実施済みだが、
   AGX Orinからの疎通は記録が見当たらない)。IK単独デモを試す前に、まずこのステップ0を
   単独で確認するべき。

---

## 6. 推奨する実施順序

1. AGX OrinにSSHし、現在のブランチ/dimos状態を確認。このセッションと同じ独立venvの原則で
   `oda/ik-only-grasp`用のworktreeを追加(共有venvにしない)。
2. ステップ0のDDS疎通確認: 軽量なブループリント(`unitree-g1-coordinator`等)で
   AGX Orin ⇔ G1のDDSが実際に届くか確認。
3. チェストZEDのhand-eyeキャリブレーション実施(SS-04ドキュメントの手法をZED用に再実施)。
4. §4の新ブループリントファイルを作成(`ZEDCamera`配線)。`dimos/robot/all_blueprints.py`の
   自動生成テストを実行して登録(本セッションで確立した手順と同じ)。
5. DRY-RUNでラップトップからビューア接続確認(AGX OrinのIPに向けて`dimos-viewer --connect`)。
6. LIVEテスト(アームのみ、グリッパはドライランのまま)― 本セッションのkp_arm=160/kd_arm=6.0の
   知見も流用可能(ハードウェア側の腕そのものは変わらないため)。

---

## 参考(このドキュメントの元になった調査)

- 本リポジトリ `dimos/hardware/sensors/camera/zed/camera.py`(`ZEDCamera`本体、既存)
- `origin/feat/g1-okra-langgraph`ブランチの`unitree_g1_okra_harvest_zed.py`(配線の実例、コピー元ではなく参考)
- vault: `G1収穫設計書/00-全体設計書.md`(§3/§4、目標アーキテクチャ)、`SS-04-粗アプローチIK.md`(キャリブレーション状況)、
  `04-Context.md`(AGX Orin接続情報)、`Free-space/ueda/2026-06-24.md`(ZED SDK導入・GPU torch罠の記録)
