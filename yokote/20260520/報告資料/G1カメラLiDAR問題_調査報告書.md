# G1ロボットにおけるカメラ・LiDAR 表示/利用不可問題の調査・解決報告

作成日: 2026-05-20

---

## 1. 要旨

G1 実機で DimOS agentic stack を起動した際、カメラ画像が真っ黒、LiDAR/点群が確認できない問題が発生していた。

調査の結果、**原因はコードにあり、ハードウェア故障ではない**ことが判明した。

- **カメラ**: G1 本体カメラを使う実装が存在せず、PC 側の Webcam を参照しようとしていたが、PC に Webcam がないため失敗していた。コードを修正し、G1 の映像ストリームを使う実装を追加した。映像が実際に流れるかの実機確認は次回対応予定。
- **LiDAR**: G1 の LiDAR を DimOS に取り込むために必要なパッケージと native binary が環境に不足していた。環境整備および blueprint 統合を行い、LiDAR 点群データが DimOS 上で流れることを実機で確認した。

---

## 2. 背景・目的

### 背景

G1 実機での DimOS 起動時に以下の状態が継続していた。

- 可視化画面のカメラ表示が真っ黒
- LiDAR/点群が可視化画面に表示されない
- その結果、視覚認識・空間認識・ナビゲーション系の skill が機能しない

Go2 では同じ DimOS stack でカメラ・LiDAR ともに正常動作しているため、G1 固有の問題であることは把握していた。

### 目的

以下の状態を達成すること。

- G1 実機起動時にカメラ画像が DimOS stream に流れる
- G1 実機起動時に LiDAR/点群データが DimOS stream に流れる
- LLM agent の視覚・空間系 skill（`take_a_picture`, `navigate_with_text` 等）が正常に動作する

---

## 3. 実施内容

### 3.1 調査環境

| 項目 | 内容 |
|------|------|
| PC IP | 192.168.123.212 |
| G1 IP | 192.168.123.161（メインコンピュータ） / 192.168.123.164（Jetson） |
| LiDAR IP | 192.168.123.120（Mid-360） |
| OS | Ubuntu 24.04.4 LTS |
| 起動 blueprint | `unitree-g1-agentic` |

### 3.2 調査手順

1. `run_robot.sh` で G1 agentic stack を起動し、起動ログを確認
2. `dimos topic echo` および `dimos lcmspy` で各 topic のデータ流通を確認
3. blueprint のソースコードを読み、カメラ・LiDAR の入力経路を特定
4. Go2 との実装差分を比較し、G1 固有の問題点を特定

---

## 4. 結果

### 4.1 カメラ黒画面問題

**原因**:

G1 の blueprint は、実機時に PC 側の Webcam（`/dev/video0`）を参照する実装になっていた。G1 本体カメラを取得する実装は存在しなかった。

```
# 起動ログに出ていたエラー
VIDEOIO(V4L2:/dev/video4): can't open camera by index
RuntimeError: Failed to open camera 4
```

`dimos topic echo` で確認したところ、`/color_image` topic にはデータが流れていなかった。一方 Go2 は本体カメラの映像を DimOS stream に流す実装を持っており、この差分が原因だった。

**対応**:

G1 のロボット接続モジュール（`G1Connection`）に、G1 本体カメラの映像を DimOS stream に流す処理を追加した。また、blueprint から PC 側 Webcam を参照していたコードを削除した。

修正後、blueprint が起動エラーなく読み込まれることを確認した。ただし、G1 との通信（WebRTC）が確立した状態での映像流通確認は次回実施予定（詳細は 7 章）。

### 4.2 LiDAR/点群問題

**前提知識**: G1 の Mid-360 LiDAR から点群・自己位置を取得するには、LiDAR の raw データをリアルタイムで処理する **FastLIO**（LiDAR Odometry アルゴリズム）が必要。

**原因**:

Go2 と異なり、G1 の接続モジュールは LiDAR stream を持っていなかった。LiDAR データを DimOS に取り込む blueprint は存在したが、動作させるために必要な以下のものが環境に不足していた。

| 不足していたもの | 内容 |
|----------------|------|
| 必要な Python パッケージ | Unitree SDK の DDS 通信パッケージが未インストール |
| LiDAR 処理 binary | FastLIO の native binary が未ビルド（ビルドツール Nix も未インストール） |
| Git LFS データ | 経路計画に必要なデータファイルが LFS サーバーから取得できない（404） |

**対応**:

段階的に環境を整備し、LFS データが不要な LiDAR 確認用の blueprint を新たに作成して動作を確認した後、`unitree-g1-agentic` に統合した。

1. 不足パッケージをインストール
2. FastLIO の native binary をビルド
3. LFS データ不足を回避するため、経路計画を除いた LiDAR 専用の診断 blueprint を作成し、LiDAR が動作することを確認
4. `unitree-g1-agentic` に LiDAR/FastLIO の経路を統合

### 4.3 SpatialMemory の起動失敗

**原因**:

`SpatialMemory`（画像の意味的な記憶を管理するモジュール）が起動時に CLIP モデルファイルを Git LFS サーバーから取得しようとしていたが、LFS サーバーが 404 を返していた。

**対応**:

HuggingFace から同等のモデルファイルを直接取得して所定の場所に配置した。これにより `SpatialMemory` が `--disable` オプションなしで起動できるようになった。

---

## 5. 確認結果

### 5.1 LiDAR（確認済み）

`unitree-g1-agentic` 起動後、以下の topic にデータが流れることを `dimos lcmspy` で確認した。

**実施画面（DimOS Viewer に LiDAR 点群が表示されている様子）:**

![LiDAR点群 実施画面](../実施メモ/実施画面.png)

| topic | 意味 | 観測周波数 |
|-------|------|-----------|
| `/lidar` | LiDAR 点群（生データ） | 約 3.8 Hz |
| `/global_map` | 累積された環境地図 | 約 3.8 Hz |
| `/global_costmap` | ナビゲーション用コスト地図 | 約 3.8 Hz |
| `/odometry` / `/odom` | FastLIO による自己位置推定 | 約 10.8 Hz |
| `/state_estimation` | 状態推定（上と同一ソース） | 約 10.8 Hz |

LiDAR データが DimOS 上で正常に流れること、および自己位置推定が機能していることを確認した。

### 5.2 カメラ（コード修正済み・実機確認未完了）

カメラ映像の流通確認には G1 との通信（WebRTC）が確立している必要があり、現時点では確認できていない。WebRTC 接続の問題は別途調査中（7 章参照）。

---

## 6. 考察

Go2 と G1 の主な実装差分は以下の通りだった。

| 項目 | Go2 | G1（修正前） |
|------|-----|-------------|
| カメラ | 本体カメラの映像を DimOS stream に直接 publish | PC 側 Webcam を参照（G1 本体カメラ未対応） |
| LiDAR | 本体 LiDAR の点群を DimOS stream に直接 publish | 外部の LiDAR 処理システムが別途必要な設計（未接続） |
| 自己位置 | WebRTC 経由で本体 odom を publish | FastLIO 経由（今回統合） |

G1 の実装が Go2 に比べて不完全だったことが根本原因であり、ハードウェアや環境固有の問題ではなかった。カメラについては、接続さえ確立できれば映像が流れる見込みは高い。

---

## 7. 課題・リスク

### 残課題

| 課題 | 詳細 | 優先度 |
|------|------|--------|
| WebRTC 接続の確立 | G1 との通信（WebRTC）が最終ステップで止まっている。原因は特定済みで次回対応予定 | 高 |
| カメラ映像の実機確認 | WebRTC 接続確立後に `/color_image` にデータが流れるかを確認する | 高（WebRTC 解決後） |
| カメラ intrinsics の精度 | G1 前面カメラのレンズパラメータに概算値を使用中。`navigate_with_text` 等の精度に影響する可能性あり | 中 |
| 経路計画データの取得 | 完全な自律ナビゲーション stack に必要なデータファイルが LFS サーバーから取得できない | 低（現状は FastLIO で代替可能） |

### 補足：WebRTC 接続問題について

G1 との通信プロトコルである WebRTC の接続は、以下の流れで行われる。

```
① 接続情報（SDP offer）をG1に送る
② G1が応答（SDP answer）を返す  ← ここまでは今回確認済み
③ 実際の通信経路（ICE）を確立する
④ データチャンネルの認証を完了する  ← ここで止まっている
⑤ 接続完了
```

②まで動作することは確認できた。③④の完了には追加の調査が必要。

### リスク

使用している通信ライブラリ（`unitree_webrtc_connect`）に2箇所のバグを発見・修正したが、現在は一時的な修正にとどまっている。DimOS の依存パッケージを更新すると修正が失われる可能性がある。

---

## 8. 今後の予定

| タスク | 内容 | 時期 |
|--------|------|------|
| WebRTC 接続の完了 | データチャンネル認証の問題を解消し、G1 との通信を確立する | 次回セッション |
| カメラ映像の実機確認 | `/color_image` topic にデータが流れることを確認する | WebRTC 解決後 |
| 自律歩行のデモ | LiDAR は取得済み。WebRTC が繋がれば自然言語でロボットを操作できる見込み | WebRTC 解決後 |

---

## 9. 相談事項

**通信ライブラリのバグ修正をどう扱うか**

G1 との通信に使用している `unitree_webrtc_connect` というオープンソースライブラリに、Go2・G1 共通の接続バグを2箇所発見した。

- **バグ1**: 接続情報に自分の通信アドレスが含まれていない状態でG1に送ってしまい、G1がタイムアウトする問題
- **バグ2**: G1からの応答を待つ時間が短すぎて（5秒）、通信確立前に接続を諦めてしまう問題

どちらも今回の環境では修正済みだが、**修正は一時的なもので、パッケージ更新時に消える**。

案1・upstream（ライブラリの公式リポジトリ）にバグ報告・PR を送る
案2・DimOS 側で修正版をフォークして管理する
案3・このまま一時修正で運用し、問題が再発したら対処する

現時点では案1が望ましいと考えているが、方針についてご意見をいただきたい。

---

## 10. 参考資料

| 資料 | 場所 |
|------|------|
| 問題の詳細調査ログ | `yokote/20260520/実施メモ/３．抱えている問題.md` |
| G1 agentic blueprint | `dimos/robot/unitree/g1/blueprints/agentic/unitree_g1_agentic.py` |
| G1 接続モジュール（修正済み） | `dimos/robot/unitree/g1/connection.py` |
| G1 primitive blueprint（修正済み） | `dimos/robot/unitree/g1/blueprints/primitive/uintree_g1_primitive_no_nav.py` |
| LiDAR 診断用 blueprint | `dimos/robot/unitree/g1/blueprints/navigation/unitree_g1_mid360_fastlio.py` |
| FastLIO odometry bridge | `dimos/robot/unitree/g1/odometry_bridge.py` |
