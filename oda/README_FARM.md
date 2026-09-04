# 農場オフライン作業手順 — ZED × YOLO × IK 自動オクラ把持

現地でネットが無くても手元で読める作業手順の親ページ。詳細は各リンク先。

## ファイルの場所

| ファイル | 中身 | 用途 |
|---|---|---|
| `oda/start_zed_yolo_auto.sh` | **実行スクリプト**（1本で全部起動） | 現地でこれを叩く |
| `oda/RUN_ZED_YOLO_IK_AUTO.md` | 手順書（手動コマンド全部＋トラブル対処） | スクリプトが詰まった時の逆引き |
| `oda/RUN_ZED_IK.md` | ZED版クリック運用の詳細＋トラブル対処 | 深掘り参照 |
| `oda/RUN_CHEATSHEET.md` | 全知見のコピペ一覧・環境変数リファレンス | 深掘り参照 |
| `oda/ZED_M_Depth_check/finetune_V5/model/okra11n-seg.pt` | YOLOオクラ検出モデル | 存在確認用 |

---

## 事前（屋内・ネットありで／テザリング可）

1. **ZEDキャッシュ温め**: `dimos run unitree-g1-zed-ik-view` で点群が出るのを確認
   （工場キャリブ／NEURAL深度モデルはシリアル毎に初回1回だけDL → **ZED個体を替えたら必須**）
2. **YOLOモデル確認**: `oda/ZED_M_Depth_check/finetune_V5/model/okra11n-seg.pt` があるか
3. **Wi-Fiを切って一度通しで動かす** ＝ 唯一のオフライン動作証明

実行時にインターネットは一切使わない（DimOS/IK/DDS/LCMは全ローカル）。テザリングは初回キャッシュ温めの保険。

---

## 現地（3ステップ）

```bash
# 0) 準備
#    G1電源ON → リモコン L2+↑ → R1+Y
#    ZEDをUSB3ポート 4-4 へ（4-1は不良）。PCはAC電源。e-stop(L2+B)を手元に。

# 1) スクリプトを叩くだけ（中で sudo3行・アプリ・ビューア・YOLOブリッジを全部起動）
cd ~/Toyota-auto-body-PoC/DimOS_oda
bash oda/start_zed_yolo_auto.sh           # 段階1 DRY-RUN（腕停止・座標だけ表示）

# 2) 座標がオクラと合うのを確認したら次へ
bash oda/start_zed_yolo_auto.sh --hover   # 段階2（5cm手前ホバー・掴まない／奥行き検証）

# 3) 奥行きOKなら本把持
bash oda/start_zed_yolo_auto.sh --live    # 段階3（腕リーチ＋自動全閉）
```

**操作**: 起動後この端末で **Enter＝オクラ1個を検出→収穫 / q+Enter＝全停止**
（q+Enterでアプリ・ビューアも自動で片付く）

グリッパのゼロ点がズレてる時は close_q を上書き:
```bash
OKRA_NOACT_CLOSE_Q=1.7 bash oda/start_zed_yolo_auto.sh --live   # 刃全閉q(≈1.8)-0.1
```

---

## 停止・非常時

- **通常停止**: `q`+Enter または Ctrl-C → ログ `/tmp/zed_yolo_auto_app.log` に
  `G1ArmSdkConnection disconnected` を確認 → 電源OFF
- **グリッパが固まった**: リモコン **L2+B**（ダンピング＝機体脱力、支え必須／復帰 L2+↑）、または
  ```bash
  DEX1_NIC=enp46s0 DEX1_OPEN_Q=3.7 .venv/bin/python oda/gripper_open.py
  ```

---

## 現地で詰まったら

| 症状 | 対処 |
|---|---|
| 点群が出ない | sudo3行が効いてない。`ip link show lo` に `MULTICAST` があるか確認 |
| YOLOが検出しない | `OKRA_YOLO_CONF=0.15 bash oda/start_zed_yolo_auto.sh` で下げる |
| 重心近傍に点群なし | `YOLO_BRIDGE_PX_RADIUS=30 bash ...`／ZEDは対象から35cm以上離す |
| `3D点が遠すぎ`で中止 | 対象に近づく（既定0.8m上限） |
| アプリ起動20秒で足りない | スクリプト内の `sleep 20` を伸ばす |
| 腕が的とズレる | ZED付け直し由来。`RUN_ZED_IK.md`の`ZED_MOUNT_XYZRPY`再計測 |
| ビューアが古い画面のまま | ビューアを閉じて開き直す（スクリプト再実行） |
| ZED映像が帯状に乱れる | USBポート4-4に挿す（4-1不良） |
| PCが落ちる | AC接続確認・`ZED_DEPTH_MODE=PERFORMANCE`で負荷1/10 |

詳しい逆引きは `oda/RUN_ZED_YOLO_IK_AUTO.md` の表。

---

## ⚠️ 重要な注意

- スクリプトは構文チェック（`bash -n`）済みだが、**実機での通し実行は未検証**。
  **必ず DRY-RUN → hover → live の順**で上げること。
- ZEDの自動クローズ把持は実機未達・ZED運用中のPCフリーズ実績あり。無理せず段階で。
- 実機で通ったら、成功したパラメータ（close_q・conf・待ち時間等）をこのページと
  `oda/RUN_ZED_YOLO_IK_AUTO.md` に反映すること。
