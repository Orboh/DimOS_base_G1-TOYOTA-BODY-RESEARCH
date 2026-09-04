# 重力補償ホールドテスト（Dex1-1 公式URDF 校正）

> 位置づけ: IK・カメラ・グリッパーに依存しない **G1ArmSdkConnection 単体の最小テストベンチ**。
> 重力フィードフォワード g(q) だけで右腕を保持できるかを、Dex1-1 装着時の質量モデルを
> 切り替えながら検証する。
> 関連: [IK_REACH_PLAN.md](IK_REACH_PLAN.md)（同じ `G1ArmSdkConnection` を使う片道リーチ計画）

---

## 背景

Unitree 公式配布URDF（`unitree_ros`）の Dex1-1 質量パラメータ（base_link + finger×2 合計
364.946g）が、公式スペックシート値550g・実機実測546gに対し約1.5倍過小評価されていた。これにより
重力補償のフィードフォワードトルクが不足し、実機テストで「肩より上に上げると右腕が垂れる」症状が
確認された。質量を550gに校正したURDF（`g1_dex1_1_calibrated_550g.urdf`）を用意し、垂れの軽減を
確認済み。

調査過程の詳細（実測手順、Isaac Sim USDとの比較など）は Git 履歴・コミットメッセージ
（`9c461603c`）を参照。

---

## 追加したファイル

| Path | 説明 |
|---|---|
| `dimos/robot/unitree/g1/g1_dex1_1_official.urdf` | Unitree公式(unitree_ros, BSD-3-Clause)のG1+Dex1-1両手URDFをそのまま導入(無改変)。 |
| `dimos/robot/unitree/g1/g1_dex1_1_calibrated_550g.urdf` | 上記のDex1-1リンク(base_link+finger×2、合計364.946g)の質量・慣性を係数 `550/364.946 = 1.507072` でスケールし、公式スペック値550gに一致させた校正版。**実機テストで「肩より上げた際の垂れ」が軽減することを確認済みで、現在はこちらを運用設定として採用**。 |
| `dimos/robot/unitree/g1/blueprints/manipulation/unitree_g1_gravity_hold_test.py` | 新規ブループリント `unitree-g1-gravity-hold-test`。IK・カメラ・グリッパー無しで `G1ArmSdkConnection` 単体を `collection_mode` 起動し、重力補償(g(q)フィードフォワード)だけで右腕を保持できるかを確認する最小テストベンチ。起動直後は現在の実測姿勢をSTIFFに保持するだけで動かない。 |
| `scripts/gravity_hold_toggle.py` | 上記ブループリント用の操作スクリプト。別ターミナルで実行し、`c`キーで `/g1/reach_done` をpublishして右腕をcompliant化(kpを1.5秒かけて0にランプダウン+重力FFトルク投入)。右腕7関節角を1Hzでターミナルに表示するので、手を離した後キープできているか目視確認できる。`q`で終了。 |
| `dimos/robot/unitree/g1/examples/start_gravity_hold_test.sh` | 起動スクリプト。`--live`無し=DRY-RUN(実機通信するがarm_sdkには書き込まない)、`--live`ありで実際に駆動。 |
| `dimos/robot/unitree/g1/blueprints/manipulation/unitree_g1_okra_collect.py`(既存への追記) | `OKRA_GRAVITY_URDF` 環境変数を追加し、既存のキネステティック収集パイプライン側でも重力モデルURDFを切り替えられるようにした。 |

---

## 環境変数

| 変数 | 既定値 | 説明 |
|---|---|---|
| `ROBOT_NIC` | `enp46s0` | G1と有線接続するNIC名。**このThinkPadでは `enx6c1ff771dc67` を明示的に指定する必要がある**(既定値のままだと `Cannot find device` エラーで起動に失敗する)。 |
| `OKRA_GRAVITY_URDF` | 空文字(=`g1.urdf`、ダミーラバーハンド170g想定) | 重力補償計算に使うURDFを切り替える。Dex1-1装着時は `dimos/robot/unitree/g1/g1_dex1_1_calibrated_550g.urdf` を指定するのが現在の推奨設定。 |
| `OKRA_GRAVITY_TAU_SCALE` | `1.0` | 重力フィードフォワードの追加スケール(通常は1.0のままでよい。550g校正版で対応済みのため)。 |
| `OKRA_COMPLIANT_KP_RAMP_S` | `1.5` | コンプライアント化にかける時間[s]。 |
| `IK_REACH_LIVE` | 未設定 | `1` を設定しないとDRY-RUN(実機は動かない)。 |

---

## 実行手順(現在の推奨運用設定)

**ターミナル1 — ブループリント起動:**

```bash
cd /home/kota-ueda/Desktop/dimos-hackathon
ROBOT_NIC=enx6c1ff771dc67 \
OKRA_GRAVITY_URDF=dimos/robot/unitree/g1/g1_dex1_1_calibrated_550g.urdf \
  bash dimos/robot/unitree/g1/examples/start_gravity_hold_test.sh --live
```

**ターミナル2 — 操作:**

```bash
cd /home/kota-ueda/Desktop/dimos-hackathon
.venv/bin/python scripts/gravity_hold_toggle.py
```

右腕を手で支えてから `c` を押すとコンプライアント化する。手を離してその場でキープできているか、
1Hzでターミナルに表示される右腕7関節角[deg]を見て目視確認する。終わったらブループリント側を
**Ctrl+Cで安全停止**(arm_sdkのweightを1→0にランプダウンしてから停止)。

**初回は必ずDRY-RUNで確認すること**(`--live`を付けない=実機とは通信するが`rt/arm_sdk`には書き込まない):

```bash
ROBOT_NIC=enx6c1ff771dc67 \
OKRA_GRAVITY_URDF=dimos/robot/unitree/g1/g1_dex1_1_calibrated_550g.urdf \
  bash dimos/robot/unitree/g1/examples/start_gravity_hold_test.sh
```

---

## 安全上の注意

- **`c` を押す前に必ず右腕を支えること**。コンプライアント化の瞬間、支えていないと落下する。
- **E-stopをすぐ押せる状態で行うこと**。
- **初回は `--live` 無し(DRY-RUN)でログだけ確認してから `--live` にすること**。
