# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""``start()`` でオクラ収穫 LangGraph フローを実行する DimOS モジュール。

収穫オーケストレーター（グラフ + スキル + SafetyMonitor + 日本語音声）を
デプロイ可能な Module としてラップし、``dimos run unitree-g1-okra-harvest``
でフロー全体を起動する。デフォルトは **DUMMY** スキル（ロボットなし）—
各アクションは ``[DUMMY]`` をログ出力し、音声行は 🔊 プレフィックス付きで表示される。

実機を動かす場合は ``use_dummy=False`` を使用する予定だが、現時点では未接続
（実際の把持 = 停止可能な okra-ACT GraspModule、検出 = YOLO+深度 等 — ``README.md`` 参照）。
見せかけを避けるため、現在は ``NotImplementedError`` を送出する。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
import threading
from threading import Thread
from typing import Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.harvest.announce import CallableAnnouncer
from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig, initial_state
from dimos.robot.unitree.g1.harvest.dummy_skills import DummyHarvestSkills
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph
from dimos.robot.unitree.g1.harvest.real_skills import build_live_harvest_skills
from dimos.robot.unitree.g1.harvest.safety import SafetyCheck, SafetyMonitor
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class HarvestModuleConfig(ModuleConfig):
    use_dummy: bool = True  # True = DUMMY（ロボットなし）; False = LIVE（実カメラで YOLO 検出）
    num_okra: int = 3  # ダミーフィールドのオクラ本数（ダミーモードのみ）
    stations: int = 1  # ダミー作業ステーション数（ダミーモードのみ）
    # LIVE 検出対象クラス。標準 yolo11n は COCO（"okra" にはファインチューニング済み重みが必要）—
    # "banana" は実カメラ→検出→選択パスの動作確認用プロキシ。okra 重み投入後は "okra"。
    target_classes: str = "banana"
    # LIVE YOLO 重み。既定は COCO yolo11n（banana プロキシ用）。オクラ専用 seg モデルは
    # HuggingFace Kota0612/okra-seg-detector（[[SS-01-オクラ検出]]）。ローカルパス or
    # ultralytics が解決できる名前を渡す。seg モデルならマスク重心+depth median で3D化。
    yolo_model: str = "yolo11n.pt"
    recursion_limit: int = 400  # LangGraph ステップ上限（ループでノードを再訪するため多め）
    # LIVE: G1 スピーカーで日本語音声再生（pyopenjtalk + PlayStream）。
    # False = コンソールにログ出力（ロボットなし / 音声依存なし）。
    use_g1_speaker: bool = False
    network_interface: str = ""  # G1 音声 DDS 用 NIC（未設定時は ROBOT_INTERFACE を使用）
    # LIVE: ローカル Ollama ビジョンモデルで verify_harvest を実行。"moondream" = 高速
    # キャプション+キーワード（約1秒）; "qwen3-vl:2b" = チャット yes/no（約5秒、多言語対応）。
    # 空文字 = プレースホルダー検証（常に True）。ollama_vlm.py 参照。
    vlm_model: str = ""
    ollama_host: str = ""  # Ollama ベース URL（空 = ollama_vlm DEFAULT_HOST / Jetson）
    # LIVE: YOLO の代わりに同じ Ollama ビジョンモデルで detect_okra を実行（存在確認）。
    # VLM がオクラを検出した場合に1本を返す — オクラ学習済み YOLO 重みなしで
    # 検出後フロー（把持/確認/記録/掃引）を動作確認できる。
    # vlm_model を使用（未設定時は "moondream"）。ollama_vlm.py 参照。
    use_vlm_detect: bool = False
    # LIVE: cmd_vel（SDK LocoClient）で実機ベースの再配置/掃引を制御。
    # ⚠️ ロボットが歩行します — デフォルトは OFF; 実機安全確認 + オペレーター立会いのもとで有効化。
    use_base_move: bool = False
    # LIVE: スポット使い切り後（左掃引完了、オクラなし）に前進して
    # 探索を継続する — ナビゲーションスタックの暫定代替（前方にオクラがある可能性）。
    # use_base_move が必要。make_search_forward 参照。
    use_forward_search: bool = False
    search_forward_step: float = 0.30  # [m] 前進探索1ステップあたりの移動距離
    max_search_advances: int = 3  # 前進探索ステップ上限（超えるとランを終了）
    # relative_move の変位[m]をタイムド速度指令へ変換する際の並進速度 [m/s]。
    # G1 LocoClient.Move(continuous_move=True) は ≥0.3 m/s でないと実際には歩き
    # 出さない（real_skills.py._BASE_SPEED 参照）。既定 0.8（ユーザー要望で
    # 0.5→0.8、2026-09-08）。
    base_speed: float = 0.8
    # LIVE: フロー開始前に最初のカメラフレームが届くまで最大この時間 [s] 待機し、
    # 最初の検出で空画像を掴まないようにする。
    first_frame_timeout_s: float = 10.0
    # LIVE: ZED 深度画像を depth_getter として使用し、YOLO 検出の 3D 位置精度を向上させる。
    # ZEDCamera が depth_image を出力するブループリント（unitree-g1-okra-harvest-zed）で使用。
    use_zed_depth: bool = False
    # LIVE: 把持に実機 okra-ACT（停止可能 ActGraspModule）を使用。⚠️ アームが動きます —
    # デフォルトは OFF; act_service + アーム/グリッパー接続の配線が必要。
    use_act_grasp: bool = False
    act_endpoint: str = "tcp://127.0.0.1:5701"  # okra-ACT 推論サービス（ZMQ REP）
    grasp_max_steps: int = 120  # ACT 到達エピソード長の上限
    # ACT モデルが手首単眼・右腕7次元（sotata/act-okura-kinesthetic-wrist-7d）か。
    # True: state/action は右腕7関節のみ、画像は手首1枚、グリッパ次元なし（切断は ACT 外）。
    # False（既定）: 旧 8次元右腕+グリッパ / 16次元両腕モデル（後方互換）。
    act_right_arm_only_7d: bool = False
    # LIVE: 把持を IK 粗アプローチ→(任意)ACT 微調整→切断可否→切断 のシーケンス
    # （GraspSequence）で行う。use_act_grasp=True と併用: ActGraspModule を
    # GraspSequence でラップし、重心への IK 接近後に ACT で切断点へ寄せ、グリッパを
    # 閉じて切断する（[[SS-04/05/06]]）。use_act_grasp=False と併用: ACT を挟まず、
    # IK 到達後そのまま切断可否チェック→グリッパを閉じる（ACT無し、スクリプト式）。
    # False（既定）なら use_act_grasp のみで従来どおり ACT 単独（後方互換）。
    use_ik_grasp_sequence: bool = False
    # LIVE + use_ik_grasp_sequence: 最初の把持ループ開始前にこの秒数だけ待つ。
    # 0（既定）= 待たない（後方互換）。
    # ⚠️ 2026-09-08 実機LIVEで判明: クリック駆動版(IkReachBridge)は人間が実際にクリック
    # するまでの自然な待ち時間が G1ArmSdkConnection の重力補償ランプ(stiff_gravity_ramp_s、
    # 既定5s で 0→100% に立ち上がる)の完了を意図せず待っていたが、この自動検出版は
    # カメラフレーム受信直後(起動から1秒未満)に即座にIK到達を開始するため、重力補償が
    # 16-41%程度しか立ち上がっていない状態で腕を動かし、重力に負けて目標に届かない
    # （中途半端な位置で止まる）現象が実機で確認された。ブループリント側で
    # G1ArmSdkConnection.stiff_gravity_ramp_s と同じ値（gravity_ff 有効時のみ）を渡す。
    pregrasp_settle_s: float = 0.0
    # LIVE + use_ik_grasp_sequence: IK 粗アプローチを「① LIFT: 現在の手先位置で真上へ
    # ② TRANSIT: 対象の真上（高度を保持）へ水平移動 ③ DESCEND: 対象へ垂直降下」の
    # 3段階Cartesian経路にする[m]（IkApproachSkill.solve_legs 参照）。0（既定）=
    # 直接一発リーチ（レガシー）。クリック駆動版(IkReachBridge.approach_above_m)と同じ
    # 考え方 — 低い休憩姿勢から一発で関節空間リーチすると、手先の実軌道が読めない弧を
    # 描き、株を払ったり不自然な軌道になったりすることが実機で確認されたため追加。
    ik_approach_above_m: float = 0.0
    # LIVE + use_ik_grasp_sequence: IK 粗アプローチを「① align: 今の奥行き(X)のまま
    # 対象の高さ(Z)と左右位置(Y)を同時に合わせる ② push: その位置から奥行き(X)方向へ
    # まっすぐ押し込む」の2段階Cartesian経路にする[m]（IkApproachSkill.solve_legs/
    # stream_legs の front_m 参照）。0（既定）= 使わない。above_m と front_m を
    # 両方>0にすると ik_approach.py 側の規約により above が優先される。クリック
    # 駆動版(IkReachBridge.approach_front_m)にのみあった方式を2026-09-11に
    # IkApproachSkill へ移植し、2026-09-12のsim比較検証（above vs front）を経て
    # unitree_g1_okra_honban.py が本番既定として採用（unitree_g1_okra_harvest_zed.py
    # 側は above_m のまま残し、いつでも above 方式へ戻せるフォールバックにしている）。
    ik_approach_front_m: float = 0.0
    # LIVE + use_ik_grasp_sequence: 切断点手前でIKを止める量 [m]（IkApproachSkill.standoff_m
    # 参照）。本来は「IKは重心へ寄せれば十分、最後の standoff_m 分は ACT が詰める」設計
    # （既定 0.05 = IkApproachSkill 既定値と同一、後方互換）。use_act_grasp=False
    # （no-ACT構成、unitree_g1_okra_honban.py）では ACT が standoff を詰めるステップが
    # 無いため、既定の 0.05 のままだと刃が莢まで届かない。no-ACT構成では 0.0 を渡し、
    # IK自体に重心（切断点）まで到達させること（2026-09-14 ユーザー指摘）。
    ik_approach_standoff_m: float = 0.05
    # LIVE + use_ik_grasp_sequence: IK到達判定の許容残差 [m]（IkApproachSkill
    # .max_reach_pos_err_m 参照。内部ソルバー自体の収束判定eps=1e-4(0.1mm)とは別物 —
    # ここが効くのは「反復上限まで解いても収束しきらなかった(best-effort)」少数
    # ケースのみ）。既定 0.003 = 2026-09-12 に莢の精度要求(3mm)へ厳格化した値と
    # 同一（後方互換）。実機再検証等でコマンドから緩めたい場合に上書きする
    # （2026-09-14 ユーザー要望。honban.py の OKRA_MAX_REACH_POS_ERR_M 参照）。
    ik_approach_max_reach_pos_err_m: float = 0.003
    # LIVE + use_ik_grasp_sequence: front方式の align フェーズで許容する最大
    # 前進量 [m]（IkApproachSkill.front_align_margin_m 参照。対象の手前この
    # 距離までは、Y,Zを合わせる際に奥行き(X)が動いてよい）。既定0.05は
    # 「align中はXを完全固定」だと体に近い浅いXから始めた際にY,Zの自由度が
    # 2軸しか無く関節可動域超過で頻繁にrejectされていた問題への対処
    # （2026-09-14 ユーザー指摘・実機LIVEで確認）。
    ik_approach_front_align_margin_m: float = 0.05
    # LIVE + use_ik_grasp_sequence: 起動直後・把持ループ開始前に一度だけ、腕をこの
    # torso座標 "x,y,z"[m] へ IK で移動させる（IkApproachSkill.solve() で解き、
    # return_to_rest.py と同じ多段補間・控えめ速度で送る）。空文字（既定）=
    # スキップ・後方互換。
    # ⚠️ 2026-09-14 実機LIVEで判明: front方式の align は「今の奥行き(X)のまま」
    # Y,Zを合わせる設計だが、G1の休憩姿勢（腕を下げた状態）の手先Xを実測すると
    # 約0.047m — IkApproachSkill のワークスペース下限 ws_x[0]=0.05m をわずかに
    # 下回っている。このため休憩姿勢から align を始めると初手から「workspace
    # 外」でreject、または途中で手首が窮屈になり関節可動域超過でrejectされ、
    # 手が一切届かない（9/12のIsaac Sim検証は肘を曲げた前倣え姿勢からの起動
    # だったため気づかれなかった）。起動時に一度、ワークスペース内で安定して
    # 到達できる位置へ動かしておくことで解消する。IKが解けない場合は警告を
    # 出して休憩姿勢のまま続行する（起動は止めない）。
    pregrasp_pose_torso_xyz: str = ""
    # LIVE + use_ik_grasp_sequence: 起動直後・把持ループ開始前に一度だけ、腕を
    # この右腕7関節角度 "q0,q1,...,q6"[rad](正準順)へ直接移動させる
    # （IKを経由しない、return_to_rest.py と同じ多段補間・控えめ速度で送る）。
    # 空文字（既定）=未指定。指定されていれば pregrasp_pose_torso_xyz より
    # こちらを優先する。
    # ⚠️ IKで座標から自動計算した準備姿勢は「解けるか」しか保証しないため、
    # 2026-09-14 ユーザー提案により、G1ArmSdkConnection.collection_mode
    # （重力補償のみのコンプライアントモード、実機検証済み）で人の手で実際に
    # 動かして決めた姿勢をそのまま使えるようにした。教示手順は
    # unitree_g1_teach_pregrasp_pose.py ブループリント参照
    # （OKRA_PREGRASP_POSE_Q7 経由でここへ渡す）。
    pregrasp_pose_q7: str = ""
    # LIVE + use_ik_grasp_sequence: IK粗アプローチを IkApproachSkill.stream_legs（密な
    # Cartesianストリーミング、クリック駆動版 IkReachBridge._stream_leg と同じ密度）で
    # 実行する。False（既定）= solve_legs（レグの端点だけを解いて関節空間補間任せに
    # する簡易版）。ユーザー要望（2026-09-08）でクリック版と同じ動きの質感に近づける
    # ために追加。この関数はストリーミング中ブロックする（中断チェックを持たないため、
    # SafetyMonitor のファイル e-stop は完了まで効かない — ハードウェアの e-stop に頼る）。
    ik_stream_legs: bool = False
    ik_stream_step_m: float = 0.035
    ik_stream_cadence_s: float = 0.18
    # ⚠️ SAFETY: G1ArmSdkConnection には publish_cmd という DRY-RUN 切替があるが、
    # G1GripperConnection にはそれに相当するゲートが無い（gripper_target を受け取れば
    # 無条件で Dex1 へ送信する）。GraspSequence.publish_gripper / basket_deposit の
    # open_gripper はこのフラグで明示的にガードし、False（既定）なら実際には publish
    # せずログのみとする。2026-09-08 実機DRY-RUNで、G1GripperConnection の起動が
    # rt/dex1/*/state 未受信でタイムアウト・クラッシュする前にグリッパ閉コマンドが
    # 発行されていたことが判明（このケースは購読登録前だったため実害なしと推定される
    # が、タイミング次第では防げなかった）。True にする前に実機のグリッパ挙動を確認すること。
    gripper_live: bool = False
    # [rad] 切断時のグリッパ閉じ位置。既定4.4はDex1-1公式サービスの仕様
    # （手で固く閉じた状態をq=0として校正=qが小さいほど閉じる、q増加が開く方向）
    # とは逆向きの値だったことが2026-09-14 実機確認(oda/gripper_move_probe.py /
    # gripper_close_probe.py)で判明。honban.py（アタッチメント無しの素のDex1-1
    # 構成）は OKRA_CUT_CLOSE_Q="0.0" で上書きして正しい方向（全閉）にしている
    # ——このデフォルト自体は他ブループリントとの後方互換のため変更していない。
    cut_close_q: float = 4.4
    # [rad] グリッパの開き方向の安全上限（機械限界 5.4 の手前、過電流フォルト
    # 回避）。qが小さいほど閉じる/大きいほど開く（上記コメント参照）ため、
    # 実質「開きすぎ防止の上限」として機能する。
    blade_max_q: float = 5.2
    # 切断（グリッパ閉）指令の後、実際に閉じきるまで待つ秒数（GraspSequence.cut_settle_s
    # 参照）。0（既定）だと use_basket_deposit=True の場合に、グリッパが閉じきる前に
    # 籠投入の開き指令が飛ぶ（2026-09-08 実機LIVEで確認）。use_basket_deposit=True と
    # 併用するなら必ず正の値（gripper_grasp_on_reach.py の grasp_settle_s と同程度、
    # 1.5s 前後）を設定すること。
    cut_settle_s: float = 0.0
    # LIVE + use_ik_grasp_sequence: 切断（グリッパ閉）の後、右腕のみで腹部固定かご
    # （basket_link, pelvis接合）へ IK 投入 → 開放（F-07, harvest/basket_deposit.py）。
    # ⚠️ MuJoCoでのみ自己衝突検証済み — 実オクラでのLIVE実行前に必ずDRY-RUNで確認
    # すること（basket_deposit.py のSAFETY注記参照）。False（既定）= 従来どおり
    # 切断後は保持したまま（プレースホルダー・F-07未接続）。
    use_basket_deposit: bool = False
    # LIVE + use_basket_deposit: 教示済みの籠投入姿勢（右腕7関節[rad]、正準順、
    # カンマ区切り、unitree-g1-teach-pregrasp-poseと同じ手法で教示）。3つとも
    # 指定されていればIKを使わずこれを直接再生する（推奨、basket_deposit.py の
    # docstring参照）。IK座標(entry_torso等)は自己干渉モデルを持たないため、
    # お腹や籠の縁に干渉する経路を解いてしまうリスクがある
    # （2026-09-14 ユーザー指摘）。空文字（既定）=IKモード（後方互換）。
    basket_entry_q7: str = ""
    basket_drop_q7: str = ""
    basket_retreat_q7: str = ""
    # LIVE + use_basket_deposit: 籠投入時にオクラをリリースする開き角度[rad]
    # （make_basket_deposit_fn の q_open 参照）。既定 3.7 = basket_deposit_bridge.py
    # 由来の実機実績値。qが大きいほど開く方向（cut_close_q コメント参照）なので、
    # 3.7 は起動時の休憩姿勢(≈3.7)と同程度に開いた状態 — 2026-09-14 ユーザー確認
    # により、リリース角度としてはこのままで十分（フルの開き上限blade_max_q=5.2
    # まで開く必要はない）。
    basket_open_q: float = 3.7
    # ZED→torso のハンドアイ外部パラメータ（重心3D を IK の torso フレームへ変換）。
    # 空 = 未校正（Step 4 で配線）。形式は [x,y,z, qx,qy,qz,qw]（torso<-camera）。
    cam_to_torso_xyzquat: str = ""
    # §6 実機安全（実機動作が有効な場合に使用）。ファイル E-stop: `touch` で一時停止。
    safety_estop_file: str = "/tmp/okra_estop"
    torque_limit: float = 0.0  # [N·m] アームトルク接触ガード; 0 = OFF（要チューニング）
    # §6 HMI: 把持/移動/籠交換など「これから物理的に動く」ノードで、音声を発してから
    # 実際に動作を送信するまでこの秒数 [s] だけ待つ。G1SpeakerAnnouncer.say() はキュー投入
    # 後すぐ返り再生を待たないため、待たないと動作が音声より先に（あるいは重なって）
    # 始まり、今何をしているか分からなくなる（HarvestConfig.voice_lead_s 参照）。
    # 0.0（既定）= 待たない（後方互換）。
    voice_lead_s: float = 0.0
    # §5 sweep（HarvestConfig.advance_step/max_empty_advances のパススルー、既定
    # None=HarvestConfigのデフォルト値のまま=後方互換）。sim検証で「10mくらい
    # 移動できるか」等、広い探索範囲を試したい場合に env から上書きできるように
    # する（2026-09-12 要望）。max_empty_advances を増やさないと、advance_step
    # 間隔でオクラが見つからない距離が続くと REVISIT に切り替わり探索が止まる。
    advance_step: float | None = None  # [m] 左sweep1回の移動量（既定 HarvestConfig 0.30m）
    max_empty_advances: int | None = None  # 連続空振り上限（既定 HarvestConfig 2）


class HarvestModule(Module):
    """デプロイ時にワーカースレッドでオクラ収穫 LangGraph フローを実行する。

    ``use_dummy=True``（デフォルト）: ロボット不要の完全自己完結型 DUMMY フロー。
    ``use_dummy=False``（LIVE）: ヘッドカメラ ``color_image`` ストリームで実 YOLO 検出を行う。
    確認/移動/把持/ナビは引き続き ``[LIVE-TODO]`` プレースホルダー（VLM 確認、
    okra-ACT GraspModule、ベース動作とナビは今後対応）のため、
    実知覚のみを実行し実機動作は行わない。
    """

    config: HarvestModuleConfig
    color_image: In[Image]  # ヘッドカメラ（LIVE モード）; ダミーモードでは未使用
    depth_image: In[
        Image
    ]  # ZED 深度画像（LIVE + use_zed_depth）; 未接続時は仮定深度にフォールバック
    camera_info: In[CameraInfo]  # ZED 内部パラメータ（LIVE + use_zed_depth、実逆投影に使用）
    cam_right_wrist: In[Image]  # 右手首カメラ（LIVE + use_act_grasp、2カメラツリーモデル）
    cmd_vel: Out[Twist]  # ベース速度（LIVE + use_base_move）-> G1Connection
    # アームストリーム（LIVE + use_act_grasp）-> G1ArmSdkConnection / G1GripperConnection
    motor_states: In[JointState]
    right_gripper_state: In[JointState]
    arm_target: Out[JointState]
    gripper_target: Out[JointState]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._thread: Thread | None = None
        self._monitor: SafetyMonitor | None = None
        self._app: Any = None
        self._voice: Any = None
        self._lock = threading.Lock()
        self._latest_image: Image | None = None
        self._latest_depth: Image | None = None
        self._latest_camera_info: CameraInfo | None = None
        self._latest_wrist: Image | None = None
        self._latest_state: JointState | None = None
        self._latest_gripper: float = 0.0
        # 起動時（重力補償ランプ待ち完了後・最初の把持前）の腕姿勢（左7+右7）。
        # 籠投入後にここへ復帰させる（return_to_rest.py）。
        self._rest_q14: list[float] | None = None
        # 直近のIK計算で狙ったtorso座標（到達確認用、2026-09-14追加）。
        # _ik_solve が計算するたびに更新し、GraspSequence.post_reach_verify_fn
        # から参照する。
        self._last_ik_target_torso: list[float] | None = None

    def _on_wrist(self, image: Image) -> None:
        with self._lock:
            self._latest_wrist = image

    def _on_state(self, state: JointState) -> None:
        with self._lock:
            self._latest_state = state

    def _on_gripper(self, state: JointState) -> None:
        pos = list(state.position)
        if pos:
            with self._lock:
                self._latest_gripper = float(pos[0])

    def _build_voice(self) -> Any:
        """デフォルトはコンソールログアナウンサー; use_g1_speaker が True なら実 G1 スピーカーを使用。"""
        if self.config.use_g1_speaker:
            from dimos.robot.unitree.g1.harvest.g1_speaker import make_g1_playstream_announcer

            nic = self.config.network_interface or os.getenv("ROBOT_INTERFACE", "")
            try:  # デプロイ環境によって DDS が既に初期化済みの場合がある
                return make_g1_playstream_announcer(nic, init_dds=True)
            except Exception as exc:
                logger.warning(
                    "G1 speaker init_dds=True failed; retry init_dds=False", error=str(exc)
                )
                try:
                    return make_g1_playstream_announcer(init_dds=False)
                except Exception as exc2:
                    logger.warning("G1 speaker unavailable; using console voice", error=str(exc2))
        return CallableAnnouncer(lambda text: logger.info(f"🔊 {text}"))

    def _build_safety_checks(self) -> list[SafetyCheck]:
        """実機動作が有効な場合は §6 実機チェック; それ以外はダミーの常時安全チェック。"""
        # use_ik_grasp_sequence moves the arm (IK reach + cut) even with use_act_grasp=False
        # (ACT-free mode, see run()) -- must count as real motion for §6 checks too.
        real_motion = (
            self.config.use_act_grasp
            or self.config.use_ik_grasp_sequence
            or self.config.use_base_move
        )
        if not real_motion:
            return [SafetyCheck("dummy_person_clear", lambda: True)]
        from dimos.robot.unitree.g1.harvest.safety_checks import FileEStop, make_torque_check

        checks = [FileEStop(self.config.safety_estop_file).as_check()]
        if self.config.torque_limit > 0:
            checks.append(
                make_torque_check(lambda: self._latest_state, limit=self.config.torque_limit)
            )
        logger.info(
            f"SafetyMonitor real checks: file e-stop={self.config.safety_estop_file!r} "
            f"(touch to pause), torque_limit={self.config.torque_limit}"
        )
        return checks

    def _parse_cam_to_torso(self, spec: str) -> Callable[[Sequence[float]], list[float]] | None:
        """``"x,y,z,qx,qy,qz,qw"`` → 関数 ``[x,y,z](camera)->[x,y,z](torso)``。

        空文字なら None（未校正; その場合 IK には camera 系座標がそのまま渡る＝Step 4 で
        実校正値を入れるまでの暫定）。実際のハンドアイ校正値は [[SS-04-粗アプローチIK]]。
        """
        spec = (spec or "").strip()
        if not spec:
            return None
        import numpy as np

        vals = [float(v) for v in spec.replace(" ", "").split(",")]
        if len(vals) != 7:
            logger.warning(
                f"cam_to_torso_xyzquat needs 7 values (x,y,z,qx,qy,qz,qw); got {len(vals)}; ignoring"
            )
            return None
        tx, ty, tz, qx, qy, qz, qw = vals
        # quaternion(xyzw) -> 回転行列（torso<-camera）
        n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5 or 1.0
        qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
        rot = np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ]
        )
        trans = np.array([tx, ty, tz])

        def _to_torso(p_cam: Sequence[float]) -> list[float]:
            return list(rot @ np.asarray(p_cam, dtype=float) + trans)

        return _to_torso

    def _on_image(self, image: Image) -> None:
        with self._lock:
            self._latest_image = image

    def _on_depth(self, image: Image) -> None:
        with self._lock:
            self._latest_depth = image

    def _on_camera_info(self, info: CameraInfo) -> None:
        with self._lock:
            self._latest_camera_info = info

    def _zed_intrinsics(self) -> tuple[float, float, float, float] | None:
        """``camera_info.K``（3x3 row-major）から (fx, fy, cx, cy) を返す。未受信なら None。"""
        with self._lock:
            info = self._latest_camera_info
        k = getattr(info, "K", None) if info is not None else None
        if not k or len(k) < 6:
            return None
        return float(k[0]), float(k[4]), float(k[2]), float(k[5])

    @rpc
    def start(self) -> None:
        super().start()
        voice = self._build_voice()
        self._voice = voice

        if self.config.use_dummy:
            skills: Any = DummyHarvestSkills(
                num_okra=self.config.num_okra, stations=self.config.stations
            )
            grasp_module = skills.grasp_module
            mode = "DUMMY（ロボットなし）"
        else:
            self.register_disposable(Disposable(self.color_image.subscribe(self._on_image)))
            targets = {c.strip() for c in self.config.target_classes.split(",") if c.strip()}

            depth_getter = None
            pixel_to_base = None
            depth_note = "depth=assumed(0.45m)"
            if self.config.use_zed_depth:
                import numpy as np

                self.register_disposable(Disposable(self.depth_image.subscribe(self._on_depth)))
                self.register_disposable(
                    Disposable(self.camera_info.subscribe(self._on_camera_info))
                )
                _FALLBACK_DEPTH_M = 0.45  # [m] ZED が値を返さない場合のフォールバック

                def _zed_depth_getter(u: float, v: float) -> float:
                    with self._lock:
                        img = self._latest_depth
                    if img is None:
                        return _FALLBACK_DEPTH_M
                    try:
                        arr = img.data  # float32 [H, W]（ZED MEASURE.DEPTH: メートル値）
                        h, w = arr.shape[:2]
                        d = float(arr[int(np.clip(v, 0, h - 1)), int(np.clip(u, 0, w - 1))])
                        return d if np.isfinite(d) and 0.05 < d < 10.0 else _FALLBACK_DEPTH_M
                    except Exception:
                        return _FALLBACK_DEPTH_M

                depth_getter = _zed_depth_getter
                depth_note = "depth=ZED"

                # ピクセル→3D は D435i（頭部）画角の当て推量ではなく、ZED 自身の camera_info
                # (K) を使った実逆投影で行う（default_pixel_to_base は intrinsics 未受信時
                # のみのフォールバック）。奥行きは上記の ZED mask-median depth のまま。
                from dimos.robot.unitree.g1.harvest.detect_yolo import make_zed_pixel_to_base

                pixel_to_base_cam = make_zed_pixel_to_base(
                    depth_getter=depth_getter,
                    intrinsics_getter=self._zed_intrinsics,
                )

                # Okra.pos_3d は「ロボット本体(torso)基準」であることが reach box
                # （graph.py の select）・IK 双方の前提（blackboard.py の Box3D docstring
                # 参照）。cam_to_torso_xyzquat が校正済みならここで検出時点に torso 座標へ
                # 変換する — reach box チェックにも IK にも同じ変換済み座標が渡るように
                # （以前は IK 直前でしか変換されておらず、reach box は未校正のカメラ座標を
                # 見て誤判定していた）。未校正（空文字）ならカメラ座標のまま通す。
                cam_to_torso = self._parse_cam_to_torso(self.config.cam_to_torso_xyzquat)
                if cam_to_torso is not None:

                    def pixel_to_base(u: float, v: float, det: Any) -> dict[str, float] | None:
                        p_cam = pixel_to_base_cam(u, v, det)
                        # pixel_to_base_cam は深度/camera_info 未受信時に None を返す
                        # 設計（detect_yolo.py 参照: 推測値で腕を動かすより捨てる方が
                        # 安全）。ここで None チェックを忘れると cam_to_torso([None...])
                        # で TypeError になり、LangGraph の detect ノードごとクラッシュ
                        # する（2026-09-08、OKRA_CAM_TO_TORSO を初めて有効にした際に
                        # 顕在化 — このパス自体がそれまで一度も実行されていなかった）。
                        if p_cam is None:
                            return None
                        # ⚠️ 座標系の取り違え（2026-09-08 実機LIVEで発覚、「近すぎる/
                        # 遠すぎる」誤判定の直接原因）: pixel_to_base_cam は harvest
                        # 独自の座標系 x=lateral(+右)/y=depth(+前)/z=height(+上)
                        # （detect_yolo.py 参照）で返すが、cam_to_torso（
                        # _parse_cam_to_torso）は REP-103 光学フレーム
                        # x=右/y=下/z=前 の点を受け取り、torso標準系
                        # x=前/y=左/z=上 を返す設計（IkReachBridge._torso_from_optical
                        # と同じ変換）。この2つの軸割り当ては全く違う（harvestのy=depth
                        # はopticalのz、harvestのz=heightはopticalの-y）ため、素通しで
                        # 渡すと明後日の座標に変換されていた。往復とも正しい軸に
                        # 並べ替える必要がある。
                        p_optical = [p_cam["x"], -p_cam["z"], p_cam["y"]]
                        x_t, y_t, z_t = cam_to_torso(p_optical)
                        # torso(x=前,y=左,z=上) -> harvest(x=lateral+右,y=depth+前,z=height+上)
                        # ＝ Okra.pos_3d / reach box / _ik_solve が前提とする座標系
                        # （HarvestConfig.reach の docstring 参照）。
                        return {"x": -y_t, "y": x_t, "z": z_t}
                else:
                    pixel_to_base = pixel_to_base_cam

            verify_fn = None
            verify_note = "verify=[LIVE-TODO] プレースホルダー"
            if self.config.vlm_model:
                from dimos.robot.unitree.g1.harvest.ollama_vlm import make_ollama_verify

                verify_fn = make_ollama_verify(
                    lambda: self._latest_image,
                    model=self.config.vlm_model,
                    host=self.config.ollama_host or None,
                )
                verify_note = f"verify=Ollama:{self.config.vlm_model}"
            # ⚠️ post_grasp_verify_fn は build_live_harvest_skills に渡す「把持後確認」
            # 専用（route_after_verify が picks カウント・「収穫成功です」発話のゲートに
            # 使う）。verify_fn 自体は GraspSequence.cut_ok_fn（切断可否ゲート）にも
            # 流用するので、ここでは上書きしない — VLM 未配線時 (verify_fn is None) の
            # フォールバックは use_ik_grasp_sequence ブロックで grasp_override 構築後に
            # 確定させる（2026-09-08 実機LIVEで判明: フォールバック無しだと
            # real_skills.py の常時 True プレースホルダーが使われ、IK到達失敗でも
            # 「収穫成功です」と発話され picks がインクリメントされていた）。
            post_grasp_verify_fn = verify_fn

            detect_override = None
            detect_note = "detect=YOLO"
            if self.config.use_vlm_detect:
                from dimos.robot.unitree.g1.harvest.ollama_vlm import make_ollama_detect_okra

                detect_model = self.config.vlm_model or "moondream"
                detect_override = make_ollama_detect_okra(
                    lambda: self._latest_image,
                    model=detect_model,
                    host=self.config.ollama_host or None,
                )
                detect_note = f"detect=Ollama:{detect_model}"

            move_cmd = None
            move_note = "move=[LIVE-TODO] プレースホルダー"
            if self.config.use_base_move:
                from dimos.robot.unitree.g1.harvest.nav_skills import make_twist_move_cmd

                move_cmd = make_twist_move_cmd(self.cmd_vel.publish)
                move_note = "move=cmd_vel(SDK)"

            next_station_override = None
            if self.config.use_forward_search and move_cmd is not None:
                from dimos.robot.unitree.g1.harvest.nav_skills import make_search_forward

                next_station_override = make_search_forward(
                    move_cmd,
                    step_m=self.config.search_forward_step,
                    max_advances=self.config.max_search_advances,
                )
                move_note += "+前進探索"

            # ⚠️ SAFETY: G1GripperConnection には publish_cmd 相当の DRY-RUN 切替が無い
            # （gripper_target を受け取れば無条件で Dex1 へ送信する）。arm_target は
            # G1ArmSdkConnection.publish_cmd（ブループリント側の IK_REACH_LIVE 等）で
            # 保護されているのに対し、グリッパ側はここで明示的にガードしないと
            # config.gripper_live を見ずに実際へ送信してしまう。
            def _publish_gripper_guarded(msg: JointState) -> None:
                if self.config.gripper_live:
                    self.gripper_target.publish(msg)
                    return
                logger.info(
                    f"[DRY-RUN] gripper_live=False: gripper_target publish suppressed "
                    f"(would send position={list(getattr(msg, 'position', []) or [])})"
                )

            grasp_override = None
            grasp_note = "grasp=DUMMY"
            if self.config.use_ik_grasp_sequence:
                # IK 粗アプローチ→(ACTありなら微調整)→切断可否→切断 を1エピソードに束ねる。
                # act_module=None（use_act_grasp=False）なら②相当: ACT を挟まず、IK到達後
                # そのまま切断可否チェック→グリッパを閉じる（GraspSequence.run_episode参照）。
                from dimos.robot.unitree.g1.harvest.grasp_sequence import GraspSequence
                from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill

                self.register_disposable(Disposable(self.motor_states.subscribe(self._on_state)))

                act_module = None
                if self.config.use_act_grasp:
                    from dimos.robot.unitree.g1.harvest.act_grasp import ActGraspModule

                    self.register_disposable(
                        Disposable(self.cam_right_wrist.subscribe(self._on_wrist))
                    )
                    self.register_disposable(
                        Disposable(self.right_gripper_state.subscribe(self._on_gripper))
                    )
                    act_module = ActGraspModule(
                        image_getter=lambda: self._latest_image,
                        wrist_getter=lambda: self._latest_wrist,  # 2カメラ / 手首単眼
                        state_getter=lambda: self._latest_state,
                        gripper_getter=lambda: self._latest_gripper,
                        publish_arm=self.arm_target.publish,
                        publish_gripper=_publish_gripper_guarded,
                        act_endpoint=self.config.act_endpoint,
                        max_steps=self.config.grasp_max_steps,
                        right_arm_only_7d=self.config.act_right_arm_only_7d,
                    )

                ik_skill = IkApproachSkill(
                    standoff_m=self.config.ik_approach_standoff_m,
                    max_reach_pos_err_m=self.config.ik_approach_max_reach_pos_err_m,
                    front_align_margin_m=self.config.ik_approach_front_align_margin_m,
                )

                place_basket_fn = None
                if self.config.use_basket_deposit:
                    from dimos.msgs.sensor_msgs.JointState import JointState as _JointState
                    from dimos.robot.unitree.g1.harvest.basket_deposit import (
                        make_basket_deposit_fn,
                    )

                    def _send_arm(arm14: list[float], _secs: float) -> None:
                        self.arm_target.publish(
                            _JointState(
                                name=list(ik_skill.joint_names),
                                position=[float(x) for x in arm14],
                                velocity=[0.0] * len(arm14),
                                effort=[0.0] * len(arm14),
                            )
                        )

                    def _open_gripper(q: float, _secs: float) -> None:
                        _publish_gripper_guarded(
                            _JointState(
                                name=["g1/right_gripper"],
                                position=[float(q)],
                                velocity=[0.0],
                                effort=[0.0],
                            )
                        )

                    def _get_measured() -> list[float]:
                        with self._lock:
                            state = self._latest_state
                        return list(state.position) if state is not None else [0.0] * 29

                    def _parse_q7(spec: str) -> list[float] | None:
                        if not spec:
                            return None
                        try:
                            q7 = [float(v) for v in spec.split(",")]
                        except ValueError:
                            logger.warning(
                                f"HarvestModule: 籠投入姿勢 {spec!r} の書式が不正 "
                                "(期待形式: カンマ区切り7要素[rad]) — 無視します"
                            )
                            return None
                        if len(q7) != 7:
                            logger.warning(
                                f"HarvestModule: 籠投入姿勢 {spec!r} が7要素でない "
                                f"({len(q7)}要素) — 無視します"
                            )
                            return None
                        return q7

                    place_basket_fn = make_basket_deposit_fn(
                        send_arm=_send_arm,
                        open_gripper=_open_gripper,
                        get_measured=_get_measured,
                        entry_q7=_parse_q7(self.config.basket_entry_q7),
                        drop_q7=_parse_q7(self.config.basket_drop_q7),
                        retreat_q7=_parse_q7(self.config.basket_retreat_q7),
                        q_open=self.config.basket_open_q,
                    )

                # 籠投入後、次のオクラ探索前に起動時の姿勢へ腕を戻す
                # （return_to_rest.py 参照。2026-09-08 実機LIVEで判明した「固定
                # モーションの最後で止まっている」ように見える問題への対処）。
                return_to_rest_fn = None
                if self.config.use_basket_deposit:
                    from dimos.msgs.sensor_msgs.JointState import JointState as _JointState2
                    from dimos.robot.unitree.g1.harvest.return_to_rest import (
                        make_return_to_rest_fn,
                    )

                    def _send_arm14_rest(q14: list[float]) -> None:
                        self.arm_target.publish(
                            _JointState2(
                                name=list(ik_skill.joint_names),
                                position=[float(x) for x in q14],
                                velocity=[0.0] * len(q14),
                                effort=[0.0] * len(q14),
                            )
                        )

                    def _get_measured29() -> list[float]:
                        with self._lock:
                            state = self._latest_state
                        return list(state.position) if state is not None else [0.0] * 29

                    return_to_rest_fn = make_return_to_rest_fn(
                        send_arm=_send_arm14_rest,
                        get_measured=_get_measured29,
                        rest_q_getter=lambda: self._rest_q14,
                    )

                def _ik_solve(okra: Any) -> Any:
                    """対象オクラの重心(pos_3d、既に torso 座標)→右腕 IK。解けなければ None。"""
                    pos = getattr(okra, "pos_3d", None) or {}
                    # pos_3d は use_zed_depth 時点で cam_to_torso 変換済み（上記 pixel_to_base
                    # 参照）だが、この収穫パイプライン独自の座標系（x=左右+右, y=奥行き+前方,
                    # z=高さ+上）のまま。IkApproachSkill（pinocchio）は標準 ROS 座標系
                    # （X=前方, Y=左, Z=上）を要求するので、ここで変換する（変換し忘れると
                    # x/y が取り違えられ、reach box は通っても IK ワークスペース判定で
                    # 「範囲外」と誤って弾かれる）。
                    x_lat = float(pos.get("x", 0.0))
                    y_depth = float(pos.get("y", 0.0))
                    z_height = float(pos.get("z", 0.0))
                    target_torso = [y_depth, -x_lat, z_height]
                    # 到達確認用（2026-09-14追加）: この呼び出しが最後に狙った
                    # 「オクラの重心そのもの」のtorso座標を保持しておく。
                    # GraspSequence.post_reach_verify_fn（_verify_reach）が、IK
                    # legs完了後にこれと実測FK位置を比較する。
                    self._last_ik_target_torso = list(target_torso)
                    okra_id = getattr(okra, "id", "?")
                    logger.info(
                        f"[ik-grasp] {okra_id}: pos_3d(harvest x=lateral,y=depth,z=height)="
                        f"{{'x': {x_lat:.3f}, 'y': {y_depth:.3f}, 'z': {z_height:.3f}}} -> "
                        f"target_torso(X前,Y左,Z上)={[round(v, 3) for v in target_torso]}"
                    )
                    with self._lock:
                        state = self._latest_state
                    if state is None:
                        logger.warning("[ik-grasp] no motor_states yet; cannot solve IK")
                        return None
                    logger.info(
                        f"[ik-grasp] {okra_id}: measured q_right(rad)="
                        f"{[round(float(x), 3) for x in list(state.position)[22:29]]}"
                    )
                    if self.config.ik_stream_legs:

                        def _send_arm14(arm14: list[float]) -> None:
                            from dimos.msgs.sensor_msgs.JointState import JointState

                            self.arm_target.publish(
                                JointState(
                                    name=list(ik_skill.joint_names),
                                    position=[float(x) for x in arm14],
                                    velocity=[0.0] * len(arm14),
                                    effort=[0.0] * len(arm14),
                                )
                            )

                        return ik_skill.stream_legs(
                            target_torso,
                            list(state.position),
                            above_m=self.config.ik_approach_above_m,
                            front_m=self.config.ik_approach_front_m,
                            send_arm=_send_arm14,
                            step_m=self.config.ik_stream_step_m,
                            cadence_s=self.config.ik_stream_cadence_s,
                        )
                    return ik_skill.solve_legs(
                        target_torso,
                        list(state.position),
                        above_m=self.config.ik_approach_above_m,
                        front_m=self.config.ik_approach_front_m,
                    )

                def _verify_reach() -> None:
                    """到達確認（2026-09-14追加）: IK legs完了直後、実測の右腕
                    関節角度からFK（tip_torso）を計算し、_ik_solve が最後に狙った
                    「オクラの重心」座標との残差をログする。IK の err（ソルバー内部
                    の収束判定）は計算上の値でしかなく、実機のPD制御が実際にそこ
                    まで追従したか・エンドエフェクタが本当に対象へ届いたかは別問題
                    — ユーザー指摘（実機で約5cmずれて把持した事例）を受けて追加。
                    残差が大きい場合、原因が「腕の追従誤差」なのか「そもそも目標
                    座標(cam_to_torso変換/検出)がずれている」のかを切り分ける
                    材料にする。
                    """
                    target = self._last_ik_target_torso
                    if target is None:
                        return
                    with self._lock:
                        state = self._latest_state
                    if state is None:
                        logger.warning(
                            "[reach-verify] motor_states 未受信のため到達確認をスキップ"
                        )
                        return
                    q_right_measured = list(state.position)[22:29]
                    tip = ik_skill.tip_torso(q_right_measured)
                    err_xyz = [float(tip[i] - target[i]) for i in range(3)]
                    err_norm = float(sum(e * e for e in err_xyz) ** 0.5)
                    logger.info(
                        f"[reach-verify] target_torso(重心)={[round(float(v), 4) for v in target]} "
                        f"actual_tip(X前,Y左,Z上)={[round(float(v), 4) for v in tip]} "
                        f"err_xyz={[round(v, 4) for v in err_xyz]} err_norm={err_norm:.4f}m"
                    )

                # 切断可否ゲート: verify_fn（moondream）を流用。未配線なら None=常許可。
                grasp_override = GraspSequence(
                    ik_solve=_ik_solve,
                    publish_arm=self.arm_target.publish,
                    act_module=act_module,
                    cut_ok_fn=verify_fn,
                    publish_gripper=_publish_gripper_guarded,
                    q_close=self.config.cut_close_q,
                    q_blade_max=self.config.blade_max_q,
                    cut_settle_s=self.config.cut_settle_s,
                    place_basket_fn=place_basket_fn,
                    return_to_rest_fn=return_to_rest_fn,
                    post_reach_verify_fn=_verify_reach,
                    announcer=voice,
                )
                grasp_note = (
                    "grasp=IK->ACT->cut(seq)" if act_module is not None else "grasp=IK->cut(no-ACT)"
                )
                if place_basket_fn is not None:
                    grasp_note += "->basket(F-07)"
                if post_grasp_verify_fn is None:
                    # VLM未配線: GraspSequence.episodes の直近の結果（IK到達・切断まで
                    # 到達したか）を「把持後確認」の代理指標として使う。VLMによる独立
                    # 検証ではないため、grasp_okra 内部のゲートを通過した以上のことは
                    # 保証しない（例えば実際に果実を掴めたかまでは確認できない）が、
                    # 「IK失敗でも収穫成功扱いになる」バグ（2026-09-08）は解消する。
                    def _grasp_sequence_verify() -> bool:
                        if not grasp_override.episodes:
                            return False
                        _, _, ok = grasp_override.episodes[-1]
                        return bool(ok)

                    post_grasp_verify_fn = _grasp_sequence_verify
                    verify_note = "verify=grasp_sequence(no-VLM, IK/cut到達を代理指標)"
            elif self.config.use_act_grasp:
                from dimos.robot.unitree.g1.harvest.act_grasp import ActGraspModule

                self.register_disposable(Disposable(self.cam_right_wrist.subscribe(self._on_wrist)))
                self.register_disposable(Disposable(self.motor_states.subscribe(self._on_state)))
                self.register_disposable(
                    Disposable(self.right_gripper_state.subscribe(self._on_gripper))
                )
                grasp_override = ActGraspModule(
                    image_getter=lambda: self._latest_image,
                    wrist_getter=lambda: self._latest_wrist,  # 2カメラ / 手首単眼
                    state_getter=lambda: self._latest_state,
                    gripper_getter=lambda: self._latest_gripper,
                    publish_arm=self.arm_target.publish,
                    publish_gripper=_publish_gripper_guarded,
                    act_endpoint=self.config.act_endpoint,
                    max_steps=self.config.grasp_max_steps,
                    right_arm_only_7d=self.config.act_right_arm_only_7d,
                )
                grasp_note = "grasp=okra-ACT(2cam)"

            skills, grasp_module = build_live_harvest_skills(
                frame_getter=lambda: self._latest_image,
                target_classes=targets,
                detect_fn=detect_override,
                verify_fn=post_grasp_verify_fn,
                move_cmd=move_cmd,
                grasp_module=grasp_override,
                next_station_fn=next_station_override,
                depth_getter=depth_getter,
                pixel_to_base=pixel_to_base,
                yolo_model=self.config.yolo_model,
                base_speed=self.config.base_speed,
            )
            gripper_live_note = f"gripper_live={self.config.gripper_live}"
            mode = (
                f"LIVE — {detect_note}; {depth_note}; {verify_note}; {move_note}; "
                f"{grasp_note}; {gripper_live_note}"
            )

        # 実機動作が有効な場合は §6 実機チェック（ファイル E-stop + トルク）; それ以外はダミー。
        self._monitor = SafetyMonitor(
            self._build_safety_checks(),
            on_pause=lambda reason: grasp_module.stop(),
            announcer=voice,
        )
        self._monitor.start()
        _hcfg_kwargs: dict[str, Any] = {"voice_lead_s": self.config.voice_lead_s}
        if self.config.advance_step is not None:
            _hcfg_kwargs["advance_step"] = self.config.advance_step
        if self.config.max_empty_advances is not None:
            _hcfg_kwargs["max_empty_advances"] = self.config.max_empty_advances
        self._app = build_harvest_graph(
            skills,
            HarvestConfig(**_hcfg_kwargs),
            announcer=voice,
            safety=self._monitor.gate,
        )
        # カメラは別ワーカー/プロセスからストリーミングされる — 最初のフレームが届くまで待機し、
        # フロー最初の検出で空画像を掴まないようにする
        # （そうしないと、フレーム到着前に picks=0 で終了してしまう）。
        if not self.config.use_dummy:
            self._await_first_frames(self.config.first_frame_timeout_s)
            if self.config.pregrasp_settle_s > 0:
                import time as _time

                logger.info(
                    f"HarvestModule: 重力補償ランプ待ち {self.config.pregrasp_settle_s:.1f}s "
                    "（G1ArmSdkConnection.stiff_gravity_ramp_s の完了を待ってから把持を開始）"
                )
                _time.sleep(self.config.pregrasp_settle_s)
            # 起動時、休憩姿勢のままだと front方式の align がworkspace外/関節限界
            # で失敗する問題への対処（config.pregrasp_pose_torso_xyz 参照）。
            # ik_skill は use_ik_grasp_sequence ブロックでのみ定義されるためガードする。
            if self.config.use_ik_grasp_sequence and (
                self.config.pregrasp_pose_q7 or self.config.pregrasp_pose_torso_xyz
            ):
                self._move_to_pregrasp_pose(ik_skill)
            # 最初の把持の直前の腕姿勢を「起動時姿勢」としてキャプチャ（左7+右7）。
            # 籠投入後、次のオクラ探索前にここへ戻る（return_to_rest.py）。
            with self._lock:
                state = self._latest_state
            if state is not None:
                pos = list(state.position)
                self._rest_q14 = pos[15:22] + pos[22:29]
                logger.info(
                    f"HarvestModule: 起動時姿勢キャプチャ q14(左7+右7)="
                    f"{[round(float(x), 3) for x in self._rest_q14]}"
                )
            else:
                logger.warning(
                    "HarvestModule: motor_states 未受信のため起動時姿勢をキャプチャ"
                    "できず（籠投入後の復帰はスキップされる）"
                )
        self._thread = Thread(target=self._run, daemon=True, name="okra-harvest")
        self._thread.start()
        logger.info(f"HarvestModule 起動 — {mode}")

    def _move_to_pregrasp_pose(self, ik_skill: Any) -> None:
        """起動直後に一度だけ、腕を準備姿勢へ移動する。

        ``config.pregrasp_pose_q7``（教示ツールで記録した右腕7関節角度）があれば
        それを優先し、IKを経由せず直接その姿勢へ移動する。無ければ
        ``config.pregrasp_pose_torso_xyz``（torso座標、IKで解く）にフォール
        バックする。休憩姿勢（腕を下げた状態）の手先Xが front-approach の align
        前提（現在のXを維持）と噛み合わずワークスペース外/関節限界で弾かれる
        問題への対処（各フィールドのコメント参照、2026-09-14）。解けない/書式
        不正なら警告を出して休憩姿勢のまま続行する（起動そのものは止めない）。
        """
        q7_spec = self.config.pregrasp_pose_q7
        if q7_spec:
            try:
                q_right_goal = [float(v) for v in q7_spec.split(",")]
            except ValueError:
                q_right_goal = []
            if len(q_right_goal) != 7:
                logger.warning(
                    f"HarvestModule: pregrasp_pose_q7={q7_spec!r} の書式が不正 "
                    "(期待形式: カンマ区切り7要素[rad]) — 準備姿勢への移動をスキップ"
                )
                return
            with self._lock:
                state = self._latest_state
            if state is None:
                logger.warning(
                    "HarvestModule: motor_states 未受信のため準備姿勢への移動をスキップ"
                )
                return
            q_left = list(state.position)[15:22]
            arm14_goal = list(q_left) + q_right_goal
            logger.info(f"HarvestModule: 準備姿勢(教示q7)へ移動開始 q_right={q_right_goal}")
            self._move_arm_to_pose(ik_skill, arm14_goal)
            return

        spec = self.config.pregrasp_pose_torso_xyz
        if not spec:
            return
        try:
            xyz = [float(v) for v in spec.split(",")]
        except ValueError:
            xyz = []
        if len(xyz) != 3:
            logger.warning(
                f"HarvestModule: pregrasp_pose_torso_xyz={spec!r} の書式が不正 "
                '(期待形式 "x,y,z") — 準備姿勢への移動をスキップ'
            )
            return
        with self._lock:
            state = self._latest_state
        if state is None:
            logger.warning(
                "HarvestModule: motor_states 未受信のため準備姿勢への移動をスキップ"
            )
            return
        pose_result = ik_skill.solve(xyz, list(state.position))
        if pose_result is None:
            logger.warning(
                f"HarvestModule: 準備姿勢 target_torso={xyz} へのIKが解けず "
                "移動をスキップ（休憩姿勢のまま続行 — front方式の align が失敗 "
                "する可能性が高い点に注意）"
            )
            return
        logger.info(f"HarvestModule: 準備姿勢(IK)へ移動開始 target_torso={xyz}")
        self._move_arm_to_pose(ik_skill, pose_result.arm14)

    def _move_arm_to_pose(self, ik_skill: Any, arm14_goal: list[float]) -> None:
        """14関節目標(左7+右7)へ、``return_to_rest.py`` と同じ多段補間・控えめ速度で移動する。"""
        from dimos.msgs.sensor_msgs.JointState import JointState as _JointState3
        from dimos.robot.unitree.g1.harvest.return_to_rest import make_return_to_rest_fn

        def _send_arm14(q14: list[float]) -> None:
            self.arm_target.publish(
                _JointState3(
                    name=list(ik_skill.joint_names),
                    position=[float(x) for x in q14],
                    velocity=[0.0] * len(q14),
                    effort=[0.0] * len(q14),
                )
            )

        def _get_measured29() -> list[float]:
            with self._lock:
                s = self._latest_state
            return list(s.position) if s is not None else [0.0] * 29

        move_fn = make_return_to_rest_fn(
            send_arm=_send_arm14,
            get_measured=_get_measured29,
            rest_q_getter=lambda: arm14_goal,
        )
        ok = move_fn()
        logger.info(f"HarvestModule: 準備姿勢への移動{'完了' if ok else '失敗'}")

    def _await_first_frames(self, timeout_s: float) -> None:
        """ヘッド（ACT 把持が有効な場合は右手首も）カメラフレームが届くまでブロックし、
        フローが画像なしで開始しないようにする。"""
        import time

        need_wrist = self.config.use_act_grasp
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            with self._lock:
                have_head = self._latest_image is not None
                have_wrist = self._latest_wrist is not None or not need_wrist
            if have_head and have_wrist:
                logger.info("HarvestModule: 最初のカメラフレーム受信 — フロー開始")
                return
            time.sleep(0.1)
        logger.warning(
            f"HarvestModule: カメラフレームが {timeout_s}s 以内に届かなかった "
            f"(head={self._latest_image is not None}, wrist_needed={need_wrist}, "
            f"wrist={self._latest_wrist is not None}) — フローを開始します"
        )

    def _run(self) -> None:
        try:
            final = self._app.invoke(
                initial_state(), {"recursion_limit": self.config.recursion_limit}
            )
            logger.info(f"HarvestModule: 収穫フロー完了 — picks={final.get('picks')}")
        except Exception:
            logger.exception("HarvestModule: 収穫フローでエラーが発生しました")

    @rpc
    def stop(self) -> None:
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)  # ダミーフローは素早く終了; それ以外はデーモン
            self._thread = None
        if self._voice is not None and hasattr(self._voice, "stop"):
            self._voice.stop()
        self._voice = None
        super().stop()


__all__ = ["HarvestModule", "HarvestModuleConfig"]
