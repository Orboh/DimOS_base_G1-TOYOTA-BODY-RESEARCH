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

import os
import threading
from threading import Thread
from typing import Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Twist import Twist
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
    max_search_advances: int = 3       # 前進探索ステップ上限（超えるとランを終了）
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
    cut_close_q: float = 4.4   # [rad] 切断時のグリッパ閉じ位置
    blade_max_q: float = 5.2   # [rad] 刃保護の上限（機械限界 5.4 の手前）
    # ZED→torso のハンドアイ外部パラメータ（重心3D を IK の torso フレームへ変換）。
    # 空 = 未校正（Step 4 で配線）。形式は [x,y,z, qx,qy,qz,qw]（torso<-camera）。
    cam_to_torso_xyzquat: str = ""
    # §6 実機安全（実機動作が有効な場合に使用）。ファイル E-stop: `touch` で一時停止。
    safety_estop_file: str = "/tmp/okra_estop"
    torque_limit: float = 0.0  # [N·m] アームトルク接触ガード; 0 = OFF（要チューニング）


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
    depth_image: In[Image]  # ZED 深度画像（LIVE + use_zed_depth）; 未接続時は仮定深度にフォールバック
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
        self._latest_wrist: Image | None = None
        self._latest_state: JointState | None = None
        self._latest_gripper: float = 0.0

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
            except Exception as exc:  # noqa: BLE001
                logger.warning("G1 speaker init_dds=True failed; retry init_dds=False", error=str(exc))
                try:
                    return make_g1_playstream_announcer(init_dds=False)
                except Exception as exc2:  # noqa: BLE001
                    logger.warning("G1 speaker unavailable; using console voice", error=str(exc2))
        return CallableAnnouncer(lambda text: logger.info(f"🔊 {text}"))

    def _build_safety_checks(self) -> list[SafetyCheck]:
        """実機動作が有効な場合は §6 実機チェック; それ以外はダミーの常時安全チェック。"""
        # use_ik_grasp_sequence moves the arm (IK reach + cut) even with use_act_grasp=False
        # (ACT-free mode, see run()) -- must count as real motion for §6 checks too.
        real_motion = (
            self.config.use_act_grasp or self.config.use_ik_grasp_sequence or self.config.use_base_move
        )
        if not real_motion:
            return [SafetyCheck("dummy_person_clear", lambda: True)]
        from dimos.robot.unitree.g1.harvest.safety_checks import FileEStop, make_torque_check

        checks = [FileEStop(self.config.safety_estop_file).as_check()]
        if self.config.torque_limit > 0:
            checks.append(make_torque_check(lambda: self._latest_state, limit=self.config.torque_limit))
        logger.info(
            f"SafetyMonitor real checks: file e-stop={self.config.safety_estop_file!r} "
            f"(touch to pause), torque_limit={self.config.torque_limit}"
        )
        return checks

    def _parse_cam_to_torso(self, spec: str):
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
            logger.warning(f"cam_to_torso_xyzquat needs 7 values (x,y,z,qx,qy,qz,qw); got {len(vals)}; ignoring")
            return None
        tx, ty, tz, qx, qy, qz, qw = vals
        # quaternion(xyzw) -> 回転行列（torso<-camera）
        n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5 or 1.0
        qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
        rot = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        trans = np.array([tx, ty, tz])

        def _to_torso(p_cam):
            return list(rot @ np.asarray(p_cam, dtype=float) + trans)

        return _to_torso

    def _on_image(self, image: Image) -> None:
        with self._lock:
            self._latest_image = image

    def _on_depth(self, image: Image) -> None:
        with self._lock:
            self._latest_depth = image

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
            depth_note = "depth=assumed(0.45m)"
            if self.config.use_zed_depth:
                import numpy as np

                self.register_disposable(Disposable(self.depth_image.subscribe(self._on_depth)))
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
                        publish_gripper=self.gripper_target.publish,
                        act_endpoint=self.config.act_endpoint,
                        max_steps=self.config.grasp_max_steps,
                        right_arm_only_7d=self.config.act_right_arm_only_7d,
                    )

                ik_skill = IkApproachSkill()
                cam_to_torso = self._parse_cam_to_torso(self.config.cam_to_torso_xyzquat)

                def _ik_solve(okra: Any) -> Any:
                    """対象オクラの重心(pos_3d)→torso 3D→右腕 IK。解けなければ None。"""
                    pos = getattr(okra, "pos_3d", None) or {}
                    p = [float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))]
                    target_torso = cam_to_torso(p) if cam_to_torso is not None else p
                    with self._lock:
                        state = self._latest_state
                    if state is None:
                        logger.warning("[ik-grasp] no motor_states yet; cannot solve IK")
                        return None
                    return ik_skill.solve(target_torso, list(state.position))

                # 切断可否ゲート: verify_fn（moondream）を流用。未配線なら None=常許可。
                grasp_override = GraspSequence(
                    ik_solve=_ik_solve,
                    publish_arm=self.arm_target.publish,
                    act_module=act_module,
                    cut_ok_fn=verify_fn,
                    publish_gripper=self.gripper_target.publish,
                    q_close=self.config.cut_close_q,
                    q_blade_max=self.config.blade_max_q,
                )
                grasp_note = "grasp=IK->ACT->cut(seq)" if act_module is not None else "grasp=IK->cut(no-ACT)"
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
                    publish_gripper=self.gripper_target.publish,
                    act_endpoint=self.config.act_endpoint,
                    max_steps=self.config.grasp_max_steps,
                    right_arm_only_7d=self.config.act_right_arm_only_7d,
                )
                grasp_note = "grasp=okra-ACT(2cam)"

            skills, grasp_module = build_live_harvest_skills(
                frame_getter=lambda: self._latest_image,
                target_classes=targets,
                detect_fn=detect_override,
                verify_fn=verify_fn,
                move_cmd=move_cmd,
                grasp_module=grasp_override,
                next_station_fn=next_station_override,
                depth_getter=depth_getter,
                yolo_model=self.config.yolo_model,
            )
            mode = f"LIVE — {detect_note}; {depth_note}; {verify_note}; {move_note}; {grasp_note}"

        # 実機動作が有効な場合は §6 実機チェック（ファイル E-stop + トルク）; それ以外はダミー。
        self._monitor = SafetyMonitor(
            self._build_safety_checks(),
            on_pause=lambda reason: grasp_module.stop(),
            announcer=voice,
        )
        self._monitor.start()
        self._app = build_harvest_graph(
            skills, HarvestConfig(), announcer=voice, safety=self._monitor.gate
        )
        # カメラは別ワーカー/プロセスからストリーミングされる — 最初のフレームが届くまで待機し、
        # フロー最初の検出で空画像を掴まないようにする
        # （そうしないと、フレーム到着前に picks=0 で終了してしまう）。
        if not self.config.use_dummy:
            self._await_first_frames(self.config.first_frame_timeout_s)
        self._thread = Thread(target=self._run, daemon=True, name="okra-harvest")
        self._thread.start()
        logger.info(f"HarvestModule 起動 — {mode}")

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
        except Exception:  # noqa: BLE001
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
