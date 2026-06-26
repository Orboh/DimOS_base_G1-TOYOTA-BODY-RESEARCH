#!/usr/bin/env python3
"""S4 検証: 安全機構（F-11）。計画書 [[12-検証計画-sim]] §6 S4 / §5 F-11。

オクラ3Dモデル・Isaac Sim・実機いずれも不要（安全ロジックだけを確かめる Tier 0 検証）。
3点を確認する:
  (a) FileEStop: ファイルを touch すると停止ゲートが trip し on_pause/アナウンスが出る。
      rm すると clear し on_resume/アナウンスが出る（= 緊急停止と再開）。
  (b) 把持中断: 停止要求が立つと GraspSequence.run_episode が開始拒否/途中中断して False。
  (c) BladeGuard: 切断角に上限超(6.0rad)を渡しても 5.2rad にクランプして publish される。

実行:
  .venv/bin/python docs/sim-setup/verify_s4_safety.py
"""
from __future__ import annotations

import os
import sys
import threading
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from dimos.robot.unitree.g1.harvest import announce
from dimos.robot.unitree.g1.harvest.announce import RecordingAnnouncer
from dimos.robot.unitree.g1.harvest.blackboard import Okra
from dimos.robot.unitree.g1.harvest.grasp_sequence import GraspSequence, _Q_BLADE_MAX
from dimos.robot.unitree.g1.harvest.safety import SafetyMonitor
from dimos.robot.unitree.g1.harvest.safety_checks import FileEStop

ESTOP_PATH = "/tmp/okra_estop_s4test"


def _sol(wait_s: float = 0.0) -> SimpleNamespace:
    """ik_solve の戻り値（arm14/joint_names/wait_s を持つ）の最小ダミー。"""
    return SimpleNamespace(arm14=[0.0] * 14, joint_names=[f"j{i}" for i in range(14)], wait_s=wait_s)


def _okra() -> Okra:
    return Okra(id="s4", pos_3d={"x": 0.30, "y": 0.45, "z": 0.80}, ripeness=1.0, reachable=True)


# --- (a) FileEStop で停止ゲートが trip/clear するか ---------------------------------
def check_file_estop() -> tuple[bool, str]:
    if os.path.exists(ESTOP_PATH):
        os.remove(ESTOP_PATH)
    estop = FileEStop(ESTOP_PATH)
    paused: list[str] = []
    resumed: list[bool] = []
    voice = RecordingAnnouncer()
    mon = SafetyMonitor(
        [estop.as_check()],
        on_pause=lambda r: paused.append(r),
        on_resume=lambda: resumed.append(True),
        announcer=voice,
    )
    mon.step()  # 初期=クリア
    init_clear = (not mon.gate.is_paused()) and mon.gate.checkpoint(timeout=0)

    open(ESTOP_PATH, "w").close()  # touch → 停止
    mon.step()
    tripped = mon.gate.is_paused() and (mon.gate.checkpoint(timeout=0) is False) and len(paused) == 1
    stop_said = any("危険を検知" in s for s in voice.said)

    os.remove(ESTOP_PATH)  # rm → 再開
    mon.step()
    cleared = (not mon.gate.is_paused()) and mon.gate.checkpoint(timeout=0) and len(resumed) == 1
    resume_said = announce.safety_resume() in voice.said

    ok = init_clear and tripped and stop_said and cleared and resume_said
    return ok, f"init_clear={init_clear} tripped={tripped} stop_said={stop_said} cleared={cleared} resume_said={resume_said}"


# --- (b) 停止要求で把持エピソードが拒否/中断するか ---------------------------------
def check_grasp_abort() -> tuple[bool, str]:
    # (b1) 開始前に stop → 即拒否
    seq1 = GraspSequence(ik_solve=lambda o: _sol(0.0), publish_gripper=lambda js: None)
    seq1.stop()
    refused = seq1.run_episode(_okra()) is False

    # (b2) 待機中に stop → 中断（切断に到達しない）
    cuts: list[float] = []
    seq2 = GraspSequence(
        ik_solve=lambda o: _sol(wait_s=1.0),  # 整定待ち中に止める
        publish_gripper=lambda js: cuts.append(js.position[0]),
    )
    result: list[bool] = []
    t = threading.Thread(target=lambda: result.append(seq2.run_episode(_okra())))
    t.start()
    # 待機に入ってから停止要求
    if not _wait_until(lambda: t.is_alive(), 1.0):
        pass
    threading.Event().wait(0.1)
    seq2.stop()
    t.join(timeout=2.0)
    aborted = result == [False] and cuts == []  # 中断＝False かつ 切断未発行

    ok = refused and aborted
    return ok, f"refuse_before_start={refused} abort_mid_wait={aborted}"


def _wait_until(pred, timeout: float) -> bool:
    import time as _t

    end = _t.time() + timeout
    while _t.time() < end:
        if pred():
            return True
        _t.sleep(0.005)
    return pred()


# --- (c) BladeGuard: 切断角クランプ ----------------------------------------------
def check_bladeguard() -> tuple[bool, str]:
    captured: list[float] = []
    over = _Q_BLADE_MAX + 0.8  # 上限(5.2)超の指令 6.0rad
    seq = GraspSequence(
        ik_solve=lambda o: _sol(0.0),
        publish_gripper=lambda js: captured.append(float(js.position[0])),
        cut_ok_fn=None,  # 切断許可
        q_close=over,
    )
    ok_run = seq.run_episode(_okra())
    clamped = bool(captured) and abs(captured[-1] - _Q_BLADE_MAX) < 1e-9
    ok = ok_run and clamped
    return ok, f"commanded={over} published={captured[-1] if captured else None} (上限={_Q_BLADE_MAX})"


def main() -> int:
    results = [
        ("(a) FileEStop 停止/再開", *check_file_estop()),
        ("(b) 停止要求で把持中断", *check_grasp_abort()),
        ("(c) BladeGuard 切断角クランプ", *check_bladeguard()),
    ]
    print("========== S4: 安全機構 検証 ==========")
    all_ok = True
    for label, ok, detail in results:
        all_ok = all_ok and ok
        print(f"  [{'OK ' if ok else 'NG '}] {label}: {detail}")
    print(f"[S4] RESULT: {'PASS ✅' if all_ok else 'FAIL ❌'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
