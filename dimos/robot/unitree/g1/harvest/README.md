# Okra harvest — LangGraph workflow orchestrator (G1)

Drives the Unitree G1 through the okra-harvest procedure defined in
`okra_harvest_workflow.md`. The **sequence is fixed in code** (a LangGraph
`StateGraph`); only the **judgment** steps (detection, ripeness, target choice,
harvest verification) defer to an LLM/VLM. This is deliberately *not* a free
ReAct loop — the model cannot skip or reorder phases.

LangGraph is already a DimOS dependency (it backs `McpClient`'s `create_agent`),
so this adds no new dependency.

## Files

| File | Role |
|---|---|
| `blackboard.py` | World state (handbook §3) as a LangGraph `State`; `Box3D` reach/FOV volumes; `HarvestConfig` holds all tunable thresholds + geometry (no magic numbers in nodes). |
| `skills.py` | `HarvestSkills` protocol = the robot/perception capabilities the graph drives; `MockHarvestSkills` = a **spatial 3D** field for dry runs / tests. |
| `announce.py` | Spoken **Japanese** status (handbook §6 HMI): `Announcer` interface + `NullAnnouncer`/`RecordingAnnouncer`/`CallableAnnouncer` and the fixed phrase templates. |
| `real_skills.py` | Real-robot wiring: `DimosHarvestSkills` (adapts the protocol to live DimOS calls; `relative_move` implemented) + `make_g1_speaker_announcer` (G1 onboard TTS). Not runtime-verified. |
| `detect_yolo.py` | Interim `detect_okra` via the DimOS YOLO detector (head camera). `make_yolo_detect_okra(...)`. ⚠️ Stock `yolo11n.pt` has no "okra" class — use a proxy class now / okra-fine-tuned weight later; 3D + ripeness are placeholders pending calibration. |
| `safety.py` | Background `SafetyMonitor` (§6) + `PauseGate`: parallel supervisor of `SafetyCheck`s that trips a gate (and stops the running action) on a hazard and clears it when safe. The graph consults the gate at each motion node. |
| `dummy_skills.py` | ⚠️ **DUMMY** full `HarvestSkills` (no robot) for end-to-end bring-up. `DummyGraspModule` is **stoppable** so the SafetyMonitor can cancel a reach mid-action; everything logs `[DUMMY]`. `make_vlm_verify_harvest` wires verify to a VLM. |
| `harvest_module.py` | `HarvestModule` — a deployable DimOS Module that runs the whole flow on `start()`. Backs the `unitree-g1-okra-harvest` blueprint (`dimos run`). Defaults to DUMMY skills. |
| `g1_speaker.py` | Japanese speech via the G1 speaker: `synth_pcm_jp` (pyopenjtalk, **local**) → `AudioClient.PlayStream`. `G1SpeakerAnnouncer` (non-blocking queue, cached). Onboard TTS can't do Japanese; this synthesises off-board and streams the PCM. Needs `pyopenjtalk` + `scipy`. |
| `ollama_vlm.py` | `verify_harvest` via a **local Ollama vision model** (`make_ollama_verify`, default `moondream`; swap to `qwen2.5vl`). Sends the head frame + a yes/no prompt to Ollama. Fail-safe (no frame / Ollama down → False). |
| `graph.py` | `build_harvest_graph(skills, config, announcer)` — the `StateGraph`: nodes = phases, edges = the fixed sequence; conditional edges = the Verify gate, the §7 retry, and the §5 grasp/approach/sweep decision. |
| `run_demo.py` | Dry-run against a mock okra row; prints the phase + base-move trace. |
| `test_harvest_graph.py` | In-reach pick, depth approach (too far / too close), left strafe, height skip, sweep-discovery, termination, retry recovery, give-up. |

## Run

```bash
# Whole flow, DUMMY skills, no robot (logs [DUMMY] + 🔊):
dimos run unitree-g1-okra-harvest

# LIVE: REAL head-cam YOLO detect + Japanese G1 speaker + Ollama-vision verify.
# move/grasp/nav still [LIVE-TODO] (no arm motion yet). Needs the robot +
# NX teleimager + ROBOT_INTERFACE + local Ollama (`ollama pull moondream`):
dimos run unitree-g1-okra-harvest-live

# Standalone dry-run script + tests:
.venv/bin/python -m dimos.robot.unitree.g1.harvest.run_demo
.venv/bin/python -m pytest dimos/robot/unitree/g1/harvest/ -q
```

## Spatial model (3D)

Robot base frame, metres: `x` lateral (+right), `y` depth (+forward), `z` height
(+up). A fruit is graspable only inside the `reach` `Box3D`. When nothing is in
reach:

- **reposition** (an okra is visible but out of the box): move the base so the
  fruit lands at the box centre — `move = position − centre`, so too-far → +forward,
  too-close → −forward (backs off the ridge, clamped by `standoff_min`), left/right → strafe.
  Re-detect after moving to correct the estimate ("compute once, then verify").
- **advance_left** (no fruit in view): harvest progresses RIGHT→LEFT, so sweep
  left to discover more. (Reach stays on the right — the okra-ACT arm/Dex1 is the
  right one — so moving left brings unpicked left-side okra into the right-side
  reach.)
- **revisit** (swept enough, but okra were left behind): the agent tracks its own
  base displacement (odometry) and remembers every ripe okra it has seen
  (`pending`, in the odometry frame). If a fruit was pushed out of view before
  being picked, it returns to that remembered position to collect it — so the
  one-way sweep never abandons fruit. Bounded by `max_revisits`. Done only when
  swept enough AND nothing is pending.
- **height**: the G1 cannot squat in this build, so an okra whose `z` is outside
  the reach box is skipped (`skipped_height`), not chased.

## Graph

```
detect ─▶ select ─┬─ in-reach target ──▶ grasp ─▶ verify ─┬ ok ─▶ record ─┐
                  │                                        │ retry≤N ─▶ grasp
                  │                                        └ exhausted ─▶ give_up ─▶ detect
                  ├─ out of reach (z ok) ─▶ reposition ───────────────────────────▶ detect
                  └─ nothing in view ─────▶ advance_left (right→left) ─────────────▶ detect
                                               │ empty ≥ N
                                               ├─ pending left behind ─▶ revisit ─▶ detect
                                               ▼ nothing pending
                                       next_station ─(more)─▶ detect
                                                    └─(none)─▶ finish ─▶ END
record ─(basket full)─▶ swap_basket ─▶ detect      record ─(else)─▶ detect
```

Figure (regenerate locally, no network): `~/Pictures/okra_langgraph.{png,dot,mmd}`.

## Spoken announcements (Japanese)

When the agent makes a decision or the world state changes, it speaks a fixed
Japanese line (handbook §6 HMI). The phrasing is templated, not LLM-generated —
status must be predictable and free. Announced events: start, grasp decision,
height skip, approach direction (前/後ろ/左/右), re-grasp, each pick + count,
"searching" sweep, "going back" revisit, moving to the next station, basket
swap, give-up, and the final summary.

The speaker is injected, so it is silent and testable by default:

```python
from dimos.robot.unitree.g1.harvest import build_harvest_graph, CallableAnnouncer

# Dry run / tests: RecordingAnnouncer captures the text (no audio).
# Real robot: wrap a speak() callable that targets the G1 speaker.
app = build_harvest_graph(skills, cfg, announcer=CallableAnnouncer(speak))
```

Japanese on the G1 speaker (verified on the real robot):
- The onboard `AudioClient.TtsMaker(text, speaker_id)` does **not** speak Japanese
  (it sounds English for all speaker_ids) — so we synthesise off-board.
- `g1_speaker.py`: `pyopenjtalk` (local, no network) → 16 kHz mono PCM →
  `AudioClient.PlayStream`. Use a UNIQUE `stream_id` per utterance (a reused id
  plays silent). `make_g1_playstream_announcer(nic)` returns the announcer; the
  live blueprint enables it via `HarvestModule(use_g1_speaker=True)`.
- Quick checks: `scripts/verify_g1_speaker.py` (onboard TTS sweep) and
  `scripts/verify_g1_playstream.py` (off-board synth → PlayStream).

## Safety monitor (§6)

`safety.py` is a **parallel supervisor** that runs alongside the workflow and can
preempt it (handbook §6). Cheap checks (person proximity, contact,
self-diagnosis, balance) run every tick; expensive checks (a VLM "is the task
still going right?" judgement) run on a slower cadence.

```python
from dimos.robot.unitree.g1.harvest import SafetyMonitor, SafetyCheck, build_harvest_graph

monitor = SafetyMonitor(
    checks=[
        SafetyCheck("person",  person_is_clear),            # cheap, every tick
        SafetyCheck("contact", no_unexpected_contact),      # cheap
        SafetyCheck("vlm_task", task_looks_ok, expensive=True),  # throttled VLM
    ],
    on_pause=lambda reason: grasp_module.stop(),  # stop the running ACT/cut Module
    on_resume=lambda: None,
    announcer=voice,                              # speaks 危険を検知… / 安全を確認…
)
monitor.start()
app = build_harvest_graph(skills, announcer=voice, safety=monitor.gate)
```

On a hazard the monitor trips `monitor.gate`; the graph **blocks at the next
motion node** (`gate.checkpoint()` in detect/grasp/reposition/advance/revisit/
next_station/swap) and the `on_pause` hook stops whatever is running. When all
checks are safe again the gate clears and the workflow resumes (re-observing from
`detect`). Failing/raising checks are treated as unsafe (fail-safe).

> ⚠️ For `on_pause` to actually stop a grasp, the real `grasp_okra` must run the
> okra-ACT as a **stoppable Module** (start/stop + `_stop_event`), not a blocking
> call — a DimOS `@skill` cannot be interrupted mid-run (design v0.7 / IO design v1).

## Current scope and known limitations

Implemented: handbook **Phases 1→8** — per-station detect→pick with the §5
approach/sweep/revisit movement (odometry-tracked, so left-behind fruit is
collected), **navigation between stations** (`go_to_next_station`), and **basket
transport/swap** when full (`swap_basket`). Fully verified in the mock.

Deferred / not yet done:
- Pedicel cutting — **the cutter is not yet on the robot**; the MVP assumes
  grasp-and-pull with the Dex1 gripper only.
- §6 background safety monitor / interrupt (`look_out_for` → preempt).

## Wiring the real robot

`real_skills.py` is the wiring harness — the graph is unchanged. ⚠️ It is not
runtime-verified (needs the G1 + live DimOS modules); the operator launches
robot motion, this only assembles the wiring.

```python
from dimos.robot.unitree.g1.harvest import (
    build_harvest_graph, build_dimos_harvest_skills, make_g1_speaker_announcer,
    initial_state,
)

skills = build_dimos_harvest_skills(
    move_cmd=g1.move,           # G1 velocity move (vx, vy, vyaw, duration)
    detect_fn=detect_okra,      # head-cam VLM + depth -> list[Okra] (rel pos_3d, ripeness, reachable)
    grasp_fn=run_act_grasp,     # one okra-ACT episode (unitree-g1-act-arm) + Dex1 force
    verify_fn=check_harvest,    # Dex1 hold + VLM "picked?"
    next_station_fn=go_next,    # nav route planning; False when field done
    swap_fn=swap_basket,        # nav to collection point, swap, return
)
voice = make_g1_speaker_announcer(network_interface="<nic>")  # G1 onboard TTS (Japanese: confirm speaker_id)
app = build_harvest_graph(skills, announcer=voice)
final = app.invoke(initial_state(), {"recursion_limit": 400})
```

`relative_move` is implemented (velocity move). The injected callables each need
a real subsystem — contracts below:

| Skill | Real backing |
|---|---|
| `detect_okra()` | **Interim wired** via `make_yolo_detect_okra` (head-cam YOLO, `detect_yolo.py`). The graph uses `pos_3d` only (not `reachable`), so calibration drives grasping. Stock weights = proxy class; swap in okra-fine-tuned weight + real intrinsics/depth + ripeness classifier for production. |
| `relative_move(lateral, forward)` | DimOS navigation skill (`relative_move` / `move`). |
| `go_to_next_station()` | DimOS nav route planning (`navigate_to` next work pose); False when the field is done. |
| `swap_basket()` | Navigate to the collection point, swap an empty basket, return. |
| `grasp_okra(okra, force)` | The okra-ACT manipulation stack (`unitree-g1-act-arm`, branch `feat/g1-act-stage-b`); `force` → Dex1 (`set_gripper`). |
| `verify_harvest()` | **Wired** to a local Ollama vision model (`make_ollama_verify`, `moondream`/`qwen2.5vl`) — frame + "picked?" → yes/no. (Future: + Dex1 hold state.) |
| `record_harvest(rec)` | `dimos/memory2` (handbook §9 record). |
| `announcer` (speak) | G1 `AudioClient.TtsMaker` / `PlayStream`, or DimOS `SpeakSkill` — see "Spoken announcements". |

> Project rule: real base/arm motion is launched by the operator, not by this
> code. A real `HarvestSkills` should make the *decision*; the operator runs the
> G1 motion command (see the okra-ACT `SETUP.md`).

## Design references

- `okra_harvest_workflow.md` — the procedure (single source of truth).
- Memory `g1-okra-harvest-workflow-langgraph` — the design decision + skill gap.
