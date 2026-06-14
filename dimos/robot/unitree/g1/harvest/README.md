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
| `graph.py` | `build_harvest_graph(skills, config, announcer)` — the `StateGraph`: nodes = phases, edges = the fixed sequence; conditional edges = the Verify gate, the §7 retry, and the §5 grasp/approach/sweep decision. |
| `run_demo.py` | Dry-run against a mock okra row; prints the phase + base-move trace. |
| `test_harvest_graph.py` | In-reach pick, depth approach (too far / too close), left strafe, height skip, sweep-discovery, termination, retry recovery, give-up. |

## Run the dry-run (no robot)

```bash
.venv/bin/python -m dimos.robot.unitree.g1.harvest.run_demo
.venv/bin/python -m pytest dimos/robot/unitree/g1/harvest/test_harvest_graph.py -q
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

Wiring `speak` to the **G1's onboard speaker** (`unitree_sdk2py.g1.audio.AudioClient`):
- `AudioClient.TtsMaker(text, speaker_id)` — onboard TTS (simplest; Japanese
  support depends on firmware/`speaker_id`).
- If onboard TTS lacks Japanese: synthesise Japanese audio off-board (the DimOS
  `OpenAITTSNode`, or `PyTTSNode`) and push the PCM with
  `AudioClient.PlayStream(app_name, stream_id, pcm_data)`.
- Or the DimOS `SpeakSkill` (OpenAI TTS, Japanese-capable) — but that plays on
  the **host's** speaker, not the robot's. Use it only if robot-side audio is
  not required. Always speak non-blocking so audio never stalls the loop.

## Current scope and known limitations

Implemented: handbook **Phases 1→8** — per-station detect→pick with the §5
approach/sweep/revisit movement (odometry-tracked, so left-behind fruit is
collected), **navigation between stations** (`go_to_next_station`), and **basket
transport/swap** when full (`swap_basket`). Fully verified in the mock.

Deferred / not yet done:
- Pedicel cutting — **the cutter is not yet on the robot**; the MVP assumes
  grasp-and-pull with the Dex1 gripper only.
- §6 background safety monitor / interrupt (`look_out_for` → preempt).

## Wiring the real robot (next milestone)

Provide a concrete `HarvestSkills` implementation; the graph is unchanged.

| Skill | Real backing |
|---|---|
| `detect_okra()` | VLM detection (`ask_vlm` / `nav_vlm` in `dimos/perception/detection/module3D.py`) + depth → fills `Okra.pos_3d` (relative x/y/z), `ripeness`, `reachable`. |
| `relative_move(lateral, forward)` | DimOS navigation skill (`relative_move` / `move`). |
| `go_to_next_station()` | DimOS nav route planning (`navigate_to` next work pose); False when the field is done. |
| `swap_basket()` | Navigate to the collection point, swap an empty basket, return. |
| `grasp_okra(okra, force)` | The okra-ACT manipulation stack (`unitree-g1-act-arm`, branch `feat/g1-act-stage-b`); `force` → Dex1 (`set_gripper`). |
| `verify_harvest()` | Dex1 hold state + a VLM "is it picked?" check. |
| `record_harvest(rec)` | `dimos/memory2` (handbook §9 record). |
| `announcer` (speak) | G1 `AudioClient.TtsMaker` / `PlayStream`, or DimOS `SpeakSkill` — see "Spoken announcements". |

> Project rule: real base/arm motion is launched by the operator, not by this
> code. A real `HarvestSkills` should make the *decision*; the operator runs the
> G1 motion command (see the okra-ACT `SETUP.md`).

## Design references

- `okra_harvest_workflow.md` — the procedure (single source of truth).
- Memory `g1-okra-harvest-workflow-langgraph` — the design decision + skill gap.
