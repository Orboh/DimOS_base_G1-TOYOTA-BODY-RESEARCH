# Okra-ACT Upper-Body Pick — End-to-End Setup Guide

**Task:** Run the okra ACT pick-up policy on a real Unitree G1 via DimOS, from
a fresh control laptop. The policy controls the upper body (14 arm joints + right
Dex1 gripper) over `rt/arm_sdk`; legs stay on the onboard balance controller
(sport mode is **not** released).

Related: [STAGE_B_PLAN.md](STAGE_B_PLAN.md)

---

## What cloning this repo gives you vs. what you must set up

Cloning the repo is **not sufficient**. The table below shows what each part
requires.

| Component | In this repo? | Manual setup needed? |
|---|---|---|
| DimOS modules (ActBridge, G1ArmSdkConnection, G1GripperConnection, TeleimagerCamera) | Yes | No (code is here) |
| dimos venv (`.venv`, Python 3.12, uv-managed) | No | `scripts/install.sh` + cyclonedds source build + teleimager install |
| ACT inference venv (`.venv_act`, Python 3.11, lerobot 0.4.1) | No | `uv venv` + `requirements-act.txt` |
| Policy checkpoint (`sotata/act-okura-pick-06102026`) | No | HuggingFace download on first run |
| Dataset stats (`Orboh/okura-sub-lerobot`) | No | HuggingFace download on first run |
| teleimager-server on the NX Jetson | No | conda env + `scripts/install_nx_teleimager_service.sh` |

---

## Architecture

Three pieces interact over two separate Python venvs and the robot LAN.

```
Control laptop
┌──────────────────────────────────────────────────────────┐
│  .venv (Python 3.12, uv, dimos)                         │
│                                                          │
│  TeleimagerCamera ──color_image──▶ ActBridge             │
│  G1ArmSdkConnection ─motor_states─▶ ActBridge            │
│  G1GripperConnection ─right_gripper_state─▶ ActBridge    │
│                                                          │
│  ActBridge ──ZMQ REQ tcp://127.0.0.1:5701──▶ act_service │
│  ActBridge ──arm_target──────────▶ G1ArmSdkConnection    │
│  ActBridge ──gripper_target──────▶ G1GripperConnection   │
│                                                          │
│  .venv_act (Python 3.11, uv, lerobot 0.4.1)             │
│  scripts/act_service.py --serve                          │
│  (preprocessor/postprocessor from Orboh/okura-sub-lerobot)│
└──────────────────────────────────────────────────────────┘
        │ CycloneDDS 0.10.5 (source build, robot LAN NIC)
        ▼
G1 Robot (192.168.123.164)
  rt/lowstate  (←  arm position source)
  rt/arm_sdk   (→  14 arm joint targets + weight)
  rt/dex1/right/state  (←  right Dex1 measured q)
  rt/dex1/right/cmd    (→  right Dex1 target q)

NX Jetson (on the robot, 192.168.123.164)
  teleimager-server --rs
  tcp://192.168.123.164:55555  (head D435i frames, 640x480 RGB ~30 fps)
  tcp://192.168.123.164:60000  (config REQ-REP)
```

**Observation layout (16-dim):**
- `[0:7]`  left arm (dimos motor index 15–21)
- `[7:14]` right arm (dimos motor index 22–28)
- `[14]`   left gripper — constant 0.0 (hardware absent; matches training)
- `[15]`   right gripper — measured from `rt/dex1/right/state`

**Action layout (16-dim):** same index order; `[14]` (left gripper) is dropped
before sending to hardware.

---

## Step A — Clone the repo

```bash
git clone https://github.com/Orboh/DimOS_base_G1-TOYOTA-BODY-.git ~/Desktop/dimos-hackathon
cd ~/Desktop/dimos-hackathon
git checkout feat/g1-act-stage-b   # or main once merged
```

---

## Step B — dimos venv (Python 3.12)

### B-1. Run the installer

```bash
bash scripts/install.sh
```

The interactive installer installs system packages (apt), creates `.venv` via
`uv`, and installs dimos. Follow the prompts and select the `unitree` extra.

After this step, confirm the venv Python version:

```bash
.venv/bin/python --version   # must print Python 3.12.x
```

**CRITICAL — always use `.venv/bin/dimos`, never the global `dimos`.**
The global `~/.local/bin/dimos` is Python 3.10. Launching it produces a
`_ARRAY_API not found` numpy/matplotlib crash on import. The repo venv launcher
is the only supported path.

### B-2. Replace cyclonedds with the 0.10.5 source build

This step is **not performed by the installer** and is required for real-robot
use. The pip wheel ships libddsc 11.0.1, which silently drops `rt/lowstate`
samples — the robot will appear connected but no joint state is received.

Build cyclonedds 0.10.5 from source and install the Python binding against it.
The C-library build commands below are a reference — the working install on this
machine lives at `~/cyclonedds-install`; cross-check against it (and the dimos
cyclonedds notes) if anything differs:

```bash
# Install build deps (Ubuntu)
sudo apt-get install -y cmake build-essential libssl-dev

# Build the C library
git clone --branch 0.10.5 https://github.com/eclipse-cyclonedds/cyclonedds.git ~/cyclonedds-src
cmake -B ~/cyclonedds-build -S ~/cyclonedds-src -DCMAKE_INSTALL_PREFIX=~/cyclonedds-install
cmake --build ~/cyclonedds-build -j$(nproc)
cmake --install ~/cyclonedds-build

# Install the Python binding against the correct C library.
# NOTE: the dimos venv is uv-managed and has NO pip — use `uv pip ... --python`.
CYCLONEDDS_HOME=~/cyclonedds-install \
  uv pip install --python .venv/bin/python --no-binary cyclonedds cyclonedds==0.10.5
```

A warning that `unitree-sdk2py` wants `cyclonedds==0.10.2` is safe to ignore —
unitree-sdk2py works against 0.10.5.

### B-3. Install teleimager into the dimos venv

teleimager (https://github.com/Orboh/teleimager) provides `ImageClient`, used
by `TeleimagerCamera`. It is a separate repo and must be installed into the dimos
venv explicitly.

```bash
# Clone teleimager next to the inference workspace
git clone https://github.com/Orboh/teleimager.git ~/act-okura/teleimager

# Install logging_mp (required by teleimager) then teleimager itself
uv pip install --python .venv/bin/python logging_mp
uv pip install --python .venv/bin/python --no-deps -e ~/act-okura/teleimager
```

`--no-deps` avoids dependency conflicts caused by teleimager declaring
`requires-python <3.12`; the `ImageClient` works fine on 3.12 in practice.

Verify the install:

```bash
.venv/bin/python -c "from teleimager.image_client import ImageClient; print('teleimager OK')"
```

---

## Step C — ACT inference venv (Python 3.11)

This venv is **separate from the dimos venv**. lerobot and torch must not be
mixed into dimos.

```bash
uv venv --python 3.11 ~/act-okura/.venv_act
uv pip install --python ~/act-okura/.venv_act/bin/python \
    -r scripts/requirements-act.txt
```

`requirements-act.txt` pins `lerobot==0.4.1` (the version that routes
normalization through `make_pre_post_processors`) and CUDA 12.6 torch via pinned
`nvidia-*-cu12` wheels.

**Do not change the lerobot version.** lerobot 0.4.1 moved normalization out of
the policy and into a preprocessor/postprocessor pipeline built from the dataset
stats. An older or different version feeds raw (un-normalized) values into the
policy and returns normalized-space outputs as raw radians — the arms move but to
garbage targets (the "moves but wrong" bug; root cause documented in FleetSeek
[exp_01KTXQ8XKD0TADTDZ8WRQ0NVD8](https://web-ebon-zeta-33.vercel.app/experience/exp_01KTXQ8XKD0TADTDZ8WRQ0NVD8)).

---

## Step D — HuggingFace assets

The policy and the dataset are downloaded automatically on first run. They live
in the HuggingFace cache (`~/.cache/huggingface/`).

| Asset | HuggingFace ID | Size |
|---|---|---|
| Policy checkpoint | `sotata/act-okura-pick-06102026` | ~200 MB |
| Dataset (normalization stats + task string) | `Orboh/okura-sub-lerobot` | ~2 GB |

The dataset is in the private Orboh org. If the download fails with a 401 error:

```bash
~/act-okura/.venv_act/bin/python -c "from huggingface_hub import login; login()"
```

To pre-cache before going to the lab (requires network access):

```bash
~/act-okura/.venv_act/bin/python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
LeRobotDataset('Orboh/okura-sub-lerobot')
"
```

---

## Step E — Robot / NX side

### E-1. teleimager-server (boot service)

The NX Jetson must serve the head D435i via teleimager-server on boot. Run this
**from the laptop** (it SSHs into the NX):

```bash
scripts/install_nx_teleimager_service.sh
# default target: unitree@192.168.123.164
# for a different address: scripts/install_nx_teleimager_service.sh unitree@<ip>
```

This installs `g1-teleimager.service` (systemd unit on the NX) and disables the
old `g1-cam-publisher` (GEAR-SONIC). The D435i is single-occupancy: both cannot
run at the same time.

**Prereq on the NX:** the `teleimager_relobot` conda environment with
`teleimager-server` on PATH, and `cam_config_server.yaml` configured for the
head D435i (480x640). See the teleimager repo README for the one-time camera
config setup required for new or reflashed NX boards.

After installation, verify the ports are open from the laptop:

```bash
# Should open in < 1s each
timeout 3 bash -c "echo > /dev/tcp/192.168.123.164/60000" && echo "port 60000 OK"
timeout 3 bash -c "echo > /dev/tcp/192.168.123.164/55555" && echo "port 55555 OK"
```

### E-2. Robot mode

The robot must be in **motion control mode** (self-balancing, not lying down)
before starting the live blueprint. arm_sdk drives joints 12–28 and weight
register 29; legs (0–11) are left to the onboard locomotion controller.

Sport mode is **not** released and must not be released.

---

## Step F — Network / per-laptop config

### F-1. Put the laptop on the robot LAN

The laptop must be on the 192.168.123.x subnet. Wired connection to the robot
LAN switch is recommended (the G1's built-in switch port works).

### F-2. Find your NIC name

```bash
ip -br addr
```

Look for the interface with address `192.168.123.x`. The name differs per
laptop and per USB-Ethernet adapter. On the current team laptop it is
`enx6c1ff771dc67`.

```
# Example output:
enx6c1ff771dc67  UP  192.168.123.10/24 ...
```

`unitree_sdk2py` ignores `CYCLONEDDS_URI`. The NIC **must** be passed via the
`ROBOT_INTERFACE` environment variable. Omitting it lets the SDK auto-select the
default route interface (usually the WiFi adapter), which will not see DDS
traffic on the robot LAN.

---

## Run order (three terminals)

Start these in order and wait for each readiness signal before proceeding.

### Terminal 1 — NX: camera server

If the boot service is installed (Step E-1), teleimager-server starts
automatically. Check its status or restart manually:

```bash
# Check boot service status from the laptop
ssh unitree@192.168.123.164 systemctl status g1-teleimager --no-pager

# Or start manually (without the boot service)
ssh unitree@192.168.123.164 "conda activate teleimager_relobot && teleimager-server --rs"
```

**Sanity-check the camera from the laptop (dimos venv, no robot motion):**

```bash
~/Desktop/dimos-hackathon/.venv/bin/python ~/act-okura/teleimager_source_check.py 192.168.123.164
```

Expected output: `640x480 RGB` frames arriving at ~30 fps.

### Terminal 2 — Laptop: ACT inference service

```bash
cd ~/Desktop/dimos-hackathon
~/act-okura/.venv_act/bin/python scripts/act_service.py --serve
```

Wait for:
```
[act] serving on tcp://127.0.0.1:5701 (Ctrl-C to stop)
```

The service loads the policy and dataset on startup (first run downloads them,
which takes a few minutes). Subsequent starts are fast (cache hit).

### Terminal 3 — Laptop: DimOS blueprint

Replace `<nic>` with your NIC name from Step F-2.

**B0 — dry-run (safe, no motion, recommended first):**

```bash
cd ~/Desktop/dimos-hackathon
ROBOT_INTERFACE=<nic> .venv/bin/dimos run unitree-g1-act-dryrun
```

B0 exercises the full pipeline — camera, state assembly, ACT inference — but
writes nothing to DDS. The arms do not move.

**B1 — LIVE (arms move):**

```bash
cd ~/Desktop/dimos-hackathon
ROBOT_INTERFACE=<nic> .venv/bin/dimos run unitree-g1-act-arm
```

**Camera source override:** the default is teleimager (the format the policy
was trained on). To fall back to the GEAR-SONIC ZMQ publisher:

```bash
ROBOT_INTERFACE=<nic> DIMOS_CAMERA_SOURCE=zmq .venv/bin/dimos run unitree-g1-act-arm
```

**Startup sequence (B1):**
1. `G1ArmSdkConnection` waits for the first `rt/lowstate` (~1 s), captures the
   current arm pose, and slews the arms to the dataset start pose
   (`_INIT_ARM_POSE` defined in the blueprint, left 7 + right 7 joints in rad).
2. `ActBridge` holds inference for `startup_delay_s = 2.5 s` while the arms
   finish the slew.
3. After 2.5 s, closed-loop inference begins at 30 Hz over ZMQ. The arm
   controller runs at 250 Hz and clips each step toward the ACT target
   relative to the measured pose (≤ 20 rad/s), preventing observation OOD drift.

**Stop with a single Ctrl-C.** The stop handler ramps the arm_sdk `weight`
register from 1 to 0 over ~2 s, handing the arms back to the onboard
controller gracefully. Pressing Ctrl-C multiple times in quick succession can
interrupt the ramp-down (NoneType warning in the log) — one press and wait
is enough.

---

## Verification

### Self-test (no robot, no camera)

Confirms that normalization is wired correctly by replaying the first frame from
the dataset and comparing the inferred action to the recorded action.

```bash
~/act-okura/.venv_act/bin/python scripts/act_service.py --selftest
```

Expected output:
```
[selftest] max|err| = 0.0xxx rad  -> OK
```

The error should be below 0.2 rad (typical: ~0.056 rad). The old raw normalization
path was off by 2.21 rad.

### B0 dry-run checks

Look for these log lines to confirm each stage is working:

| Log line | Confirms |
|---|---|
| `TeleimagerCamera first frame received — publishing on color_image size=640x480` | Camera path working |
| `arm_sdk ready (mode_machine=...)` | `rt/lowstate` arriving; NIC correct |
| `Dex1 right gripper ready; holding current q=...` | `rt/dex1/right/state` arriving |
| `[dry-run] ACT action #1: ...` | Full pipeline running; inference OK |
| `arm track: max\|target-measured\|=... DRY` | State assembly OK (no motion) |

### B1 LIVE checks

After the arm slew, look for:

- `arm track: max|target-measured|=0.0x-0.4x rad weight=1.00 LIVE` — arm
  following the ACT target. A value below ~0.5 rad with no growth is healthy.
  A value growing toward 1+ rad each line signals OOD drift; press Ctrl-C.
- Right gripper: `right_gripper_state` in the range 0 to ~5.4 rad (open to
  close). Left gripper target (`action[14]`) is intentionally dropped.

---

## Safety

**Real arm motion is physical and irreversible.**

- Clear a 60 cm radius around both arms before starting B1.
- Keep the Unitree remote in hand with **L2+B** (e-stop) ready at all times.
- The robot must be in **motion control mode** (standing, self-balancing) —
  not lying down, not in sport mode.
- Run B0 dry-run first on every new setup to confirm the camera and state
  pipeline before enabling motion.
- Stop with a **single Ctrl-C** and wait for the weight ramp-down (~2 s).
  The ramp-down is what safely returns arm authority to the onboard controller.
- If the arm tracking error grows uncontrolled, press Ctrl-C immediately. Do
  not attempt to catch the arm.

---

## Troubleshooting

### No `rt/lowstate` received — arm_sdk fails to start

Symptoms: `G1ArmSdkConnection` times out with `No LowState received; cannot
start arm_sdk safely`.

Checks in order:
1. `ROBOT_INTERFACE` is set to the correct NIC name (`ip -br addr`).
2. The laptop has a 192.168.123.x address on that interface.
3. The robot is powered on and standing (motion control mode).
4. Ping the robot: `ping 192.168.123.164`.
5. A stale multicast route can hijack DDS traffic. Check:
   ```bash
   ip route | grep 224.0.0.0
   ```
   If a `224.0.0.0/4 dev lo` route is present, remove it:
   ```bash
   sudo ip route del 224.0.0.0/4 dev lo
   ```
   This was observed to affect the `unitree_lerobot` venv; the dimos venv was
   unaffected in our tests, but it is worth checking on a fresh machine.

### "Moves but wrong" / large flailing arm motion

Root cause: lerobot version mismatch in `.venv_act`. The policy receives
un-normalized inputs and returns normalized-space outputs as raw radians. The
arm moves but to garbage targets (off by ~2.21 rad).

Fix: ensure `.venv_act` has `lerobot==0.4.1`:
```bash
~/act-okura/.venv_act/bin/python -c "import lerobot; print(lerobot.__version__)"
```

If the version differs, rebuild `.venv_act` from `scripts/requirements-act.txt`.

### dimos crashes on import — `_ARRAY_API not found`

You ran the global `dimos` (Python 3.10). Use the repo venv:

```bash
# Wrong:  dimos run ...
# Right:
cd ~/Desktop/dimos-hackathon
.venv/bin/dimos run unitree-g1-act-dryrun
```

### "ACT service timeout" — ActBridge keeps reconnecting

`scripts/act_service.py --serve` is not running, crashed, or is listening on a
different endpoint.

Check Terminal 2 for errors. The service must print `[act] serving on
tcp://127.0.0.1:5701` before the blueprint is started. The ZMQ endpoint is
`tcp://127.0.0.1:5701` (loopback only; laptop and service must be on the same
machine).

### `rt/dex1/right/state` timeout — gripper fails to start

```
No rt/dex1/right/state received; cannot start gripper safely
```

The right Dex1 is not connected or not powered. Check the hand cable. If the
gripper is intentionally absent for a test, use the dry-run blueprint
(`unitree-g1-act-dryrun`) which runs both modules with `publish_cmd=False` and
will still wait for the gripper state.

### TeleimagerCamera receives 0 fps — no camera frames

`teleimager-server` is not running on the NX, or the boot service failed.

Check on the NX:
```bash
ssh unitree@192.168.123.164 journalctl -u g1-teleimager -n 30 --no-pager
```

The most common cause on a new NX is that `cam_config_server.yaml` was not
configured before installing the service, or the `teleimager_relobot` conda
environment is missing. See `scripts/install_nx_teleimager_service.sh` and the
teleimager repo README.

### Ctrl-C produces `NoneType` warning during stop

One Ctrl-C was followed by more key presses before the weight ramp-down (2 s)
completed. The ramp-down calls into `_low_cmd` which was already set to `None`
by a concurrent stop. This is a minor cosmetic issue; the arms are handed back
safely regardless. Use one Ctrl-C and wait.

---

## File reference

| Path | Purpose |
|---|---|
| `dimos/robot/unitree/g1/act/act_bridge.py` | Assembles 16-dim observation, ZMQ REQ to act_service, publishes arm_target / gripper_target |
| `dimos/robot/unitree/g1/act/g1_arm_sdk_connection.py` | Publishes rt/arm_sdk at 250 Hz; clip-from-measured anti-drift; weight ramp |
| `dimos/robot/unitree/g1/act/g1_gripper_connection.py` | rt/dex1/right/cmd publisher + state subscriber |
| `dimos/robot/unitree/g1/act/dds_init.py` | Thread-safe one-shot CycloneDDS channel factory init |
| `dimos/robot/unitree/g1/act/STAGE_B_PLAN.md` | Design rationale, task breakdown, verification stages (Japanese) |
| `dimos/robot/unitree/g1/camera/teleimager_camera_module.py` | TeleimagerCamera module — color_image stream from the NX D435i |
| `dimos/robot/unitree/g1/blueprints/manipulation/unitree_g1_act_arm.py` | B1 LIVE blueprint (arms + gripper move) |
| `dimos/robot/unitree/g1/blueprints/manipulation/unitree_g1_act_dryrun.py` | B0 dry-run blueprint (no DDS writes) |
| `scripts/act_service.py` | ACT inference service (ZMQ REP, runs in .venv_act) |
| `scripts/requirements-act.txt` | Frozen deps for .venv_act (lerobot 0.4.1, torch CUDA 12.6) |
| `scripts/install_nx_teleimager_service.sh` | Installs g1-teleimager.service on the NX via SSH |
| `scripts/g1-teleimager.service` | systemd unit file (NX) |
| `scripts/g1-teleimager-run.sh` | Conda activation wrapper for the systemd unit |
