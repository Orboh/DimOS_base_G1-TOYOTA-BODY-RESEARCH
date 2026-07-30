# ZED SDK + Neural Depth Setup — ZED Mini

## System info (checked 2026-07-15)

- **OS**: Ubuntu 24.04.3 LTS
- **GPU**: NVIDIA GeForce RTX 5080 (Blackwell)
- **Driver**: 580.159.03 (supports up to CUDA 13.0)
- **CUDA toolkit (nvcc) found separately installed**: 12.0 (older than driver — fine, not used by ZED SDK install)
- **Camera**: ZED Mini (USB 3.0)
- **Project dir**: `/home/techshare/user/Okra_detaction/ZED_M_Depth_check`

## SDK version chosen

- **ZED SDK 5.4** (latest)
- Build: **Ubuntu 24 / CUDA 13.0 / TensorRT 10.13**
- Why CUDA 13.0: RTX 5080 is Blackwell, needs CUDA ≥ 12.8 for full support; driver already supports CUDA 13.0, so this build matches exactly.
- Installer file (already downloaded into project dir):
  `ZED_SDK_Ubuntu24_cuda13.0_tensorrt10.13_v5.4.0.zstd.run` (~1.6 GB)
- Download source: https://www.stereolabs.com/developers/release (select Ubuntu 24.04 + CUDA 12.8/13.0)

> Note: the SDK binaries are **not** on GitHub. `stereolabs/zed-sdk` on GitHub only has sample code / API wrappers, which require the real SDK to be installed first from stereolabs.com.

## Install steps

1. Install the `zstd` dependency (needed to unpack the installer):
   ```bash
   sudo apt update && sudo apt install -y zstd
   ```
   ✅ Done — confirmed `zstd` present (v1.5.7).

2. Make the installer executable:
   ```bash
   cd /home/techshare/user/Okra_detaction/ZED_M_Depth_check
   chmod +x ZED_SDK_Ubuntu24_cuda13.0_tensorrt10.13_v5.4.0.zstd.run
   ```
   ✅ Done.

3. Run the installer (requires sudo password — must be run directly in your terminal, not by the assistant):
   ```bash
   cd /home/techshare/user/Okra_detaction/ZED_M_Depth_check
   sudo ./ZED_SDK_Ubuntu24_cuda13.0_tensorrt10.13_v5.4.0.zstd.run
   ```
   - Press `q` after the license text
   - Answer `y` to samples / tools / Python API prompts

   **Status: ✅ Done.** First attempt hung on an interactive prompt; retried with silent mode and it completed successfully.
   ```bash
   cd /home/techshare/user/Okra_detaction/ZED_M_Depth_check
   sudo ./ZED_SDK_Ubuntu24_cuda13.0_tensorrt10.13_v5.4.0.zstd.run -- silent
   ```
   Optional silent flags that can be appended: `runtime_only`, `skip_cuda`, `skip_od_module`, `skip_python`, `skip_tools`.

   Python API note: installer printed a manual reinstall command if ever needed:
   ```bash
   python -m pip install --ignore-installed /tmp/selfgz*/pyzed-5.4-cp312-cp312-linux_x86_64.whl
   ```

4. `/usr/local/zed` is owned by `root:zed` with mode `0770` — your user needs to be in the `zed` group to access it without sudo:
   ```bash
   sudo usermod -aG zed $USER
   ```
   ✅ Done. Group membership normally only takes effect after logging out/in or rebooting, but since a reboot wasn't possible yet, commands below are run via `sg zed -c "..."` which applies the group to just that command in the current login session.

   Note: this SDK build (5.4) doesn't ship a `VERSION.txt` in `/usr/local/zed` — that's expected, not an error.

5. Reboot (still outstanding) to make the `zed` group membership permanent for all future shells/sessions, so `sg zed -c` is no longer needed:
   ```bash
   sudo reboot
   ```

## Neural depth model

- AI models (incl. NEURAL depth) are **not bundled** in the installer — downloaded from Stereolabs' servers and optimized for the specific GPU, either automatically on first use or pre-downloaded manually.
- Pre-download / optimize manually:
  ```bash
  sg zed -c "cd /usr/local/zed/tools && ./ZED_Diagnostic --ais 0"
  ```
  ✅ Done. Downloaded and confirmed present in `/usr/local/zed/resources/`:
  - `neural_depth_5.3.model`
  - `neural_depth_light_5.3.model`

  The tool also ran a GPU optimization pass for `objects_performance_3.2` (object detection), completed 100%.

## Testing

1. Plug in ZED Mini via USB 3.0. ✅ Detected — `lsusb` shows `STEREOLABS ZED-M HID Interface`.
2. Confirm camera stream:
   ```bash
   sg zed -c "/usr/local/zed/tools/ZED_Explorer"
   ```
3. Confirm neural depth:
   ```bash
   sg zed -c "/usr/local/zed/tools/ZED_Depth_Viewer"
   ```
   Switch depth mode to **NEURAL** and confirm the depth map renders.

   Note: actual binary name is `ZED_Depth_Viewer` (underscore), not `"ZED Depth Viewer"`.

## Outstanding / next step

- [x] Re-run installer in silent mode — SDK 5.4 installed successfully to `/usr/local/zed`.
- [x] Add user to `zed` group (`sudo usermod -aG zed $USER`) — done; reboot still needed to make it permanent (using `sg zed -c` as a workaround meanwhile).
- [x] Pre-download neural depth model — done, confirmed in `/usr/local/zed/resources/`.
- [x] Python API (`pyzed`) installed into a dedicated `zed` conda env (Python 3.12, matches the SDK 5.4 cp312 wheel). Also has `ultralytics`, `opencv-python`, `python-dotenv`, `torch` (CUDA 13.0) for running the okra detection + depth script.
- [x] Combined YOLO detection + ZED NEURAL depth script written: `finetune_V5/zed_depth_yolo.py` (loads the fine-tuned `output/okra_finetune_v5/weights/best.pt`, overlays per-detection distance in meters read from the ZED depth map).
- [ ] **Blocked: camera hardware fault (not software).** See below.
- [ ] Reboot when convenient to drop the `sg zed -c` workaround (once the camera issue is resolved).

## Known issue: ZED Mini video interface not enumerating (2026-07-15)

- Symptom: `sl.Camera().open()` fails with `CAMERA STREAM FAILED TO START`.
- USB diagnosis (`lsusb`, `journalctl -k`): the camera only ever enumerates its **HID (motion sensor) interface** (`2b03:f681`, "ZED-M HID Interface") at **full-speed (USB 1.1, 12 Mbps)**. The video/UVC interface never appears at all.
- Tried 4 different physical ports across both the USB 2.0 controller (Bus 001, 480M) and confirming the USB 3.0 controller (Bus 002, 10000M, where a flash drive works fine at 5000M) — every attempt gave the identical result (HID only, full-speed, no video interface).
- Since port/bus and (reported) cable changes made no difference and the failure pattern is identical every time, this points to a **hardware fault in the camera's own USB video circuitry** rather than a cabling/port/software problem. The IMU/sensor board still works because it's a separate, low-bandwidth USB function inside the composite device.
- Next step (up to user): test the camera on a different computer to rule out this machine's USB3 controllers, or pursue Stereolabs support/warranty if confirmed faulty.
- Software side is fully ready — once a working camera is connected on a real USB 3.0 port, `zed_depth_yolo.py` (see below) should run immediately.

## Running the okra detection + depth script

```bash
source /home/techshare/miniconda3/etc/profile.d/conda.sh
conda activate zed
cd /home/techshare/user/Okra_detaction/ZED_M_Depth_check/finetune_V5
sg zed -c "$(which python) zed_depth_yolo.py"
```

- `sg zed -c "..."` is required until the machine is rebooted (so the `zed` group membership takes effect for the whole login session); after reboot it can be dropped.
- Press `Q` in the window to quit.
- Make sure no other ZED tool (`ZED_Explorer`, `ZED_Depth_Viewer`, etc.) is running — they hold the camera open exclusively and will cause `open()` to fail with the same error.
